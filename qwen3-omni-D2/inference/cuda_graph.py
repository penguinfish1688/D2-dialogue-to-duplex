"""Fixed-shape CUDA graphs for the D2 macro inference policy.

The graph boundary follows the model layout instead of the transport packet
shape:

* one Thinker graph appends the fixed ``2k`` environment/self audio prefix;
* one Thinker graph appends a single autoregressive text token;
* one Talker graph appends a single Talker token; and
* two small CodePredictor graphs cover its two-token start and one-token
  residual steps.

The Thinker graphs share one fixed-address cache and the Talker graph owns a
second fixed-address cache.  All masks have the full configured KV width.
Unwritten future slots are always false, so preallocation cannot expose future
keys to attention.

This module intentionally has no eager fallback.  Callers choose whether to
construct it (the CPU reference path does not); once construction is requested,
any unsupported kernel or failed capture raises :class:`CudaGraphCaptureError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from d2_qwen.model.constants import CODEBOOKS, CODEBOOK_SIZE
from d2_qwen.model.utils import embed_on, module_device
from .cache import split_dtype_static_cache
from .streaming import (
    inference_autocast,
    model_compute_dtype,
    record_component_trace,
    sample_logits,
)
from .timing import ForwardTimings, measure_forward
from transformers.cache_utils import Cache


class CudaGraphCaptureError(RuntimeError):
    """A requested D2 CUDA graph could not be captured safely."""


@dataclass(frozen=True, slots=True)
class D2GraphCapacity:
    """Static KV shapes for one latency-specific D2 model."""

    frames_per_unit: int
    native_frames: int

    def __post_init__(self) -> None:
        k = int(self.frames_per_unit)
        frames = int(self.native_frames)
        if k < 1 or frames < 1:
            raise ValueError("CUDA graph dimensions must be positive")
        if frames % k:
            raise ValueError(f"KV capacity {frames} must contain complete {k}-frame macro units")

    @property
    def thinker_tokens(self) -> int:
        return 3 * int(self.native_frames)

    @property
    def talker_tokens(self) -> int:
        return int(self.native_frames) + 1

    @property
    def macro_units(self) -> int:
        return int(self.native_frames) // int(self.frames_per_unit)


def _copy_full_width_mask(target: torch.Tensor, source: torch.Tensor) -> None:
    if source.ndim != 4 or source.shape[:-1] != target.shape[:-1]:
        raise ValueError(
            "CUDA graph mask prefix shape changed: "
            f"source={tuple(source.shape)} target={tuple(target.shape)}"
        )
    live_tokens = int(source.shape[-1])
    if live_tokens > int(target.shape[-1]):
        raise ValueError(f"live KV width {live_tokens} exceeds graph width {target.shape[-1]}")
    target.zero_()
    target[..., :live_tokens].copy_(source.to(device=target.device, dtype=torch.bool))


def _last_hidden(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, (tuple, list)) and output:
        hidden = output[0]
    if not torch.is_tensor(hidden):
        raise TypeError(f"expected a hidden-state tensor, got {type(output)!r}")
    return hidden


def _model_device(model: torch.nn.Module) -> torch.device:
    device = module_device(model)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise CudaGraphCaptureError("D2 CUDA graphs require a CUDA-resident model")
    return device


def _layer_count(model: torch.nn.Module) -> int:
    layers = getattr(model, "layers", None)
    if layers is None or len(layers) < 1:
        raise TypeError("CUDA graph model must expose non-empty .layers")
    configured = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if configured is not None and int(configured) != len(layers):
        raise RuntimeError(f"model layer inventory {len(layers)} != configured {configured}")
    return len(layers)


def _hidden_size(model: torch.nn.Module) -> int:
    value = getattr(getattr(model, "config", None), "hidden_size", None)
    if value is None or int(value) < 1:
        raise TypeError("CUDA graph model config must expose hidden_size")
    return int(value)


@torch.no_grad()
def _set_cache_cursor(cache: Cache, value: int) -> None:
    layers = getattr(cache, "layers", None)
    if not layers:
        raise TypeError("static CUDA graph cache has no layers")
    for layer in layers:
        cursor = getattr(layer, "cumulative_length", None)
        if not torch.is_tensor(cursor):
            raise TypeError("static CUDA graph cache has no tensor cursor")
        cursor.fill_(int(value))


def _uniform_cache_cursor(cache: Cache) -> int:
    """Read a cache cursor for end-of-session validation only."""

    layers = getattr(cache, "layers", None)
    if not layers:
        raise TypeError("static CUDA graph cache has no layers")
    values: list[int] = []
    for layer in layers:
        cursor = getattr(layer, "cumulative_length", None)
        if not torch.is_tensor(cursor):
            raise TypeError("static CUDA graph cache has no tensor cursor")
        values.append(int(cursor.item()))
    if len(set(values)) != 1:
        raise RuntimeError(f"static cache layer cursors diverged: {values}")
    return values[0]


class D2ThinkerCudaGraphs:
    """A fixed ``2k`` prefill graph and one-token graph sharing Thinker KV."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        accepted_hidden_layer: int,
        capacity: D2GraphCapacity,
        dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.capacity = capacity
        self.dtype = dtype
        self.accepted_hidden_layer = int(accepted_hidden_layer)
        layers = _layer_count(model)
        if not 1 <= self.accepted_hidden_layer <= layers:
            raise ValueError(
                "accepted_hidden_layer must index a completed Thinker layer: "
                f"{self.accepted_hidden_layer}/{layers}"
            )
        self.device = _model_device(model)
        self.cache = split_dtype_static_cache(
            layers=layers,
            max_cache_len=capacity.thinker_tokens,
        )
        k = int(capacity.frames_per_unit)
        hidden = _hidden_size(model)
        maximum = int(capacity.thinker_tokens)

        self.audio_inputs = torch.zeros((1, 2 * k, hidden), device=self.device, dtype=dtype)
        self.audio_mask = torch.zeros((1, 1, 2 * k, maximum), device=self.device, dtype=torch.bool)
        self.audio_positions = torch.zeros((4, 1, 2 * k), device=self.device, dtype=torch.long)
        self.audio_cache_positions = torch.arange(2 * k, device=self.device, dtype=torch.long)

        self.text_input = torch.zeros((1, 1, hidden), device=self.device, dtype=dtype)
        self.text_mask = torch.zeros((1, 1, 1, maximum), device=self.device, dtype=torch.bool)
        self.text_positions = torch.zeros((4, 1, 1), device=self.device, dtype=torch.long)
        self.text_cache_position = torch.zeros(1, device=self.device, dtype=torch.long)

        self.audio_graph = torch.cuda.CUDAGraph()
        self.text_graph = torch.cuda.CUDAGraph()
        self.audio_last_hidden: torch.Tensor | None = None
        self.text_last_hidden: torch.Tensor | None = None
        self.text_context_hidden: torch.Tensor | None = None
        self._tokens = 0
        self.audio_replays = 0
        self.text_replays = 0
        self._capture()

    def _audio_forward(self) -> Any:
        return self.model(
            inputs_embeds=self.audio_inputs,
            attention_mask=self.audio_mask,
            position_ids=self.audio_positions,
            cache_position=self.audio_cache_positions,
            past_key_values=self.cache,
            use_cache=True,
            output_hidden_states=False,
        )

    def _text_forward(self) -> Any:
        return self.model(
            inputs_embeds=self.text_input,
            attention_mask=self.text_mask,
            position_ids=self.text_positions,
            cache_position=self.text_cache_position,
            past_key_values=self.cache,
            use_cache=True,
            output_hidden_states=False,
        )

    @torch.no_grad()
    def _capture(self) -> None:
        k = int(self.capacity.frames_per_unit)
        # Warmup also performs the StaticLayer lazy allocation.  A causal mask
        # over the first audio prefix is numerically valid but its values are
        # irrelevant; capture is followed by a complete cache reset.
        self.audio_mask[..., : 2 * k].copy_(
            torch.ones((2 * k, 2 * k), device=self.device, dtype=torch.bool).tril()
        )
        with inference_autocast(self.device, self.dtype):
            self._audio_forward()
        self.cache.reset()
        _set_cache_cursor(self.cache, 0)

        try:
            torch.cuda.synchronize(self.device)
            with (
                inference_autocast(self.device, self.dtype),
                torch.cuda.graph(
                    self.audio_graph,
                    capture_error_mode="thread_local",
                ),
            ):
                output = self._audio_forward()
            self.audio_last_hidden = output.last_hidden_state
        except Exception as error:
            raise CudaGraphCaptureError(
                f"failed to capture fixed {2 * k}-token Thinker audio graph"
            ) from error

        self.cache.reset()
        # The text graph is captured at the first legal post-audio cursor.  KV
        # values may be zero during capture; the full mask and static shapes
        # are identical to replay and actual audio KV is written before use.
        _set_cache_cursor(self.cache, 2 * k)
        self.text_cache_position.fill_(2 * k)
        self.text_positions.fill_(2 * k)
        self.text_mask[..., : 2 * k + 1] = True
        accepted_layer = self.model.layers[self.accepted_hidden_layer - 1]
        captured: dict[str, torch.Tensor] = {}

        def capture_context(_module: Any, _inputs: Any, output: Any) -> None:
            captured["context"] = _last_hidden(output)

        # One eager warmup exercises the single-token kernels after the static
        # cache has already been initialized by the audio warmup.
        with inference_autocast(self.device, self.dtype):
            self._text_forward()
        self.cache.reset()
        _set_cache_cursor(self.cache, 2 * k)

        handle = accepted_layer.register_forward_hook(capture_context)
        try:
            torch.cuda.synchronize(self.device)
            with (
                inference_autocast(self.device, self.dtype),
                torch.cuda.graph(
                    self.text_graph,
                    capture_error_mode="thread_local",
                ),
            ):
                output = self._text_forward()
            self.text_last_hidden = output.last_hidden_state
            self.text_context_hidden = captured.get("context")
        except Exception as error:
            raise CudaGraphCaptureError("failed to capture one-token Thinker text graph") from error
        finally:
            handle.remove()

        if (
            self.audio_last_hidden is None
            or self.text_last_hidden is None
            or self.text_context_hidden is None
        ):
            raise CudaGraphCaptureError("Thinker graphs did not retain all required output buffers")
        self.reset()

    @torch.no_grad()
    def reset(self) -> None:
        self.cache.reset()
        _set_cache_cursor(self.cache, 0)
        self._tokens = 0

    @torch.no_grad()
    def prefill_system_prompt(
        self,
        *,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Any:
        if self._tokens != 0:
            raise RuntimeError("system prompt requires a fresh Thinker cache")
        if inputs_embeds.ndim != 3 or int(inputs_embeds.shape[0]) != 1:
            raise ValueError("system prompt embeddings must be [1,tokens,hidden]")
        prompt_tokens = int(inputs_embeds.shape[1])
        if prompt_tokens < 1 or prompt_tokens > self.capacity.thinker_tokens:
            raise ValueError("system prompt length exceeds Thinker KV capacity")
        if tuple(position_ids.shape) != (4, 1, prompt_tokens):
            raise ValueError("system prompt positions must be [4,1,tokens]")
        attention_mask = torch.zeros(
            (1, 1, prompt_tokens, self.capacity.thinker_tokens),
            device=self.device,
            dtype=torch.bool,
        )
        attention_mask[..., :prompt_tokens].copy_(
            torch.ones(
                (prompt_tokens, prompt_tokens),
                device=self.device,
                dtype=torch.bool,
            ).tril()
        )
        cache_position = torch.arange(
            prompt_tokens,
            device=self.device,
            dtype=torch.long,
        )
        with inference_autocast(self.device, self.dtype):
            output = self.model(
                inputs_embeds=inputs_embeds.to(device=self.device, dtype=self.dtype),
                attention_mask=attention_mask,
                position_ids=position_ids.to(device=self.device, dtype=torch.long),
                cache_position=cache_position,
                past_key_values=self.cache,
                use_cache=True,
                output_hidden_states=False,
            )
        _set_cache_cursor(self.cache, prompt_tokens)
        self._tokens = prompt_tokens
        return output

    @torch.no_grad()
    def replay_audio(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cached_tokens: int,
    ) -> Any:
        cached_tokens = int(cached_tokens)
        k = int(self.capacity.frames_per_unit)
        if cached_tokens != self._tokens:
            raise RuntimeError(
                f"Thinker audio graph expected cursor {self._tokens}, got {cached_tokens}"
            )
        next_tokens = cached_tokens + 2 * k
        if next_tokens > self.capacity.thinker_tokens:
            raise RuntimeError(
                f"Thinker KV capacity exceeded: {next_tokens} > {self.capacity.thinker_tokens}"
            )
        if tuple(inputs_embeds.shape) != tuple(self.audio_inputs.shape):
            raise ValueError(
                f"Thinker audio graph input changed shape: {tuple(inputs_embeds.shape)}"
            )
        if tuple(position_ids.shape) != tuple(self.audio_positions.shape):
            raise ValueError(f"Thinker audio positions changed shape: {tuple(position_ids.shape)}")
        self.audio_inputs.copy_(inputs_embeds.to(device=self.device, dtype=self.audio_inputs.dtype))
        self.audio_positions.copy_(position_ids.to(device=self.device, dtype=torch.long))
        self.audio_cache_positions.copy_(
            torch.arange(
                cached_tokens,
                next_tokens,
                device=self.device,
                dtype=torch.long,
            )
        )
        _copy_full_width_mask(self.audio_mask, attention_mask)
        self.audio_graph.replay()
        self._tokens = next_tokens
        self.audio_replays += 1
        return SimpleNamespace(
            last_hidden_state=self.audio_last_hidden,
            past_key_values=self.cache,
        )

    @torch.no_grad()
    def replay_text(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cached_tokens: int,
    ) -> Any:
        cached_tokens = int(cached_tokens)
        if cached_tokens != self._tokens:
            raise RuntimeError(
                f"Thinker text graph expected cursor {self._tokens}, got {cached_tokens}"
            )
        next_tokens = cached_tokens + 1
        if next_tokens > self.capacity.thinker_tokens:
            raise RuntimeError(
                f"Thinker KV capacity exceeded: {next_tokens} > {self.capacity.thinker_tokens}"
            )
        if tuple(inputs_embeds.shape) != tuple(self.text_input.shape):
            raise ValueError(
                f"Thinker text graph input changed shape: {tuple(inputs_embeds.shape)}"
            )
        if tuple(position_ids.shape) != tuple(self.text_positions.shape):
            raise ValueError(f"Thinker text positions changed shape: {tuple(position_ids.shape)}")
        self.text_input.copy_(inputs_embeds.to(device=self.device, dtype=self.text_input.dtype))
        self.text_positions.copy_(position_ids.to(device=self.device, dtype=torch.long))
        self.text_cache_position.fill_(cached_tokens)
        _copy_full_width_mask(self.text_mask, attention_mask)
        self.text_graph.replay()
        self._tokens = next_tokens
        self.text_replays += 1
        hidden_states: list[torch.Tensor | None] = [None] * (self.accepted_hidden_layer + 1)
        hidden_states[self.accepted_hidden_layer] = self.text_context_hidden
        return SimpleNamespace(
            last_hidden_state=self.text_last_hidden,
            hidden_states=tuple(hidden_states),
            past_key_values=self.cache,
        )


class D2TalkerCudaGraph:
    """One-token Talker graph with a fixed speaker-prefix cache slot."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        speaker_prefix: torch.Tensor,
        capacity: D2GraphCapacity,
        dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.capacity = capacity
        self.dtype = dtype
        self.device = _model_device(model)
        self.cache = split_dtype_static_cache(
            layers=_layer_count(model),
            max_cache_len=capacity.talker_tokens,
        )
        hidden = _hidden_size(model)
        self.speaker_prefix = (
            speaker_prefix.detach().to(device=self.device, dtype=dtype).contiguous()
        )
        if tuple(self.speaker_prefix.shape) != (1, 1, hidden):
            raise ValueError(
                "Talker speaker prefix must be [1,1,hidden], got "
                f"{tuple(self.speaker_prefix.shape)}"
            )
        maximum = int(capacity.talker_tokens)
        self.input_embed = torch.zeros((1, 1, hidden), device=self.device, dtype=dtype)
        self.attention_mask = torch.zeros((1, 1, 1, maximum), device=self.device, dtype=torch.bool)
        self.position_ids = torch.ones((1, 1), device=self.device, dtype=torch.long)
        self.cache_position = torch.ones(1, device=self.device, dtype=torch.long)
        self.graph = torch.cuda.CUDAGraph()
        self.last_hidden_state: torch.Tensor | None = None
        self._frames = 0
        self.replays = 0
        self._capture()

    def _prefix_forward(self) -> Any:
        prefix_mask = torch.zeros(
            (1, 1, 1, self.capacity.talker_tokens),
            device=self.device,
            dtype=torch.bool,
        )
        prefix_mask[..., 0] = True
        return self.model(
            inputs_embeds=self.speaker_prefix,
            attention_mask=prefix_mask,
            position_ids=torch.zeros((1, 1), device=self.device, dtype=torch.long),
            cache_position=torch.zeros(1, device=self.device, dtype=torch.long),
            past_key_values=self.cache,
            use_cache=True,
        )

    def _forward(self) -> Any:
        return self.model(
            inputs_embeds=self.input_embed,
            attention_mask=self.attention_mask,
            position_ids=self.position_ids,
            cache_position=self.cache_position,
            past_key_values=self.cache,
            use_cache=True,
        )

    @torch.no_grad()
    def _restore_prefix(self) -> None:
        self.cache.reset()
        _set_cache_cursor(self.cache, 0)
        with inference_autocast(self.device, self.dtype):
            self._prefix_forward()
        _set_cache_cursor(self.cache, 1)

    @torch.no_grad()
    def _capture(self) -> None:
        self._restore_prefix()
        self.attention_mask[..., :2] = True
        with inference_autocast(self.device, self.dtype):
            self._forward()
        self._restore_prefix()
        try:
            torch.cuda.synchronize(self.device)
            with (
                inference_autocast(self.device, self.dtype),
                torch.cuda.graph(
                    self.graph,
                    capture_error_mode="thread_local",
                ),
            ):
                output = self._forward()
            self.last_hidden_state = output.last_hidden_state
        except Exception as error:
            raise CudaGraphCaptureError("failed to capture one-token Talker graph") from error
        if self.last_hidden_state is None:
            raise CudaGraphCaptureError("Talker graph retained no output buffer")
        self.reset()

    @torch.no_grad()
    def reset(self) -> None:
        self._restore_prefix()
        self._frames = 0

    @torch.no_grad()
    def replay(
        self,
        *,
        inputs_embeds: torch.Tensor,
        position_id: int,
        cached_tokens: int,
    ) -> Any:
        cached_tokens = int(cached_tokens)
        expected = self._frames + 1
        if cached_tokens != expected:
            raise RuntimeError(
                f"Talker graph expected {expected} cached tokens, got {cached_tokens}"
            )
        if self._frames >= self.capacity.native_frames:
            raise RuntimeError(f"Talker KV capacity exceeded at frame {self._frames}")
        if tuple(inputs_embeds.shape) != tuple(self.input_embed.shape):
            raise ValueError(f"Talker graph input changed shape: {tuple(inputs_embeds.shape)}")
        self.input_embed.copy_(inputs_embeds.to(device=self.device, dtype=self.input_embed.dtype))
        self.position_ids.fill_(int(position_id))
        self.cache_position.fill_(cached_tokens)
        self.attention_mask.zero_()
        self.attention_mask[..., : cached_tokens + 1] = True
        self.graph.replay()
        self._frames += 1
        self.replays += 1
        return SimpleNamespace(
            last_hidden_state=self.last_hidden_state,
            past_key_values=self.cache,
        )


class D2CodePredictorCudaGraphs:
    """Fixed two-token start and one-token residual CodePredictor graphs."""

    def __init__(self, model: torch.nn.Module, *, dtype: torch.dtype) -> None:
        self.model = model
        self.dtype = dtype
        self.device = _model_device(model)
        self.sequence_length = int(CODEBOOKS)
        self.cache = split_dtype_static_cache(
            layers=_layer_count(model),
            max_cache_len=self.sequence_length,
        )
        hidden = _hidden_size(model)
        self.start_inputs = torch.zeros((1, 2, hidden), device=self.device, dtype=dtype)
        self.start_mask = torch.zeros(
            (1, 1, 2, self.sequence_length),
            device=self.device,
            dtype=torch.bool,
        )
        self.start_mask[..., 0, 0] = True
        self.start_mask[..., 1, :2] = True
        self.start_positions = torch.arange(2, device=self.device, dtype=torch.long).view(1, 2)
        self.start_cache_positions = torch.arange(2, device=self.device, dtype=torch.long)
        self.step_input = torch.zeros((1, 1, hidden), device=self.device, dtype=dtype)
        self.step_mask = torch.zeros(
            (1, 1, 1, self.sequence_length),
            device=self.device,
            dtype=torch.bool,
        )
        self.step_position = torch.full((1, 1), 2, device=self.device, dtype=torch.long)
        self.step_cache_position = torch.full((1,), 2, device=self.device, dtype=torch.long)
        self.start_graph = torch.cuda.CUDAGraph()
        self.step_graph = torch.cuda.CUDAGraph()
        self.start_hidden: torch.Tensor | None = None
        self.step_hidden: torch.Tensor | None = None
        self._next_position = 0
        self.start_replays = 0
        self.step_replays = 0
        self._capture()

    def _start_forward(self) -> Any:
        return self.model(
            inputs_embeds=self.start_inputs,
            attention_mask=self.start_mask,
            position_ids=self.start_positions,
            cache_position=self.start_cache_positions,
            past_key_values=self.cache,
            use_cache=True,
        )

    def _step_forward(self) -> Any:
        return self.model(
            inputs_embeds=self.step_input,
            attention_mask=self.step_mask,
            position_ids=self.step_position,
            cache_position=self.step_cache_position,
            past_key_values=self.cache,
            use_cache=True,
        )

    @torch.no_grad()
    def _capture(self) -> None:
        with inference_autocast(self.device, self.dtype):
            self._start_forward()
        self.cache.reset()
        _set_cache_cursor(self.cache, 0)
        try:
            torch.cuda.synchronize(self.device)
            with (
                inference_autocast(self.device, self.dtype),
                torch.cuda.graph(
                    self.start_graph,
                    capture_error_mode="thread_local",
                ),
            ):
                output = self._start_forward()
            self.start_hidden = output.last_hidden_state
        except Exception as error:
            raise CudaGraphCaptureError("failed to capture CodePredictor start graph") from error

        self.cache.reset()
        _set_cache_cursor(self.cache, 0)
        with inference_autocast(self.device, self.dtype):
            self._start_forward()
        self.step_mask[..., :3] = True
        self.step_position.fill_(2)
        self.step_cache_position.fill_(2)
        _set_cache_cursor(self.cache, 2)
        with inference_autocast(self.device, self.dtype):
            self._step_forward()
        self.cache.reset()
        _set_cache_cursor(self.cache, 2)
        try:
            torch.cuda.synchronize(self.device)
            with (
                inference_autocast(self.device, self.dtype),
                torch.cuda.graph(
                    self.step_graph,
                    capture_error_mode="thread_local",
                ),
            ):
                output = self._step_forward()
            self.step_hidden = output.last_hidden_state
        except Exception as error:
            raise CudaGraphCaptureError("failed to capture CodePredictor residual graph") from error
        if self.start_hidden is None or self.step_hidden is None:
            raise CudaGraphCaptureError("CodePredictor graphs retained no output buffers")
        self.reset()

    @torch.no_grad()
    def reset(self) -> None:
        self.cache.reset()
        _set_cache_cursor(self.cache, 0)
        self._next_position = 0

    @torch.no_grad()
    def start(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        if tuple(inputs_embeds.shape) != tuple(self.start_inputs.shape):
            raise ValueError(
                f"CodePredictor start input changed shape: {tuple(inputs_embeds.shape)}"
            )
        self.reset()
        self.start_inputs.copy_(inputs_embeds.to(device=self.device, dtype=self.start_inputs.dtype))
        self.start_graph.replay()
        self._next_position = 2
        self.start_replays += 1
        return self.start_hidden[:, -1, :]

    @torch.no_grad()
    def advance(self, input_embed: torch.Tensor, *, position: int) -> torch.Tensor:
        position = int(position)
        if position != self._next_position:
            raise RuntimeError(
                f"CodePredictor graph expected position {self._next_position}, got {position}"
            )
        if not 2 <= position < self.sequence_length:
            raise ValueError(f"invalid CodePredictor position {position}")
        if tuple(input_embed.shape) == (1, _hidden_size(self.model)):
            input_embed = input_embed[:, None, :]
        if tuple(input_embed.shape) != tuple(self.step_input.shape):
            raise ValueError(
                f"CodePredictor residual input changed shape: {tuple(input_embed.shape)}"
            )
        self.step_input.copy_(input_embed.to(device=self.device, dtype=self.step_input.dtype))
        self.step_mask.zero_()
        self.step_mask[..., : position + 1] = True
        self.step_position.fill_(position)
        self.step_cache_position.fill_(position)
        self.step_graph.replay()
        self._next_position = position + 1
        self.step_replays += 1
        return self.step_hidden[:, -1, :]


class D2CudaGraphBackend:
    """Process-wide graph owner reused by sequential D2 sessions."""

    def __init__(
        self,
        model: Any,
        *,
        frames_per_unit: int,
        capacity_frames: int,
        fuse_audio_text: bool = False,
    ) -> None:
        self.model = model
        self.capacity = D2GraphCapacity(
            frames_per_unit=int(frames_per_unit),
            native_frames=int(capacity_frames),
        )
        self.dtype = model_compute_dtype(model)
        device = module_device(model.qwen.thinker.model)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise CudaGraphCaptureError("D2 graph backend requires CUDA")
        if module_device(model.qwen.talker.model) != device:
            raise CudaGraphCaptureError(
                "Thinker and Talker must share one CUDA device for graph replay"
            )

        with inference_autocast(device, self.dtype):
            speaker_prefix = model._talker_speaker_prefix(
                batch_size=1,
                target_device=device,
                target_dtype=self.dtype,
            )
        try:
            thinker_class = D2ThinkerCudaGraphs
            if fuse_audio_text:
                from .live_graph import FusedNativeThinker

                thinker_class = FusedNativeThinker
            self.thinker = thinker_class(
                model.qwen.thinker.model,
                accepted_hidden_layer=int(model.accept_hidden_layer),
                capacity=self.capacity,
                dtype=self.dtype,
            )
            self.talker = D2TalkerCudaGraph(
                model.qwen.talker.model,
                speaker_prefix=speaker_prefix,
                capacity=self.capacity,
                dtype=self.dtype,
            )
            self.code_predictor = D2CodePredictorCudaGraphs(
                model.qwen.talker.code_predictor.model,
                dtype=self.dtype,
            )
        except CudaGraphCaptureError:
            raise
        except Exception as error:
            raise CudaGraphCaptureError("requested D2 CUDA graph initialization failed") from error
        self.sessions = 0
        self.fuse_audio_text = fuse_audio_text

    @property
    def thinker_cache(self) -> Cache:
        return self.thinker.cache

    @property
    def talker_cache(self) -> Cache:
        return self.talker.cache

    @property
    def code_predictor_graph_enabled(self) -> bool:
        return True

    @property
    def code2wav_graph_enabled(self) -> bool:
        return False

    @torch.no_grad()
    def reset_session(self) -> tuple[Cache, Cache]:
        self.thinker.reset()
        self.talker.reset()
        self.code_predictor.reset()
        self.sessions += 1
        return self.thinker.cache, self.talker.cache

    @torch.no_grad()
    def prefill_system_prompt(
        self,
        *,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Any:
        return self.thinker.prefill_system_prompt(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )

    def replay_counters(self) -> dict[str, int]:
        """Return monotonic counters suitable for per-session deltas."""

        return {
            "audio_prefill": int(self.thinker.audio_replays),
            "text_decode": int(self.thinker.text_replays),
            "talker_decode": int(self.talker.replays),
            "code_predictor_start": int(self.code_predictor.start_replays),
            "code_predictor_step": int(self.code_predictor.step_replays),
        }

    def validate_session_cursors(
        self,
        *,
        native_frames: int,
        thinker_prefix_tokens: int = 0,
    ) -> dict[str, int]:
        """Verify device-side graph mutations after the caller synchronizes."""

        native_frames = int(native_frames)
        if not 0 <= native_frames <= self.capacity.native_frames:
            raise ValueError(f"invalid completed frame count {native_frames}")
        actual = {
            "thinker_tokens": _uniform_cache_cursor(self.thinker.cache),
            "talker_tokens": _uniform_cache_cursor(self.talker.cache),
        }
        expected = {
            "thinker_tokens": int(thinker_prefix_tokens) + 3 * native_frames,
            "talker_tokens": native_frames + 1,
        }
        if actual != expected:
            raise RuntimeError(f"captured cache cursors {actual} do not match expected {expected}")
        return actual

    def audio_prefill_forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cached_tokens: int,
    ) -> Any:
        return self.thinker.replay_audio(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cached_tokens=cached_tokens,
        )

    def text_forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cached_tokens: int,
    ) -> Any:
        return self.thinker.replay_text(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cached_tokens=cached_tokens,
        )

    def talker_forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        position_id: int,
        cached_tokens: int,
    ) -> Any:
        return self.talker.replay(
            inputs_embeds=inputs_embeds,
            position_id=position_id,
            cached_tokens=cached_tokens,
        )

    @torch.no_grad()
    def generate_residual_codes(
        self,
        model: Any,
        *,
        talker_hidden: torch.Tensor,
        code0: torch.Tensor,
        residual_temperature: float,
        residual_top_k: int,
        residual_top_p: float,
        timings: ForwardTimings | None,
        trace: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        predictor = model.qwen.talker.code_predictor
        predictor_device = module_device(predictor.model, talker_hidden.device)
        codes = [code0]
        with measure_forward(timings, "code_predictor", predictor_device):
            first_inputs = torch.cat(
                (
                    talker_hidden.unsqueeze(1),
                    embed_on(
                        model.qwen.talker.model.codec_embedding,
                        code0,
                        target_device=talker_hidden.device,
                    ).unsqueeze(1),
                ),
                dim=1,
            )
            hidden = self.code_predictor.start(first_inputs)
            residual_embeddings = predictor.get_input_embeddings()
            for group in range(1, CODEBOOKS):
                record_component_trace(
                    trace,
                    f"code_predictor/codebook_{group:02d}_hidden",
                    hidden,
                )
                head = predictor.lm_head[group - 1]
                head_device = module_device(head, hidden.device)
                with inference_autocast(head_device, self.dtype):
                    logits = head(hidden.to(device=head_device))[..., :CODEBOOK_SIZE]
                record_component_trace(
                    trace,
                    f"code_predictor/codebook_{group:02d}_logits",
                    logits,
                )
                code = sample_logits(
                    logits,
                    temperature=float(residual_temperature),
                    top_k=int(residual_top_k),
                    top_p=float(residual_top_p),
                )
                codes.append(code)
                if group == CODEBOOKS - 1:
                    break
                next_embed = embed_on(
                    residual_embeddings[group - 1],
                    code,
                    target_device=hidden.device,
                )
                hidden = self.code_predictor.advance(
                    next_embed,
                    position=group + 1,
                )
        return torch.stack(codes, dim=-1).view(CODEBOOKS).detach().cpu()

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "required": True,
            "frames_per_unit": self.capacity.frames_per_unit,
            "capacity_frames": self.capacity.native_frames,
            "thinker_capacity_tokens": self.capacity.thinker_tokens,
            "talker_capacity_tokens": self.capacity.talker_tokens,
            "audio_prefill_query_tokens": 2 * self.capacity.frames_per_unit,
            "text_query_tokens": 1,
            "talker_query_tokens": 1,
            "future_kv_slots_masked": True,
            "static_cache": "hf_split_dtype_static_cache",
            "decode_boundary": "fused_env_self_previous_text_then_talker"
            if self.fuse_audio_text
            else "separate_thinker_talker_graphs",
            "fused_thinker_parity": getattr(self.thinker, "parity", None),
            "replay_counters": self.replay_counters(),
        }


__all__ = [
    "CudaGraphCaptureError",
    "D2CodePredictorCudaGraphs",
    "D2CudaGraphBackend",
    "D2GraphCapacity",
    "D2TalkerCudaGraph",
    "D2ThinkerCudaGraphs",
]
