from __future__ import annotations
from array import array
from dataclasses import dataclass, replace
import math
import sys
import time
from typing import Any
import torch
from d2_qwen.model.config import frames_for_latency
from d2_qwen.model.constants import AUDIO_HZ, CODEBOOKS, SAMPLE_RATE_CODEC
from d2_qwen.model.utils import module_device
from .layout import (
    AudioPrefillResult,
    MacroInferenceState,
    TalkerStepResult,
    TextStepResult,
    run_macro_unit,
)
from .control import ResponseTransition, response_transition
from .streaming import (
    IncrementalCode2WavPlayback,
    IncrementalCode2WavRenderer,
    StreamingPcmBoundaryDezipper,
    cache_sequence_length,
    embed_codec_frame,
    generate_codec_frame,
    inference_autocast,
    model_compute_dtype,
    prefill_talker_speaker,
    sample_logits,
)
from .timing import ForwardTimings, measure_forward


INPUT_SAMPLE_RATE_HZ = 16_000


OUTPUT_SAMPLE_RATE_HZ = 24_000


INPUT_FRAME_SAMPLES = 1_280


OUTPUT_FRAME_SAMPLES = 1_920


CODEC_SILENCE_FRAME = (
    1049,
    1700,
    1626,
    546,
    306,
    1443,
    1871,
    2008,
    1866,
    374,
    662,
    1383,
    1123,
    1430,
    644,
    610,
)


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Reference async-inference sampling defaults."""

    text_temperature: float = 0.0
    text_top_k: int = 0
    text_top_p: float = 1.0
    codec_temperature: float = 0.7
    codec_top_k: int = 50
    codec_top_p: float = 1.0
    codec_repetition_penalty: float = 1.1
    residual_temperature: float = 0.7
    residual_top_k: int = 50
    residual_top_p: float = 0.8


@dataclass(frozen=True, slots=True)
class OfflineResult:
    """One offline rollout, matching the reference inference result surface."""

    pcm_s16le: bytes
    frame_events: tuple[dict[str, Any], ...]
    session_summary: dict[str, Any]
    source_samples: int
    padded_samples: int


@dataclass(frozen=True, slots=True)
class LiveBatch:
    """Complete native output frames produced by zero or more macro steps."""

    pcm_frames: tuple[bytes, ...]
    frame_events: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _TalkerContext:
    outside_context: torch.Tensor
    transition: ResponseTransition
    text_done_after: bool
    text_id: int
    text_delta: str
    sequence: int


def _macro_pcm_payloads(
    pcm_s16le: bytes,
    *,
    frames_per_unit: int,
) -> tuple[tuple[bytes, ...], int, int, int]:
    """Pad native PCM frames to complete macro units.

    Returns ``(macro_payloads, source_frames, processed_frames, padded_samples)``.
    An empty utterance intentionally produces no model calls.
    """

    if not isinstance(pcm_s16le, (bytes, bytearray, memoryview)):
        raise TypeError("pcm_s16le must be bytes-like")
    payload = bytes(pcm_s16le)
    if len(payload) % 2:
        raise ValueError("PCM16 input must contain complete samples")
    k = int(frames_per_unit)
    frames_for_latency(k * 80)
    source_samples = len(payload) // 2
    source_frames = math.ceil(source_samples / INPUT_FRAME_SAMPLES) if source_samples else 0
    processed_frames = math.ceil(source_frames / k) * k if source_frames else 0
    processed_samples = processed_frames * INPUT_FRAME_SAMPLES
    padded = payload + bytes(max(0, processed_samples * 2 - len(payload)))
    macro_bytes = k * INPUT_FRAME_SAMPLES * 2
    macros = tuple(
        padded[offset : offset + macro_bytes] for offset in range(0, len(padded), macro_bytes)
    )
    return macros, source_frames, processed_frames, processed_samples - source_samples


def _pcm_tensor(pcm_s16le: bytes) -> torch.Tensor:
    samples = array("h")
    samples.frombytes(pcm_s16le)
    if sys.byteorder != "little":
        samples.byteswap()
    return torch.tensor(samples, dtype=torch.float32).div_(32768.0)


def _encode_pcm(wav: torch.Tensor) -> bytes:
    values = (
        wav.detach()
        .cpu()
        .float()
        .clamp_(-1.0, 1.0)
        .mul_(32767.0)
        .round_()
        .to(dtype=torch.int16)
        .tolist()
    )
    samples = array("h", values)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


def _async_text_special_ids(model: Any) -> set[int]:
    ids = {int(value) for value in model.processor.tokenizer.all_special_ids}
    ids.update(
        (
            int(model.pad_id),
            int(model.assistant_start_id),
            int(model.assistant_end_id),
        )
    )
    ids.update(int(value) for value in model.forbidden_text_special_ids)
    ids.discard(int(model.space_id))
    return ids


class _MacroSession:
    """One stream; all latency variation is the scalar ``k``."""

    def __init__(
        self,
        *,
        model: Any,
        device: torch.device,
        frames_per_unit: int,
        max_context_frames: int,
        kv_cache_capacity_frames: int,
        kv_cache_capacity_effective_ms: int,
        require_cuda_graphs: bool,
        cuda_graph_backend: Any | None,
        sampling: SamplingConfig,
        checkpoint_metadata: dict[str, Any],
        logical_frames: int,
        session_id: str,
        system_prompt_kind: str | None,
        system_prompt_input_ids: tuple[int, ...] | None,
        live_audio_graphs: Any | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.k = int(frames_per_unit)
        self.max_context_frames = int(max_context_frames)
        self.kv_cache_capacity_frames = int(kv_cache_capacity_frames)
        self.kv_cache_capacity_effective_ms = int(kv_cache_capacity_effective_ms)
        self.require_cuda_graphs = bool(require_cuda_graphs)
        self.cuda_graph_backend = cuda_graph_backend
        if (
            self.device.type == "cuda"
            and self.require_cuda_graphs
            and self.cuda_graph_backend is None
        ):
            raise RuntimeError(
                "CUDA graph capture is required but no captured backend is installed"
            )
        self.sampling = sampling
        self.checkpoint_metadata = checkpoint_metadata
        self.logical_frames = int(logical_frames)
        self.session_id = session_id
        self.system_prompt_kind = system_prompt_kind
        self.system_prompt_input_ids = tuple(system_prompt_input_ids or ())
        self.system_prompt_tokens = len(self.system_prompt_input_ids)
        self.timings = ForwardTimings()
        self.env_stream = model.audio_encoder.new_stream(device)
        self.self_stream = model.audio_encoder.new_stream(device)
        self.live_audio_graphs = live_audio_graphs
        if live_audio_graphs is not None:
            self.env_stream, self.self_stream = live_audio_graphs.reset(
                self.env_stream, self.self_stream
            )
        if self.cuda_graph_backend is None:
            thinker_kv = None
            talker_kv = prefill_talker_speaker(model, timings=self.timings)
            self._graph_replay_start: dict[str, int] | None = None
        else:
            if (
                int(self.cuda_graph_backend.capacity.frames_per_unit) != self.k
                or int(self.cuda_graph_backend.capacity.native_frames)
                != self.kv_cache_capacity_frames
            ):
                raise RuntimeError("CUDA graph backend capacity does not match session")
            thinker_kv, talker_kv = self.cuda_graph_backend.reset_session()
            self._graph_replay_start = self.cuda_graph_backend.replay_counters()
        if self.system_prompt_tokens:
            prompt_ids = torch.tensor(
                [self.system_prompt_input_ids],
                dtype=torch.long,
                device=self.device,
            )
            prompt_embeddings = model.embed_text(prompt_ids).to(dtype=model_compute_dtype(model))
            order = torch.arange(
                self.system_prompt_tokens,
                dtype=torch.long,
                device=self.device,
            )
            prompt_positions = order[None, None, :].expand(4, 1, -1).contiguous()
            if self.cuda_graph_backend is None:
                prompt_mask = torch.ones(
                    (self.system_prompt_tokens, self.system_prompt_tokens),
                    dtype=torch.bool,
                    device=self.device,
                ).tril()[None, None]
                with inference_autocast(
                    prompt_embeddings.device,
                    model_compute_dtype(model),
                ):
                    output = model.qwen.thinker.model(
                        inputs_embeds=prompt_embeddings,
                        attention_mask=prompt_mask,
                        position_ids=prompt_positions,
                        past_key_values=None,
                        use_cache=True,
                        output_hidden_states=False,
                    )
                thinker_kv = output.past_key_values
            else:
                output = self.cuda_graph_backend.prefill_system_prompt(
                    inputs_embeds=prompt_embeddings,
                    position_ids=prompt_positions,
                )
                thinker_kv = output.past_key_values
        self.state = MacroInferenceState(
            thinker_kv=thinker_kv,
            talker_kv=talker_kv,
            frames_per_unit=self.k,
        )
        self.renderer = IncrementalCode2WavRenderer(
            model,
            left_context_frames=25,
            parity_atol=float("inf"),
            parity_mean_atol=float("inf"),
            verify_reference=False,
            capture_components=False,
            retain_raw_prefix=False,
            timings=self.timings,
            forward_backend=live_audio_graphs.waveform if live_audio_graphs is not None else None,
        )
        self.playback = IncrementalCode2WavPlayback(self_delay_samples=0)
        self.dezipper = StreamingPcmBoundaryDezipper(
            transition_samples=round(SAMPLE_RATE_CODEC * 0.5 / 1_000.0),
            minimum_jump=0.03,
        )
        self.responding = False
        self.text_done = True
        self.talker_responding = False
        self.text_special_ids = _async_text_special_ids(model)
        self.generated_text: list[int] = []
        self.text_trace: list[dict[str, Any]] = []
        self.response_patterns: list[torch.Tensor] = []
        self.response_codec0_history: list[int] = []
        self.pending_self_pcm: torch.Tensor | None = None
        self._unit_physical: list[torch.Tensor] = []
        self._unit_active: list[bool] = []
        self._unit_records: list[dict[str, Any]] = []
        self.macro_wall_ms: list[float] = []
        self.closed = False

    def _encode_audio_macro(
        self,
        environment_pcm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, bool]:
        expected = self.k * INPUT_FRAME_SAMPLES
        if environment_pcm.ndim != 1 or int(environment_pcm.numel()) != expected:
            raise ValueError(
                f"macro PCM has {environment_pcm.numel()} samples; expected {expected}"
            )
        encoder = self.model.audio_encoder.encoder
        encoder_device = module_device(encoder, self.device)
        compute_dtype = model_compute_dtype(self.model)
        if self.pending_self_pcm is None:
            with measure_forward(self.timings, "environment_aut", encoder_device):
                with inference_autocast(encoder_device, compute_dtype):
                    environment = self.env_stream.push_audio(environment_pcm)
            hidden = int(self.model.self_silence_embed.numel())
            own = (
                self.model.self_silence_embed.detach()
                .to(
                    device=self.device,
                    dtype=torch.float32,
                )
                .view(1, hidden)
                .expand(self.k, hidden)
            )
            previous = None
            batched = False
        else:
            if int(self.pending_self_pcm.numel()) != expected:
                raise RuntimeError("pending self-feedback does not span one macro unit")
            with measure_forward(self.timings, "batched_aut", encoder_device):
                with inference_autocast(encoder_device, compute_dtype):
                    if self.live_audio_graphs is None:
                        environment, own = self.model.audio_encoder.push_stream_batch(
                            [self.env_stream, self.self_stream],
                            [environment_pcm, self.pending_self_pcm],
                        )
                    else:
                        environment = self.env_stream.push_audio(environment_pcm)
                        own = self.self_stream.push_audio(self.pending_self_pcm)
            # Online at unit t, the newly encoded playback is unit t-k and is
            # exactly the history consumed by the model's representation delay.
            previous = own.unsqueeze(0)
            batched = True
        expected_shape = (self.k, int(environment.shape[-1]))
        if tuple(environment.shape) != expected_shape:
            raise RuntimeError(
                f"environment AuT emitted {tuple(environment.shape)}; expected {expected_shape}"
            )
        if tuple(own.shape) != expected_shape:
            raise RuntimeError(f"self AuT emitted {tuple(own.shape)}; expected {expected_shape}")
        self.pending_self_pcm = None
        return environment.unsqueeze(0), own.unsqueeze(0), previous, batched

    def _audio_prefill(
        self,
        *,
        environment_audio: torch.Tensor,
        self_audio: torch.Tensor,
        previous_self_audio: torch.Tensor | None,
        unit_index: int,
        thinker_kv: Any,
    ) -> AudioPrefillResult:
        if self.cuda_graph_backend is None:
            cached = cache_sequence_length(thinker_kv) or 0
        else:
            # Every completed macro contributes 2k audio plus k text tokens.
            # Keeping this cursor on the CPU avoids synchronizing on the
            # StaticLayer's device-side cumulative_length before each replay.
            cached = self.system_prompt_tokens + int(unit_index) * 3 * self.k
        prepared = self.model.inference_audio_prefill(
            environment_audio=environment_audio,
            self_audio=self_audio,
            previous_self_audio=previous_self_audio,
            unit_offset=unit_index,
            cached_tokens=cached,
            valid_cached_tokens=cached,
            position_offset_tokens=self.system_prompt_tokens,
        )
        dtype = model_compute_dtype(self.model)
        with measure_forward(self.timings, "thinker_audio_prefill", prepared.inputs_embeds.device):
            if self.cuda_graph_backend is None:
                with inference_autocast(prepared.inputs_embeds.device, dtype):
                    output = self.model.qwen.thinker.model(
                        inputs_embeds=prepared.inputs_embeds,
                        attention_mask=prepared.attention_mask,
                        position_ids=prepared.position_ids,
                        past_key_values=thinker_kv,
                        use_cache=True,
                        output_hidden_states=False,
                    )
            else:
                if thinker_kv is not self.cuda_graph_backend.thinker_cache:
                    raise RuntimeError("session lost the captured Thinker cache")
                output = self.cuda_graph_backend.audio_prefill_forward(
                    inputs_embeds=prepared.inputs_embeds,
                    attention_mask=prepared.attention_mask,
                    position_ids=prepared.position_ids,
                    cached_tokens=cached,
                )
        next_cache = output.past_key_values
        if self.cuda_graph_backend is None:
            length = cache_sequence_length(next_cache)
            if length is not None and length != cached + 2 * self.k:
                raise RuntimeError(f"Thinker audio KV length {length} != {cached + 2 * self.k}")
        return AudioPrefillResult(
            thinker_kv=next_cache,
            next_self_audio_history=prepared.next_self_history,
        )

    def _text_delta(self, text_id: int) -> str:
        if int(text_id) in self.text_special_ids:
            return ""
        return self.model.processor.tokenizer.decode([int(text_id)], skip_special_tokens=True)

    def _text_step(
        self,
        *,
        text_index: int,
        text_input: Any,
        unit_index: int,
        thinker_kv: Any,
    ) -> TextStepResult:
        if self.cuda_graph_backend is None:
            cached = cache_sequence_length(thinker_kv) or 0
        else:
            cached = (
                self.system_prompt_tokens
                + int(unit_index) * 3 * self.k
                + 2 * self.k
                + int(text_index)
            )
        ids = torch.tensor([[int(text_input)]], dtype=torch.long, device=self.device)
        prepared = self.model.inference_text_step(
            previous_text_ids=ids,
            text_index=text_index,
            unit_offset=unit_index,
            cached_tokens=cached,
            valid_cached_tokens=cached,
            position_offset_tokens=self.system_prompt_tokens,
        )
        dtype = model_compute_dtype(self.model)
        with measure_forward(self.timings, "thinker_text", prepared.inputs_embeds.device):
            if self.cuda_graph_backend is None:
                with inference_autocast(prepared.inputs_embeds.device, dtype):
                    output = self.model.qwen.thinker.model(
                        inputs_embeds=prepared.inputs_embeds,
                        attention_mask=prepared.attention_mask,
                        position_ids=prepared.position_ids,
                        past_key_values=thinker_kv,
                        use_cache=True,
                        output_hidden_states=True,
                    )
            else:
                if thinker_kv is not self.cuda_graph_backend.thinker_cache:
                    raise RuntimeError("session lost the captured Thinker cache")
                output = self.cuda_graph_backend.text_forward(
                    inputs_embeds=prepared.inputs_embeds,
                    attention_mask=prepared.attention_mask,
                    position_ids=prepared.position_ids,
                    cached_tokens=cached,
                )
        if output.hidden_states is None:
            raise RuntimeError("Thinker did not return Talker context layer")
        next_cache = output.past_key_values
        if self.cuda_graph_backend is None:
            length = cache_sequence_length(next_cache)
            if length is not None and length != cached + 1:
                raise RuntimeError(f"Thinker text KV length {length} != {cached + 1}")
        text_hidden = output.last_hidden_state[:, -1, :]
        outside = output.hidden_states[self.model.accept_hidden_layer][:, -1:, :]
        with measure_forward(self.timings, "text_head", text_hidden.device):
            with inference_autocast(text_hidden.device, dtype):
                logits = self.model.text_logits(text_hidden)
        masked = self.model.mask_text_logits(
            logits,
            speech_open=torch.tensor([self.responding], dtype=torch.bool, device=logits.device),
            text_done=torch.tensor([self.text_done], dtype=torch.bool, device=logits.device),
            previous_text_id=torch.tensor(
                [int(text_input)], dtype=torch.long, device=logits.device
            ),
        )
        text_tensor = sample_logits(
            masked,
            temperature=self.sampling.text_temperature,
            top_k=self.sampling.text_top_k,
            top_p=self.sampling.text_top_p,
        )
        text_id = int(text_tensor.item())
        transition, text_done_after = response_transition(
            responding=self.responding,
            text_done=self.text_done,
            text_id=text_id,
            pad_id=self.model.pad_id,
            assistant_start_id=self.model.assistant_start_id,
            assistant_end_id=self.model.assistant_end_id,
            talker_responding=self.talker_responding,
        )
        sequence = unit_index * self.k + text_index
        context = _TalkerContext(
            outside_context=outside,
            transition=transition,
            text_done_after=text_done_after,
            text_id=text_id,
            text_delta=self._text_delta(text_id),
            sequence=sequence,
        )
        return TextStepResult(
            generated_text=text_id,
            thinker_kv=next_cache,
            talker_context=context,
        )

    def _silence_pattern(self) -> torch.Tensor:
        return self.model.codec_silence_frame.detach().long().cpu().view(CODEBOOKS).clone()

    def _talker_step(
        self,
        *,
        frame_index: int,
        generated_text: Any,
        talker_context: _TalkerContext,
        codec_input: Any,
        unit_index: int,
        talker_kv: Any,
    ) -> TalkerStepResult:
        del frame_index
        text_id = int(generated_text)
        transition = talker_context.transition
        dtype = model_compute_dtype(self.model)
        if transition.use_text_context:
            text = torch.tensor([[text_id]], dtype=torch.long, device=self.device)
            text_embed = self.model.embed_text(text).to(dtype=dtype)
            projection = self.model.qwen.talker.text_projection
            projection_device = module_device(projection, text_embed.device)
            with measure_forward(self.timings, "talker_context_projection", projection_device):
                with inference_autocast(projection_device, dtype):
                    context = projection(text_embed.to(projection_device))
            previous = self._silence_pattern() if codec_input is None else codec_input
            if text_id == int(self.model.response_id):
                previous = self._silence_pattern()
                previous[0] = int(self.model.talker_codec_bos_id)
            codec_context = embed_codec_frame(self.model, previous)
        else:
            projection = self.model.qwen.talker.hidden_projection
            projection_device = module_device(projection, talker_context.outside_context.device)
            with measure_forward(self.timings, "talker_context_projection", projection_device):
                with inference_autocast(projection_device, dtype):
                    context = projection(
                        talker_context.outside_context.to(device=projection_device, dtype=dtype)
                    )
            codec_context = torch.zeros_like(context)
        talker_input = codec_context.to(device=context.device, dtype=context.dtype) + context
        if self.cuda_graph_backend is None:
            cached = cache_sequence_length(talker_kv) or 0
        else:
            # One immutable speaker-prefix token precedes native frame tokens.
            cached = int(talker_context.sequence) + 1
        with measure_forward(self.timings, "talker", talker_input.device):
            if self.cuda_graph_backend is None:
                with inference_autocast(talker_input.device, dtype):
                    output = self.model.qwen.talker.model(
                        inputs_embeds=talker_input,
                        attention_mask=torch.ones(
                            (1, cached + 1),
                            dtype=torch.long,
                            device=talker_input.device,
                        ),
                        position_ids=torch.tensor(
                            [[talker_context.sequence + 1]],
                            dtype=torch.long,
                            device=talker_input.device,
                        ),
                        past_key_values=talker_kv,
                        use_cache=True,
                    )
            else:
                if talker_kv is not self.cuda_graph_backend.talker_cache:
                    raise RuntimeError("session lost the captured Talker cache")
                output = self.cuda_graph_backend.talker_forward(
                    inputs_embeds=talker_input,
                    position_id=talker_context.sequence + 1,
                    cached_tokens=cached,
                )
        next_cache = output.past_key_values
        if self.cuda_graph_backend is None:
            length = cache_sequence_length(next_cache)
            if length is not None and length != cached + 1:
                raise RuntimeError(f"Talker KV length {length} != {cached + 1}")
        hidden = output.last_hidden_state[:, -1, :]

        if text_id == int(self.model.response_id):
            pattern = self._silence_pattern()
            pattern[0] = int(self.model.talker_codec_bos_id)
        elif text_id == int(self.model.interrupt_id) and bool(transition.talker_before):
            pattern = self._silence_pattern()
            pattern[0] = int(self.model.talker_codec_eos_id)
        elif transition.emit_audio:
            pattern = generate_codec_frame(
                self.model,
                hidden,
                codec0_history=self.response_codec0_history,
                codec_repetition_penalty=self.sampling.codec_repetition_penalty,
                codec_temperature=self.sampling.codec_temperature,
                codec_top_k=self.sampling.codec_top_k,
                codec_top_p=self.sampling.codec_top_p,
                residual_temperature=self.sampling.residual_temperature,
                residual_top_k=self.sampling.residual_top_k,
                residual_top_p=self.sampling.residual_top_p,
                timings=self.timings,
                forward_backend=self.cuda_graph_backend,
                allow_codec_eos=True,
            )
            if int(pattern[0]) == int(self.model.talker_codec_eos_id):
                transition = replace(transition, emit_audio=False, talker_after=False)
        else:
            pattern = self._silence_pattern()

        self.responding = bool(transition.after)
        self.text_done = bool(talker_context.text_done_after)
        self.talker_responding = bool(transition.talker_after)
        self.generated_text.append(text_id)
        if transition.use_text_context:
            if text_id == int(self.model.assistant_start_id):
                self.response_patterns.clear()
                self.response_codec0_history.clear()
            self.response_patterns.append(pattern)
            if transition.emit_audio:
                self.response_codec0_history.append(int(pattern[0]))
        physical = pattern.clone() if transition.emit_audio else self._silence_pattern()
        if not self.talker_responding:
            self.response_patterns.clear()
            self.response_codec0_history.clear()
        self._unit_physical.append(physical)
        self._unit_active.append(bool(transition.emit_audio))

        if text_id == int(self.model.pad_id):
            kind = None
        elif text_id == int(self.model.assistant_start_id):
            kind = "response"
        elif text_id == int(self.model.assistant_end_id):
            kind = "interrupt"
        else:
            kind = "text"
        record = {
            "sequence": talker_context.sequence,
            "token_id": text_id,
            "text_delta": talker_context.text_delta,
            "response_open": self.responding,
            "talker_open": self.talker_responding,
            "text_done": self.text_done,
            "emit_audio": bool(transition.emit_audio),
        }
        self._unit_records.append(record)
        if kind is not None and talker_context.sequence < self.logical_frames:
            self.text_trace.append(
                {
                    "frame": talker_context.sequence,
                    "seconds": round(talker_context.sequence / float(AUDIO_HZ), 3),
                    "kind": kind,
                    "token_id": text_id,
                    "text": talker_context.text_delta,
                    "response_open": self.responding,
                    "talker_open": self.talker_responding,
                    "text_done": self.text_done,
                }
            )
        next_codec_input = pattern if self.talker_responding else self._silence_pattern()
        return TalkerStepResult(
            codec_frame=pattern,
            talker_kv=next_cache,
            next_codec_input=next_codec_input,
        )

    @torch.no_grad()
    def step_macro(
        self,
        environment_pcm: torch.Tensor,
    ) -> tuple[list[bytes], list[dict[str, Any]]]:
        if self.closed:
            raise RuntimeError("session is closed")
        unit_index = int(self.state.unit_index)
        if (unit_index + 1) * self.k > self.kv_cache_capacity_frames:
            raise ValueError("session exceeds fixed inference KV capacity")
        started = time.perf_counter()
        environment, own, effective_previous, used_batched = self._encode_audio_macro(
            environment_pcm
        )
        # The feedback emitted by the preceding macro becomes the explicit
        # representation history consumed by this macro's helper.
        state = self.state
        if effective_previous is not None:
            state = replace(state, self_audio_history=effective_previous)
        self._unit_physical = []
        self._unit_active = []
        self._unit_records = []
        output = run_macro_unit(
            frames_per_unit=self.k,
            initial_text=int(self.model.pad_id),
            environment_audio=environment,
            self_audio=own,
            state=state,
            audio_prefill=self._audio_prefill,
            text_step=self._text_step,
            talker_step=self._talker_step,
        )
        self.state = output.state
        if not (
            len(self._unit_physical) == len(self._unit_active) == len(self._unit_records) == self.k
        ):
            raise RuntimeError("macro runner did not emit exactly k native frames")

        pcm_frames: list[bytes] = []
        feedback_frames: list[torch.Tensor] = []
        for physical, active in zip(self._unit_physical, self._unit_active, strict=True):
            raw = self.renderer.append_frame(physical)
            playback = self.playback.append(raw, active=active)
            audible = self.dezipper.append(playback.playback_24k)
            if int(audible.numel()) != OUTPUT_FRAME_SAMPLES:
                raise RuntimeError(
                    f"Code2Wav emitted {audible.numel()} samples; expected {OUTPUT_FRAME_SAMPLES}"
                )
            if int(playback.self_feedback_16k.numel()) != INPUT_FRAME_SAMPLES:
                raise RuntimeError("self feedback does not span one native frame")
            pcm_frames.append(_encode_pcm(audible))
            feedback_frames.append(playback.self_feedback_16k)
        self.pending_self_pcm = torch.cat(feedback_frames).contiguous()
        macro_ms = (time.perf_counter() - started) * 1_000.0
        self.macro_wall_ms.append(macro_ms)
        per_frame_ms = macro_ms / self.k
        events: list[dict[str, Any]] = []
        for record in self._unit_records:
            event: dict[str, Any] = {
                "type": "frame",
                "sequence": int(record["sequence"]),
                "forward_ms": per_frame_ms,
                "target_forward_ms": 1_000.0 / float(AUDIO_HZ),
                "token_id": int(record["token_id"]),
                "metadata": {
                    "latency_ms": self.k * 80,
                    "frames_per_unit": self.k,
                    "macro_index": unit_index,
                    "batched_environment_self_aut": used_batched,
                    "response_open": bool(record["response_open"]),
                    "talker_open": bool(record["talker_open"]),
                    "text_done": bool(record["text_done"]),
                    "emit_audio": bool(record["emit_audio"]),
                },
            }
            if record["text_delta"]:
                event["text_delta"] = str(record["text_delta"])
            events.append(event)
        return pcm_frames, events

    def close(self) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("session is already closed")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.closed = True
        wall = self.macro_wall_ms
        frame_count = int(self.state.unit_index) * self.k
        if self.cuda_graph_backend is None:
            graph_summary: dict[str, Any] = {
                "enabled": False,
                "required": self.require_cuda_graphs,
                "reason": (
                    "cpu_reference_eager" if self.device.type != "cuda" else "capture_not_required"
                ),
            }
        else:
            if self._graph_replay_start is None:
                raise RuntimeError("captured session has no replay counter baseline")
            replay_counters = self.cuda_graph_backend.replay_counters()
            session_replays = {
                name: int(value) - int(self._graph_replay_start[name])
                for name, value in replay_counters.items()
            }
            expected_replays = {
                "audio_prefill": int(self.state.unit_index),
                "text_decode": frame_count,
                "talker_decode": frame_count,
            }
            for name, expected in expected_replays.items():
                if session_replays[name] != expected:
                    raise RuntimeError(
                        f"CUDA graph replay count {name}={session_replays[name]} "
                        f"does not match expected {expected}"
                    )
            cache_cursors = self.cuda_graph_backend.validate_session_cursors(
                native_frames=frame_count,
                thinker_prefix_tokens=self.system_prompt_tokens,
            )
            graph_summary = self.cuda_graph_backend.metadata()
            graph_summary.update(
                {
                    "session_replays": session_replays,
                    "expected_session_replays": expected_replays,
                    "session_replays_verified": True,
                    "session_cache_cursors": cache_cursors,
                    "session_cache_cursors_verified": True,
                }
            )
        return {
            "session_id": self.session_id,
            "system_prompt_kind": self.system_prompt_kind,
            "system_prompt_tokens": self.system_prompt_tokens,
            "inference_type": "async_macro",
            "latency_ms": self.k * 80,
            "frames_per_unit": self.k,
            "frames": min(frame_count, self.logical_frames),
            "physical_frames": frame_count,
            "macro_units": int(self.state.unit_index),
            "input_limit_frames": self.max_context_frames,
            "kv_cache_capacity_frames": self.kv_cache_capacity_frames,
            "kv_cache_capacity_effective_ms": self.kv_cache_capacity_effective_ms,
            "cuda_graphs_required": self.require_cuda_graphs,
            "cuda_graphs": graph_summary,
            "control_signal_contract": self.model.control_signal_contract,
            "checkpoint": self.checkpoint_metadata,
            "sampling": {
                name: getattr(self.sampling, name) for name in self.sampling.__dataclass_fields__
            },
            "text_trace": self.text_trace,
            "generated_text_tokens": self.generated_text[: self.logical_frames],
            "environment_stream": self.env_stream.state_summary(),
            "self_stream": self.self_stream.state_summary(),
            "playback": self.playback.state_summary(),
            "playback_dezipper": self.dezipper.state_summary(),
            "renderer": {
                "code_frames": self.renderer.code_frames,
                "transformer_cache_frames": self.renderer.transformer_cache_frames,
                "forward_calls": self.renderer.forward_calls,
                "max_conv_window_frames": self.renderer.max_conv_window_frames,
            },
            "macro_wall_ms": {
                "count": len(wall),
                "total": float(sum(wall)),
                "average": float(sum(wall) / len(wall)) if wall else 0.0,
                "minimum": float(min(wall)) if wall else 0.0,
                "maximum": float(max(wall)) if wall else 0.0,
            },
            "forward_timings": self.timings.summary(),
        }


class D2LiveSession:
    """Buffer browser PCM into trained macro units without resetting model state."""

    def __init__(
        self,
        owner: Any,
        macro_session: _MacroSession,
        *,
        seed: int,
    ) -> None:
        self.owner = owner
        self.macro_session = macro_session
        self.seed = int(seed)
        self.buffer = bytearray()
        self.closed = False
        self.summary: dict[str, Any] | None = None

    @property
    def macro_bytes(self) -> int:
        return self.macro_session.k * INPUT_FRAME_SAMPLES * 2

    async def push_pcm16(self, payload: bytes) -> LiveBatch:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("PCM16 input must be bytes-like")
        packet = bytes(payload)
        if len(packet) % 2:
            raise ValueError("PCM16 input must contain complete samples")
        return await self.owner._execute(self._push_sync, packet)

    def _push_sync(self, packet: bytes) -> LiveBatch:
        if self.closed:
            raise RuntimeError("live session is closed")
        self.buffer.extend(packet)
        pcm_frames: list[bytes] = []
        frame_events: list[dict[str, Any]] = []
        while len(self.buffer) >= self.macro_bytes:
            if (
                self.macro_session.state.unit_index + 1
            ) * self.macro_session.k > self.owner.max_context_frames:
                raise ValueError(
                    "Live session reached its fixed context limit; stop and start a new conversation"
                )
            payload = bytes(self.buffer[: self.macro_bytes])
            del self.buffer[: self.macro_bytes]
            produced_pcm, produced_events = self.macro_session.step_macro(_pcm_tensor(payload))
            pcm_frames.extend(produced_pcm)
            frame_events.extend(produced_events)
        return LiveBatch(tuple(pcm_frames), tuple(frame_events))

    async def close(self) -> dict[str, Any]:
        return await self.owner._execute(self._close_sync)

    def _close_sync(self) -> dict[str, Any]:
        if self.closed:
            return dict(self.summary or {})
        try:
            summary = self.macro_session.close()
            summary.update(
                {
                    "seed": self.seed,
                    "dropped_partial_input_samples": len(self.buffer) // 2,
                }
            )
            self.summary = summary
            return dict(summary)
        finally:
            self.buffer.clear()
            self.closed = True
            self.owner._active_live_session = False
