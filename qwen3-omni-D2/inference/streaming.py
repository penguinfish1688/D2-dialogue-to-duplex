from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any
import torch
from d2_qwen.model.constants import (
    AUDIO_HZ,
    CODEBOOKS,
    CODEBOOK_SIZE,
    CODEC_HZ,
    SAMPLE_RATE_AUDIO,
    SAMPLE_RATE_CODEC,
)
from d2_qwen.model.utils import module_device, embed_on
from .audio import (
    OMNI_CODE2WAV_CAUSAL_TAIL_SAMPLES_24K,
    OMNI_CODE2WAV_PLAYBACK_DELAY_SAMPLES_16K,
    code2wav_output_samples,
)
from .timing import ForwardTimings, measure_forward
from .cache import compact_transformer_cache


CODE2WAV_MIN_PREFIX_FRAMES = 1


def inference_autocast(device: torch.device, dtype: torch.dtype):
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda" and dtype in (torch.bfloat16, torch.float16),
    )


def model_compute_dtype(model: Any) -> torch.dtype:
    dtype = getattr(model, "compute_dtype", None)
    if dtype is None:
        return next(model.qwen.parameters()).dtype
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError(f"model.compute_dtype must be a floating torch.dtype, got {dtype!r}")
    return dtype


def record_component_trace(
    trace: dict[str, torch.Tensor] | None,
    name: str,
    tensor: torch.Tensor,
) -> None:
    if trace is not None:
        trace[name] = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()


def sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    diagnostic_token_id: int | None = None,
    diagnostics: dict[str, float | int | bool] | None = None,
) -> torch.Tensor:
    if (diagnostic_token_id is None) != (diagnostics is None):
        raise ValueError("diagnostic_token_id and diagnostics must be provided together")
    diagnostic_logits: torch.Tensor | None = None
    if diagnostics is not None:
        if logits.numel() != logits.shape[-1]:
            raise ValueError("sampling diagnostics require one logit row")
        token_id = int(diagnostic_token_id)
        if token_id < 0 or token_id >= int(logits.shape[-1]):
            raise ValueError(f"diagnostic token ID is out of range: {token_id}")
        diagnostic_logits = logits.reshape(-1)
        token_logit = diagnostic_logits[token_id]
        best_logit, best_token = diagnostic_logits.max(dim=-1)
        diagnostics.update(
            {
                "token_id": token_id,
                "logit": float(token_logit.item()),
                "best_token_id": int(best_token.item()),
                "best_logit": float(best_logit.item()),
                "margin_to_best": float((token_logit - best_logit).item()),
                "rank": int((diagnostic_logits > token_logit).sum().item()) + 1,
            }
        )
    if temperature <= 0.0:
        sampled = logits.argmax(dim=-1)
        if diagnostics is not None:
            diagnostics.update(
                {
                    "sampling_probability": float(
                        int(sampled.reshape(-1)[0].item()) == int(diagnostic_token_id)
                    ),
                    "selected": bool(
                        int(sampled.reshape(-1)[0].item()) == int(diagnostic_token_id)
                    ),
                }
            )
        return sampled
    scaled = logits / float(temperature)
    if 0 < int(top_k) < int(scaled.shape[-1]):
        cutoff = torch.topk(scaled, int(top_k), dim=-1).values[..., -1:]
        scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
    if not 0.0 < float(top_p) <= 1.0:
        raise ValueError(f"top_p must be in (0,1], got {top_p}")
    if float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        remove = sorted_probs.cumsum(dim=-1) - sorted_probs > float(top_p)
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(scaled, float("-inf"))
        scaled = filtered.scatter(-1, sorted_indices, sorted_logits)
    probabilities = torch.softmax(scaled, dim=-1)
    sampled = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
    if diagnostics is not None:
        diagnostics.update(
            {
                "sampling_probability": float(
                    probabilities.reshape(-1)[int(diagnostic_token_id)].item()
                ),
                "selected": bool(int(sampled.reshape(-1)[0].item()) == int(diagnostic_token_id)),
            }
        )
    return sampled


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_history: list[int] | tuple[int, ...],
    penalty: float,
) -> torch.Tensor:
    """Match HF's repetition penalty over previously generated CB0 IDs."""
    penalty = float(penalty)
    if penalty <= 0.0:
        raise ValueError(f"repetition penalty must be positive, got {penalty}")
    if penalty == 1.0 or not token_history:
        return logits
    token_ids = sorted(
        {int(token_id) for token_id in token_history if 0 <= int(token_id) < int(logits.shape[-1])}
    )
    if not token_ids:
        return logits
    out = logits.clone()
    indices = torch.tensor(token_ids, dtype=torch.long, device=out.device)
    scores = out.index_select(-1, indices)
    scores = torch.where(scores < 0, scores * penalty, scores / penalty)
    out.index_copy_(-1, indices, scores)
    return out


def cache_sequence_length(cache: Any) -> int | None:
    if cache is None or not hasattr(cache, "get_seq_length"):
        return None
    return int(cache.get_seq_length())


def embed_codec_frame(model: Any, codes: torch.Tensor) -> torch.Tensor:
    """Embed one complete physical frame for the next Talker clock."""
    codes = codes.long().view(1, 1, CODEBOOKS)
    embedded = embed_on(model.qwen.talker.model.codec_embedding, codes[:, :, 0])
    target_device = embedded.device
    residual_embeddings = model.qwen.talker.code_predictor.get_input_embeddings()
    for group in range(1, CODEBOOKS):
        embedded = embedded + embed_on(
            residual_embeddings[group - 1],
            codes[:, :, group],
            target_device=target_device,
        )
    return embedded


@torch.no_grad()
def generate_codec_frame(
    model: Any,
    talker_hidden: torch.Tensor,
    *,
    codec0_history: list[int] | tuple[int, ...],
    codec_repetition_penalty: float,
    codec_temperature: float,
    codec_top_k: int,
    codec_top_p: float,
    residual_temperature: float,
    residual_top_k: int,
    residual_top_p: float,
    initial_residual_seed: torch.Tensor | None = None,
    forced_code0: int | None = None,
    timings: ForwardTimings | None = None,
    forward_backend: Any | None = None,
    trace: dict[str, torch.Tensor] | None = None,
    allow_codec_eos: bool = False,
    codec_eos_diagnostics: dict[str, float | int | bool] | None = None,
) -> torch.Tensor:
    """Generate one complete physical frame ``CB0..CB15[t]``.

    This is the official Qwen recurrence: CB0 comes from the main Talker head,
    then the CodePredictor autoregresses all residual books of the same frame.
    """
    dtype = model_compute_dtype(model)
    codec0_head = model.qwen.talker.codec_head
    codec0_device = module_device(codec0_head, talker_hidden.device)
    with measure_forward(timings, "codec0_head", codec0_device):
        with inference_autocast(codec0_device, dtype):
            codec0_logits = codec0_head(talker_hidden.to(device=codec0_device))[
                ...,
                : (int(model.talker_codec_vocab_size) if allow_codec_eos else CODEBOOK_SIZE),
            ]
    if allow_codec_eos:
        eos_id = int(model.talker_codec_eos_id)
        special_mask = (
            torch.arange(
                int(codec0_logits.shape[-1]),
                device=codec0_logits.device,
            )
            >= CODEBOOK_SIZE
        )
        special_mask[eos_id] = False
        codec0_logits = codec0_logits.masked_fill(special_mask, float("-inf"))
    codec0_logits = apply_repetition_penalty(
        codec0_logits,
        codec0_history,
        codec_repetition_penalty,
    )
    record_component_trace(trace, "talker/codec0_logits", codec0_logits)
    if forced_code0 is None:
        active_eos_diagnostics = codec_eos_diagnostics if allow_codec_eos else None
        code0 = sample_logits(
            codec0_logits,
            temperature=codec_temperature,
            top_k=codec_top_k,
            top_p=codec_top_p,
            diagnostic_token_id=(
                int(model.talker_codec_eos_id) if active_eos_diagnostics is not None else None
            ),
            diagnostics=active_eos_diagnostics,
        )
    else:
        code0 = torch.full(
            codec0_logits.shape[:-1],
            int(forced_code0),
            dtype=torch.long,
            device=codec0_logits.device,
        )
    if allow_codec_eos and bool((code0 == int(model.talker_codec_eos_id)).all().item()):
        eos_pattern = model.codec_silence_frame.detach().long().cpu().view(CODEBOOKS).clone()
        eos_pattern[0] = int(model.talker_codec_eos_id)
        return eos_pattern
    if initial_residual_seed is not None:
        seed = initial_residual_seed.detach().long().cpu().view(CODEBOOKS)
        return torch.cat((code0.detach().cpu().view(1), seed[1:]), dim=0)
    if forward_backend is not None and forward_backend.code_predictor_graph_enabled:
        return forward_backend.generate_residual_codes(
            model,
            talker_hidden=talker_hidden,
            code0=code0,
            residual_temperature=residual_temperature,
            residual_top_k=residual_top_k,
            residual_top_p=residual_top_p,
            timings=timings,
            trace=trace,
        )
    codes = [code0]

    cp_model = model.qwen.talker.code_predictor.model
    cp_device = module_device(cp_model, talker_hidden.device)
    with measure_forward(timings, "code_predictor", cp_device):
        residual_embeddings = model.qwen.talker.code_predictor.get_input_embeddings()
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
        with inference_autocast(first_inputs.device, dtype):
            cp_outputs = cp_model(
                inputs_embeds=first_inputs,
                attention_mask=torch.ones((1, 2), dtype=torch.long, device=first_inputs.device),
                use_cache=True,
            )
        cp_cache = cp_outputs.past_key_values
        for group in range(1, CODEBOOKS):
            hidden = cp_outputs.last_hidden_state[:, -1, :]
            record_component_trace(
                trace,
                f"code_predictor/codebook_{group:02d}_hidden",
                hidden,
            )
            head = model.qwen.talker.code_predictor.lm_head[group - 1]
            head_device = module_device(head, hidden.device)
            # Training evaluates every residual head inside the outer BF16
            # autocast region. Keep that same numerical contract in inference,
            # including for the FP32-master trainable heads.
            with inference_autocast(head_device, dtype):
                logits = head(hidden.to(device=head_device))[..., :CODEBOOK_SIZE]
            record_component_trace(
                trace,
                f"code_predictor/codebook_{group:02d}_logits",
                logits,
            )
            code = sample_logits(
                logits,
                temperature=residual_temperature,
                top_k=residual_top_k,
                top_p=residual_top_p,
            )
            codes.append(code)
            if group == CODEBOOKS - 1:
                break
            next_embed = embed_on(
                residual_embeddings[group - 1],
                code,
                target_device=hidden.device,
            ).unsqueeze(1)
            with inference_autocast(next_embed.device, dtype):
                cp_outputs = cp_model(
                    inputs_embeds=next_embed,
                    attention_mask=torch.ones(
                        (1, group + 2),
                        dtype=torch.long,
                        device=next_embed.device,
                    ),
                    past_key_values=cp_cache,
                    use_cache=True,
                )
            cp_cache = cp_outputs.past_key_values
    return torch.stack(codes, dim=-1).view(CODEBOOKS).detach().cpu()


@torch.no_grad()
def prefill_talker_speaker(
    model: Any,
    timings: ForwardTimings | None = None,
) -> Any:
    dtype = model_compute_dtype(model)
    prefix_device = module_device(model.qwen.talker.model)
    # The prefix includes the FP32-master text-projection path and is built
    # inside the outer autocast region during training.
    with inference_autocast(prefix_device, dtype):
        prefix = model._talker_speaker_prefix(
            batch_size=1,
            target_device=prefix_device,
            target_dtype=dtype,
        )
    with measure_forward(timings, "talker_speaker_prefill", prefix.device):
        with inference_autocast(prefix.device, dtype):
            outputs = model.qwen.talker.model(
                inputs_embeds=prefix,
                attention_mask=torch.ones((1, 1), dtype=torch.long, device=prefix.device),
                use_cache=True,
            )
    cache = outputs.past_key_values
    cache_len = cache_sequence_length(cache)
    if cache_len is not None and cache_len != 1:
        raise RuntimeError(f"Talker speaker prefill KV length {cache_len} != 1")
    return cache


@torch.no_grad()
def decode_omni_code_prefix(model: Any, codes: torch.Tensor) -> torch.Tensor:
    if codes.ndim != 2 or int(codes.shape[1]) != CODEBOOKS:
        raise ValueError(f"Expected native Mimi codes [T,{CODEBOOKS}], got {tuple(codes.shape)}")
    if codes.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    if int(codes.min()) < 0 or int(codes.max()) >= CODEBOOK_SIZE:
        raise ValueError("Generated native Mimi ID is outside the Omni codebook range")
    code2wav = model.qwen.code2wav
    code_device = module_device(code2wav)
    input_codes = codes.transpose(0, 1).unsqueeze(0).to(device=code_device, dtype=torch.long)
    # A direct causal prefix avoids reintroducing one right-tail crop at every
    # artificial chunk boundary.  At the default 260-frame ceiling this is the
    # same single forward used by official chunked_decode.
    wav = code2wav(input_codes)
    expected_samples = expected_code2wav_samples(code2wav, int(codes.shape[0]))
    if int(wav.shape[-1]) != expected_samples:
        raise RuntimeError(
            f"Code2Wav causal output length {int(wav.shape[-1])} != architecture-derived "
            f"{expected_samples} for {int(codes.shape[0])} code frames"
        )
    return wav.detach().float().cpu().view(-1).contiguous()


def expected_code2wav_samples(code2wav: Any, code_frames: int) -> int:
    return code2wav_output_samples(
        int(code_frames),
        upsampling_ratios=tuple(int(value) for value in code2wav.config.upsampling_ratios),
        upsample_rates=tuple(int(value) for value in code2wav.config.upsample_rates),
    )


class IncrementalCode2WavRenderer:
    """KV-cache the transformer and recompute only bounded convolution context."""

    def __init__(
        self,
        model: Any,
        *,
        left_context_frames: int,
        parity_atol: float,
        parity_mean_atol: float,
        verify_reference: bool,
        capture_components: bool,
        retain_raw_prefix: bool = True,
        timings: ForwardTimings | None = None,
        forward_backend: Any | None = None,
    ) -> None:
        self.model = model
        self.code2wav = model.qwen.code2wav
        self.left_context_frames = int(left_context_frames)
        if self.left_context_frames < 1:
            raise ValueError("Code2Wav left context must be positive")
        self.parity_atol = float(parity_atol)
        self.parity_mean_atol = float(parity_mean_atol)
        self.verify_reference = bool(verify_reference)
        self.retain_raw_prefix = bool(retain_raw_prefix)
        self.timings = timings
        self.forward_backend = forward_backend
        if self.parity_atol <= 0 or self.parity_mean_atol <= 0:
            raise ValueError("Code2Wav parity tolerances must be positive")
        if self.verify_reference and not self.retain_raw_prefix:
            raise ValueError("Code2Wav reference checks require the raw prefix")
        self.transformer_cache: Any = None
        self.transformer_cache_frames = 0
        self.transformed_frames: list[torch.Tensor] = []
        self.raw_prefix = torch.empty(0, dtype=torch.float32)
        self._stability_prefix = torch.empty(0, dtype=torch.float32)
        self.raw_samples_emitted = 0
        self.code_frames = 0
        self.forward_calls = 0
        self.max_conv_window_frames = 0
        self.parity_checks: list[dict[str, Any]] = []
        self.component_trace: dict[str, torch.Tensor] | None = {} if capture_components else None

    @torch.no_grad()
    def compact_transformer_context(self, *, memory_frames: int) -> None:
        memory_frames = int(memory_frames)
        if not 0 <= memory_frames <= self.transformer_cache_frames:
            raise ValueError(
                f"invalid Code2Wav memory {memory_frames} for "
                f"{self.transformer_cache_frames} cached frames"
            )
        if self.transformer_cache is None:
            if memory_frames:
                raise RuntimeError("Code2Wav cache is empty")
            return
        if self.forward_backend is not None and self.forward_backend.code2wav_graph_enabled:
            self.forward_backend.compact_code2wav_transformer(
                memory_frames=memory_frames,
            )
        else:
            compact_transformer_cache(
                self.transformer_cache,
                target_tokens=memory_frames,
            )
        self.transformer_cache_frames = memory_frames

    @torch.no_grad()
    def prefill_constant_frame(
        self,
        code_frame: torch.Tensor,
        *,
        frames: int,
    ) -> torch.Tensor:
        """Bulk-prefill a fresh renderer with one repeated physical frame."""

        count = int(frames)
        if count < 1:
            raise ValueError("Code2Wav prefill frames must be positive")
        if code_frame.numel() != CODEBOOKS:
            raise ValueError(
                f"Expected one Code2Wav frame [{CODEBOOKS}], got {tuple(code_frame.shape)}"
            )
        if (
            self.transformer_cache is not None
            or self.transformer_cache_frames
            or self.transformed_frames
            or self.raw_samples_emitted
            or self.code_frames
        ):
            raise RuntimeError("Code2Wav bulk prefill requires a fresh renderer")
        if self.forward_backend is not None:
            raise ValueError("Code2Wav bulk prefill does not support a graph backend")

        code_device = module_device(self.code2wav)
        repeated = (
            code_frame.detach()
            .long()
            .view(1, CODEBOOKS, 1)
            .expand(1, CODEBOOKS, count)
            .to(code_device)
        )
        with measure_forward(self.timings, "code2wav_transformer", code_device):
            hidden = self.code2wav.code_embedding(repeated + self.code2wav.code_offset).mean(1)
            outputs = self.code2wav.pre_transformer(
                inputs_embeds=hidden,
                use_cache=True,
            )
        transformed = outputs.last_hidden_state.detach()
        if int(transformed.shape[1]) != count:
            raise RuntimeError("Code2Wav bulk prefill length changed")
        self.transformer_cache = outputs.past_key_values
        cache_len = cache_sequence_length(self.transformer_cache)
        if cache_len is not None and cache_len != count:
            raise RuntimeError(f"Code2Wav bulk-prefill KV length {cache_len} != {count}")
        self.transformer_cache_frames = count
        retained = min(count, self.left_context_frames + 1)
        self.transformed_frames = [
            transformed[:, index : index + 1, :].clone() for index in range(count - retained, count)
        ]
        self.code_frames = count
        self.raw_samples_emitted = expected_code2wav_samples(
            self.code2wav,
            count,
        )

        if self.retain_raw_prefix or count <= self.left_context_frames + 1:
            decoded = self._decode_transformed(transformed)
            if int(decoded.numel()) != self.raw_samples_emitted:
                raise RuntimeError("Code2Wav bulk-prefill output length changed")
            if self.retain_raw_prefix:
                self.raw_prefix = decoded
            else:
                self._stability_prefix = decoded
        else:
            window = torch.cat(self.transformed_frames, dim=1)
            decoded = self._decode_transformed(window)
            expected_window = expected_code2wav_samples(
                self.code2wav,
                retained,
            )
            if int(decoded.numel()) != expected_window:
                raise RuntimeError("Code2Wav bulk-prefill window length changed")

        last_samples = (
            expected_code2wav_samples(self.code2wav, 1)
            if count == 1
            else int(round(SAMPLE_RATE_CODEC / CODEC_HZ))
        )
        return decoded[-last_samples:].contiguous()

    @torch.no_grad()
    def _transform_one(self, code_frame: torch.Tensor) -> torch.Tensor:
        code_device = module_device(self.code2wav)
        with measure_forward(self.timings, "code2wav_transformer", code_device):
            codes = code_frame.long().view(1, CODEBOOKS, 1).to(code_device)
            hidden = self.code2wav.code_embedding(codes + self.code2wav.code_offset).mean(1)
            record_component_trace(
                self.component_trace,
                "code2wav/code_embedding",
                hidden,
            )
            if self.forward_backend is not None and self.forward_backend.code2wav_graph_enabled:
                outputs = self.forward_backend.code2wav_transformer_forward(
                    hidden,
                    past_key_values=self.transformer_cache,
                    frame_index=self.code_frames,
                )
            else:
                outputs = self.code2wav.pre_transformer(
                    inputs_embeds=hidden,
                    past_key_values=self.transformer_cache,
                    use_cache=True,
                )
        self.transformer_cache = outputs.past_key_values
        record_component_trace(
            self.component_trace,
            "code2wav/transformer_hidden",
            outputs.last_hidden_state,
        )
        cache_len = cache_sequence_length(self.transformer_cache)
        ring_capacity = (
            None
            if self.forward_backend is None
            else getattr(self.forward_backend, "fixed_ring_cache_frames", None)
        )
        expected_cache_frames = (
            self.transformer_cache_frames + 1
            if ring_capacity is None
            else min(int(ring_capacity), self.transformer_cache_frames + 1)
        )
        if cache_len is not None and cache_len != expected_cache_frames:
            raise RuntimeError(
                f"Code2Wav transformer KV length {cache_len} != {expected_cache_frames}"
            )
        self.transformer_cache_frames = expected_cache_frames
        return outputs.last_hidden_state.detach().clone()

    @torch.no_grad()
    def _decode_transformed(self, transformed: torch.Tensor) -> torch.Tensor:
        with measure_forward(
            self.timings,
            "code2wav_decoder",
            module_device(self.code2wav, transformed.device),
        ):
            if self.forward_backend is not None and self.forward_backend.code2wav_graph_enabled:
                wav = self.forward_backend.code2wav_decoder_forward(transformed)
            else:
                hidden = transformed.permute(0, 2, 1)
                for blocks in self.code2wav.upsample:
                    for block in blocks:
                        hidden = block(hidden)
                wav = hidden
                for block in self.code2wav.decoder:
                    wav = block(wav)
        self.forward_calls += 1
        self.max_conv_window_frames = max(
            self.max_conv_window_frames,
            int(transformed.shape[1]),
        )
        record_component_trace(
            self.component_trace,
            "code2wav/waveform",
            wav,
        )
        return wav.clamp(min=-1, max=1).detach().float().cpu().view(-1).contiguous()

    @torch.no_grad()
    def check_reference(self, all_codes: torch.Tensor, label: str) -> None:
        if not self.retain_raw_prefix:
            raise RuntimeError("Code2Wav reference checks require the raw prefix")
        reference = decode_omni_code_prefix(self.model, all_codes)
        if reference.shape != self.raw_prefix.shape:
            raise RuntimeError(
                f"Incremental Code2Wav {label} length {self.raw_prefix.numel()} != "
                f"full-prefix {reference.numel()}"
            )
        diff = (reference - self.raw_prefix).abs()
        max_abs = float(diff.max()) if diff.numel() else 0.0
        mean_abs = float(diff.mean()) if diff.numel() else 0.0
        check = {
            "label": label,
            "frames": self.code_frames,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
        }
        self.parity_checks.append(check)
        if max_abs > self.parity_atol or mean_abs > self.parity_mean_atol:
            raise RuntimeError(
                f"Incremental/full Code2Wav parity failed at {label}: {check}, "
                f"max_atol={self.parity_atol} mean_atol={self.parity_mean_atol}"
            )

    @torch.no_grad()
    def _append_code_frame(self, code_frame: torch.Tensor) -> torch.Tensor:
        if code_frame.numel() != CODEBOOKS:
            raise ValueError(
                f"Expected one generated code frame [{CODEBOOKS}], got {tuple(code_frame.shape)}"
            )
        total_frames = self.code_frames + 1
        transformed = self._transform_one(code_frame)
        self.transformed_frames.append(transformed)
        if len(self.transformed_frames) > self.left_context_frames + 1:
            del self.transformed_frames[0]
        self.code_frames = total_frames
        if total_frames < CODE2WAV_MIN_PREFIX_FRAMES:
            return torch.empty(0, dtype=torch.float32)

        window_frames = min(total_frames, self.left_context_frames + 1)
        transformed_window = torch.cat(self.transformed_frames[-window_frames:], dim=1)
        window_wav = self._decode_transformed(transformed_window)
        expected_window = expected_code2wav_samples(self.code2wav, window_frames)
        if int(window_wav.numel()) != expected_window:
            raise RuntimeError(
                f"Incremental Code2Wav window emitted {window_wav.numel()} != {expected_window}"
            )
        if total_frames == 1:
            raw_chunk = window_wav
        else:
            samples_per_code = int(round(SAMPLE_RATE_CODEC / CODEC_HZ))
            if total_frames <= self.left_context_frames + 1:
                old_reference = window_wav[:-samples_per_code]
                previous_prefix = (
                    self.raw_prefix if self.retain_raw_prefix else self._stability_prefix
                )
                diff = (old_reference - previous_prefix).abs()
                max_abs = float(diff.max()) if diff.numel() else 0.0
                if old_reference.shape != previous_prefix.shape or max_abs > self.parity_atol:
                    raise RuntimeError(
                        "Code2Wav growing-prefix stability failed: "
                        f"frames={total_frames} old={previous_prefix.numel()} "
                        f"reference={old_reference.numel()} max_abs={max_abs} "
                        f"atol={self.parity_atol}"
                    )
            raw_chunk = window_wav[-samples_per_code:]

        self.raw_samples_emitted += int(raw_chunk.numel())
        if self.retain_raw_prefix:
            self.raw_prefix = torch.cat((self.raw_prefix, raw_chunk), dim=0)
        elif total_frames <= self.left_context_frames + 1:
            self._stability_prefix = torch.cat(
                (self._stability_prefix, raw_chunk),
                dim=0,
            )
        else:
            self._stability_prefix = torch.empty(0, dtype=torch.float32)

        expected_total = expected_code2wav_samples(self.code2wav, total_frames)
        if self.raw_samples_emitted != expected_total:
            raise RuntimeError(
                f"Incremental Code2Wav emitted {self.raw_samples_emitted} != "
                f"{expected_total} samples"
            )
        if self.retain_raw_prefix and int(self.raw_prefix.numel()) != expected_total:
            raise RuntimeError("Retained Code2Wav prefix length is inconsistent")
        return raw_chunk

    @torch.no_grad()
    def append(self, all_codes: torch.Tensor) -> torch.Tensor:
        if not self.retain_raw_prefix:
            raise RuntimeError("Use append_frame() when raw-prefix retention is disabled")
        if all_codes.ndim != 2 or int(all_codes.shape[1]) != CODEBOOKS:
            raise ValueError(
                f"Expected generated codes [T,{CODEBOOKS}], got {tuple(all_codes.shape)}"
            )
        total_frames = int(all_codes.shape[0])
        if total_frames != self.code_frames + 1:
            raise RuntimeError(
                f"Incremental Code2Wav expected frame {self.code_frames + 1}, got {total_frames}"
            )
        self._append_code_frame(all_codes[-1])
        if self.verify_reference and total_frames == CODE2WAV_MIN_PREFIX_FRAMES:
            self.check_reference(all_codes, "startup")
        if self.verify_reference and total_frames == self.left_context_frames + 2:
            self.check_reference(all_codes, "first_rolling_window")
        return self.raw_prefix

    @torch.no_grad()
    def append_frame(self, code_frame: torch.Tensor) -> torch.Tensor:
        if self.retain_raw_prefix:
            raise RuntimeError("append_frame() requires raw-prefix retention to be disabled")
        return self._append_code_frame(code_frame)


@dataclass(frozen=True)
class IncrementalPlaybackFrame:
    playback_24k: torch.Tensor
    self_feedback_16k: torch.Tensor
    playback_latency_samples: int


class IncrementalCode2WavPlayback:
    """Publish one bounded, prefix-equivalent playback frame at a time."""

    def __init__(
        self,
        *,
        self_delay_samples: int = OMNI_CODE2WAV_PLAYBACK_DELAY_SAMPLES_16K,
    ) -> None:
        self.frames = 0
        self.playback_samples_per_frame = int(round(SAMPLE_RATE_CODEC / AUDIO_HZ))
        self.feedback_samples_per_frame = int(round(SAMPLE_RATE_AUDIO / AUDIO_HZ))
        self.playback_latency_samples = OMNI_CODE2WAV_CAUSAL_TAIL_SAMPLES_24K
        self.self_delay_samples = int(self_delay_samples)
        if self.self_delay_samples < 0:
            raise ValueError("self feedback delay must be non-negative")
        numerators = (
            torch.arange(self.feedback_samples_per_frame, dtype=torch.long) * SAMPLE_RATE_CODEC
        )
        self._resample_left = torch.div(
            numerators,
            SAMPLE_RATE_AUDIO,
            rounding_mode="floor",
        )
        self._resample_right = self._resample_left + 1
        self._resample_weight = torch.remainder(numerators, SAMPLE_RATE_AUDIO).to(
            torch.float32
        ) / float(SAMPLE_RATE_AUDIO)
        self._previous_source_sample: torch.Tensor | None = None
        self._self_delay_tail = torch.zeros(
            self.self_delay_samples,
            dtype=torch.float32,
        )
        self.max_raw_chunk_samples = 0

    def _aligned_playback_frame(self, raw_chunk: torch.Tensor) -> torch.Tensor:
        values = raw_chunk.detach().cpu().float().flatten().contiguous()
        expected = self.playback_samples_per_frame
        if self.frames == 0:
            expected -= self.playback_latency_samples
        if int(values.numel()) != expected:
            raise ValueError(
                f"Code2Wav frame {self.frames} has {values.numel()} raw samples; "
                f"expected {expected}"
            )
        self.max_raw_chunk_samples = max(
            self.max_raw_chunk_samples,
            int(values.numel()),
        )
        if self.frames:
            return values
        return torch.cat(
            (
                values.new_zeros(self.playback_latency_samples),
                values,
            ),
            dim=0,
        )

    def _resample_feedback_frame(self, playback: torch.Tensor) -> torch.Tensor:
        previous = self._previous_source_sample
        if previous is None:
            previous = playback[:1]
        source = torch.cat((previous, playback), dim=0)
        weight = self._resample_weight.to(dtype=source.dtype)
        return (
            source[self._resample_left] * (1.0 - weight) + source[self._resample_right] * weight
        ).contiguous()

    def prefill_inactive(
        self,
        last_raw_chunk: torch.Tensor,
        *,
        physical_frames: int,
    ) -> None:
        """Restore the playback boundary after silent physical prefix frames."""

        frames = int(physical_frames)
        if frames < 1:
            raise ValueError("playback prefill frames must be positive")
        if self.frames or self._previous_source_sample is not None:
            raise RuntimeError("playback prefill requires a fresh assembler")
        values = last_raw_chunk.detach().cpu().float().flatten().contiguous()
        expected = self.playback_samples_per_frame
        if frames == 1:
            expected -= self.playback_latency_samples
        if int(values.numel()) != expected:
            raise ValueError(
                f"last Code2Wav prefix chunk has {values.numel()} samples; expected {expected}"
            )
        playback = values
        if frames == 1:
            playback = torch.cat(
                (
                    values.new_zeros(self.playback_latency_samples),
                    values,
                ),
                dim=0,
            )
        self._previous_source_sample = playback[-1:].clone()
        self.frames = frames
        self.max_raw_chunk_samples = int(values.numel())

    def append(
        self,
        raw_chunk: torch.Tensor,
        *,
        active: bool,
    ) -> IncrementalPlaybackFrame:
        playback = self._aligned_playback_frame(raw_chunk)
        feedback = self._resample_feedback_frame(playback)
        activity = float(bool(active))
        masked_playback = (playback * activity).contiguous()
        masked_feedback = (feedback * activity).contiguous()
        delay = self.self_delay_samples
        if delay:
            delayed_feedback = torch.cat(
                (self._self_delay_tail, masked_feedback[:-delay]),
                dim=0,
            )
            self._self_delay_tail = masked_feedback[-delay:].clone()
        else:
            delayed_feedback = masked_feedback
        self._previous_source_sample = playback[-1:].clone()
        self.frames += 1
        return IncrementalPlaybackFrame(
            playback_24k=masked_playback,
            self_feedback_16k=delayed_feedback,
            playback_latency_samples=self.playback_latency_samples,
        )

    def state_summary(self) -> dict[str, int]:
        return {
            "frames": self.frames,
            "retained_source_samples": int(
                self._previous_source_sample.numel()
                if self._previous_source_sample is not None
                else 0
            ),
            "retained_self_delay_samples": int(self._self_delay_tail.numel()),
            "max_raw_chunk_samples": self.max_raw_chunk_samples,
        }


class StreamingPcmBoundaryDezipper:
    """Causally remove large frame-edge steps from audible playback only."""

    def __init__(self, *, transition_samples: int, minimum_jump: float) -> None:
        self.transition_samples = int(transition_samples)
        self.minimum_jump = float(minimum_jump)
        if self.transition_samples < 0:
            raise ValueError("transition_samples must be non-negative")
        if not math.isfinite(self.minimum_jump) or self.minimum_jump < 0.0:
            raise ValueError("minimum_jump must be non-negative")
        if self.transition_samples > 1:
            phase = torch.arange(self.transition_samples, dtype=torch.float32)
            phase.mul_(math.pi / float(self.transition_samples - 1))
            self._weights = phase.cos_().add_(1.0).mul_(0.5)
        elif self.transition_samples == 1:
            self._weights = torch.ones(1, dtype=torch.float32)
        else:
            self._weights = torch.empty(0, dtype=torch.float32)
        self.previous_sample: float | None = None
        self.boundaries = 0
        self.corrected_boundaries = 0
        self.maximum_jump = 0.0
        self.maximum_correction = 0.0

    def append(self, frame: torch.Tensor) -> torch.Tensor:
        output = frame.detach().cpu().float().flatten().contiguous().clone()
        if output.numel() == 0:
            return output
        if self.previous_sample is not None:
            self.boundaries += 1
            correction = self.previous_sample - float(output[0])
            jump = abs(correction)
            self.maximum_jump = max(self.maximum_jump, jump)
            if self.transition_samples and jump > self.minimum_jump:
                samples = min(self.transition_samples, int(output.numel()))
                weights = self._weights[:samples]
                if samples != self.transition_samples and samples > 1:
                    phase = torch.arange(samples, dtype=torch.float32)
                    phase.mul_(math.pi / float(samples - 1))
                    weights = phase.cos_().add_(1.0).mul_(0.5)
                output[:samples].add_(weights * correction).clamp_(-1.0, 1.0)
                self.corrected_boundaries += 1
                self.maximum_correction = max(self.maximum_correction, jump)
        self.previous_sample = float(output[-1])
        return output

    def state_summary(self) -> dict[str, int | float]:
        return {
            "transition_samples": self.transition_samples,
            "minimum_jump": self.minimum_jump,
            "boundaries": self.boundaries,
            "corrected_boundaries": self.corrected_boundaries,
            "maximum_jump": self.maximum_jump,
            "maximum_correction": self.maximum_correction,
            "retained_samples": int(self.previous_sample is not None),
        }
