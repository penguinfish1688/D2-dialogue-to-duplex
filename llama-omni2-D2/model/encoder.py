"""Stage-1 block-causal Whisper encoder and distillation objective.

The implementation wraps the released OpenAI Whisper encoder and speech
projector without importing either sibling D2 project.  It batches one
``tail-2 + current macro`` convolution window per macro, then applies a
block-causal mask to the released Transformer blocks.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from d2_llama.core.lora import LoRALinear, install_lora
from d2_llama.model.official import (
    OFFICIAL_MODEL_ID,
    OFFICIAL_MODEL_REVISION,
    OFFICIAL_REPOSITORY_REVISION,
)

from d2_llama.core.constants import (
    ADAPTOR_GROUP_ROWS,
    ADAPTOR_HIDDEN_SIZE,
    MEL_FRAMES_PER_FRAME,
    NATIVE_FRAME_MS,
    THINKER_HIDDEN_SIZE,
    WHISPER_MAX_ROWS,
    WHISPER_ROWS_PER_FRAME,
    WHISPER_WIDTH,
    frames_for_latency,
)


AUT_FORMAT = "d2.llama_omni2.block_causal_whisper.v1"


AUT_TEACHER_CONTRACT_FORMAT = "native_waveform_pad_before_official_logmel_v1"


AUT_ATTENTION_POLICY = "current_5k_bidirectional_completed_macros_causal_v1"


AUT_FRONTEND_POLICY = "official_centered_conv_tail2_take_last_5k_v1"


AUT_RELEASE_POLICY = "official_group5_projector_10hz_v1"


FRONTEND_TAIL_MEL_FRAMES = 2


def _position_table(encoder: nn.Module) -> torch.Tensor:
    positions = getattr(encoder, "positional_embedding", None)
    if positions is None:
        positions = getattr(encoder, "embed_positions", None)
    positions = getattr(positions, "weight", positions)
    if not torch.is_tensor(positions) or positions.ndim != 2:
        raise TypeError("Whisper encoder must expose a rank-2 position table")
    return positions


def _blocks(encoder: nn.Module) -> tuple[nn.Module, ...]:
    blocks = getattr(encoder, "blocks", None)
    if blocks is None:
        blocks = getattr(encoder, "layers", None)
    if blocks is None:
        raise TypeError("Whisper encoder must expose blocks or layers")
    return tuple(blocks)


def _openai_attention_with_mask(
    attention: nn.Module,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the exact additive mask to an OpenAI Whisper attention block.

    OpenAI Whisper's SDPA fast path only checks whether ``mask`` is non-null
    and then sets ``is_causal=True``.  It therefore silently turns our
    block-causal mask into an ordinary token-causal mask.  Project Q/K/V here
    and pass the real visibility matrix to SDPA instead.
    """

    required = ("query", "key", "value", "out", "n_head")
    if any(not hasattr(attention, name) for name in required):
        raise TypeError("OpenAI Whisper attention geometry is unavailable")
    heads = getattr(attention, "n_head")
    if isinstance(heads, bool) or not isinstance(heads, int) or heads < 1:
        raise TypeError("OpenAI Whisper attention head count is invalid")
    batch, rows, width = values.shape
    if width % heads:
        raise ValueError("Whisper attention width is not divisible by its heads")
    expected_mask = (rows, rows)
    if tuple(mask.shape) != expected_mask:
        raise ValueError(f"Whisper attention mask must have shape {expected_mask}")

    def split_heads(projected: torch.Tensor) -> torch.Tensor:
        return projected.reshape(batch, rows, heads, width // heads).transpose(1, 2)

    query = split_heads(attention.query(values))
    key = split_heads(attention.key(values))
    value = split_heads(attention.value(values))
    # Boolean SDPA masks use True for visible entries.  This is both cheaper
    # than the fp32 additive matrix inside the kernel and safe for bf16/fp16.
    visible = torch.isfinite(mask).to(device=query.device)[None, None]
    attended = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=visible,
        dropout_p=0.0,
        is_causal=False,
    )
    attended = attended.transpose(1, 2).reshape(batch, rows, width)
    return attention.out(attended.contiguous())


def _openai_block_with_mask(
    block: nn.Module,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Run one released OpenAI Whisper encoder block block-causally."""

    required = ("attn", "attn_ln", "mlp", "mlp_ln")
    if any(not hasattr(block, name) for name in required):
        raise TypeError("released Whisper block geometry is unavailable")
    hidden = values + _openai_attention_with_mask(
        block.attn,
        block.attn_ln(values),
        mask,
    )
    return hidden + block.mlp(block.mlp_ln(hidden))


def block_causal_mask(
    row_count: int,
    frames_per_unit: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Additive mask: bidirectional within a macro, causal between macros."""

    if row_count < 1:
        raise ValueError("row_count must be positive")
    block = WHISPER_ROWS_PER_FRAME * frames_per_unit
    if row_count % block:
        raise ValueError(f"row_count must contain complete {block}-row macros")
    query = torch.arange(row_count, device=device).unsqueeze(1)
    key = torch.arange(row_count, device=device).unsqueeze(0)
    visible_end = (torch.div(query, block, rounding_mode="floor") + 1) * block
    allowed = key < visible_end
    result = torch.zeros((row_count, row_count), device=device, dtype=dtype)
    return result.masked_fill(~allowed, float("-inf"))


class GroupedSpeechProjector(nn.Module):
    """The released concat-5 -> 2048 -> Thinker-width speech adaptor."""

    def __init__(
        self,
        output_width: int = THINKER_HIDDEN_SIZE,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if output_width < 1:
            raise ValueError("output_width must be positive")
        factory = {"device": device, "dtype": dtype}
        self.k = ADAPTOR_GROUP_ROWS
        self.encoder_dim = WHISPER_WIDTH
        self.llm_dim = int(output_width)
        self.linear1 = nn.Linear(
            ADAPTOR_GROUP_ROWS * WHISPER_WIDTH,
            ADAPTOR_HIDDEN_SIZE,
            **factory,
        )
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(
            ADAPTOR_HIDDEN_SIZE,
            self.llm_dim,
            **factory,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != WHISPER_WIDTH:
            raise ValueError("speech projector input must be [batch,time,1280]")
        batch, rows, width = values.shape
        if rows < self.k or rows % self.k:
            raise ValueError("speech projector requires complete groups of five")
        grouped = values.reshape(batch, rows // self.k, self.k * width)
        return self.linear2(self.relu(self.linear1(grouped)))

    @classmethod
    def from_official(cls, projector: nn.Module) -> "GroupedSpeechProjector":
        linears = tuple(m for m in projector.modules() if isinstance(m, nn.Linear))
        if len(linears) != 2:
            raise ValueError("official speech projector must contain two Linear layers")
        first, second = linears
        if tuple(first.weight.shape) != (
            ADAPTOR_HIDDEN_SIZE,
            ADAPTOR_GROUP_ROWS * WHISPER_WIDTH,
        ):
            raise ValueError("official speech projector first layer geometry changed")
        result = cls(
            output_width=int(second.weight.shape[0]),
            device=first.weight.device,
            dtype=first.weight.dtype,
        )
        result.linear1.load_state_dict(first.state_dict())
        result.linear2.load_state_dict(second.state_dict())
        return result


@dataclass
class AuTState:
    """One lane's raw convolution history; lanes must never share this."""

    mel_tail: torch.Tensor | None = None
    frontend_rows: torch.Tensor | None = None
    completed_units: int = 0
    reset_count: int = 0

    def clear(self) -> None:
        self.mel_tail = None
        self.frontend_rows = None
        self.completed_units = 0
        self.reset_count = 0


@dataclass(frozen=True)
class AuTOutput:
    embeddings: torch.Tensor
    whisper_rows: torch.Tensor
    state: AuTState
    reset_before: bool


class BlockCausalWhisper(nn.Module):
    """Trainable released Whisper modules under D2 causal scheduling."""

    def __init__(
        self,
        speech_encoder: nn.Module,
        speech_projector: nn.Module,
        *,
        latency_ms: int = NATIVE_FRAME_MS,
    ) -> None:
        super().__init__()
        self.frames_per_unit = frames_for_latency(latency_ms)
        self.speech_encoder = speech_encoder
        self.speech_projector = speech_projector
        if not hasattr(speech_encoder, "conv1") or not hasattr(speech_encoder, "conv2"):
            raise TypeError("Whisper encoder must expose conv1 and conv2")
        _blocks(speech_encoder)
        positions = _position_table(speech_encoder)
        if positions.shape[0] < self.segment_rows:
            raise ValueError("Whisper position table is shorter than D2 segment")

    @classmethod
    def from_official_model(
        cls,
        official_model: nn.Module,
        *,
        latency_ms: int = NATIVE_FRAME_MS,
    ) -> "BlockCausalWhisper":
        owner = (
            official_model.get_model()
            if callable(getattr(official_model, "get_model", None))
            else official_model
        )
        tower = getattr(owner, "speech_encoder", None)
        projector = getattr(owner, "speech_projector", None)
        if tower is None or projector is None:
            raise TypeError("official LLaMA-Omni2 speech modules are unavailable")
        encoder = getattr(tower, "encoder", tower)
        return cls(
            deepcopy(encoder),
            GroupedSpeechProjector.from_official(projector),
            latency_ms=latency_ms,
        )

    @property
    def macro_mel_frames(self) -> int:
        return MEL_FRAMES_PER_FRAME * self.frames_per_unit

    @property
    def macro_rows(self) -> int:
        return WHISPER_ROWS_PER_FRAME * self.frames_per_unit

    @property
    def segment_rows(self) -> int:
        return WHISPER_MAX_ROWS // self.macro_rows * self.macro_rows

    @property
    def contract(self) -> dict[str, Any]:
        return {
            "format": AUT_FORMAT,
            "latency_ms": self.frames_per_unit * NATIVE_FRAME_MS,
            "frames_per_unit": self.frames_per_unit,
            "macro_mel_frames": self.macro_mel_frames,
            "macro_rows": self.macro_rows,
            "segment_rows": self.segment_rows,
            "attention_policy": AUT_ATTENTION_POLICY,
            "frontend_policy": AUT_FRONTEND_POLICY,
            "release_policy": AUT_RELEASE_POLICY,
            "adaptor_group_rows": ADAPTOR_GROUP_ROWS,
            "adaptor_hidden_size": ADAPTOR_HIDDEN_SIZE,
            "output_width": int(getattr(self.speech_projector, "llm_dim", THINKER_HIDDEN_SIZE)),
            "source_model": OFFICIAL_MODEL_ID,
            "source_model_revision": OFFICIAL_MODEL_REVISION,
            "source_repository_revision": OFFICIAL_REPOSITORY_REVISION,
        }

    def new_state(self) -> AuTState:
        return AuTState()

    def _frontend(
        self,
        features: torch.Tensor,
        mel_tail: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] % self.macro_mel_frames:
            raise ValueError("features must be [batch,128,complete-macro-mels]")
        if features.shape[1] != 128:
            raise ValueError("Whisper-large-v3 requires 128 mel bins")
        batch, bins, frames = features.shape
        tail = (
            features.new_zeros(batch, bins, FRONTEND_TAIL_MEL_FRAMES)
            if mel_tail is None
            else mel_tail.to(device=features.device, dtype=features.dtype)
        )
        if tuple(tail.shape) != (batch, bins, FRONTEND_TAIL_MEL_FRAMES):
            raise ValueError("mel tail is incompatible with current features")
        padded = torch.cat((tail, features), dim=-1)
        windows = padded.unfold(
            -1,
            self.macro_mel_frames + FRONTEND_TAIL_MEL_FRAMES,
            self.macro_mel_frames,
        )
        units = frames // self.macro_mel_frames
        windows = windows.permute(0, 2, 1, 3).reshape(
            batch * units,
            bins,
            self.macro_mel_frames + FRONTEND_TAIL_MEL_FRAMES,
        )
        conv1 = self.speech_encoder.conv1
        dtype = conv1.weight.dtype
        device = conv1.weight.device
        hidden = F.gelu(conv1(windows.to(device=device, dtype=dtype)))
        hidden = F.gelu(self.speech_encoder.conv2(hidden)).permute(0, 2, 1)
        if hidden.shape[1] != self.macro_rows + 1:
            raise RuntimeError("official Whisper convolution clock changed")
        rows = hidden[:, -self.macro_rows :, :].reshape(batch, units * self.macro_rows, -1)
        return rows, features[..., -FRONTEND_TAIL_MEL_FRAMES:]

    def _transform(self, frontend_rows: torch.Tensor) -> torch.Tensor:
        positions = _position_table(self.speech_encoder)
        rows = frontend_rows.shape[1]
        hidden = frontend_rows + positions[:rows].to(
            device=frontend_rows.device,
            dtype=frontend_rows.dtype,
        )
        mask = block_causal_mask(
            rows,
            self.frames_per_unit,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        for block in _blocks(self.speech_encoder):
            if all(hasattr(block, name) for name in ("attn", "attn_ln", "mlp", "mlp_ln")):
                hidden = _openai_block_with_mask(block, hidden, mask)
            else:
                try:
                    hidden = block(hidden, mask=mask)
                except TypeError:
                    hidden = block(hidden, attention_mask=mask)
            if isinstance(hidden, (tuple, list)):
                hidden = hidden[0]
        norm = getattr(self.speech_encoder, "ln_post", None)
        if norm is None:
            norm = getattr(self.speech_encoder, "layer_norm", None)
        return hidden if norm is None else norm(hidden)

    def forward(
        self,
        features: torch.Tensor,
        state: AuTState | None = None,
        *,
        detach_state: bool = True,
    ) -> AuTOutput:
        state = self.new_state() if state is None else state
        current_rows, next_tail = self._frontend(features, state.mel_tail)
        reset = False
        history = state.frontend_rows
        if history is not None and history.shape[0] != current_rows.shape[0]:
            raise ValueError("AuT state batch size changed")
        offset = 0
        current_outputs: list[torch.Tensor] = []
        while offset < current_rows.shape[1]:
            history_rows = 0 if history is None else int(history.shape[1])
            if history_rows == self.segment_rows:
                history = None
                history_rows = 0
                reset = True
                state.reset_count += 1
            capacity = self.segment_rows - history_rows
            take = min(capacity, int(current_rows.shape[1]) - offset)
            # Both capacity and the remaining current sequence are macro
            # aligned, so no segment can split a D2 unit.
            if take < 1 or take % self.macro_rows:
                raise AssertionError("Whisper segment split a macro unit")
            chunk = current_rows[:, offset : offset + take]
            all_rows = chunk if history is None else torch.cat((history, chunk), dim=1)
            transformed = self._transform(all_rows)
            current_outputs.append(transformed[:, -take:, :])
            history = all_rows.detach() if detach_state else all_rows
            offset += take
            if offset < current_rows.shape[1]:
                history = None
                reset = True
                state.reset_count += 1
        current_hidden = torch.cat(current_outputs, dim=1)
        embeddings = self.speech_projector(current_hidden)
        expected = current_rows.shape[1] // ADAPTOR_GROUP_ROWS
        if embeddings.shape[1] != expected:
            raise RuntimeError("speech projector changed the 10-Hz release clock")
        state.frontend_rows = history
        state.mel_tail = next_tail.detach() if detach_state else next_tail
        state.completed_units += current_rows.shape[1] // self.macro_rows
        return AuTOutput(embeddings, current_hidden, state, reset)

    def checkpoint_state_dict(self) -> dict[str, Any]:
        return {"contract": self.contract, "model": self.state_dict()}

    def load_checkpoint_state_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("contract") != self.contract:
            raise ValueError("AuT checkpoint contract does not match this latency")
        self.load_state_dict(payload["model"], strict=True)


def native_teacher_log_mel(waveform: torch.Tensor) -> torch.Tensor:
    """Released preprocessing: pad 16-kHz PCM silence before log-mel.

    Each utterance is normalized independently, exactly as the official
    inference loader. Normalized mel values are never padded with zero.
    """
    from whisper.audio import N_SAMPLES, log_mel_spectrogram, pad_or_trim

    if waveform.ndim != 1 or not 0 < waveform.numel() <= N_SAMPLES:
        raise ValueError("native teacher expects a nonempty mono segment of at most 30 seconds")
    pcm = waveform.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(pcm).all()):
        raise ValueError("native teacher PCM contains a nonfinite sample")
    padded_pcm = pad_or_trim(pcm, length=N_SAMPLES)
    features = log_mel_spectrogram(padded_pcm, n_mels=128)
    if tuple(features.shape) != (128, 3000):
        raise RuntimeError("released teacher log-mel geometry changed")
    return features


def native_teacher_contract(frames_per_unit: int) -> dict[str, Any]:
    if type(frames_per_unit) is not int or frames_per_unit not in (1, 2, 4, 8):
        raise ValueError("production teacher supports k in {1,2,4,8}")
    return {
        "format": AUT_TEACHER_CONTRACT_FORMAT,
        "sample_rate_hz": 16000,
        "padded_waveform_samples": 480000,
        "mel_bins": 128,
        "mel_frames": 3000,
        "center": True,
        "normalization": "released_whisper_per_utterance_logmax",
        "padding": "silent_pcm_before_log_mel_never_zero_mel",
        "segment_frames": 300 // frames_per_unit * frames_per_unit,
        "frames_per_unit": frames_per_unit,
        "targets": "complete_student_macros_only_no_padded_tail_targets",
        "source_model": OFFICIAL_MODEL_ID,
        "source_model_revision": OFFICIAL_MODEL_REVISION,
        "source_repository_revision": OFFICIAL_REPOSITORY_REVISION,
    }


class OfficialWhisperTeacher(nn.Module):
    """Frozen released bidirectional encoder using its native PCM frontend."""

    def __init__(
        self,
        speech_encoder: nn.Module,
        speech_projector: nn.Module,
        *,
        frames_per_unit: int,
    ) -> None:
        super().__init__()
        self.speech_encoder = speech_encoder
        self.speech_projector = speech_projector
        self.frames_per_unit = int(frames_per_unit)
        if self.frames_per_unit not in (1, 2, 4, 8):
            raise ValueError("production teacher supports k in {1,2,4,8}")
        if _position_table(speech_encoder).shape[0] != WHISPER_MAX_ROWS:
            raise ValueError("released teacher must retain its 1500-row/30-second context")
        self.segment_rows = (
            WHISPER_MAX_ROWS // (WHISPER_ROWS_PER_FRAME * self.frames_per_unit)
        ) * (WHISPER_ROWS_PER_FRAME * self.frames_per_unit)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_official_model(
        cls,
        official_model: nn.Module,
        *,
        frames_per_unit: int,
    ) -> "OfficialWhisperTeacher":
        owner = (
            official_model.get_model()
            if callable(getattr(official_model, "get_model", None))
            else official_model
        )
        tower = getattr(owner, "speech_encoder", None)
        projector = getattr(owner, "speech_projector", None)
        if tower is None or projector is None:
            raise TypeError("official LLaMA-Omni2 speech modules are unavailable")
        return cls(
            getattr(tower, "encoder", tower),
            projector,
            frames_per_unit=frames_per_unit,
        )

    @property
    def contract(self) -> dict[str, Any]:
        return native_teacher_contract(self.frames_per_unit)

    def forward(self, waveform: torch.Tensor, *, valid_frames: int) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2 or waveform.shape[0] < 1:
            raise ValueError("teacher input must be clean 16-kHz PCM [batch,samples]")
        expected_frames = waveform.shape[-1] // (1600 * self.frames_per_unit) * self.frames_per_unit
        if type(valid_frames) is not int or valid_frames < 1 or valid_frames != expected_frames:
            raise ValueError("teacher target length must match the complete student macro count")
        segment_frames = self.segment_rows // WHISPER_ROWS_PER_FRAME
        parameter = next(self.speech_encoder.parameters())
        outputs: list[torch.Tensor] = []
        for start in range(0, valid_frames, segment_frames):
            current = waveform[:, start * 1600 : (start + segment_frames) * 1600]
            features = torch.stack([native_teacher_log_mel(pcm) for pcm in current])
            features = features.to(device=parameter.device, dtype=parameter.dtype)
            hidden = self.speech_encoder(features)
            if isinstance(hidden, (tuple, list)):
                hidden = hidden[0]
            valid_rows = min(segment_frames, valid_frames - start) * WHISPER_ROWS_PER_FRAME
            hidden = hidden[:, :valid_rows]
            outputs.append(self.speech_projector(hidden))
        return torch.cat(outputs, dim=1)


@dataclass(frozen=True)
class AuTTrainability:
    non_attention_lora_modules: tuple[str, ...]
    parameter_groups: tuple[dict[str, Any], ...]


def configure_aut_trainability(
    student: BlockCausalWhisper,
    config: Mapping[str, Any],
) -> AuTTrainability:
    """Apply the Stage-1 dense-frontend/dense-attention/LoRA-MLP policy."""

    for parameter in student.parameters():
        parameter.requires_grad_(False)
    for frontend in (student.speech_encoder.conv1, student.speech_encoder.conv2):
        for parameter in frontend.parameters():
            parameter.requires_grad_(True)
    for block in _blocks(student.speech_encoder):
        attention = getattr(block, "attn", None)
        if attention is None:
            attention = getattr(block, "self_attn", None)
        if attention is None:
            raise TypeError("Whisper block does not expose self attention")
        for parameter in attention.parameters():
            parameter.requires_grad_(True)

    rank = int(config["lora_rank"])
    alpha = float(config["lora_alpha"])
    dropout = float(config["lora_dropout"])
    mlp_names = install_lora(
        student.speech_encoder,
        target_suffixes=("mlp.0", "mlp.2", "fc1", "fc2"),
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    adaptor_names = install_lora(
        student.speech_projector,
        target_suffixes=("linear1", "linear2"),
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )

    def lora_at(root: nn.Module, paths: tuple[str, ...]) -> list[nn.Parameter]:
        result: list[nn.Parameter] = []
        for path in paths:
            module = root
            for piece in path.split("."):
                module = getattr(module, piece)
            if not isinstance(module, LoRALinear):
                raise RuntimeError("non-attention LoRA installation changed")
            result.extend((module.lora_a, module.lora_b))
        return result

    frontend_params = [
        parameter
        for module in (student.speech_encoder.conv1, student.speech_encoder.conv2)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    attention_params = [
        parameter
        for block in _blocks(student.speech_encoder)
        for attention in (getattr(block, "attn", None) or getattr(block, "self_attn", None),)
        for parameter in attention.parameters()
        if parameter.requires_grad
    ]
    lora_params = [
        *lora_at(student.speech_encoder, mlp_names),
        *lora_at(student.speech_projector, adaptor_names),
    ]
    groups = (
        {
            "name": "frontend",
            "params": frontend_params,
            "lr": float(config["frontend_lr"]),
        },
        {
            "name": "attention",
            "params": attention_params,
            "lr": float(config["attention_lr"]),
        },
        {
            "name": "non_attention_lora",
            "params": lora_params,
            "lr": float(config["non_attention_lora_lr"]),
        },
    )
    if any(not group["params"] for group in groups):
        raise RuntimeError("Stage-1 optimizer contains an empty parameter group")
    return AuTTrainability(mlp_names + adaptor_names, groups)


def build_aut_optimizer(
    trainability: AuTTrainability,
    config: Mapping[str, Any],
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        list(trainability.parameter_groups),
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=float(config["weight_decay"]),
    )


@dataclass(frozen=True)
class DistillationTerms:
    loss: torch.Tensor
    l1: torch.Tensor
    cosine_similarity: torch.Tensor
    cosine_penalty: torch.Tensor


def feature_distillation_terms(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> DistillationTerms:
    if student.shape != teacher.shape or student.ndim != 3:
        raise ValueError("student and teacher must have identical [batch,time,width] shapes")
    left, right = student.float(), teacher.float()
    l1 = (left - right).abs().mean()
    similarity = F.cosine_similarity(left, right, dim=-1).mean()
    penalty = -F.logsigmoid(F.cosine_similarity(left, right, dim=-1)).mean()
    return DistillationTerms(l1 + penalty, l1, similarity, penalty)


def trainable_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in module.parameters() if parameter.requires_grad)
