from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from .constants import CODEBOOKS, CODEBOOK_SIZE


CONFIGURED_TEXT_WEIGHT_POLICY = "configured_pad_everywhere"


SPLIT_PAD_TEXT_WEIGHT_POLICY = "split_pad_by_response_span"


RESPONSE_TEXT_WEIGHT_POLICY = CONFIGURED_TEXT_WEIGHT_POLICY


SUPPORTED_TEXT_WEIGHT_POLICIES = {
    CONFIGURED_TEXT_WEIGHT_POLICY,
    SPLIT_PAD_TEXT_WEIGHT_POLICY,
}


def _detach_transformer_cache(cache: Any) -> Any:
    if cache is None:
        return None
    if torch.is_tensor(cache):
        return cache.detach()
    if isinstance(cache, tuple):
        return tuple(_detach_transformer_cache(value) for value in cache)
    if isinstance(cache, list):
        return [_detach_transformer_cache(value) for value in cache]
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for layer in layers:
            for name in ("keys", "values"):
                value = getattr(layer, name, None)
                if torch.is_tensor(value):
                    setattr(layer, name, value.detach())
        return cache
    for name in ("key_cache", "value_cache"):
        values = getattr(cache, name, None)
        if isinstance(values, list):
            for index, value in enumerate(values):
                if torch.is_tensor(value):
                    values[index] = value.detach()
    return cache


def _compact_transformer_cache(
    cache: Any,
    *,
    target_tokens: int,
    preserved_prefix_tokens: int = 0,
) -> Any:
    if cache is None:
        raise ValueError("Cannot compact an empty Transformer cache")
    target_tokens = int(target_tokens)
    preserved_prefix_tokens = int(preserved_prefix_tokens)
    if target_tokens < 0:
        raise ValueError(
            f"Transformer cache target length must be non-negative, got {target_tokens}"
        )
    if not 0 <= preserved_prefix_tokens <= target_tokens:
        raise ValueError(
            "Transformer cache preserved prefix must be within the target: "
            f"prefix={preserved_prefix_tokens}, target={target_tokens}"
        )
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise TypeError("Fixed-memory Transformer-XL requires a cache with mutable layers")
    temporal_target = target_tokens - preserved_prefix_tokens
    for layer_index, layer in enumerate(layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if not torch.is_tensor(keys) or not torch.is_tensor(values):
            raise TypeError(
                "Fixed-memory Transformer-XL cache layer "
                f"{layer_index} does not expose tensor keys and values"
            )
        if keys.ndim < 2 or values.ndim < 2:
            raise ValueError(
                "Transformer cache tensors must include sequence and feature "
                f"dimensions, got keys={tuple(keys.shape)}, "
                f"values={tuple(values.shape)}"
            )
        if int(keys.shape[-2]) != int(values.shape[-2]):
            raise ValueError(
                "Transformer cache key/value lengths differ at layer "
                f"{layer_index}: {int(keys.shape[-2])} != "
                f"{int(values.shape[-2])}"
            )
        source_tokens = int(keys.shape[-2])
        if source_tokens < preserved_prefix_tokens:
            raise ValueError(
                "Transformer cache is shorter than its preserved prefix at "
                f"layer {layer_index}: {source_tokens} < "
                f"{preserved_prefix_tokens}"
            )

        def compact_tensor(tensor: torch.Tensor) -> torch.Tensor:
            detached = tensor.detach()
            prefix = detached[..., :preserved_prefix_tokens, :]
            temporal = detached[..., preserved_prefix_tokens:, :]
            kept_tokens = min(int(temporal.shape[-2]), temporal_target)
            suffix = temporal[..., -kept_tokens:, :] if kept_tokens > 0 else temporal[..., :0, :]
            padding_shape = list(detached.shape)
            padding_shape[-2] = temporal_target - kept_tokens
            padding = detached.new_zeros(padding_shape)
            return torch.cat((prefix, padding, suffix), dim=-2).detach().clone()

        layer.keys = compact_tensor(keys)
        layer.values = compact_tensor(values)
        if hasattr(layer, "is_initialized"):
            layer.is_initialized = True
    get_seq_length = getattr(cache, "get_seq_length", None)
    if callable(get_seq_length) and int(get_seq_length()) != target_tokens:
        raise RuntimeError(
            "Compacted Transformer cache reports the wrong length: "
            f"{int(get_seq_length())} != {target_tokens}"
        )
    return cache


def _transformer_xl_past_attention_mask(
    *,
    batch_size: int,
    cached_tokens: int,
    valid_memory_frames: int | None,
    tokens_per_memory_frame: int,
    preserved_prefix_tokens: int,
    device: torch.device,
) -> torch.Tensor | None:
    batch_size = int(batch_size)
    cached_tokens = int(cached_tokens)
    tokens_per_memory_frame = int(tokens_per_memory_frame)
    preserved_prefix_tokens = int(preserved_prefix_tokens)
    if cached_tokens < 0:
        raise ValueError(f"Transformer cache length must be non-negative, got {cached_tokens}")
    if tokens_per_memory_frame <= 0:
        raise ValueError(
            f"Transformer memory token rate must be positive, got {tokens_per_memory_frame}"
        )
    if not 0 <= preserved_prefix_tokens <= cached_tokens:
        raise ValueError(
            "Transformer cache prefix exceeds the available cache: "
            f"{preserved_prefix_tokens} > {cached_tokens}"
        )
    if valid_memory_frames is None:
        return torch.ones(
            (batch_size, cached_tokens),
            dtype=torch.long,
            device=device,
        )
    valid_memory_frames = int(valid_memory_frames)
    valid_memory_tokens = valid_memory_frames * tokens_per_memory_frame
    temporal_cache_tokens = cached_tokens - preserved_prefix_tokens
    if valid_memory_tokens < 0 or valid_memory_tokens > temporal_cache_tokens:
        raise ValueError(
            "Transformer valid memory exceeds its temporal cache: "
            f"{valid_memory_tokens} valid tokens for "
            f"{temporal_cache_tokens} cached temporal tokens"
        )
    if valid_memory_tokens == temporal_cache_tokens:
        return None
    return torch.cat(
        (
            torch.ones(
                (batch_size, preserved_prefix_tokens),
                dtype=torch.long,
                device=device,
            ),
            torch.zeros(
                (batch_size, temporal_cache_tokens - valid_memory_tokens),
                dtype=torch.long,
                device=device,
            ),
            torch.ones(
                (batch_size, valid_memory_tokens),
                dtype=torch.long,
                device=device,
            ),
        ),
        dim=1,
    )


@dataclass
class QwenDuplexTransformerXLState:
    """Detached recurrence carried between chunks of one logical sample."""

    thinker_cache: Any = None
    talker_cache: Any = None
    frame_offset: int = 0
    memory_frames: int | None = None
    valid_memory_frames: int = 0
    previous_text_target: torch.Tensor | None = None
    previous_self_audio: torch.Tensor | None = None
    response_open: torch.Tensor | None = None
    text_done: torch.Tensor | None = None
    previous_codec_target: torch.Tensor | None = None
    previous_response_start: torch.Tensor | None = None
    talker_open: torch.Tensor | None = None

    def detach_(self) -> "QwenDuplexTransformerXLState":
        self.thinker_cache = _detach_transformer_cache(self.thinker_cache)
        self.talker_cache = _detach_transformer_cache(self.talker_cache)
        for name in (
            "previous_text_target",
            "previous_self_audio",
            "response_open",
            "text_done",
            "previous_codec_target",
            "previous_response_start",
            "talker_open",
        ):
            value = getattr(self, name)
            if torch.is_tensor(value):
                setattr(self, name, value.detach())
        return self


def shift_right_text(
    text_target: torch.Tensor,
    pad_id: int,
    *,
    previous_text_target: torch.Tensor | None = None,
) -> torch.Tensor:
    out = torch.empty_like(text_target)
    if previous_text_target is None:
        out[:, 0] = int(pad_id)
    else:
        if tuple(previous_text_target.shape) != (int(text_target.shape[0]),):
            raise ValueError(
                "Previous text target must have shape "
                f"[{text_target.shape[0]}], got {tuple(previous_text_target.shape)}"
            )
        out[:, 0] = previous_text_target.to(
            device=out.device,
            dtype=out.dtype,
        )
    out[:, 1:] = text_target[:, :-1]
    return out


def mask_async_text_logits(
    logits: torch.Tensor,
    *,
    speech_open: torch.Tensor,
    text_done: torch.Tensor,
    previous_text_id: torch.Tensor,
    pad_id: int,
    assistant_start_id: int,
    assistant_end_id: int,
    special_ids: tuple[int, ...] | list[int] = (),
) -> torch.Tensor:
    """Apply the INTERRUPT/RESPONSE async text state machine."""

    state_shape = logits.shape[:-1]
    for name, value in (
        ("speech_open", speech_open),
        ("text_done", text_done),
        ("previous_text_id", previous_text_id),
    ):
        if tuple(value.shape) != tuple(state_shape):
            raise ValueError(
                f"{name} shape {tuple(value.shape)} != logits prefix {tuple(state_shape)}"
            )
    vocab_size = int(logits.shape[-1])
    control_ids = (
        int(pad_id),
        int(assistant_start_id),
        int(assistant_end_id),
    )
    if len(set(control_ids)) != 3 or min(control_ids) < 0 or max(control_ids) >= vocab_size:
        raise ValueError(f"PAD/RESPONSE/INTERRUPT IDs {control_ids} do not fit vocab {vocab_size}")

    speech_open = speech_open.to(device=logits.device, dtype=torch.bool)
    text_done = text_done.to(device=logits.device, dtype=torch.bool)
    previous_text_id = previous_text_id.to(device=logits.device)
    just_started = speech_open & (previous_text_id == int(assistant_start_id))
    lexical_allowed = speech_open & ~text_done
    allowed = lexical_allowed.unsqueeze(-1).expand_as(logits).clone()
    invalid_special_ids = sorted(set(int(token_id) for token_id in special_ids))
    if invalid_special_ids:
        if min(invalid_special_ids) < 0 or max(invalid_special_ids) >= vocab_size:
            raise ValueError(
                f"Special token IDs {invalid_special_ids} do not fit vocab {vocab_size}"
            )
        allowed[..., invalid_special_ids] = False
    allowed[..., int(pad_id)] = (~speech_open) | (speech_open & ~just_started)
    allowed[..., int(assistant_start_id)] = ~speech_open
    allowed[..., int(assistant_end_id)] = True
    return logits.masked_fill(~allowed, float("-inf"))


def validate_async_text_targets(
    text_target: torch.Tensor,
    *,
    pad_id: int,
    assistant_start_id: int,
    assistant_end_id: int,
    initial_speech_open: torch.Tensor | None = None,
    initial_text_done: torch.Tensor | None = None,
    initial_previous_text_id: torch.Tensor | None = None,
    immediate_pad_allowed: torch.Tensor | None = None,
    forbidden_ids: tuple[int, ...] | list[int] = (),
    text_tokenizer_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reject interleaving outside the INTERRUPT/RESPONSE contract."""

    if text_target.ndim != 2:
        raise ValueError(f"Expected async text target [B,T], got {tuple(text_target.shape)}")
    batch_size = int(text_target.shape[0])
    state_shape = (batch_size,)
    if initial_speech_open is None:
        initial_speech_open = torch.zeros(state_shape, dtype=torch.bool)
    if initial_text_done is None:
        initial_text_done = torch.ones(state_shape, dtype=torch.bool)
    if initial_previous_text_id is None:
        initial_previous_text_id = torch.full(
            state_shape,
            int(pad_id),
            dtype=torch.long,
        )
    if immediate_pad_allowed is None:
        immediate_pad_allowed = torch.zeros_like(text_target, dtype=torch.bool)
    if tuple(immediate_pad_allowed.shape) != tuple(text_target.shape):
        raise ValueError(
            "immediate_pad_allowed shape "
            f"{tuple(immediate_pad_allowed.shape)} != {tuple(text_target.shape)}"
        )
    for name, value in (
        ("initial_speech_open", initial_speech_open),
        ("initial_text_done", initial_text_done),
        ("initial_previous_text_id", initial_previous_text_id),
    ):
        if tuple(value.shape) != state_shape:
            raise ValueError(f"{name} shape {tuple(value.shape)} != {state_shape}")

    target = text_target.detach().cpu().long()
    initial_open_cpu = initial_speech_open.detach().cpu().bool()
    initial_done_cpu = initial_text_done.detach().cpu().bool()
    initial_previous_cpu = initial_previous_text_id.detach().cpu().long()
    immediate_pad_allowed_cpu = immediate_pad_allowed.detach().cpu().bool()
    ending_open = torch.empty(state_shape, dtype=torch.bool)
    ending_done = torch.empty(state_shape, dtype=torch.bool)
    forbidden = set(int(token_id) for token_id in forbidden_ids)
    forbidden.difference_update((int(pad_id), int(assistant_start_id), int(assistant_end_id)))
    for batch_index in range(int(target.shape[0])):
        speech_open = bool(initial_open_cpu[batch_index])
        text_done = bool(initial_done_cpu[batch_index])
        just_started = speech_open and (
            int(initial_previous_cpu[batch_index]) == int(assistant_start_id)
        )
        for frame_index in range(int(target.shape[1])):
            token_id = int(target[batch_index, frame_index])
            location = f"batch={batch_index} frame={frame_index}"
            if token_id < 0 or (
                text_tokenizer_size is not None and token_id >= int(text_tokenizer_size)
            ):
                raise ValueError(
                    f"Text token {token_id} is outside the tokenizer vocabulary at {location}"
                )
            if token_id in forbidden:
                raise ValueError(f"Forbidden special token {token_id} in async text at {location}")
            if not speech_open:
                if token_id == int(pad_id):
                    continue
                if token_id == int(assistant_end_id):
                    text_done = True
                    just_started = False
                    continue
                if token_id != int(assistant_start_id):
                    raise ValueError(
                        f"Closed async text permits only PAD/RESPONSE/INTERRUPT at {location}"
                    )
                speech_open = True
                text_done = False
                just_started = True
                continue
            if token_id == int(assistant_start_id):
                raise ValueError(f"Nested RESPONSE in async text at {location}")
            if token_id == int(assistant_end_id):
                speech_open = False
                text_done = True
                just_started = False
                continue
            if token_id == int(pad_id):
                if just_started and not bool(immediate_pad_allowed_cpu[batch_index, frame_index]):
                    raise ValueError(f"PAD cannot immediately follow RESPONSE at {location}")
                text_done = True
                just_started = False
                continue
            if text_done:
                raise ValueError(f"Lexical text cannot resume after terminal PAD at {location}")
            just_started = False
        ending_open[batch_index] = speech_open
        ending_done[batch_index] = text_done
    return (
        ending_open.to(device=text_target.device),
        ending_done.to(device=text_target.device),
    )


def response_state_mask_with_initial(
    text_target: torch.Tensor,
    assistant_start_id: int,
    assistant_end_id: int,
    *,
    initial_response_open: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(assistant_start_id) == int(assistant_end_id):
        raise ValueError("RESPONSE and INTERRUPT IDs must be distinct")
    if text_target.ndim != 2 or int(text_target.shape[1]) < 1:
        raise ValueError(f"Expected non-empty text target [B,T], got {tuple(text_target.shape)}")
    if initial_response_open is None:
        initial_response_open = torch.zeros(
            (int(text_target.shape[0]),),
            dtype=torch.bool,
            device=text_target.device,
        )
    if tuple(initial_response_open.shape) != (int(text_target.shape[0]),):
        raise ValueError(
            "Initial response state must have shape "
            f"[{text_target.shape[0]}], got {tuple(initial_response_open.shape)}"
        )
    starts = text_target == int(assistant_start_id)
    ends = text_target == int(assistant_end_id)
    positions = torch.arange(
        1,
        int(text_target.shape[1]) + 1,
        device=text_target.device,
        dtype=torch.long,
    ).view(1, -1)
    start_base = torch.where(
        initial_response_open.to(device=text_target.device, dtype=torch.bool),
        torch.zeros_like(initial_response_open, dtype=torch.long),
        torch.full_like(initial_response_open, -1, dtype=torch.long),
    ).unsqueeze(1)
    end_base = torch.where(
        initial_response_open.to(device=text_target.device, dtype=torch.bool),
        torch.full_like(initial_response_open, -1, dtype=torch.long),
        torch.zeros_like(initial_response_open, dtype=torch.long),
    ).unsqueeze(1)
    last_start = torch.cummax(
        torch.where(starts, positions, start_base),
        dim=1,
    ).values
    last_end = torch.cummax(
        torch.where(ends, positions, end_base),
        dim=1,
    ).values
    open_after = last_start > last_end
    open_before = torch.cat(
        (
            initial_response_open.to(
                device=text_target.device,
                dtype=torch.bool,
            ).unsqueeze(1),
            open_after[:, :-1],
        ),
        dim=1,
    )
    return open_after | (ends & open_before), open_after[:, -1]


def talker_state_mask_with_initial(
    codec0_target: torch.Tensor,
    *,
    codec_bos_id: int,
    codec_eos_id: int,
    initial_talker_open: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return inclusive Talker episodes delimited by codec BOS and EOS."""

    if codec0_target.ndim != 2 or int(codec0_target.shape[1]) < 1:
        raise ValueError(
            f"Expected non-empty codec0 target [B,T], got {tuple(codec0_target.shape)}"
        )
    if int(codec_bos_id) == int(codec_eos_id):
        raise ValueError("Talker codec BOS and EOS IDs must be distinct")
    batch_size = int(codec0_target.shape[0])
    if initial_talker_open is None:
        initial_talker_open = torch.zeros(
            (batch_size,),
            dtype=torch.bool,
            device=codec0_target.device,
        )
    if tuple(initial_talker_open.shape) != (batch_size,):
        raise ValueError(
            "Initial Talker state must have shape "
            f"[{batch_size}], got {tuple(initial_talker_open.shape)}"
        )
    starts = (codec0_target == int(codec_bos_id)).to(dtype=torch.int32)
    ends = (codec0_target == int(codec_eos_id)).to(dtype=torch.int32)
    initial_depth = initial_talker_open.to(
        device=codec0_target.device,
        dtype=torch.int32,
    ).unsqueeze(1)
    depth_after = initial_depth + starts.cumsum(dim=1) - ends.cumsum(dim=1)
    if bool(((depth_after < 0) | (depth_after > 1)).any().item()):
        raise ValueError("Talker codec BOS/EOS spans are unbalanced or nested")
    return (
        (depth_after + ends).to(dtype=torch.bool),
        depth_after[:, -1].to(dtype=torch.bool),
    )


def weighted_text_ce_terms(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    response_mask: torch.Tensor,
    pad_id: int,
    pad_weight: float,
    assistant_start_id: int,
    assistant_start_weight: float,
    assistant_end_id: int,
    assistant_end_weight: float,
    response_text_weight_policy: str = RESPONSE_TEXT_WEIGHT_POLICY,
    inside_response_pad_weight: float | None = None,
    token_weight_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if response_mask.shape != target.shape:
        raise ValueError(
            f"Response mask shape {tuple(response_mask.shape)} != target shape "
            f"{tuple(target.shape)}"
        )
    response_mask = response_mask.to(device=target.device, dtype=torch.bool)
    response_text_weight_policy = str(response_text_weight_policy)
    if response_text_weight_policy not in SUPPORTED_TEXT_WEIGHT_POLICIES:
        raise ValueError(
            "Unsupported response text weight policy "
            f"{response_text_weight_policy!r}; expected one of "
            f"{sorted(SUPPORTED_TEXT_WEIGHT_POLICIES)}"
        )
    if response_text_weight_policy == SPLIT_PAD_TEXT_WEIGHT_POLICY:
        if inside_response_pad_weight is None:
            raise ValueError("inside_response_pad_weight is required for split PAD weighting")
        inside_response_pad_weight = float(inside_response_pad_weight)
        if not math.isfinite(inside_response_pad_weight) or inside_response_pad_weight <= 0.0:
            raise ValueError(
                "inside_response_pad_weight must be finite and positive, got "
                f"{inside_response_pad_weight}"
            )
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction="none",
    ).reshape_as(target)
    token_weight = torch.ones_like(ce)
    if response_text_weight_policy == SPLIT_PAD_TEXT_WEIGHT_POLICY:
        pad_token_weight = torch.where(
            response_mask,
            token_weight.new_full((), inside_response_pad_weight),
            token_weight.new_full((), float(pad_weight)),
        )
    else:
        pad_token_weight = token_weight.new_full((), float(pad_weight))
    token_weight = torch.where(
        target == int(pad_id),
        pad_token_weight,
        token_weight,
    )
    token_weight = torch.where(
        target == int(assistant_start_id),
        token_weight.new_full((), float(assistant_start_weight)),
        token_weight,
    )
    token_weight = torch.where(
        target == int(assistant_end_id),
        token_weight.new_full((), float(assistant_end_weight)),
        token_weight,
    )
    if token_weight_override is not None:
        if token_weight_override.shape != target.shape:
            raise ValueError(
                "Token-weight override shape "
                f"{tuple(token_weight_override.shape)} != target shape "
                f"{tuple(target.shape)}"
            )
        override = token_weight_override.to(
            device=token_weight.device,
            dtype=token_weight.dtype,
        )
        if bool((~torch.isfinite(override) | (override < 0.0)).any().item()):
            raise ValueError("Token-weight overrides must be finite and non-negative")
        token_weight = torch.where(override > 0.0, override, token_weight)
    weight = mask.float() * token_weight
    weight_sum = weight.sum()
    numerator = (ce * weight).sum()
    return numerator / weight_sum.clamp_min(1.0), numerator, weight_sum


def masked_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    frame_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    target = target.to(device=logits.device)
    mask = mask.to(device=logits.device)
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction="none",
    ).reshape_as(target)
    weight = mask.float()
    if frame_weight is not None:
        weight = weight * frame_weight.to(device=weight.device, dtype=weight.dtype)
    return (ce * weight).sum() / weight.sum().clamp_min(1.0)


def two_signal_codec_supervision(
    codec_target: torch.Tensor,
    frame_mask: torch.Tensor,
    talker_mask: torch.Tensor,
    codec_bos_targets: torch.Tensor,
    codec_eos_targets: torch.Tensor,
    silence_frame: torch.Tensor,
    *,
    codec_eos_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = frame_mask.to(device=codec_target.device, dtype=torch.bool)
    bos = codec_bos_targets.to(device=codec_target.device, dtype=torch.bool)
    eos = codec_eos_targets.to(device=codec_target.device, dtype=torch.bool)
    open_frames = talker_mask.to(device=codec_target.device, dtype=torch.bool)
    effective_target = codec_target.clone()
    closed = valid & ~open_frames
    effective_target[closed] = silence_frame.to(
        device=codec_target.device,
        dtype=codec_target.dtype,
    ).view(1, -1)
    codec0_mask = valid & ~bos
    residual_mask = codec0_mask & ~eos
    codec0_frame_weight = torch.ones_like(frame_mask, dtype=torch.float32)
    codec0_frame_weight = torch.where(
        eos,
        codec0_frame_weight.new_full((), float(codec_eos_weight)),
        codec0_frame_weight,
    )
    return effective_target, codec0_mask, residual_mask, codec0_frame_weight


def select_talker_context(
    *,
    text_context: torch.Tensor,
    hidden_context: torch.Tensor,
    text_target: torch.Tensor,
    assistant_start_id: int,
    assistant_end_id: int,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if text_context.shape != hidden_context.shape:
        raise ValueError(
            f"Talker text/hidden context mismatch: {text_context.shape} vs {hidden_context.shape}"
        )
    mask = response_mask
    if mask.shape != text_target.shape:
        raise ValueError(
            f"Response mask {tuple(mask.shape)} does not match text target "
            f"{tuple(text_target.shape)}"
        )
    mask = mask.to(device=text_target.device, dtype=torch.bool)
    selected = torch.where(mask.unsqueeze(-1), text_context, hidden_context)
    return selected, mask


def zero_codec_lane_outside_response(
    codec_context: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Return literal Talker-hidden zeros outside active pre-END speech."""
    if response_mask.shape != codec_context.shape[:2]:
        raise ValueError(
            f"Response mask {tuple(response_mask.shape)} does not match "
            f"codec context {tuple(codec_context.shape[:2])}"
        )
    return codec_context * response_mask.to(
        device=codec_context.device,
        dtype=codec_context.dtype,
    ).unsqueeze(-1)


def reset_codec_history_at_response_starts(
    codec_in: torch.Tensor,
    text_target: torch.Tensor,
    assistant_start_id: int,
    silence_frame: torch.Tensor,
    codec_bos_id: int | None = None,
) -> torch.Tensor:
    """Force response-local canonical history on every AUDIO_START clock."""
    if codec_in.ndim != 3 or int(codec_in.shape[-1]) != CODEBOOKS:
        raise ValueError(f"Expected codec input [B,T,{CODEBOOKS}], got {tuple(codec_in.shape)}")
    if text_target.shape != codec_in.shape[:2]:
        raise ValueError(
            f"Text target {tuple(text_target.shape)} does not match codec input "
            f"{tuple(codec_in.shape[:2])}"
        )
    if tuple(silence_frame.shape) != (CODEBOOKS,):
        raise ValueError(f"Expected silence frame [{CODEBOOKS}], got {tuple(silence_frame.shape)}")
    start_mask = text_target == int(assistant_start_id)
    start_frame = silence_frame.to(
        device=codec_in.device,
        dtype=codec_in.dtype,
    ).clone()
    if codec_bos_id is not None:
        start_frame[0] = int(codec_bos_id)
    return torch.where(
        start_mask.to(device=codec_in.device).unsqueeze(-1),
        start_frame.view(1, 1, CODEBOOKS),
        codec_in,
    )


def assistant_turn_diagnostic_masks(
    *,
    assistant_mask: torch.Tensor | None,
    text_target: torch.Tensor,
    pad_id: int,
    assistant_start_id: int,
    assistant_end_id: int,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return response-start and lexical diagnostics for each assistant turn."""
    shape = text_target.shape
    response_text = torch.zeros(shape, dtype=torch.bool)
    first_five_lexical = torch.zeros(shape, dtype=torch.bool)
    first_lexical = torch.zeros(shape, dtype=torch.bool)
    response_start = torch.zeros(shape, dtype=torch.bool)
    del assistant_mask
    response_state = response_mask.detach().cpu()
    if response_state.shape != text_target.shape:
        raise ValueError(
            f"Response mask {tuple(response_state.shape)} does not match text "
            f"target {tuple(text_target.shape)}"
        )
    text_cpu = text_target.detach().cpu()
    for batch_index in range(int(shape[0])):
        lexical_rank = 0
        found_lexical = False
        for frame_index in range(int(shape[1])):
            if not bool(response_state[batch_index, frame_index]):
                lexical_rank = 0
                found_lexical = False
                continue
            token_id = int(text_cpu[batch_index, frame_index])
            if token_id == int(assistant_start_id):
                lexical_rank = 0
                found_lexical = False
                response_start[batch_index, frame_index] = True
                response_text[batch_index, frame_index] = True
            if token_id not in {
                int(pad_id),
                int(assistant_start_id),
                int(assistant_end_id),
            }:
                if not found_lexical:
                    first_lexical[batch_index, frame_index] = True
                    found_lexical = True
                if lexical_rank < 5:
                    first_five_lexical[batch_index, frame_index] = True
                lexical_rank += 1
    return response_text, first_five_lexical, first_lexical, response_start


def module_device(module: nn.Module, fallback: torch.device | None = None) -> torch.device:
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return fallback or torch.device("cpu")


def embed_on(
    embedding: nn.Module,
    ids: torch.Tensor,
    *,
    target_device: torch.device | None = None,
) -> torch.Tensor:
    out = embedding(ids.to(device=module_device(embedding, ids.device)))
    if target_device is not None:
        out = out.to(device=target_device)
    return out


def _resolve_compute_dtype(
    qwen: nn.Module,
    compute_dtype: torch.dtype | str | None,
) -> torch.dtype:
    if isinstance(compute_dtype, str):
        resolved = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
            compute_dtype
        ]
    elif compute_dtype is None:
        resolved = next(
            (parameter.dtype for parameter in qwen.parameters() if parameter.is_floating_point()),
            None,
        )
        if resolved is None:
            raise ValueError(
                "compute_dtype must be explicit when Qwen has no floating-point parameters"
            )
    elif isinstance(compute_dtype, torch.dtype):
        resolved = compute_dtype
    else:
        raise TypeError(
            "compute_dtype must be a torch.dtype, dtype name, or None, got "
            f"{type(compute_dtype).__name__}"
        )
    if resolved not in {torch.bfloat16, torch.float16, torch.float32}:
        raise ValueError(f"Unsupported compute dtype: {resolved}")
    return resolved


class QwenDuplexModel(nn.Module):
    @property
    def compute_dtype(self) -> torch.dtype:
        configured = self.__dict__.get("_compute_dtype")
        if configured is not None:
            return configured
        return _resolve_compute_dtype(self.qwen, None)

    @property
    def control_signal_contract(self) -> str:
        return self._control_signal_contract

    @property
    def is_dispatched(self) -> bool:
        return bool(getattr(self.qwen, "hf_device_map", None))

    def prepare_runtime_device(self, device: torch.device) -> None:
        self.audio_encoder.to(device)
        self.stream_type_embedding.to(device)
        self.text_control_rows.to(device)
        self.audio_control_rows.to(device)
        self.self_silence_embed.data = self.self_silence_embed.data.to(device=device)
        self.codec_silence_frame.data = self.codec_silence_frame.data.to(device=device)
        self.codec_codebook_weights.data = self.codec_codebook_weights.data.to(device=device)

    def set_codec_silence_frame(self, frame: torch.Tensor) -> None:
        if frame.shape != (CODEBOOKS,):
            raise ValueError(
                f"Expected codec silence frame [{CODEBOOKS}], got {tuple(frame.shape)}"
            )
        self.codec_silence_frame.data.copy_(
            frame.detach().cpu().long().to(self.codec_silence_frame.device)
        )

    def embed_text(
        self, ids: torch.Tensor, *, target_device: torch.device | None = None
    ) -> torch.Tensor:
        out = self.text_control_rows.embed(self.qwen.thinker.get_input_embeddings(), ids)
        out = self.audio_control_rows.embed(out, ids)
        if target_device is not None:
            out = out.to(device=target_device)
        return out

    def text_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.text_control_rows.logits(self.qwen.thinker.lm_head, hidden)
        logits = self.audio_control_rows.logits(logits, hidden)
        invalid_vocab_mask = torch.zeros(
            int(logits.shape[-1]), dtype=torch.bool, device=logits.device
        )
        if self.forbidden_text_special_ids:
            invalid_vocab_mask[
                torch.tensor(
                    self.forbidden_text_special_ids, dtype=torch.long, device=logits.device
                )
            ] = True
        invalid_vocab_mask[self.text_tokenizer_size :] = True
        return logits.masked_fill(invalid_vocab_mask, float("-inf"))

    def mask_text_logits(
        self,
        logits: torch.Tensor,
        *,
        speech_open: torch.Tensor,
        text_done: torch.Tensor,
        previous_text_id: torch.Tensor,
    ) -> torch.Tensor:
        return mask_async_text_logits(
            logits,
            speech_open=speech_open,
            text_done=text_done,
            previous_text_id=previous_text_id,
            pad_id=self.pad_id,
            assistant_start_id=self.assistant_start_id,
            assistant_end_id=self.assistant_end_id,
            special_ids=self.forbidden_text_special_ids,
        )

    def _talker_speaker_prefix(
        self,
        *,
        batch_size: int,
        target_device: torch.device,
        target_dtype: torch.dtype,
        speaker_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the one-time official speaker-position embedding.

        Omni represents the speaker position as projected TTS_PAD text plus a
        codec-vocabulary speaker ID.  The prefix is Talker-only and carries no
        timeline target or loss.
        """
        text_id = torch.tensor(
            [[int(self.qwen.config.tts_pad_token_id)]],
            dtype=torch.long,
            device=module_device(self.qwen.thinker.get_input_embeddings()),
        )
        text_embed = self.qwen.thinker.get_input_embeddings()(text_id)
        text_proj_device = module_device(self.qwen.talker.text_projection, text_embed.device)
        text_part = self.qwen.talker.text_projection(text_embed.to(device=text_proj_device))
        codec_embedding = self.qwen.talker.model.codec_embedding
        codec_device = module_device(codec_embedding)
        if speaker_ids is None:
            resolved_speaker_ids = torch.full(
                (int(batch_size),),
                int(self.talker_speaker_id),
                dtype=torch.long,
                device=codec_device,
            )
        else:
            if tuple(speaker_ids.shape) != (int(batch_size),):
                raise ValueError(
                    f"Talker speaker IDs must have shape [{batch_size}], got {tuple(speaker_ids.shape)}"
                )
            resolved_speaker_ids = speaker_ids.to(device=codec_device, dtype=torch.long)
        if resolved_speaker_ids.numel() and (
            int(resolved_speaker_ids.detach().min().cpu()) < 0
            or int(resolved_speaker_ids.detach().max().cpu()) >= int(codec_embedding.num_embeddings)
        ):
            raise ValueError(
                f"Talker speaker IDs exceed the codec embedding vocabulary: min={int(resolved_speaker_ids.detach().min().cpu())} max={int(resolved_speaker_ids.detach().max().cpu())} vocab={codec_embedding.num_embeddings}"
            )
        codec_part = codec_embedding(resolved_speaker_ids[:, None])
        prefix = text_part.to(device=target_device, dtype=target_dtype).expand(
            int(batch_size), -1, -1
        ) + codec_part.to(device=target_device, dtype=target_dtype)
        return prefix.contiguous()

    def _shift_right_codec(
        self, codec_target: torch.Tensor, *, previous_codec_target: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Shift the continuous full-duplex codec stream by exactly one frame."""
        seed = self.codec_silence_frame.to(device=codec_target.device).view(1, 1, CODEBOOKS)
        out = seed.expand_as(codec_target).clone()
        if previous_codec_target is not None:
            expected = (int(codec_target.shape[0]), CODEBOOKS)
            if tuple(previous_codec_target.shape) != expected:
                raise ValueError(
                    f"Previous codec target shape {tuple(previous_codec_target.shape)} != {expected}"
                )
            out[:, 0, :] = previous_codec_target.to(device=out.device, dtype=out.dtype)
        out[:, 1:, :] = codec_target[:, :-1, :]
        return out

    def _run_talker(
        self,
        *,
        text_target: torch.Tensor,
        thinker_layer24: torch.Tensor,
        codec_input: torch.Tensor,
        response_mask: torch.Tensor,
        dtype: torch.dtype,
        speaker_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        frame_offset: int = 0,
        valid_memory_frames: int | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Fuse text/layer-24 context with codec history and run the Talker."""
        batch_size, sequence_length = text_target.shape
        target_text_embedding = self.embed_text(text_target).to(dtype=dtype)
        text_projection_device = module_device(
            self.qwen.talker.text_projection, target_text_embedding.device
        )
        text_context = self.qwen.talker.text_projection(
            target_text_embedding.to(device=text_projection_device)
        )
        hidden_projection_device = module_device(
            self.qwen.talker.hidden_projection, thinker_layer24.device
        )
        hidden_context = self.qwen.talker.hidden_projection(
            thinker_layer24.to(device=hidden_projection_device, dtype=dtype)
        ).to(device=text_context.device, dtype=text_context.dtype)
        talker_context, use_text_context = select_talker_context(
            text_context=text_context,
            hidden_context=hidden_context,
            text_target=text_target.to(device=text_context.device),
            assistant_start_id=self.assistant_start_id,
            assistant_end_id=self.assistant_end_id,
            response_mask=response_mask.to(device=text_context.device),
        )
        codec_embedding = embed_on(self.qwen.talker.model.codec_embedding, codec_input[:, :, 0])
        codec_embedding_device = codec_embedding.device
        code_embeddings = self.qwen.talker.code_predictor.get_input_embeddings()
        for codebook in range(1, CODEBOOKS):
            codec_embedding = codec_embedding + embed_on(
                code_embeddings[codebook - 1],
                codec_input[:, :, codebook],
                target_device=codec_embedding_device,
            )
        codec_embedding = zero_codec_lane_outside_response(codec_embedding, use_text_context)
        talker_inputs = (
            codec_embedding.to(device=talker_context.device, dtype=talker_context.dtype)
            + talker_context
        )
        speaker_prefix = (
            self._talker_speaker_prefix(
                batch_size=batch_size,
                target_device=talker_inputs.device,
                target_dtype=talker_inputs.dtype,
                speaker_ids=speaker_ids,
            )
            if past_key_values is None
            else talker_inputs.new_empty((batch_size, 0, talker_inputs.shape[-1]))
        )
        talker_sequence = torch.cat((speaker_prefix, talker_inputs), dim=1)
        cached_tokens = (
            int(past_key_values.get_seq_length())
            if past_key_values is not None and hasattr(past_key_values, "get_seq_length")
            else 0
        )
        position_start = 0 if past_key_values is None else int(frame_offset) + 1
        position_ids = (
            torch.arange(
                position_start,
                position_start + int(talker_sequence.shape[1]),
                dtype=torch.long,
                device=talker_sequence.device,
            )
            .view(1, -1)
            .expand(batch_size, -1)
        )
        past_attention_mask = _transformer_xl_past_attention_mask(
            batch_size=batch_size,
            cached_tokens=cached_tokens,
            valid_memory_frames=valid_memory_frames,
            tokens_per_memory_frame=1,
            preserved_prefix_tokens=int(past_key_values is not None),
            device=talker_sequence.device,
        )
        attention_mask = None
        if past_attention_mask is not None:
            attention_mask = torch.cat(
                (
                    past_attention_mask,
                    torch.ones(
                        (batch_size, int(talker_sequence.shape[1])),
                        dtype=torch.long,
                        device=talker_sequence.device,
                    ),
                ),
                dim=1,
            )
        talker_outputs = self.qwen.talker.model(
            inputs_embeds=talker_sequence,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        talker_hidden = talker_outputs.last_hidden_state[:, int(speaker_prefix.shape[1]) :, :]
        codec0_logits = self.qwen.talker.codec_head(talker_hidden)[
            ..., : self.talker_codec_vocab_size
        ]
        return (talker_hidden, codec0_logits, use_text_context, talker_outputs.past_key_values)

    def _run_code_predictor(
        self,
        *,
        talker_hidden: torch.Tensor,
        codec_target: torch.Tensor,
        loss_mask: torch.Tensor,
        assistant_mask: torch.Tensor,
        nonassistant_mask: torch.Tensor,
        loss_codec0: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Teacher-force codebooks 1-15 within each causal audio frame."""
        flat_hidden = talker_hidden.reshape(-1, int(talker_hidden.shape[-1]))
        flat_codes = codec_target.to(device=talker_hidden.device).reshape(-1, CODEBOOKS)
        code_embeddings = self.qwen.talker.code_predictor.get_input_embeddings()
        predictor_parts = [
            flat_hidden.unsqueeze(1),
            embed_on(
                self.qwen.talker.model.codec_embedding,
                flat_codes[:, 0],
                target_device=flat_hidden.device,
            ).unsqueeze(1),
        ]
        for codebook in range(1, CODEBOOKS - 1):
            predictor_parts.append(
                embed_on(
                    code_embeddings[codebook - 1],
                    flat_codes[:, codebook],
                    target_device=flat_hidden.device,
                ).unsqueeze(1)
            )
        predictor_inputs = torch.cat(predictor_parts, dim=1)
        predictor_outputs = self.qwen.talker.code_predictor.model(
            inputs_embeds=predictor_inputs,
            attention_mask=torch.ones(
                predictor_inputs.shape[:2], dtype=torch.long, device=predictor_inputs.device
            ),
            use_cache=False,
        )
        predictor_hidden = predictor_outputs.last_hidden_state
        residual_losses = []
        residual_accuracies = []
        assistant_accuracies = []
        nonassistant_accuracies = []
        for codebook in range(1, CODEBOOKS):
            head = self.qwen.talker.code_predictor.lm_head[codebook - 1]
            head_device = module_device(head, predictor_hidden.device)
            logits = head(predictor_hidden[:, codebook, :].to(device=head_device))[
                ..., :CODEBOOK_SIZE
            ]
            code_mask = loss_mask.to(device=logits.device, dtype=logits.dtype)
            cross_entropy = F.cross_entropy(
                logits, flat_codes[:, codebook].to(device=logits.device), reduction="none"
            )
            residual_losses.append(
                ((cross_entropy * code_mask).sum() / code_mask.sum().clamp_min(1.0)).to(
                    device=loss_codec0.device
                )
            )
            correct = (
                logits.argmax(dim=-1) == flat_codes[:, codebook].to(device=logits.device)
            ).float()
            assistant_code_mask = assistant_mask.to(device=logits.device, dtype=logits.dtype)
            nonassistant_code_mask = nonassistant_mask.to(device=logits.device, dtype=logits.dtype)
            residual_accuracies.append(
                ((correct * code_mask).sum() / code_mask.sum().clamp_min(1.0))
                .to(device=loss_codec0.device)
                .detach()
            )
            assistant_accuracies.append(
                ((correct * assistant_code_mask).sum() / assistant_code_mask.sum().clamp_min(1.0))
                .to(device=loss_codec0.device)
                .detach()
            )
            nonassistant_accuracies.append(
                (
                    (correct * nonassistant_code_mask).sum()
                    / nonassistant_code_mask.sum().clamp_min(1.0)
                )
                .to(device=loss_codec0.device)
                .detach()
            )
        return (residual_losses, residual_accuracies, assistant_accuracies, nonassistant_accuracies)

    def forward(
        self,
        *,
        env_audio: torch.Tensor,
        self_audio: torch.Tensor,
        text_target: torch.Tensor,
        codec_target: torch.Tensor,
        frame_mask: torch.Tensor,
        assistant_mask: torch.Tensor | None = None,
        interrupt_end_weight_class: torch.Tensor | None = None,
        speaker_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs, _state = self._forward_impl(
            env_audio=env_audio,
            self_audio=self_audio,
            text_target=text_target,
            codec_target=codec_target,
            frame_mask=frame_mask,
            assistant_mask=assistant_mask,
            interrupt_end_weight_class=interrupt_end_weight_class,
            speaker_ids=speaker_ids,
            transformer_xl_state=None,
        )
        return outputs

    def forward_chunk(
        self,
        *,
        env_audio: torch.Tensor,
        self_audio: torch.Tensor,
        text_target: torch.Tensor,
        codec_target: torch.Tensor,
        frame_mask: torch.Tensor,
        assistant_mask: torch.Tensor | None = None,
        interrupt_end_weight_class: torch.Tensor | None = None,
        speaker_ids: torch.Tensor | None = None,
        state: QwenDuplexTransformerXLState | None = None,
        memory_frames: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], QwenDuplexTransformerXLState]:
        if memory_frames is not None:
            if isinstance(memory_frames, bool) or not isinstance(memory_frames, int):
                raise TypeError(
                    f"Transformer-XL memory_frames must be a non-negative integer or None, got {memory_frames!r}"
                )
            if memory_frames < 0:
                raise ValueError(
                    f"Transformer-XL memory_frames must be non-negative, got {memory_frames}"
                )
        if state is None:
            state = QwenDuplexTransformerXLState()
        if state.frame_offset == 0:
            if state.memory_frames is not None and state.memory_frames != memory_frames:
                raise ValueError(
                    f"Transformer-XL state memory does not match memory_frames: {state.memory_frames} != {memory_frames}"
                )
            state.memory_frames = memory_frames
        elif state.memory_frames != memory_frames:
            raise ValueError(
                f"Transformer-XL continuation memory_frames changed: {state.memory_frames} != {memory_frames}"
            )
        outputs, next_state = self._forward_impl(
            env_audio=env_audio,
            self_audio=self_audio,
            text_target=text_target,
            codec_target=codec_target,
            frame_mask=frame_mask,
            assistant_mask=assistant_mask,
            interrupt_end_weight_class=interrupt_end_weight_class,
            speaker_ids=speaker_ids,
            transformer_xl_state=state,
            transformer_xl_memory_frames=memory_frames,
        )
        if next_state is None:
            raise RuntimeError("Transformer-XL forward did not return recurrent state")
        return (outputs, next_state)

    def _forward_impl(
        self,
        *,
        env_audio: torch.Tensor,
        self_audio: torch.Tensor,
        text_target: torch.Tensor,
        codec_target: torch.Tensor,
        frame_mask: torch.Tensor,
        assistant_mask: torch.Tensor | None,
        interrupt_end_weight_class: torch.Tensor | None,
        speaker_ids: torch.Tensor | None,
        transformer_xl_state: QwenDuplexTransformerXLState | None,
        transformer_xl_memory_frames: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], QwenDuplexTransformerXLState | None]:
        if text_target.ndim != 2:
            raise ValueError(f"Expected text target [B,T], got {tuple(text_target.shape)}")
        bsz, seq_len = text_target.shape
        if env_audio.shape != self_audio.shape:
            raise ValueError(
                f"env/self audio shape mismatch: {tuple(env_audio.shape)} vs {tuple(self_audio.shape)}"
            )
        if env_audio.shape[:2] != (bsz, seq_len):
            raise ValueError(
                f"env/self audio prefix shape {tuple(env_audio.shape[:2])} != {(bsz, seq_len)}"
            )
        if codec_target.shape != (bsz, seq_len, CODEBOOKS):
            raise ValueError(
                f"codec target shape {tuple(codec_target.shape)} != {(bsz, seq_len, CODEBOOKS)}"
            )
        if codec_target.numel():
            codec_min = int(codec_target.detach().min().cpu())
            codec0_max = int(codec_target[:, :, 0].detach().max().cpu())
            residual_max = int(codec_target[:, :, 1:].detach().max().cpu())
            codec0_limit = self.talker_codec_vocab_size
            if codec_min < 0 or codec0_max >= codec0_limit or residual_max >= CODEBOOK_SIZE:
                raise ValueError(
                    f"Codec targets violate the Talker vocabulary contract: min={codec_min} codec0_max={codec0_max} codec0_limit={codec0_limit} residual_max={residual_max} residual_limit={CODEBOOK_SIZE}"
                )
        if frame_mask.shape != (bsz, seq_len):
            raise ValueError(f"frame mask shape {tuple(frame_mask.shape)} != {(bsz, seq_len)}")
        if assistant_mask is not None and assistant_mask.shape != (bsz, seq_len):
            raise ValueError(
                f"assistant mask shape {tuple(assistant_mask.shape)} != {(bsz, seq_len)}"
            )
        if interrupt_end_weight_class is not None and interrupt_end_weight_class.shape != (
            bsz,
            seq_len,
        ):
            raise ValueError(
                f"interrupt END weight-class shape {tuple(interrupt_end_weight_class.shape)} != {(bsz, seq_len)}"
            )
        if interrupt_end_weight_class is not None:
            weight_class = interrupt_end_weight_class.to(device=text_target.device)
            invalid_class = (weight_class < 0) | (weight_class > 2)
            if bool(invalid_class.any().item()):
                raise ValueError("interrupt END weight classes must be one of {0,1,2}")
            annotated_interrupt = weight_class > 0
            if bool(
                (
                    annotated_interrupt
                    & (
                        (text_target != self.assistant_end_id)
                        | ~frame_mask.to(device=text_target.device, dtype=torch.bool)
                    )
                )
                .any()
                .item()
            ):
                raise ValueError(
                    "interrupt END weight classes may annotate only active INTERRUPT targets"
                )
            active_interrupt_targets = frame_mask.to(
                device=text_target.device, dtype=torch.bool
            ) & text_target.eq(self.assistant_end_id)
            if not torch.equal(annotated_interrupt, active_interrupt_targets):
                raise ValueError(
                    "interrupt END weight classes must annotate every active INTERRUPT target and no other frame"
                )
        if speaker_ids is not None and tuple(speaker_ids.shape) != (bsz,):
            raise ValueError(f"speaker IDs shape {tuple(speaker_ids.shape)} != {(bsz,)}")
        is_transformer_xl = transformer_xl_state is not None
        frame_offset = (
            int(transformer_xl_state.frame_offset) if transformer_xl_state is not None else 0
        )
        if frame_offset < 0:
            raise ValueError(
                f"Transformer-XL frame offset must be non-negative, got {frame_offset}"
            )
        valid_memory_frames = (
            int(transformer_xl_state.valid_memory_frames) if transformer_xl_state is not None else 0
        )
        if valid_memory_frames < 0:
            raise ValueError(
                f"Transformer-XL valid memory must be non-negative, got {valid_memory_frames}"
            )
        if transformer_xl_memory_frames is None:
            if valid_memory_frames != 0:
                raise ValueError(
                    f"Unbounded Transformer-XL state cannot carry bounded-memory validity ({valid_memory_frames} frames)"
                )
        else:
            expected_valid_memory_frames = min(int(transformer_xl_memory_frames), frame_offset)
            if valid_memory_frames != expected_valid_memory_frames:
                raise ValueError(
                    f"Transformer-XL bounded-memory validity is inconsistent with its frame offset: {valid_memory_frames} != {expected_valid_memory_frames}"
                )
        if transformer_xl_state is not None and frame_offset > 0:
            required_state = [
                "previous_text_target",
                "previous_self_audio",
                "response_open",
                "text_done",
                "previous_codec_target",
                "previous_response_start",
            ]
            required_state.append("talker_open")
            missing_state = [
                name for name in required_state if getattr(transformer_xl_state, name) is None
            ]
            if missing_state:
                raise ValueError(
                    f"Transformer-XL continuation is missing boundary state: {missing_state}"
                )
            if (
                transformer_xl_state.thinker_cache is None
                or transformer_xl_state.talker_cache is None
            ):
                raise ValueError(
                    "Transformer-XL continuation requires both Thinker and Talker caches"
                )
            if transformer_xl_memory_frames is not None:
                thinker_cache_tokens = int(transformer_xl_state.thinker_cache.get_seq_length())
                expected_thinker_tokens = int(transformer_xl_memory_frames) * 3
                if thinker_cache_tokens != expected_thinker_tokens:
                    raise ValueError(
                        f"Bounded Thinker cache length does not match memory_frames: {thinker_cache_tokens} != {expected_thinker_tokens}"
                    )
                talker_cache_tokens = int(transformer_xl_state.talker_cache.get_seq_length())
                expected_talker_tokens = int(transformer_xl_memory_frames) + 1
                if talker_cache_tokens != expected_talker_tokens:
                    raise ValueError(
                        f"Bounded Talker cache length does not match memory_frames: {talker_cache_tokens} != {expected_talker_tokens}"
                    )
        dtype = self.compute_dtype
        codec0_target = codec_target[:, :, 0]
        immediate_pad_allowed = torch.zeros_like(text_target, dtype=torch.bool)
        talker_codec_bos_id = getattr(self, "talker_codec_bos_id", None)
        talker_codec_eos_id = getattr(self, "talker_codec_eos_id", None)
        if (
            int(text_target.shape[1]) > 1
            and talker_codec_bos_id is not None
            and (talker_codec_eos_id is not None)
        ):
            immediate_pad_allowed[:, 1:] = codec0_target[:, :-1].eq(
                int(talker_codec_bos_id)
            ) & codec0_target[:, 1:].eq(int(talker_codec_eos_id))
        if (
            int(text_target.shape[1]) > 0
            and talker_codec_bos_id is not None
            and (talker_codec_eos_id is not None)
            and (transformer_xl_state is not None)
            and (transformer_xl_state.previous_codec_target is not None)
        ):
            immediate_pad_allowed[:, 0] = transformer_xl_state.previous_codec_target[:, 0].to(
                device=codec0_target.device
            ).eq(int(talker_codec_bos_id)) & codec0_target[:, 0].eq(int(talker_codec_eos_id))
        response_mask, response_open = response_state_mask_with_initial(
            text_target,
            self.assistant_start_id,
            self.assistant_end_id,
            initial_response_open=None
            if transformer_xl_state is None
            else transformer_xl_state.response_open,
        )
        validated_response_open, text_done = validate_async_text_targets(
            text_target,
            pad_id=self.pad_id,
            assistant_start_id=self.assistant_start_id,
            assistant_end_id=self.assistant_end_id,
            initial_speech_open=None
            if transformer_xl_state is None
            else transformer_xl_state.response_open,
            initial_text_done=None
            if transformer_xl_state is None
            else transformer_xl_state.text_done,
            initial_previous_text_id=None
            if transformer_xl_state is None
            else transformer_xl_state.previous_text_target,
            immediate_pad_allowed=immediate_pad_allowed,
            forbidden_ids=self.forbidden_text_special_ids,
            text_tokenizer_size=self.text_tokenizer_size,
        )
        if not torch.equal(validated_response_open.to(device=response_open.device), response_open):
            raise RuntimeError("Async text validator response state drifted")
        response_text_mask, first_five_mask, first_lexical_mask, response_start_mask = (
            assistant_turn_diagnostic_masks(
                assistant_mask=assistant_mask,
                text_target=text_target,
                pad_id=self.pad_id,
                assistant_start_id=self.assistant_start_id,
                assistant_end_id=self.assistant_end_id,
                response_mask=response_mask,
            )
        )
        if assistant_mask is None:
            raise ValueError(
                "INTERRUPT/RESPONSE training requires the authoritative assistant mask"
            )
        talker_mask, talker_open = talker_state_mask_with_initial(
            codec_target[:, :, 0],
            codec_bos_id=self.talker_codec_bos_id,
            codec_eos_id=self.talker_codec_eos_id,
            initial_talker_open=None
            if transformer_xl_state is None
            else transformer_xl_state.talker_open,
        )
        response_targets = text_target == int(self.response_id)
        codec_bos_targets = codec_target[:, :, 0] == int(self.talker_codec_bos_id)
        if not torch.equal(response_targets, codec_bos_targets):
            raise ValueError("Every RESPONSE frame must carry exactly one Talker codec BOS")
        codec_eos_targets = codec_target[:, :, 0] == int(self.talker_codec_eos_id)
        assistant_activity = assistant_mask.to(device=text_target.device, dtype=torch.bool)
        if bool((assistant_activity & ~talker_mask).any().item()):
            raise ValueError("Assistant codec activity escapes its Talker episode")
        if bool((assistant_activity & (codec_bos_targets | codec_eos_targets)).any().item()):
            raise ValueError("Talker BOS/EOS frames cannot carry physical audio")
        codec_activity = assistant_activity
        talker_context_mask = talker_mask
        effective_codec_target, codec_loss_mask, residual_codec_loss_mask, codec0_frame_weight = (
            two_signal_codec_supervision(
                codec_target,
                frame_mask,
                talker_mask,
                codec_bos_targets,
                codec_eos_targets,
                self.codec_silence_frame,
                codec_eos_weight=self.talker_codec_eos_weight,
            )
        )
        codec_activity_mask = (
            (
                codec_activity.to(device=frame_mask.device, dtype=torch.bool)
                & frame_mask.to(dtype=torch.bool)
            )
            .unsqueeze(-1)
            .expand(-1, -1, CODEBOOKS)
        )
        codec_response_start_mask = torch.zeros_like(response_start_mask)
        if (
            transformer_xl_state is not None
            and transformer_xl_state.previous_response_start is not None
        ):
            codec_response_start_mask[:, 0] = transformer_xl_state.previous_response_start.to(
                device=response_start_mask.device, dtype=torch.bool
            ) & codec_activity[:, 0].to(device=response_start_mask.device, dtype=torch.bool)
        if int(seq_len) > 1:
            codec_response_start_mask[:, 1:] = response_start_mask[:, :-1] & codec_activity[
                :, 1:
            ].to(device=response_start_mask.device, dtype=torch.bool)
        codec_in = self._shift_right_codec(
            effective_codec_target,
            previous_codec_target=None
            if transformer_xl_state is None
            else transformer_xl_state.previous_codec_target,
        )
        codec_in = reset_codec_history_at_response_starts(
            codec_in,
            text_target,
            self.assistant_start_id,
            self.codec_silence_frame,
            codec_bos_id=self.talker_codec_bos_id,
        )
        (
            thinker_layer24,
            text_logits,
            event_control_logits,
            loss_text_raw,
            loss_text_raw_numerator,
            loss_text_denominator,
            thinker_cache,
        ) = self._run_thinker(
            text_target=text_target,
            env_audio=env_audio,
            self_audio=self_audio,
            frame_mask=frame_mask,
            response_mask=response_mask,
            interrupt_end_weight_class=interrupt_end_weight_class,
            dtype=dtype,
            past_key_values=None
            if transformer_xl_state is None
            else transformer_xl_state.thinker_cache,
            frame_offset=frame_offset,
            previous_text_target=None
            if transformer_xl_state is None
            else transformer_xl_state.previous_text_target,
            previous_self_audio=None
            if transformer_xl_state is None
            else transformer_xl_state.previous_self_audio,
            valid_memory_frames=None
            if transformer_xl_memory_frames is None
            else valid_memory_frames,
            use_cache=is_transformer_xl,
        )
        talker_hidden, codec0_logits, use_text_context, talker_cache = self._run_talker(
            text_target=text_target,
            thinker_layer24=thinker_layer24,
            codec_input=codec_in,
            dtype=dtype,
            speaker_ids=speaker_ids,
            response_mask=talker_context_mask,
            past_key_values=None
            if transformer_xl_state is None
            else transformer_xl_state.talker_cache,
            frame_offset=frame_offset,
            valid_memory_frames=None
            if transformer_xl_memory_frames is None
            else valid_memory_frames,
            use_cache=is_transformer_xl,
        )
        loss_codec_denominator = codec_loss_mask.float().sum()
        loss_codec0 = masked_ce(
            codec0_logits,
            effective_codec_target[:, :, 0],
            codec_loss_mask,
            frame_weight=codec0_frame_weight,
        )
        flat_loss_mask = (
            residual_codec_loss_mask.float().to(device=talker_hidden.device).reshape(-1)
        )
        flat_assistant_mask = (
            codec_activity_mask[:, :, 1]
            .to(device=talker_hidden.device, dtype=torch.float32)
            .reshape(-1)
        )
        flat_nonassistant_mask = flat_loss_mask * (1.0 - flat_assistant_mask)
        (
            residual_losses,
            residual_accuracies,
            residual_assistant_accuracies,
            residual_nonassistant_accuracies,
        ) = self._run_code_predictor(
            talker_hidden=talker_hidden,
            codec_target=effective_codec_target,
            loss_mask=flat_loss_mask,
            assistant_mask=flat_assistant_mask,
            nonassistant_mask=flat_nonassistant_mask,
            loss_codec0=loss_codec0,
        )
        codec_loss_terms = torch.stack([loss_codec0, *residual_losses])
        codec_weights = self.codec_codebook_weights.to(
            device=codec_loss_terms.device, dtype=codec_loss_terms.dtype
        )
        codec_weight_sum = codec_weights.sum().clamp_min(1e-06)
        loss_codec_raw = (codec_loss_terms * codec_weights).sum() / codec_weight_sum
        loss_codec = self.codec_loss_weight * loss_codec_raw
        loss_text = self.text_loss_weight * loss_text_raw
        loss_text_numerator = self.text_loss_weight * loss_text_raw_numerator
        loss = loss_text + loss_codec
        outputs = {
            "loss": loss,
            "loss_text": loss_text.detach(),
            "loss_codec": loss_codec.detach(),
            "loss_text_local_numerator": loss_text_numerator,
            "loss_text_local_denominator": loss_text_denominator.detach(),
            "loss_codec_local_numerator": loss_codec * loss_codec_denominator,
            "loss_codec_local_denominator": loss_codec_denominator.detach(),
            "loss_codec0": loss_codec0.detach(),
            "codec_per_codebook_loss": codec_loss_terms.detach(),
        }
        if transformer_xl_state is None:
            return (outputs, None)
        transformer_xl_state.thinker_cache = thinker_cache
        transformer_xl_state.talker_cache = talker_cache
        transformer_xl_state.frame_offset = frame_offset + int(seq_len)
        if transformer_xl_memory_frames is not None:
            transformer_xl_state.thinker_cache = _compact_transformer_cache(
                transformer_xl_state.thinker_cache,
                target_tokens=int(transformer_xl_memory_frames) * 3,
            )
            transformer_xl_state.talker_cache = _compact_transformer_cache(
                transformer_xl_state.talker_cache,
                target_tokens=int(transformer_xl_memory_frames) + 1,
                preserved_prefix_tokens=1,
            )
            transformer_xl_state.valid_memory_frames = min(
                int(transformer_xl_memory_frames), transformer_xl_state.frame_offset
            )
        transformer_xl_state.previous_text_target = text_target[:, -1]
        transformer_xl_state.previous_self_audio = self_audio[:, -1, :]
        transformer_xl_state.response_open = response_open
        transformer_xl_state.text_done = text_done
        transformer_xl_state.previous_codec_target = effective_codec_target[:, -1, :]
        transformer_xl_state.previous_response_start = response_start_mask[:, -1]
        transformer_xl_state.talker_open = talker_open
        return (outputs, transformer_xl_state.detach_())
