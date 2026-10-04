from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .encoder_base import (
    Encoder as CausalAutEncoder,
    CausalAutLayerState,
    CausalAutStreamState,
    LoRALinear,
)
from .encoder_config import MacroAuTContract, NATIVE_MEL_FRAMES


FRONTEND_MODULES = ("conv2d1", "conv2d2", "conv2d3", "conv_out")


ATTENTION_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
)


NON_ATTENTION_LORA_TARGETS = ("fc1", "fc2", "proj1", "proj2")


AUT_FRONTEND_LR = 5.0e-5


AUT_ATTENTION_LR = 2.0e-6


AUT_LORA_LR = 1.0e-4


AUT_WEIGHT_DECAY = 0.01


FULL_SEQUENCE_QUERY_BLOCK_TOKENS = 1040


SFT_ATTENTION_LORA_PARAMETERS = 10_485_760


SFT_NON_ATTENTION_LORA_PARAMETERS = 13_295_616


SFT_FRONTEND_PARAMETERS = 13_983_360


SFT_ENCODER_PARAMETERS = (
    SFT_ATTENTION_LORA_PARAMETERS + SFT_NON_ATTENTION_LORA_PARAMETERS + SFT_FRONTEND_PARAMETERS
)


@dataclass
class MacroAuTStreamState(CausalAutStreamState):
    """Bounded state for a complete-macro convolution and Transformer stream."""

    @property
    def pending_native_tokens(self) -> int:
        return self.mel_frames_since_emit // NATIVE_MEL_FRAMES


@dataclass(frozen=True)
class MacroAttentionBlock:
    """One bounded slice of full-sequence macro attention.

    Query bounds are macro-unit aligned.  Key bounds are the union of the
    bounded windows needed by those queries, rather than the complete input.
    """

    query_start: int
    query_end: int
    key_start: int
    key_end: int

    @property
    def query_tokens(self) -> int:
        return self.query_end - self.query_start

    @property
    def key_tokens(self) -> int:
        return self.key_end - self.key_start

    @property
    def mask_shape(self) -> tuple[int, int]:
        return (self.query_tokens, self.key_tokens)


def plan_macro_attention_blocks(
    token_count: int,
    *,
    frames_per_unit: int,
    window_tokens: int,
    query_block_tokens: int = FULL_SEQUENCE_QUERY_BLOCK_TOKENS,
) -> tuple[MacroAttentionBlock, ...]:
    """Plan bounded attention blocks without allocating a ``T x T`` mask."""

    contract = MacroAuTContract(
        frames_per_unit=frames_per_unit,
        attention_window_tokens=window_tokens,
    )
    if token_count < 1:
        raise ValueError("cannot plan attention for an empty sequence")
    if token_count % contract.frames_per_unit:
        raise ValueError(
            f"token_count={token_count} ends in a partial "
            f"{contract.frames_per_unit}-token macro unit"
        )
    if query_block_tokens < 1:
        raise ValueError("query_block_tokens must be positive")
    if query_block_tokens % contract.frames_per_unit:
        raise ValueError("query_block_tokens must be divisible by frames_per_unit")

    blocks: list[MacroAttentionBlock] = []
    for query_start in range(0, token_count, query_block_tokens):
        query_end = min(token_count, query_start + query_block_tokens)
        key_start = max(0, query_start - contract.max_past_tokens)
        blocks.append(
            MacroAttentionBlock(
                query_start=query_start,
                query_end=query_end,
                key_start=key_start,
                key_end=query_end,
            )
        )
    return tuple(blocks)


def macro_attention_block_mask(
    block: MacroAttentionBlock,
    *,
    frames_per_unit: int,
    window_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a compact boolean SDPA mask (``True`` means visible)."""

    contract = MacroAuTContract(
        frames_per_unit=frames_per_unit,
        attention_window_tokens=window_tokens,
    )
    query = torch.arange(block.query_start, block.query_end, device=device)
    key = torch.arange(block.key_start, block.key_end, device=device)
    unit_start = (
        torch.div(query, contract.frames_per_unit, rounding_mode="floor") * contract.frames_per_unit
    )
    unit_end = unit_start + contract.frames_per_unit
    first_visible = torch.clamp(
        unit_start - contract.max_past_tokens,
        min=0,
    )
    return (key.unsqueeze(0) >= first_visible.unsqueeze(1)) & (
        key.unsqueeze(0) < unit_end.unsqueeze(1)
    )


def macro_audio_attention_full(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    *,
    frames_per_unit: int,
    window_tokens: int,
    query_block_tokens: int = FULL_SEQUENCE_QUERY_BLOCK_TOKENS,
) -> torch.Tensor:
    """Exact full-sequence macro attention with bounded temporary memory.

    Q/K/V are projected once for the complete sequence.  Each SDPA call sees
    only the union of key windows required by an aligned query block, so the
    temporary mask is bounded by ``query_block_tokens x
    (query_block_tokens + window_tokens - frames_per_unit)`` instead of
    growing as ``T x T``.
    """

    if hidden_states.ndim != 2 or int(hidden_states.shape[0]) < 1:
        raise ValueError(
            f"expected non-empty hidden states [T,D], got {tuple(hidden_states.shape)}"
        )
    token_count = int(hidden_states.shape[0])
    blocks = plan_macro_attention_blocks(
        token_count,
        frames_per_unit=frames_per_unit,
        window_tokens=window_tokens,
        query_block_tokens=query_block_tokens,
    )

    query = attention.q_proj(hidden_states).reshape(token_count, attention.num_heads, -1)
    key = attention.k_proj(hidden_states).reshape(token_count, attention.num_heads, -1)
    value = attention.v_proj(hidden_states).reshape(token_count, attention.num_heads, -1)
    query = query.transpose(0, 1).unsqueeze(0)
    key = key.transpose(0, 1).unsqueeze(0)
    value = value.transpose(0, 1).unsqueeze(0)

    output_blocks: list[torch.Tensor] = []
    for block in blocks:
        mask = macro_attention_block_mask(
            block,
            frames_per_unit=frames_per_unit,
            window_tokens=window_tokens,
            device=hidden_states.device,
        )
        output_blocks.append(
            F.scaled_dot_product_attention(
                query[..., block.query_start : block.query_end, :],
                key[..., block.key_start : block.key_end, :],
                value[..., block.key_start : block.key_end, :],
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
            )
        )

    output = torch.cat(output_blocks, dim=-2)
    output = output.squeeze(0).transpose(0, 1).reshape(token_count, -1)
    return attention.out_proj(output.contiguous())


def macro_encoder_layer_full(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    *,
    frames_per_unit: int,
    window_tokens: int,
    query_block_tokens: int = FULL_SEQUENCE_QUERY_BLOCK_TOKENS,
) -> torch.Tensor:
    """Apply one AuT layer using memory-bounded full-sequence attention."""

    residual = hidden_states
    hidden_states = layer.self_attn_layer_norm(hidden_states)
    hidden_states = macro_audio_attention_full(
        layer.self_attn,
        hidden_states,
        frames_per_unit=frames_per_unit,
        window_tokens=window_tokens,
        query_block_tokens=query_block_tokens,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = layer.final_layer_norm(hidden_states)
    hidden_states = layer.fc1(hidden_states)
    hidden_states = layer.activation_fn(hidden_states)
    hidden_states = layer.fc2(hidden_states)
    hidden_states = residual + hidden_states
    if hidden_states.dtype == torch.float16:
        limit = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-limit, max=limit)
    return hidden_states


def macro_audio_attention_chunk(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    *,
    past_key: torch.Tensor | None,
    past_value: torch.Tensor | None,
    frames_per_unit: int,
    window_tokens: int,
    query_block_tokens: int = FULL_SEQUENCE_QUERY_BLOCK_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attend one trainable chunk to detached bounded KV memory.

    The current chunk remains block-causal by macro unit.  Returned K/V are
    the current projections, allowing the caller to retain a detached suffix
    for the next optimizer step without recomputing the complete sample.
    """

    if hidden_states.ndim != 2 or int(hidden_states.shape[0]) < 1:
        raise ValueError(
            f"expected non-empty hidden states [T,D], got {tuple(hidden_states.shape)}"
        )
    contract = MacroAuTContract(
        frames_per_unit=frames_per_unit,
        attention_window_tokens=window_tokens,
    )
    token_count = int(hidden_states.shape[0])
    if token_count % contract.frames_per_unit:
        raise ValueError("SFT AuT chunk ends in a partial macro unit")
    if query_block_tokens < 1 or query_block_tokens % contract.frames_per_unit:
        raise ValueError("query_block_tokens must be macro-unit aligned")
    if (past_key is None) != (past_value is None):
        raise RuntimeError("SFT AuT memory must contain both key and value")

    query = attention.q_proj(hidden_states).reshape(token_count, attention.num_heads, -1)
    current_key = attention.k_proj(hidden_states).reshape(token_count, attention.num_heads, -1)
    current_value = attention.v_proj(hidden_states).reshape(token_count, attention.num_heads, -1)
    query = query.transpose(0, 1).unsqueeze(0)
    current_key = current_key.transpose(0, 1).unsqueeze(0)
    current_value = current_value.transpose(0, 1).unsqueeze(0)

    memory_tokens = 0
    if past_key is not None and past_value is not None:
        if (
            past_key.ndim != 4
            or tuple(past_key.shape) != tuple(past_value.shape)
            or int(past_key.shape[0]) != 1
            or int(past_key.shape[1]) != int(current_key.shape[1])
            or int(past_key.shape[-1]) != int(current_key.shape[-1])
        ):
            raise ValueError(
                "SFT AuT memory shape differs from current K/V: "
                f"past={tuple(past_key.shape)} current={tuple(current_key.shape)}"
            )
        memory_tokens = int(past_key.shape[-2])
        if memory_tokens > contract.max_past_tokens:
            raise ValueError(
                f"SFT AuT memory has {memory_tokens} tokens, maximum is {contract.max_past_tokens}"
            )
        key = torch.cat((past_key, current_key), dim=-2)
        value = torch.cat((past_value, current_value), dim=-2)
    else:
        key = current_key
        value = current_value

    output_blocks: list[torch.Tensor] = []
    for query_start in range(0, token_count, query_block_tokens):
        query_end = min(token_count, query_start + query_block_tokens)
        key_start = max(
            0,
            memory_tokens + query_start - contract.max_past_tokens,
        )
        key_end = memory_tokens + query_end
        query_positions = torch.arange(
            query_start,
            query_end,
            device=hidden_states.device,
        )
        key_positions = torch.arange(
            key_start,
            key_end,
            device=hidden_states.device,
        )
        unit_start = (
            torch.div(
                query_positions,
                contract.frames_per_unit,
                rounding_mode="floor",
            )
            * contract.frames_per_unit
        )
        first_visible = torch.clamp(
            memory_tokens + unit_start - contract.max_past_tokens,
            min=0,
        )
        visible_end = memory_tokens + unit_start + contract.frames_per_unit
        mask = (key_positions.unsqueeze(0) >= first_visible.unsqueeze(1)) & (
            key_positions.unsqueeze(0) < visible_end.unsqueeze(1)
        )
        output_blocks.append(
            F.scaled_dot_product_attention(
                query[..., query_start:query_end, :],
                key[..., key_start:key_end, :],
                value[..., key_start:key_end, :],
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
            )
        )

    output = torch.cat(output_blocks, dim=-2)
    output = output.squeeze(0).transpose(0, 1).reshape(token_count, -1)
    return (
        attention.out_proj(output.contiguous()),
        current_key,
        current_value,
    )


def macro_encoder_layer_chunk(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    *,
    past_key: torch.Tensor | None,
    past_value: torch.Tensor | None,
    frames_per_unit: int,
    window_tokens: int,
    query_block_tokens: int = FULL_SEQUENCE_QUERY_BLOCK_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one AuT layer to a chunk and return its current K/V."""

    residual = hidden_states
    normalized = layer.self_attn_layer_norm(hidden_states)
    hidden_states, current_key, current_value = macro_audio_attention_chunk(
        layer.self_attn,
        normalized,
        past_key=past_key,
        past_value=past_value,
        frames_per_unit=frames_per_unit,
        window_tokens=window_tokens,
        query_block_tokens=query_block_tokens,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = layer.final_layer_norm(hidden_states)
    hidden_states = layer.fc1(hidden_states)
    hidden_states = layer.activation_fn(hidden_states)
    hidden_states = layer.fc2(hidden_states)
    hidden_states = residual + hidden_states
    if hidden_states.dtype == torch.float16:
        limit = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-limit, max=limit)
    return hidden_states, current_key, current_value


def _claim_macro_cache(state: CausalAutLayerState) -> None:
    backend = "d2-macro-torch-sdpa"
    if state.cache_backend not in (None, backend):
        raise RuntimeError(f"AuT stream cache belongs to {state.cache_backend!r}, not {backend!r}")
    state.cache_backend = backend
    state.cache_token_axis = -2


def macro_audio_attention_step(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    state: CausalAutLayerState,
    *,
    max_past_tokens: int,
) -> torch.Tensor:
    """Attend one complete macro unit to bounded past KV and itself."""

    if hidden_states.ndim != 2 or int(hidden_states.shape[0]) < 1:
        raise ValueError(
            f"expected a non-empty macro token block [k,D], got {tuple(hidden_states.shape)}"
        )
    if max_past_tokens < 0:
        raise ValueError("max_past_tokens must be non-negative")
    _claim_macro_cache(state)
    if (state.key is None) != (state.value is None):
        raise RuntimeError("AuT attention state must contain both key and value")

    unit_tokens = int(hidden_states.shape[0])
    query = attention.q_proj(hidden_states).reshape(unit_tokens, attention.num_heads, -1)
    current_key = attention.k_proj(hidden_states).reshape(unit_tokens, attention.num_heads, -1)
    current_value = attention.v_proj(hidden_states).reshape(unit_tokens, attention.num_heads, -1)
    query = query.transpose(0, 1).unsqueeze(0)
    current_key = current_key.transpose(0, 1).unsqueeze(0)
    current_value = current_value.transpose(0, 1).unsqueeze(0)

    valid = int(state.valid_tokens)
    if valid < 0 or valid > max_past_tokens:
        raise RuntimeError(f"invalid macro AuT cache length {valid}/{max_past_tokens}")
    if valid:
        if state.key is None or state.value is None:
            raise RuntimeError("macro AuT cache length is nonzero but tensors are absent")
        expected_prefix = (
            1,
            int(current_key.shape[1]),
            max_past_tokens,
            int(current_key.shape[-1]),
        )
        if tuple(state.key.shape) != expected_prefix:
            raise RuntimeError(f"macro AuT KV shape {tuple(state.key.shape)} != {expected_prefix}")
        past_key = state.key[..., :valid, :]
        past_value = state.value[..., :valid, :]
        key = torch.cat((past_key, current_key), dim=-2)
        value = torch.cat((past_value, current_value), dim=-2)
    else:
        key = current_key
        value = current_value

    # Every query in this just-finalized unit sees all current keys.  There is
    # no future unit in ``key``, so no additional mask is required.
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )

    if max_past_tokens:
        expected = (
            1,
            int(current_key.shape[1]),
            max_past_tokens,
            int(current_key.shape[-1]),
        )
        if (
            state.key is None
            or tuple(state.key.shape) != expected
            or state.key.device != current_key.device
            or state.key.dtype != current_key.dtype
        ):
            state.key = current_key.new_zeros(expected)
            state.value = current_value.new_zeros(expected)
        if state.value is None:
            raise RuntimeError("macro AuT value cache was not allocated")
        kept_key = key[..., -max_past_tokens:, :].detach()
        kept_value = value[..., -max_past_tokens:, :].detach()
        kept = int(kept_key.shape[-2])
        state.key.zero_()
        state.value.zero_()
        state.key[..., :kept, :].copy_(kept_key)
        state.value[..., :kept, :].copy_(kept_value)
        state.valid_tokens = kept
        state.cursor = kept % max_past_tokens
    else:
        state.key = None
        state.value = None
        state.valid_tokens = 0
        state.cursor = 0

    output = output.squeeze(0).transpose(0, 1).reshape(unit_tokens, -1)
    return attention.out_proj(output.contiguous())


def macro_encoder_layer_step(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    state: CausalAutLayerState,
    *,
    max_past_tokens: int,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = layer.self_attn_layer_norm(hidden_states)
    hidden_states = macro_audio_attention_step(
        layer.self_attn,
        hidden_states,
        state,
        max_past_tokens=max_past_tokens,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = layer.final_layer_norm(hidden_states)
    hidden_states = layer.fc1(hidden_states)
    hidden_states = layer.activation_fn(hidden_states)
    hidden_states = layer.fc2(hidden_states)
    hidden_states = residual + hidden_states
    if hidden_states.dtype == torch.float16:
        limit = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-limit, max=limit)
    return hidden_states


class MacroAuTEncoder(CausalAutEncoder):
    def __init__(self, source_audio_tower, *, latency_ms, encoder_state=None, sft=True):
        nn.Module.__init__(self)
        self.audio_tower = source_audio_tower
        self.config = source_audio_tower.config
        self.window_tokens = 104
        self.macro_contract = MacroAuTContract.from_latency(latency_ms)
        self.gradient_checkpointing = True
        self.training_lora_gradient_checkpointing = True
        self._streaming_attention_backend = None
        self._uniform_forward_backend = None
        self.lora_metadata = self._apply_lora(
            rank=32, alpha=64.0, dropout=0.0, target_suffixes=NON_ATTENTION_LORA_TARGETS
        )
        self._configure_training_topology(trainable_dtype=torch.float32)
        self.training_lora_metadata = {
            "rank": 32,
            "alpha": 64.0,
            "dropout": 0.0,
            "target_modules": list(NON_ATTENTION_LORA_TARGETS),
            "wrapped_modules": self.lora_metadata["wrapped_modules"],
        }
        self._trainable_names = tuple(n for n, p in self.named_parameters() if p.requires_grad)
        self._sft_trainable_names = ()
        self.sft_training_metadata = None
        if encoder_state is not None:
            from d2.weights import restore_parameters

            restore_parameters(self, encoder_state, expected=set(self._trainable_names))
        if sft:
            self.configure_sft_training()

    def forward_uniform(self, *, input_features, feature_lens, **kwargs):
        offset, outputs = 0, []
        for length in feature_lens.tolist():
            outputs.append(self._forward_uniform_one(input_features[:, offset : offset + length]))
            offset += length
        if offset != input_features.shape[-1]:
            raise ValueError("Feature lengths do not cover the input")
        return torch.cat(outputs, dim=0)

    """Qwen causal AuT with a configurable bidirectional macro-unit window."""

    @property
    def frames_per_unit(self) -> int:
        return self.macro_contract.frames_per_unit

    @property
    def latency_ms(self) -> int:
        return self.macro_contract.latency_ms

    @property
    def max_past_tokens(self) -> int:
        return self.macro_contract.max_past_tokens

    def _configure_training_topology(self, *, trainable_dtype: torch.dtype | None = None) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for name in FRONTEND_MODULES:
            module = getattr(self.audio_tower, name)
            if trainable_dtype is not None:
                module.to(dtype=trainable_dtype)
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        for name, module in self.audio_tower.named_modules():
            if isinstance(module, nn.Linear) and any(
                (name.endswith(target) for target in ATTENTION_TARGETS)
            ):
                if trainable_dtype is not None:
                    module.to(dtype=trainable_dtype)
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
            elif isinstance(module, LoRALinear):
                for parameter in module.lora_a.parameters():
                    parameter.requires_grad_(True)
                for parameter in module.lora_b.parameters():
                    parameter.requires_grad_(True)

    def _parameter_group_names(self) -> dict[str, tuple[str, ...]]:
        groups: dict[str, list[str]] = {
            "frontend_full": [],
            "attention_full": [],
            "non_attention_lora": [],
        }
        for name in self._trainable_names:
            relative = name.removeprefix("audio_tower.")
            if any(
                (
                    relative == prefix or relative.startswith(prefix + ".")
                    for prefix in FRONTEND_MODULES
                )
            ):
                groups["frontend_full"].append(name)
            elif ".lora_a." in name or ".lora_b." in name:
                groups["non_attention_lora"].append(name)
            elif any(
                (
                    relative.endswith(target + ".weight") or relative.endswith(target + ".bias")
                    for target in ATTENTION_TARGETS
                )
            ):
                groups["attention_full"].append(name)
            else:
                raise RuntimeError(f"unclassified trainable AuT parameter: {name}")
        return {name: tuple(values) for name, values in groups.items()}

    def _validate_training_topology(self) -> None:
        if self.training_lora_metadata["rank"] != 32:
            raise ValueError("D2 AuT non-attention LoRA must use rank 32")
        if self.training_lora_metadata["alpha"] != 64.0:
            raise ValueError("D2 AuT non-attention LoRA must use alpha 64")
        if self.training_lora_metadata["dropout"] != 0.0:
            raise ValueError("D2 AuT non-attention LoRA dropout must be zero")
        groups = self._parameter_group_names()
        empty = [name for name, values in groups.items() if not values]
        if empty:
            raise RuntimeError(f"empty D2 AuT parameter groups: {empty}")
        current = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        if current != set(self._trainable_names):
            raise RuntimeError("D2 AuT trainable inventory changed unexpectedly")

    def enable_training_lora_parameters(self) -> None:
        """Restore the complete D2 topology, not merely its LoRA subset."""
        parameters = dict(self.named_parameters())
        for parameter in parameters.values():
            parameter.requires_grad_(False)
        for name in self._trainable_names:
            parameters[name].requires_grad_(True)

    def configure_training_lora_mode(self) -> None:
        self.train()

    def configure_sft_training(
        self,
        *,
        rank: int = 32,
        alpha: float = 64.0,
        dropout: float = 0.0,
        gradient_checkpointing: bool = True,
    ) -> dict[str, Any]:
        """Turn the distilled encoder into the compact SFT topology.

        The distilled dense attention is the frozen base.  SFT adds a fresh
        attention adapter, continues the existing MLP/projection adapters,
        and trains the convolution frontend.  The distilled runtime payload
        is loaded before this transformation and is never mutated on disk.
        """
        if self.sft_training_metadata is not None:
            raise RuntimeError("D2 AuT SFT topology is already configured")
        if (int(rank), float(alpha), float(dropout)) != (32, 64.0, 0.0):
            raise ValueError("D2 AuT SFT requires rank=32, alpha=64, dropout=0")
        attention = self._apply_lora(
            rank=int(rank),
            alpha=float(alpha),
            dropout=float(dropout),
            target_suffixes=ATTENTION_TARGETS,
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for name in FRONTEND_MODULES:
            module = getattr(self.audio_tower, name)
            module.to(dtype=torch.float32)
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        for module in self.audio_tower.modules():
            if isinstance(module, LoRALinear):
                for parameter in (*module.lora_a.parameters(), *module.lora_b.parameters()):
                    parameter.requires_grad_(True)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.training_lora_gradient_checkpointing = bool(gradient_checkpointing)
        self._sft_trainable_names = tuple(
            (name for name, parameter in self.named_parameters() if parameter.requires_grad)
        )
        groups = self.sft_parameter_group_names()
        counts = {
            name: sum((dict(self.named_parameters())[item].numel() for item in names))
            for name, names in groups.items()
        }
        expected = {
            "attention_lora": SFT_ATTENTION_LORA_PARAMETERS,
            "non_attention_lora": SFT_NON_ATTENTION_LORA_PARAMETERS,
            "frontend_full": SFT_FRONTEND_PARAMETERS,
        }
        if counts != expected:
            raise RuntimeError(f"D2 AuT SFT parameter inventory changed: {counts} != {expected}")
        self.sft_training_metadata = {
            "rank": int(rank),
            "alpha": float(alpha),
            "dropout": float(dropout),
            "gradient_checkpointing": bool(gradient_checkpointing),
            "attention_modules": list(attention["wrapped_modules"]),
            "trainable_parameters": SFT_ENCODER_PARAMETERS,
        }
        return dict(self.sft_training_metadata)

    def sft_parameter_group_names(self) -> dict[str, tuple[str, ...]]:
        groups: dict[str, list[str]] = {
            "frontend_full": [],
            "attention_lora": [],
            "non_attention_lora": [],
        }
        for name in self._sft_trainable_names:
            relative = name.removeprefix("audio_tower.")
            if any(
                (
                    relative == prefix or relative.startswith(prefix + ".")
                    for prefix in FRONTEND_MODULES
                )
            ):
                groups["frontend_full"].append(name)
            elif any(
                (
                    relative.startswith(target + ".") or f".{target}." in relative
                    for target in ATTENTION_TARGETS
                )
            ) and (".lora_a." in name or ".lora_b." in name):
                groups["attention_lora"].append(name)
            elif ".lora_a." in name or ".lora_b." in name:
                groups["non_attention_lora"].append(name)
            else:
                raise RuntimeError(f"unclassified D2 AuT SFT parameter: {name}")
        return {name: tuple(values) for name, values in groups.items()}

    def enable_sft_training_parameters(self) -> None:
        if self.sft_training_metadata is None:
            raise RuntimeError("D2 AuT SFT topology is not configured")
        parameters = dict(self.named_parameters())
        for parameter in parameters.values():
            parameter.requires_grad_(False)
        for name in self._sft_trainable_names:
            parameters[name].requires_grad_(True)

    def configure_sft_training_mode(self) -> None:
        self.eval()
        for module in self.audio_tower.modules():
            if isinstance(module, LoRALinear):
                module.dropout.train()

    def optimizer_parameter_groups(
        self, *, weight_decay: float = AUT_WEIGHT_DECAY
    ) -> tuple[dict[str, Any], ...]:
        """Return the three AdamW groups consumed by the shared trainer."""
        names = self._parameter_group_names()
        parameters = dict(self.named_parameters())
        learning_rates = {
            "frontend_full": AUT_FRONTEND_LR,
            "attention_full": AUT_ATTENTION_LR,
            "non_attention_lora": AUT_LORA_LR,
        }
        return tuple(
            (
                {
                    "name": group_name,
                    "params": [parameters[name] for name in names[group_name]],
                    "lr": learning_rates[group_name],
                    "peak_lr": learning_rates[group_name],
                    "weight_decay": float(weight_decay),
                }
                for group_name in ("frontend_full", "attention_full", "non_attention_lora")
            )
        )

    def trainable_named_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        parameters = dict(self.named_parameters())
        return ((name, parameters[name]) for name in self._trainable_names)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        parameters = dict(self.named_parameters())
        return {name: parameters[name].detach().cpu() for name in self._trainable_names}

    @staticmethod
    def trainable_topology_contract() -> dict[str, Any]:
        return {
            "frontend": list(FRONTEND_MODULES),
            "attention": list(ATTENTION_TARGETS),
            "non_attention_lora": list(NON_ATTENTION_LORA_TARGETS),
            "rank": 32,
            "alpha": 64.0,
            "dropout": 0.0,
        }

    def _conv_macro_window_batch(self, windows: list[torch.Tensor]) -> torch.Tensor:
        """Convolve each 104-mel window once and select its final ``k`` slots."""
        if not windows:
            raise ValueError("macro AuT convolution batch is empty")
        mel_bins = int(windows[0].shape[0])
        window_frames = self.macro_contract.conv_window_mel_frames
        if any(
            (
                window.ndim != 2
                or int(window.shape[0]) != mel_bins
                or int(window.shape[1]) != window_frames
                for window in windows
            )
        ):
            raise ValueError("macro AuT convolution requires aligned 104-mel windows")
        slot_count = (
            self.macro_contract.conv_window_mel_frames // self.macro_contract.mel_frames_per_token
        )
        first_slot = slot_count - self.frames_per_unit
        positions = torch.arange(first_slot, slot_count, device=windows[0].device, dtype=torch.long)
        positional = self.audio_tower.positional_embedding.positional_embedding.index_select(
            0, positions
        )
        batch_size = max(1, int(self.audio_tower.conv_chunksize))
        selected_units: list[torch.Tensor] = []
        for start in range(0, len(windows), batch_size):
            x = torch.stack(windows[start : start + batch_size], dim=0).unsqueeze(1)
            x = F.gelu(self.audio_tower.conv2d1(x))
            x = F.gelu(self.audio_tower.conv2d2(x))
            x = F.gelu(self.audio_tower.conv2d3(x))
            unit_batch, channels, frequency, time = x.size()
            if int(time) != slot_count:
                raise RuntimeError(
                    f"macro AuT 104-mel convolution emitted {time} slots; expected {slot_count}"
                )
            tokens = self.audio_tower.conv_out(
                x.permute(0, 3, 1, 2).contiguous().view(unit_batch, time, channels * frequency)
            )
            selected = tokens.index_select(1, positions)
            selected_units.append(
                selected + positional.to(device=selected.device, dtype=selected.dtype).unsqueeze(0)
            )
        return torch.cat(selected_units, dim=0).flatten(0, 1)

    def _conv_embed_macro_units(
        self, input_features: torch.Tensor, *, first_unit_end: int, unit_count: int
    ) -> torch.Tensor:
        """Embed complete macro units ending at the specified mel positions."""
        if input_features.ndim != 2:
            raise ValueError(f"expected features [mel,frames], got {tuple(input_features.shape)}")
        if unit_count < 0:
            raise ValueError("macro AuT unit_count must be non-negative")
        if unit_count == 0:
            return input_features.new_empty((0, int(self.audio_tower.config.d_model)))
        macro_frames = self.macro_contract.macro_mel_frames
        if first_unit_end < macro_frames:
            raise ValueError("first macro AuT unit ends before one complete unit")
        last_unit_end = first_unit_end + (unit_count - 1) * macro_frames
        if last_unit_end > int(input_features.shape[1]):
            raise ValueError("macro AuT convolution would inspect future mel frames")
        window_frames = self.macro_contract.conv_window_mel_frames
        windows = []
        for unit_index in range(unit_count):
            end = first_unit_end + unit_index * macro_frames
            start = max(0, end - window_frames)
            windows.append(self._uniform_conv_window(input_features[:, start:end].contiguous()))
        return self._conv_macro_window_batch(windows)

    def _conv_embed_one_uniform(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run one 104-mel convolution per complete latency macro unit."""
        macro_frames = self.macro_contract.macro_mel_frames
        unit_count = int(input_features.shape[1]) // macro_frames
        return self._conv_embed_macro_units(
            input_features, first_unit_end=macro_frames, unit_count=unit_count
        )

    def _forward_uniform_one(self, input_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self._conv_embed_one_uniform(input_features)
        complete_tokens = int(hidden_states.shape[0]) // self.frames_per_unit * self.frames_per_unit
        hidden_states = hidden_states[:complete_tokens]
        if hidden_states.numel() == 0:
            return hidden_states.new_empty((0, int(self.audio_tower.config.output_dim)))
        for layer in self.audio_tower.layers:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                hidden_states = activation_checkpoint(
                    lambda values, current=layer: macro_encoder_layer_full(
                        current,
                        values,
                        frames_per_unit=self.frames_per_unit,
                        window_tokens=self.macro_contract.attention_window_tokens,
                    ),
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = macro_encoder_layer_full(
                    layer,
                    hidden_states,
                    frames_per_unit=self.frames_per_unit,
                    window_tokens=self.macro_contract.attention_window_tokens,
                )
        hidden_states = self.audio_tower.ln_post(hidden_states)
        hidden_states = self.audio_tower.proj1(hidden_states)
        hidden_states = self.audio_tower.act(hidden_states)
        return self.audio_tower.proj2(hidden_states)

    def _forward_one(self, input_features: torch.Tensor) -> torch.Tensor:
        return self._forward_uniform_one(input_features)

    @staticmethod
    def _replace_sft_layer_cache(
        state: CausalAutLayerState,
        *,
        past_key: torch.Tensor | None,
        past_value: torch.Tensor | None,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        max_past_tokens: int,
    ) -> None:
        """Publish a new detached cache without mutating tensors used by backward."""
        _claim_macro_cache(state)
        if max_past_tokens == 0:
            state.key = None
            state.value = None
            state.valid_tokens = 0
            state.cursor = 0
            return
        keys = current_key if past_key is None else torch.cat((past_key, current_key), dim=-2)
        values = (
            current_value if past_value is None else torch.cat((past_value, current_value), dim=-2)
        )
        kept = min(int(keys.shape[-2]), int(max_past_tokens))
        kept_key = keys[..., -kept:, :].detach()
        kept_value = values[..., -kept:, :].detach()
        padding = int(max_past_tokens) - kept
        state.key = F.pad(kept_key, (0, 0, 0, padding)).clone()
        state.value = F.pad(kept_value, (0, 0, 0, padding)).clone()
        state.valid_tokens = kept
        state.cursor = kept % int(max_past_tokens)

    def forward_sft_mel_chunk(
        self, input_features: torch.Tensor, state: MacroAuTStreamState
    ) -> torch.Tensor:
        """Train one macro-aligned mel chunk with bounded detached recurrence.

        The frontend retains exactly the latest 104 mel frames.  Each
        Transformer layer retains ``104-k`` K/V tokens, where ``k`` is the
        configured macro-unit size.  The retained tensors are detached because
        SFT commits an optimizer step after every XL chunk.
        """
        if input_features.ndim != 2:
            raise ValueError(
                f"expected SFT features [mel,frames], got {tuple(input_features.shape)}"
            )
        if not isinstance(state, MacroAuTStreamState):
            raise TypeError("SFT AuT requires MacroAuTStreamState")
        mel_frames = int(input_features.shape[1])
        if mel_frames < 1 or mel_frames % self.macro_contract.macro_mel_frames:
            raise ValueError(
                f"SFT AuT mel chunk must be divisible by {self.macro_contract.macro_mel_frames}, got {mel_frames}"
            )
        if state.mel_frames_since_emit:
            raise RuntimeError("SFT AuT state contains a partial streaming unit")
        if len(state.layer_states) != len(self.audio_tower.layers):
            raise ValueError("SFT AuT state layer count changed")
        cached_tokens = state.cached_tokens
        if any((layer.tokens != cached_tokens for layer in state.layer_states)):
            raise RuntimeError("SFT AuT layers have inconsistent KV lengths")
        if cached_tokens > self.max_past_tokens:
            raise RuntimeError("SFT AuT state exceeds its bounded KV window")
        history = state.mel_chunk
        if history is not None:
            if (
                history.ndim != 2
                or int(history.shape[0]) != int(input_features.shape[0])
                or history.device != input_features.device
                or (history.dtype != input_features.dtype)
                or (int(history.shape[1]) > self.macro_contract.conv_window_mel_frames)
            ):
                raise ValueError("SFT AuT mel history differs from the current chunk")
            combined_features = torch.cat((history, input_features), dim=1)
        else:
            combined_features = input_features
        macro_frames = self.macro_contract.macro_mel_frames
        current_units = mel_frames // macro_frames
        current_tokens = current_units * self.frames_per_unit
        history_frames = 0 if history is None else int(history.shape[1])

        def frontend_forward(features: torch.Tensor) -> torch.Tensor:
            return self._conv_embed_macro_units(
                features, first_unit_end=history_frames + macro_frames, unit_count=current_units
            )

        if self.gradient_checkpointing and torch.is_grad_enabled():
            hidden_states = activation_checkpoint(
                frontend_forward, combined_features, use_reentrant=False
            )
        else:
            hidden_states = frontend_forward(combined_features)
        if int(hidden_states.shape[0]) != current_tokens:
            raise RuntimeError("SFT AuT frontend emitted the wrong number of current tokens")
        cache_updates: list[
            tuple[
                CausalAutLayerState,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor,
                torch.Tensor,
            ]
        ] = []
        for layer, layer_state in zip(self.audio_tower.layers, state.layer_states, strict=True):
            _claim_macro_cache(layer_state)
            valid = int(layer_state.valid_tokens)
            if valid:
                if layer_state.key is None or layer_state.value is None:
                    raise RuntimeError("SFT AuT valid cache has no K/V tensors")
                past_key = layer_state.key[..., :valid, :]
                past_value = layer_state.value[..., :valid, :]
            else:
                past_key = None
                past_value = None

            def layer_forward(
                values: torch.Tensor,
                current: nn.Module = layer,
                key: torch.Tensor | None = past_key,
                value: torch.Tensor | None = past_value,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                return macro_encoder_layer_chunk(
                    current,
                    values,
                    past_key=key,
                    past_value=value,
                    frames_per_unit=self.frames_per_unit,
                    window_tokens=self.macro_contract.attention_window_tokens,
                )

            if self.gradient_checkpointing and torch.is_grad_enabled():
                hidden_states, current_key, current_value = activation_checkpoint(
                    layer_forward, hidden_states, use_reentrant=False
                )
            else:
                hidden_states, current_key, current_value = layer_forward(hidden_states)
            cache_updates.append((layer_state, past_key, past_value, current_key, current_value))
        hidden_states = self.audio_tower.ln_post(hidden_states)
        hidden_states = self.audio_tower.proj1(hidden_states)
        hidden_states = self.audio_tower.act(hidden_states)
        output = self.audio_tower.proj2(hidden_states)
        for layer_state, past_key, past_value, current_key, current_value in cache_updates:
            self._replace_sft_layer_cache(
                layer_state,
                past_key=past_key,
                past_value=past_value,
                current_key=current_key,
                current_value=current_value,
                max_past_tokens=self.max_past_tokens,
            )
        state.mel_chunk = (
            combined_features[:, -self.macro_contract.conv_window_mel_frames :].detach().clone()
        )
        state.max_mel_frames = max(state.max_mel_frames, int(state.mel_chunk.shape[1]))
        state.emitted_tokens += current_tokens
        state.max_kv_tokens = max(state.max_kv_tokens, state.cached_tokens)
        state.latest_output = output[-1:].detach().clone()
        return output

    def new_stream_state(self) -> MacroAuTStreamState:
        return MacroAuTStreamState(
            layer_states=[CausalAutLayerState() for _ in self.audio_tower.layers]
        )

    def _stream_transformer_macro(
        self, hidden_states: torch.Tensor, state: MacroAuTStreamState
    ) -> torch.Tensor:
        if int(hidden_states.shape[0]) != self.frames_per_unit:
            raise ValueError(
                f"macro AuT expected {self.frames_per_unit} tokens, got {hidden_states.shape[0]}"
            )
        if len(state.layer_states) != len(self.audio_tower.layers):
            raise ValueError("macro AuT stream layer count changed")
        cached = state.cached_tokens
        if any((layer.tokens != cached for layer in state.layer_states)):
            raise RuntimeError("macro AuT Transformer layers have inconsistent KV")
        if cached > self.max_past_tokens:
            raise RuntimeError("macro AuT cache exceeded its bounded past window")
        for layer, layer_state in zip(self.audio_tower.layers, state.layer_states, strict=True):
            hidden_states = macro_encoder_layer_step(
                layer, hidden_states, layer_state, max_past_tokens=self.max_past_tokens
            )
        state.max_kv_tokens = max(state.max_kv_tokens, state.cached_tokens)
        hidden_states = self.audio_tower.ln_post(hidden_states)
        hidden_states = self.audio_tower.proj1(hidden_states)
        hidden_states = self.audio_tower.act(hidden_states)
        return self.audio_tower.proj2(hidden_states)

    @torch.no_grad()
    def stream_uniform_mel(
        self, input_features: torch.Tensor, state: MacroAuTStreamState
    ) -> torch.Tensor:
        """Consume mel frames and release output only at complete-unit boundaries."""
        if input_features.ndim != 2:
            raise ValueError(
                f"expected new features [mel,frames], got {tuple(input_features.shape)}"
            )
        if not isinstance(state, MacroAuTStreamState):
            raise TypeError("MacroAuTEncoder requires MacroAuTStreamState")
        if state.mel_chunk is not None and (
            state.mel_chunk.device != input_features.device
            or state.mel_chunk.dtype != input_features.dtype
            or int(state.mel_chunk.shape[0]) != int(input_features.shape[0])
        ):
            raise ValueError("new mel frames do not match the macro AuT stream")
        emitted: list[torch.Tensor] = []
        cursor = 0
        total = int(input_features.shape[1])
        macro_frames = self.macro_contract.macro_mel_frames
        while cursor < total:
            needed = macro_frames - state.mel_frames_since_emit
            take = min(needed, total - cursor)
            self._append_uniform_mel(state, input_features[:, cursor : cursor + take])
            cursor += take
            state.mel_frames_since_emit += take
            if state.mel_frames_since_emit < macro_frames:
                continue
            if state.mel_chunk is None:
                raise RuntimeError("macro AuT convolution has no macro window")
            window = self._uniform_conv_window(state.mel_chunk)
            unit = self._conv_macro_window_batch([window])
            state.mel_frames_since_emit = 0
            output = self._stream_transformer_macro(unit, state)
            state.emitted_tokens += self.frames_per_unit
            state.latest_output = output[-1:].detach()
            emitted.append(output)
        if emitted:
            return torch.cat(emitted, dim=0)
        return input_features.new_empty((0, int(self.audio_tower.config.output_dim)))

    @torch.no_grad()
    def stream_uniform_mel_batch(
        self, input_features: list[torch.Tensor], states: list[MacroAuTStreamState]
    ) -> list[torch.Tensor]:
        if not input_features or len(input_features) != len(states):
            raise ValueError("macro AuT batch requires one feature tensor per state")
        return [
            self.stream_uniform_mel(features, state)
            for features, state in zip(input_features, states, strict=True)
        ]
