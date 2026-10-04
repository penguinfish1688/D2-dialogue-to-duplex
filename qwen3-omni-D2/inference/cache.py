from __future__ import annotations
from typing import Any
import torch
from transformers.cache_utils import Cache, StaticLayer


class SplitDtypeStaticLayer(StaticLayer):
    """Static KV storage for attention modules with different key/value dtypes."""

    def lazy_initialization(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        self.dtype = key_states.dtype
        self.value_dtype = value_states.dtype
        self.device = key_states.device
        self.batch_size, self.num_heads = key_states.shape[:2]
        self.k_head_dim = key_states.shape[-1]
        self.v_head_dim = value_states.shape[-1]
        self.keys = torch.zeros(
            (self.batch_size, self.num_heads, self.max_cache_len, self.k_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.batch_size, self.num_heads, self.max_cache_len, self.v_head_dim),
            dtype=self.value_dtype,
            device=self.device,
        )
        self.cumulative_length = self.cumulative_length.to(self.device)
        if not torch.compiler.is_compiling():
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)
            torch._dynamo.mark_static_address(self.cumulative_length)
        self.is_initialized = True


def split_dtype_static_cache(*, layers: int, max_cache_len: int) -> Cache:
    if layers < 1:
        raise ValueError("Static cache must contain at least one layer")
    if max_cache_len < 1:
        raise ValueError("Static cache length must be positive")
    return Cache(layers=[SplitDtypeStaticLayer(max_cache_len) for _ in range(layers)])


def compact_transformer_cache(
    cache: Any,
    *,
    target_tokens: int,
    preserved_prefix_tokens: int = 0,
) -> Any:
    target_tokens = int(target_tokens)
    preserved_prefix_tokens = int(preserved_prefix_tokens)
    if target_tokens < 0:
        raise ValueError("target_tokens must be nonnegative")
    if not 0 <= preserved_prefix_tokens <= target_tokens:
        raise ValueError("preserved_prefix_tokens must be within the compacted cache")
    layers = getattr(cache, "layers", None)
    get_seq_length = getattr(cache, "get_seq_length", None)
    if layers is None or not callable(get_seq_length):
        raise TypeError("unsupported Transformer cache")
    source_tokens = int(get_seq_length())
    if source_tokens < target_tokens:
        raise ValueError(f"cannot expand Transformer cache from {source_tokens} to {target_tokens}")
    if source_tokens < preserved_prefix_tokens:
        raise ValueError(
            f"cache length {source_tokens} is shorter than its preserved prefix "
            f"{preserved_prefix_tokens}"
        )
    temporal_target = target_tokens - preserved_prefix_tokens
    for layer_index, layer in enumerate(layers):
        if not getattr(layer, "is_initialized", False):
            continue
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if not torch.is_tensor(keys) or not torch.is_tensor(values):
            raise TypeError(f"cache layer {layer_index} has no tensor KV storage")
        if int(keys.shape[-2]) < source_tokens or int(values.shape[-2]) < source_tokens:
            raise ValueError(
                f"cache layer {layer_index} storage is shorter than its logical length"
            )

        def compact_tensor(tensor: torch.Tensor) -> torch.Tensor:
            prefix = tensor[..., :preserved_prefix_tokens, :].detach().clone()
            temporal = tensor[..., preserved_prefix_tokens:source_tokens, :]
            kept_tokens = min(int(temporal.shape[-2]), temporal_target)
            suffix = (
                temporal[..., -kept_tokens:, :].detach().clone()
                if kept_tokens
                else temporal[..., :0, :].detach().clone()
            )
            padding_shape = list(tensor.shape)
            padding_shape[-2] = temporal_target - kept_tokens
            padding = tensor.new_zeros(padding_shape)
            return torch.cat((prefix, padding, suffix), dim=-2)

        compacted_keys = compact_tensor(keys)
        compacted_values = compact_tensor(values)
        cumulative_length = getattr(layer, "cumulative_length", None)
        if torch.is_tensor(cumulative_length):
            keys.zero_()
            values.zero_()
            keys[..., :target_tokens, :].copy_(compacted_keys)
            values[..., :target_tokens, :].copy_(compacted_values)
            cumulative_length.fill_(target_tokens)
        else:
            layer.keys = compacted_keys
            layer.values = compacted_values
            if isinstance(cumulative_length, int):
                layer.cumulative_length = target_tokens
            if hasattr(layer, "cumulative_length_int"):
                layer.cumulative_length_int = target_tokens
    if int(get_seq_length()) != target_tokens:
        raise RuntimeError(
            f"compacted Transformer cache reports {int(get_seq_length())} "
            f"tokens instead of {target_tokens}"
        )
    return cache
