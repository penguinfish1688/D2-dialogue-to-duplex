from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F

UNIFORM_CONV_WINDOW_FRAMES = 104


@dataclass
class CausalAutLayerState:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    cursor: int = 0
    valid_tokens: int = 0
    cache_backend: str | None = None
    cache_token_axis: int = -2
    backend_state: Any = field(default=None, repr=False)

    @property
    def tokens(self) -> int:
        return self.valid_tokens

    @property
    def allocated_tokens(self) -> int:
        if self.key is None:
            return 0
        return int(self.key.shape[self.cache_token_axis])


@dataclass
class CausalAutStreamState:
    layer_states: list[CausalAutLayerState] = field(default_factory=list)
    mel_chunk: torch.Tensor | None = None
    next_token_pos: int = 0
    mel_frames_since_emit: int = 0
    emitted_tokens: int = 0
    latest_output: torch.Tensor | None = None
    max_mel_frames: int = 0
    max_kv_tokens: int = 0

    @property
    def mel_chunk_frames(self) -> int:
        return 0 if self.mel_chunk is None else int(self.mel_chunk.shape[1])

    @property
    def cached_tokens(self) -> int:
        return max((layer.tokens for layer in self.layer_states), default=0)

    @property
    def allocated_cache_tokens(self) -> int:
        return max(
            (layer.allocated_tokens for layer in self.layer_states),
            default=0,
        )

    @property
    def latest_token_index(self) -> int | None:
        return self.emitted_tokens - 1 if self.emitted_tokens else None


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout))
        device = base.weight.device
        self.lora_a = nn.Linear(
            base.in_features, self.rank, bias=False, dtype=torch.float32, device=device
        )
        self.lora_b = nn.Linear(
            self.rank, base.out_features, bias=False, dtype=torch.float32, device=device
        )
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for param in self.base.parameters():
            param.requires_grad_(False)
        self._merged_into_base = False
        self._adapter_enabled = True

    @torch.no_grad()
    def merge_frozen_into_base(self) -> None:
        if self._merged_into_base:
            return
        if any(parameter.requires_grad for parameter in self.lora_a.parameters()) or any(
            parameter.requires_grad for parameter in self.lora_b.parameters()
        ):
            raise RuntimeError("LoRA parameters must be frozen before inference merging")
        delta = (
            torch.matmul(
                self.lora_b.weight.float(),
                self.lora_a.weight.float(),
            )
            * self.scaling
        )
        self.base.weight.add_(
            delta.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        )
        self._merged_into_base = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if self._merged_into_base or not self._adapter_enabled:
            return base_out
        lora_in = self.dropout(x).to(dtype=self.lora_a.weight.dtype)
        lora_out = self.lora_b(self.lora_a(lora_in)) * self.scaling
        return base_out + lora_out.to(dtype=base_out.dtype)


class Encoder(nn.Module):
    """Causal/bi-conv Qwen3 AuT audio tower with the official audio-tower API."""

    @property
    def dtype(self) -> torch.dtype:
        dtype = getattr(self.audio_tower, "dtype", None)
        if dtype is not None:
            return dtype
        return next(self.audio_tower.parameters()).dtype

    @staticmethod
    def _get_child_module(root: nn.Module, path: str) -> nn.Module:
        module = root
        if path:
            for part in path.split("."):
                module = getattr(module, part)
        return module

    @staticmethod
    def _set_child_module(root: nn.Module, path: str, child: nn.Module) -> None:
        parent_path, _, child_name = path.rpartition(".")
        parent = Encoder._get_child_module(root, parent_path)
        setattr(parent, child_name, child)

    def _unfreeze_causal_frontend(self) -> None:
        for name in ("conv2d1", "conv2d2", "conv2d3", "conv_out"):
            module = getattr(self.audio_tower, name)
            for param in module.parameters():
                param.requires_grad_(True)

    def _apply_lora(
        self, *, rank: int, alpha: float, dropout: float, target_suffixes: tuple[str, ...]
    ) -> dict[str, Any]:
        replacements: list[tuple[str, nn.Linear]] = []
        for name, module in self.audio_tower.named_modules():
            if isinstance(module, nn.Linear) and any(
                (name.endswith(suffix) for suffix in target_suffixes)
            ):
                replacements.append((name, module))
        if not replacements:
            raise RuntimeError(
                f"No Qwen AuT Linear modules matched LoRA targets: {list(target_suffixes)}"
            )
        wrapped: list[str] = []
        for name, module in replacements:
            self._set_child_module(
                self.audio_tower, name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
            )
            wrapped.append(name)
        return {
            "rank": int(rank),
            "alpha": float(alpha),
            "dropout": float(dropout),
            "wrapped_modules": wrapped,
        }

    @staticmethod
    def _append_uniform_mel(state: CausalAutStreamState, features: torch.Tensor) -> None:
        if state.mel_chunk is None:
            state.mel_chunk = features.new_empty((int(features.shape[0]), 0))
        state.mel_chunk = torch.cat((state.mel_chunk, features), dim=1)
        if state.mel_chunk_frames > UNIFORM_CONV_WINDOW_FRAMES:
            state.mel_chunk = state.mel_chunk[:, -UNIFORM_CONV_WINDOW_FRAMES:]
        state.mel_chunk = state.mel_chunk.contiguous()
        state.max_mel_frames = max(state.max_mel_frames, state.mel_chunk_frames)

    @staticmethod
    def _uniform_conv_window(features: torch.Tensor) -> torch.Tensor:
        frames = int(features.shape[1])
        if frames < 1 or frames > UNIFORM_CONV_WINDOW_FRAMES:
            raise ValueError(
                f"Uniform AuT window has {frames} mel frames; expected 1..{UNIFORM_CONV_WINDOW_FRAMES}"
            )
        return F.pad(features, (UNIFORM_CONV_WINDOW_FRAMES - frames, 0)).contiguous()
