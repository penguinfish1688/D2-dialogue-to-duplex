from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Iterable
import torch
from torch import nn


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_suffixes: tuple[str, ...] = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    )
    expected_wrapped_module_count: int | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "LoRAConfig":
        value = value or {}
        return cls(
            rank=int(value.get("rank", 16)),
            alpha=float(value.get("alpha", 32.0)),
            dropout=float(value.get("dropout", 0.05)),
            target_suffixes=tuple(value.get("target_suffixes") or cls.target_suffixes),
            expected_wrapped_module_count=(
                int(value["expected_wrapped_module_count"])
                if value.get("expected_wrapped_module_count") is not None
                else None
            ),
        )


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
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
        self.fresh_lora_a: nn.Linear | None = None
        self.fresh_lora_b: nn.Linear | None = None
        self.fresh_scaling = 0.0
        self.fresh_dropout = nn.Dropout(0.0)

    def enable_fresh_adapter(self, *, rank: int, alpha: float, dropout: float) -> None:
        """Add a zero-function trainable branch above frozen carried LoRA."""
        if self.fresh_lora_a is not None or self.fresh_lora_b is not None:
            raise RuntimeError("Fresh LoRA adapter is already enabled")
        if int(rank) < 1:
            raise ValueError(f"Fresh LoRA rank must be positive, got {rank}")
        rank = int(rank)
        device = self.base.weight.device
        self.fresh_lora_a = nn.Linear(
            self.base.in_features,
            rank,
            bias=False,
            dtype=torch.float32,
            device=device,
        )
        self.fresh_lora_b = nn.Linear(
            rank,
            self.base.out_features,
            bias=False,
            dtype=torch.float32,
            device=device,
        )
        nn.init.kaiming_uniform_(self.fresh_lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fresh_lora_b.weight)
        self.fresh_scaling = float(alpha) / float(rank)
        self.fresh_dropout = nn.Dropout(float(dropout))

    @torch.no_grad()
    def merge_frozen_into_base(self) -> None:
        if self._merged_into_base:
            return
        if any(parameter.requires_grad for parameter in self.lora_a.parameters()) or any(
            parameter.requires_grad for parameter in self.lora_b.parameters()
        ):
            raise RuntimeError("LoRA parameters must be frozen before merging into the base")
        delta = torch.matmul(self.lora_b.weight.float(), self.lora_a.weight.float()) * self.scaling
        self.base.weight.add_(
            delta.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        )
        self._merged_into_base = True

    @torch.no_grad()
    def disable_zero_adapter(self) -> None:
        if bool(torch.count_nonzero(self.lora_b.weight.detach()).item()):
            raise RuntimeError("Cannot disable a nonzero LoRA adapter without merging it")
        self._merged_into_base = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        out = base_out
        if not self._merged_into_base:
            lora_in = self.dropout(x).to(dtype=self.lora_a.weight.dtype)
            lora_out = self.lora_b(self.lora_a(lora_in)) * self.scaling
            out = out + lora_out.to(dtype=base_out.dtype)
        if self.fresh_lora_a is not None and self.fresh_lora_b is not None:
            fresh_in = self.fresh_dropout(x).to(dtype=self.fresh_lora_a.weight.dtype)
            fresh_out = self.fresh_lora_b(self.fresh_lora_a(fresh_in)) * self.fresh_scaling
            out = out + fresh_out.to(dtype=base_out.dtype)
        return out


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def _get_child_module(root: nn.Module, path: str) -> nn.Module:
    module = root
    if path:
        for part in path.split("."):
            module = getattr(module, part)
    return module


def _set_child_module(root: nn.Module, path: str, child: nn.Module) -> None:
    parent_path, _, child_name = path.rpartition(".")
    parent = _get_child_module(root, parent_path)
    setattr(parent, child_name, child)


def apply_lora_to_suffixes(
    module: nn.Module,
    config: LoRAConfig,
    *,
    include_prefixes: Iterable[str] | None = None,
) -> dict[str, Any]:
    prefixes = tuple(include_prefixes or ())
    replacements: list[tuple[str, nn.Linear]] = []
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        if prefixes and not any(name.startswith(prefix) for prefix in prefixes):
            continue
        if any(name.endswith(suffix) for suffix in config.target_suffixes):
            replacements.append((name, child))
    if not replacements:
        raise RuntimeError(
            "No Linear modules matched LoRA targets "
            f"prefixes={prefixes} suffixes={config.target_suffixes}"
        )
    if (
        config.expected_wrapped_module_count is not None
        and len(replacements) != config.expected_wrapped_module_count
    ):
        raise RuntimeError(
            "LoRA target count mismatch: "
            f"matched={len(replacements)} expected={config.expected_wrapped_module_count} "
            f"prefixes={prefixes} suffixes={config.target_suffixes}"
        )
    wrapped = []
    for name, child in replacements:
        _set_child_module(
            module,
            name,
            LoRALinear(child, rank=config.rank, alpha=config.alpha, dropout=config.dropout),
        )
        wrapped.append(name)
    return {
        "rank": config.rank,
        "alpha": config.alpha,
        "dropout": config.dropout,
        "target_suffixes": list(config.target_suffixes),
        "expected_wrapped_module_count": config.expected_wrapped_module_count,
        "wrapped_modules": wrapped,
        "wrapped_module_count": len(wrapped),
    }
