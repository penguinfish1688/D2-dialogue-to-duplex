"""Small self-contained LoRA implementation used by Stage 1 and Stage 3."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("invalid LoRA rank, alpha, or dropout")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        base = self.base(values)
        # Stage-3 v9 keeps FP32 trainable masters while retaining native
        # activation/compute precision. Casts remain differentiable: AdamW
        # updates the FP32 leaves, not a permanently rounded BF16 parameter.
        dropped = self.dropout(values)
        update = F.linear(
            F.linear(dropped, self.lora_a.to(dtype=dropped.dtype)),
            self.lora_b.to(dtype=dropped.dtype),
        )
        return base + update * self.scaling


def _parent_and_name(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    pieces = path.split(".")
    owner = root
    for piece in pieces[:-1]:
        owner = getattr(owner, piece)
    return owner, pieces[-1]


def install_lora(
    root: nn.Module,
    *,
    target_suffixes: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> tuple[str, ...]:
    """Wrap matching Linear modules and return their stable full names."""

    suffixes = tuple(target_suffixes)
    if not suffixes:
        raise ValueError("target_suffixes cannot be empty")
    matches = [
        name
        for name, module in root.named_modules()
        if name and isinstance(module, nn.Linear) and name.endswith(suffixes)
    ]
    if not matches:
        raise ValueError("LoRA target suffixes matched no Linear modules")
    for path in matches:
        owner, name = _parent_and_name(root, path)
        base = getattr(owner, name)
        if not isinstance(base, nn.Linear):
            raise RuntimeError(f"LoRA target {path} changed during installation")
        setattr(
            owner,
            name,
            LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout),
        )
    return tuple(matches)


def lora_parameters(root: nn.Module) -> Iterable[nn.Parameter]:
    for module in root.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_a
            yield module.lora_b


__all__ = ["LoRALinear", "install_lora", "lora_parameters"]
