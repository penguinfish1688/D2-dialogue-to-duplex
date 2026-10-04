"""Strict tensor-only release weights."""

from __future__ import annotations

import torch


@torch.no_grad()
def restore_parameters(model, state, *, expected=None):
    parameters = dict(model.named_parameters())
    expected = set(parameters) if expected is None else set(expected)
    if set(state) != expected:
        raise ValueError(
            f"Weight inventory differs: missing={sorted(expected - set(state))[:8]}, "
            f"unexpected={sorted(set(state) - expected)[:8]}"
        )
    for name, tensor in state.items():
        parameter = parameters[name]
        if tensor.shape != parameter.shape or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid checkpoint tensor: {name}")
        parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


@torch.no_grad()
def freeze_for_inference(model):
    model.eval().requires_grad_(False)
    model.to(dtype=torch.bfloat16)
    for module in model.modules():
        if callable(getattr(module, "merge_frozen_into_base", None)):
            module.merge_frozen_into_base()
            if getattr(module, "fresh_lora_a", None) is not None:
                delta = module.fresh_lora_b.weight.float() @ module.fresh_lora_a.weight.float()
                module.base.weight.add_((delta * module.fresh_scaling).to(module.base.weight.dtype))
                module.fresh_lora_a = module.fresh_lora_b = None
    return model
