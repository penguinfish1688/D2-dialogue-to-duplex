from __future__ import annotations
import torch
from d2_llama.core.lora import LoRALinear


@torch.no_grad()
def merge_lora_for_inference(root):
    """Accumulate deltas in FP32, then round once to the base weight dtype.

    This mutates only a loaded inference model, never a checkpoint. It is not
    numerically identical to adding a separately computed low-rank activation.
    """
    if root.training or any(p.requires_grad for p in root.parameters()):
        raise ValueError("LoRA merging requires a frozen eval model")
    count = 0
    for name, child in list(root.named_children()):
        if isinstance(child, LoRALinear):
            with torch.autocast(device_type=child.base.weight.device.type, enabled=False):
                weight = (
                    child.base.weight.float()
                    + (child.lora_b.float() @ child.lora_a.float()) * child.scaling
                )
            if not torch.isfinite(weight).all():
                raise ValueError(f"non-finite merged LoRA weight: {name}")
            child.base.weight.copy_(weight.to(child.base.weight.dtype))
            setattr(root, name, child.base)
            count += 1
        else:
            count += merge_lora_for_inference(child)
    return count
