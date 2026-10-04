"""Compile the native grouped MoE so it can run inside a CUDA graph."""

import torch
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS, grouped_mm_experts_forward


def inference_experts():
    name = "d2_grouped_mm"
    if name not in ALL_EXPERTS_FUNCTIONS:
        ALL_EXPERTS_FUNCTIONS.register(
            name,
            torch.compile(
                grouped_mm_experts_forward,
                fullgraph=True,
                dynamic=False,
                options={
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "triton.cudagraphs": False,
                },
            ),
        )
    return name
