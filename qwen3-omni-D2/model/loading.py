"""Load the native backbone and the current D2 release weights."""

import torch
from safetensors.torch import load_file

from d2.hub import asset, load_release
from d2.weights import freeze_for_inference
from .audio_encoder import AudioEncoder
from .duplex import D2Qwen3OmniModel


def load_model(source, *, device="cuda", revision=None, offline=False, training=False):
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    root, config = load_release(source, family="qwen", revision=revision, offline=offline)
    encoder_file = root / "encoder.safetensors"
    if not encoder_file.is_file():
        raise FileNotFoundError(f"Release is missing {encoder_file}")
    base = asset(config, "base_model", offline=offline)
    processor = Qwen3OmniMoeProcessor.from_pretrained(base, local_files_only=True)
    if training:
        experts = "eager"
    else:
        from d2_qwen.inference.moe import inference_experts

        experts = inference_experts()
    qwen = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        base,
        dtype=torch.bfloat16,
        device_map=None if training else {"": device},
        attn_implementation="sdpa",
        experts_implementation=experts,
        local_files_only=True,
    )
    # D2 uses the audio pathway. Removing the unused visual tower reduces VRAM.
    if hasattr(qwen.thinker, "visual"):
        del qwen.thinker.visual
    encoder = AudioEncoder(
        qwen.thinker.audio_tower,
        processor.feature_extractor,
        latency_ms=config["latency_ms"],
        encoder_state=load_file(str(encoder_file)),
    )
    qwen.thinker.audio_tower = encoder.encoder
    model = D2Qwen3OmniModel(qwen, processor, encoder, latency_ms=config["latency_ms"])
    model.load_checkpoint_state_dict(load_file(str(root / "d2.safetensors")))
    model.to(device=device)
    if training:
        model.train()
    else:
        freeze_for_inference(model)
    return model, config
