"""Restore the native backbone, distilled encoder, and D2 adapters."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from d2.hub import load_release
from d2.weights import restore_parameters
from d2_llama.prompts import configure_task_prompts
from .assets import resolve_assets
from .encoder import BlockCausalWhisper, configure_aut_trainability
from .official import load_official_bundle
from .precision import promote_aut_masters
from .trainability import configure_stage3_trainability
from .xl import DynamicD2Model


def training_config():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/train.json").read_text())


def load_model(source, *, device="cuda", revision=None, offline=False, training=False):
    root, config = load_release(source, family="llama", revision=revision, offline=offline)
    assets = resolve_assets(config, offline=offline, renderer=False)
    tokenizer, official = load_official_bundle(**assets, dtype=torch.bfloat16)
    recipe = training_config()
    tower = BlockCausalWhisper.from_official_model(official, latency_ms=config["latency_ms"])
    configure_aut_trainability(tower, recipe["aut_optim"])
    promote_aut_masters(tower)
    state = load_file(str(root / "encoder.safetensors"))
    # The encoder contains learned dense attention as well as LoRA tensors.
    # Retain each saved tensor's precision before installing SFT adapters.
    tower.load_state_dict(state, strict=True, assign=True)
    official.get_model().speech_encoder = None
    official.get_model().speech_projector = None
    model = DynamicD2Model(official, tower, thinker_alignment=True)
    trainability = configure_stage3_trainability(model, tokenizer, recipe)
    configure_task_prompts(model, tokenizer)
    restore_parameters(
        model,
        load_file(str(root / "d2.safetensors")),
        expected={n for n, p in model.named_parameters() if p.requires_grad},
    )
    model.to(device=device)
    model.tokenizer, model.trainability = tokenizer, trainability
    if training:
        model.train()
    else:
        from d2_llama.inference.acceleration import merge_lora_for_inference

        model.eval().requires_grad_(False)
        merge_lora_for_inference(model)
        # Whisper casts these weights to BF16 on every forward. Store that
        # same cast once for inference; keep normalization precision intact.
        for module in model.audio_tower.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
                module.to(dtype=torch.bfloat16)
    return model, config
