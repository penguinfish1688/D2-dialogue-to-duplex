"""Fine-tune the current D2 losses and trainable parameter groups."""

import json
from pathlib import Path
import torch
from d2.prompts import task_system_prompt_token_ids
from d2.training import run, scheduler
from d2_qwen.model.loading import load_model


def configure_optimizer(model):
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/train.json").read_text())
    rates = config["learning_rates"]
    groups = [
        dict(name=name, params=parameters, lr=rates[name])
        for name, parameters in model.optimizer_parameter_groups().items()
    ]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    ratios = [
        config["minimum_learning_rates"].get(g["name"], g["lr"] * 0.1) / g["lr"] for g in groups
    ]
    return optimizer, scheduler(optimizer, minimum_ratios=ratios)


def forward(model, tensors, row):
    required = {
        "env_mel",
        "self_mel",
        "text_target",
        "codec_target",
        "frame_mask",
        "assistant_mask",
    }
    allowed = required | {"interrupt_end_weight_class", "speaker_ids"}
    if not required <= set(tensors) or set(tensors) - allowed:
        raise ValueError(
            f"Qwen sample fields must include {sorted(required)} and only {sorted(allowed)}"
        )
    frames = tensors["text_target"].shape[1]
    if frames % model.frames_per_unit or frames > model.d2_config.xl_chunk_frames:
        raise ValueError("Sample must be one complete, macro-aligned XL chunk")
    prompt = task_system_prompt_token_ids(model.processor.tokenizer)[
        row.get("task", "conversation")
    ]
    ids = (
        torch.tensor(prompt, device=tensors["text_target"].device)
        .unsqueeze(0)
        .expand(tensors["text_target"].shape[0], -1)
    )
    result, _ = model(
        **tensors,
        audio_start_frame=0,
        memory_frames=model.d2_config.xl_chunk_frames,
        system_prompt_input_ids=ids,
    )
    return result["loss"], dict(text_loss=result["loss_text"], codec_loss=result["loss_codec"])


def train(args):
    run(
        args,
        family="qwen",
        load_model=load_model,
        configure_optimizer=configure_optimizer,
        forward=forward,
    )
