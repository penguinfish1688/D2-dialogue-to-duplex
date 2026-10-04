"""Audited Stage-3 freezing, LoRA installation, and optimizer groups."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn

from d2_llama.core.controls import (
    ControlTokenIds,
    TrainableControlRows,
    control_initialization_token_id,
    install_control_tokens,
)
from d2_llama.core.lora import LoRALinear, install_lora, lora_parameters
from d2_llama.core.constants import TTS_LORA_TARGET_SUFFIXES


@dataclass(frozen=True)
class Stage3Trainability:
    controls: ControlTokenIds
    audio_attention_lora_modules: tuple[str, ...]
    audio_non_attention_lora_modules: tuple[str, ...]
    thinker_lora_modules: tuple[str, ...]
    tts_lora_modules: tuple[str, ...]
    gradient_hooks: tuple[Any, ...]
    parameter_groups: tuple[dict[str, Any], ...]


def _parameters(module: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _promote_stage3_masters(model: nn.Module) -> tuple[Any, ...]:
    """Preserve native compute while giving every v9 trainable FP32 storage.

    AuT selects its input dtype from conv1.weight. Its released Whisper
    Conv1d casts weights to the input dtype, so preserve the pre-promotion
    compute dtype at that entry point. This hook is recreated at construction
    and changes neither the released module nor the Stage-1 checkpoint.
    """
    frontend = model.audio_tower.speech_encoder.conv1
    compute_dtype = getattr(frontend, "_d2_compute_dtype", frontend.weight.dtype)
    handles: tuple[Any, ...] = ()
    if compute_dtype != torch.float32:
        if (type(frontend).__module__, type(frontend).__name__) != ("whisper.model", "Conv1d"):
            raise TypeError(
                "mixed-precision Stage-3 frontend requires released Whisper Conv1d weight casting"
            )
        frontend._d2_compute_dtype = compute_dtype
        if not hasattr(frontend, "_d2_compute_dtype_hook"):

            def preserve_compute(module, arguments):
                if not arguments or not torch.is_tensor(arguments[0]):
                    raise TypeError("Whisper frontend requires a tensor positional input")
                return (arguments[0].to(dtype=module._d2_compute_dtype), *arguments[1:])

            frontend._d2_compute_dtype_hook = frontend.register_forward_pre_hook(preserve_compute)
        handles = (frontend._d2_compute_dtype_hook,)
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.detach().float()
    return handles


def _module_at(root: nn.Module, path: str) -> nn.Module:
    module = root
    for piece in path.split("."):
        module = getattr(module, piece)
    return module


def _installed_lora_parameters(root: nn.Module, paths: tuple[str, ...]) -> list[nn.Parameter]:
    result: list[nn.Parameter] = []
    for path in paths:
        module = _module_at(root, path)
        if not isinstance(module, LoRALinear):
            raise RuntimeError(f"{path} is not an installed LoRA module")
        module.lora_a.requires_grad_(True)
        module.lora_b.requires_grad_(True)
        result.extend((module.lora_a, module.lora_b))
    return result


def _install_or_reuse_lora(
    root: nn.Module,
    *,
    target_suffixes: tuple[str, ...],
    rank: int,
    alpha: float,
    dropout: float,
) -> tuple[str, ...]:
    """Install LoRA, or reuse the identical Stage-1 wrappers after resume."""

    existing = tuple(
        name
        for name, module in root.named_modules()
        if name and isinstance(module, LoRALinear) and name.endswith(target_suffixes)
    )
    dense = tuple(
        name
        for name, module in root.named_modules()
        if name and isinstance(module, nn.Linear) and name.endswith(target_suffixes)
    )
    if existing and dense:
        raise RuntimeError("LoRA target family is only partially installed")
    if not existing:
        return install_lora(
            root,
            target_suffixes=target_suffixes,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
    for path in existing:
        module = _module_at(root, path)
        if module.rank != rank or module.alpha != alpha or float(module.dropout.p) != dropout:
            raise RuntimeError(f"resumed LoRA contract changed at {path}")
    return existing


def configure_stage3_trainability(
    model: nn.Module,
    tokenizer: Any,
    config: Mapping[str, Any],
) -> Stage3Trainability:
    """Freeze released dense weights and enable only the documented deltas."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    official = model.official_model
    controls = install_control_tokens(tokenizer, official)
    model.control_rows = TrainableControlRows(
        controls=controls,
        base_embedding=official.get_input_embeddings(),
        base_head=official.get_output_embeddings(),
        initialization_token_id=control_initialization_token_id(tokenizer),
    )

    thinker_layers = model.thinker.layers
    thinker_cfg = config["model"]["thinker_lora"]
    thinker_names = install_lora(
        thinker_layers,
        target_suffixes=thinker_cfg["target_suffixes"],
        rank=int(thinker_cfg["rank"]),
        alpha=float(thinker_cfg["alpha"]),
        dropout=float(thinker_cfg["dropout"]),
    )
    tts_backbone = model.speech_generator.model.model.layers
    tts_cfg = config["model"]["tts_lora"]
    tts_names = install_lora(
        tts_backbone,
        target_suffixes=tts_cfg["target_suffixes"],
        rank=int(tts_cfg["rank"]),
        alpha=float(tts_cfg["alpha"]),
        dropout=float(tts_cfg["dropout"]),
    )

    audio_cfg = config["aut_optim"]
    audio_attention = install_lora(
        model.audio_tower.speech_encoder,
        target_suffixes=(
            "attn.query",
            "attn.key",
            "attn.value",
            "attn.out",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
        ),
        rank=int(audio_cfg["lora_rank"]),
        alpha=float(audio_cfg["lora_alpha"]),
        dropout=float(audio_cfg["lora_dropout"]),
    )
    audio_mlp = _install_or_reuse_lora(
        model.audio_tower.speech_encoder,
        target_suffixes=("mlp.0", "mlp.2", "fc1", "fc2"),
        rank=int(audio_cfg["lora_rank"]),
        alpha=float(audio_cfg["lora_alpha"]),
        dropout=float(audio_cfg["lora_dropout"]),
    )
    audio_adaptor = _install_or_reuse_lora(
        model.audio_tower.speech_projector,
        target_suffixes=("linear1", "linear2"),
        rank=int(audio_cfg["lora_rank"]),
        alpha=float(audio_cfg["lora_alpha"]),
        dropout=float(audio_cfg["lora_dropout"]),
    )
    for frontend in (
        model.audio_tower.speech_encoder.conv1,
        model.audio_tower.speech_encoder.conv2,
    ):
        for parameter in frontend.parameters():
            parameter.requires_grad_(True)

    expected_tts = {
        f"{index}.{suffix}"
        for index in range(len(tts_backbone))
        for suffix in TTS_LORA_TARGET_SUFFIXES
    }
    if set(tts_names) != expected_tts:
        raise RuntimeError(
            "Stage-3 TTS attention+MLP LoRA inventory differs from every released layer"
        )
    for bridge in (model.speech_generator.input_proj, model.speech_generator.gate):
        if any(parameter.requires_grad for parameter in bridge.parameters()):
            raise RuntimeError(
                "Stage-3 released TTS input projection and fusion gate must stay frozen"
            )
    tts_trainable_ids = {id(parameter) for parameter in lora_parameters(tts_backbone)}
    actual_tts_ids = {
        id(parameter)
        for parameter in model.speech_generator.parameters()
        if parameter.requires_grad
    }
    if actual_tts_ids != tts_trainable_ids:
        raise RuntimeError(
            "Stage-3 TTS may train only attention+MLP LoRA, not released dense weights"
        )

    hooks: tuple[Any, ...] = ()
    rates = config["optim"]["learning_rates"]
    groups = (
        {
            "name": "audio_frontend_full",
            "params": [
                *_parameters(model.audio_tower.speech_encoder.conv1),
                *_parameters(model.audio_tower.speech_encoder.conv2),
            ],
            "lr": float(rates["audio_frontend_full"]),
        },
        {
            "name": "audio_attention_lora",
            "params": _installed_lora_parameters(model.audio_tower.speech_encoder, audio_attention),
            "lr": float(rates["audio_attention_lora"]),
        },
        {
            "name": "audio_non_attention_lora",
            "params": [
                *_installed_lora_parameters(model.audio_tower.speech_encoder, audio_mlp),
                *_installed_lora_parameters(model.audio_tower.speech_projector, audio_adaptor),
            ],
            "lr": float(rates["audio_non_attention_lora"]),
        },
        {
            "name": "thinker_lora",
            "params": list(lora_parameters(thinker_layers)),
            "lr": float(rates["thinker_lora"]),
        },
        {
            "name": "tts_lora",
            "params": list(lora_parameters(tts_backbone)),
            "lr": float(rates["tts_lora"]),
        },
        {
            "name": "control_rows",
            "params": list(model.control_rows.parameters()),
            "lr": float(rates["control_rows"]),
        },
    )
    streams = getattr(model, "stream_type_embedding", None)
    if streams is not None:
        streams.weight.requires_grad_(True)
        groups += (
            {
                "name": "stream_type_embedding",
                "params": list(streams.parameters()),
                "lr": float(rates["control_rows"]),
            },
        )
    if getattr(model, "thinker_alignment", False):
        hooks += _promote_stage3_masters(model)
    for group in groups:
        if not group["params"]:
            raise RuntimeError(f"empty Stage-3 parameter group {group['name']}")
    grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
    trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != trainable_ids:
        raise RuntimeError(
            "Stage-3 optimizer groups must cover each trainable parameter exactly once"
        )
    return Stage3Trainability(
        controls=controls,
        audio_attention_lora_modules=audio_attention,
        audio_non_attention_lora_modules=audio_mlp + audio_adaptor,
        thinker_lora_modules=thinker_names,
        tts_lora_modules=tts_names,
        gradient_hooks=hooks,
        parameter_groups=groups,
    )


def build_stage3_optimizer(
    trainability: Stage3Trainability,
    config: Mapping[str, Any],
) -> torch.optim.AdamW:
    optim = config["optim"]
    return torch.optim.AdamW(
        list(trainability.parameter_groups),
        betas=tuple(float(value) for value in optim["betas"]),
        eps=float(optim["eps"]),
        weight_decay=float(optim["weight_decay"]),
    )


def build_stage3_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup, then cosine decay to the configured LR ratio.

    The lambda is indexed by the number of *completed* updates.  PyTorch
    evaluates index zero at construction, so ``+1`` makes optimizer update 1
    use exactly ``1 / warmup_updates`` of each parameter group's base LR.
    """

    optim = config["optim"]
    warmup = int(optim["warmup_updates"])
    cosine_end = int(optim["cosine_end_update"])
    minimum = float(optim["minimum_lr_ratio"])
    if warmup < 1 or cosine_end <= warmup or not 0.0 <= minimum <= 1.0:
        raise ValueError("invalid Stage-3 warmup/cosine schedule")

    def factor(scheduler_index: int) -> float:
        update = int(scheduler_index) + 1
        if update <= warmup:
            return update / warmup
        if update >= cosine_end:
            return minimum
        progress = (update - warmup) / (cosine_end - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum + (1.0 - minimum) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


__all__ = [
    "Stage3Trainability",
    "build_stage3_optimizer",
    "build_stage3_scheduler",
    "configure_stage3_trainability",
]
