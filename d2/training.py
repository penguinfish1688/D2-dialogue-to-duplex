"""A small single-GPU fine-tuning loop for prepared native-clock samples."""

import json
import math
from pathlib import Path
import random
import shutil
import tempfile
import os

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .hub import load_release


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_samples(path, *, family, latency_ms):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if (
        manifest.get("format") != "d2.samples.v1"
        or manifest.get("family") != family
        or manifest.get("latency_ms") != latency_ms
    ):
        raise ValueError("Sample family and interaction granularity must match the checkpoint")
    rows = manifest.get("samples", [])
    if not rows:
        raise ValueError("The sample manifest is empty")
    for row in rows:
        if row.get("task", "conversation") not in ("conversation", "avqa"):
            raise ValueError("Training task must be conversation or avqa")
        if not (path.parent / row["tensors"]).is_file():
            raise FileNotFoundError(path.parent / row["tensors"])
    return rows


def run(args, *, family, load_model, configure_optimizer, forward):
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("Fine-tuning requires an NVIDIA CUDA GPU")
    device = torch.device(args.device)
    torch.cuda.set_device(device.index if device.index is not None else 0)
    root, config = load_release(
        args.model, family=family, revision=args.revision, offline=args.offline
    )
    rows = read_samples(args.data, family=family, latency_ms=config["latency_ms"])
    seed_everything(args.seed)
    model, _ = load_model(
        args.model, device=device, revision=args.revision, offline=args.offline, training=True
    )
    optimizer, scheduler = configure_optimizer(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    metrics = []
    for step in range(1, args.steps + 1):
        row = rows[(step - 1) % len(rows)]
        tensors = load_file(str(args.data.parent / row["tensors"]), device=str(device))
        if any(t.is_floating_point() and not torch.isfinite(t).all() for t in tensors.values()):
            raise ValueError("Sample contains non-finite tensors")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = forward(model, tensors, row)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss before update {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        if not any(p.grad is not None and torch.count_nonzero(p.grad).item() for p in parameters):
            raise RuntimeError("Backward produced no nonzero trainable gradients")
        optimizer.step()
        scheduler.step()
        metric = dict(
            update=step,
            loss=float(loss.detach()),
            grad_norm=float(norm),
            **{name: float(value.detach()) for name, value in parts.items()},
        )
        metrics.append(metric)
        print(json.dumps(metric), flush=True)
    # Publish only a complete portable inference/fine-tuning bundle.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        target = Path(temp) / "release"
        target.mkdir()
        save_file(
            {
                name: p.detach().cpu().contiguous()
                for name, p in model.named_parameters()
                if p.requires_grad
            },
            str(target / "d2.safetensors"),
        )
        try:
            os.link(root / "encoder.safetensors", target / "encoder.safetensors")
        except OSError:
            shutil.copyfile(root / "encoder.safetensors", target / "encoder.safetensors")
        (target / "d2.json").write_text(json.dumps(config, indent=2))
        (target / "training.json").write_text(
            json.dumps(dict(seed=args.seed, updates=args.steps, metrics=metrics), indent=2)
        )
        target.rename(args.output)
    print(f"Saved release: {args.output}", flush=True)


def scheduler(optimizer, *, warmup=200, end=2000, minimum_ratios=None):
    ratios = minimum_ratios or [0.1] * len(optimizer.param_groups)

    def schedule(ratio):
        def factor(index):
            update = index + 1
            if update <= warmup:
                return update / warmup
            progress = min(1.0, (update - warmup) / (end - warmup))
            return ratio + (1 - ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        return factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, [schedule(ratio) for ratio in ratios])
