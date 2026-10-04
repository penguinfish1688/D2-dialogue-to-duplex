"""Pinned evaluation protocol and small file helpers."""

import hashlib
import json
from pathlib import Path

TASKS = (
    "alpacaeval",
    "commoneval",
    "wildvoice",
    "sd-qa",
    "mmsu",
    "openbookqa",
    "bbh",
    "ifeval",
    "advbench",
)
DATASET_REVISION = "b02edcef1330480be3a11bd6f7434ac32f05ad08"
VOICEBENCH_COMMIT = "3c3b0d3a7a956f745305eb348f5e03ce7ec73dad"
DEFAULT_SAMPLE_LIMIT = 200
ROLLOUT_FRAMES = 250
VOICEBENCH_MANIFEST_CONTRACT = "d2_voicebench_fixed_subset_manifest_v1"
FDB_CATEGORIES = {
    "candor_turn_taking": 119,
    "synthetic_user_interruption": 200,
    "candor_pause_handling": 216,
}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def response_text(summary, start_frame, tokenizer):
    """Use only the response beginning after the last audible user frame."""
    started = False
    tokens = []
    for event in summary["text_trace"]:
        kind = event["kind"]
        if not started:
            started = kind == "response" and event["frame"] >= start_frame
        elif kind == "interrupt":
            break
        elif kind == "text":
            tokens.append(event["token_id"])
    return tokenizer.decode(tokens, skip_special_tokens=True) if tokens else ""


def read_benchmark_audio(path):
    """Keep the original benchmark's linear resampler and PCM rounding."""
    import torch
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = torch.from_numpy(audio.mean(axis=1).copy())
    if not len(wave) or not torch.isfinite(wave).all():
        raise ValueError(f"Invalid audio: {path}")
    if rate != 16000:
        wave = torch.nn.functional.interpolate(
            wave[None, None],
            size=max(1, round(len(wave) * 16000 / rate)),
            mode="linear",
            align_corners=False,
        ).flatten()
    pcm = torch.round(wave.clamp(-1, 1) * 32767).to(torch.int16).numpy().astype("<i2").tobytes()
    return pcm, wave


def last_audible_frame(wave):
    """Last 80 ms frame in a >=20 ms run above -60 dBFS, before PCM rounding."""
    import torch

    blocks = torch.nn.functional.pad(wave, (0, -len(wave) % 160)).view(-1, 160)
    active = (blocks.double().square().mean(1) >= 1e-6).tolist()
    run = 0
    last = None
    for index, value in enumerate(active):
        run = run + 1 if value else 0
        if run >= 2:
            last = min(len(wave), (index + 1) * 160) - 1
    if last is None:
        raise ValueError("Input has no qualifying audible run")
    return last // 1280
