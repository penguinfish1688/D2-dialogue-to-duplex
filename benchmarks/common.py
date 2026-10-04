"""Shared benchmark file and audio helpers."""

import hashlib
import json
from pathlib import Path
import subprocess


def verify_upstream(path, commit):
    """Require the pinned, unmodified official checkout."""
    path = Path(path).resolve()
    if not (path / ".git").exists():
        raise ValueError("Initialize the official repositories: git submodule update --init")
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != commit:
        raise ValueError(f"Expected official revision {commit}, found {actual}")
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if dirty:
        raise ValueError(f"Official checkout has local modifications: {path}")
    return path


def run_official(command, log, *, cwd=None):
    """Save the official program's output without changing its scoring logic."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running official evaluator; log: {log}", flush=True)
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
    output = log.read_text()
    print("\n".join(output.splitlines()[-8:]), flush=True)
    result.check_returncode()
    return output


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


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
