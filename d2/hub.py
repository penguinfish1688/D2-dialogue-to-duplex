"""Resolve release bundles from a local directory or Hugging Face."""

from __future__ import annotations

import json
from pathlib import Path

MODEL_IDS = {
    "qwen": "HF_ORG/Qwen3-Omni-D2",
    "llama": "HF_ORG/LLaMA-Omni2-D2",
}
FORMAT = "d2.release.v1"


def snapshot(source: str, *, revision: str | None = None, offline: bool = False) -> Path:
    path = Path(source).expanduser()
    if path.is_dir():
        return path.resolve()
    if "HF_ORG" in source or "PLACEHOLDER" in source:
        raise ValueError(
            "D2 checkpoints have not been uploaded to Hugging Face yet. "
            "Pass --model /path/to/a/local/release, or replace the placeholder "
            "with the published repository ID when available."
        )
    if path.is_absolute() or source.startswith(("./", "../", "~")):
        raise FileNotFoundError(f"Model directory does not exist: {path}")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=source, revision=revision, local_files_only=offline))


def load_release(
    source: str, *, family: str, revision: str | None = None, offline: bool = False
) -> tuple[Path, dict]:
    root = snapshot(source, revision=revision, offline=offline)
    config = json.loads((root / "d2.json").read_text())
    if config.get("format") != FORMAT or config.get("family") != family:
        raise ValueError(f"Expected a {family} {FORMAT} release")
    allowed = {"qwen": (80, 160, 320, 640, 1040), "llama": (100, 200, 400, 800)}
    if type(config.get("latency_ms")) is not int or config["latency_ms"] not in allowed[family]:
        raise ValueError("Unsupported checkpoint interaction granularity")
    if not (root / "d2.safetensors").is_file():
        raise FileNotFoundError(f"Release is missing {root / 'd2.safetensors'}")
    return root, config


def asset(config: dict, name: str, *, offline: bool = False) -> Path:
    spec = config[name]
    if not isinstance(spec, dict) or not isinstance(spec.get("repo_id"), str):
        raise ValueError(f"Invalid asset specification: {name}")
    return snapshot(spec["repo_id"], revision=spec.get("revision"), offline=offline)
