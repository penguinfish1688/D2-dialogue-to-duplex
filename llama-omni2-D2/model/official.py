"""Pinned loader for the released LLaMA-Omni2 implementation and weights."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from d2_llama.core.constants import (
    THINKER_HIDDEN_SIZE,
    TTS_EXTENDED_VOCAB_SIZE,
    TTS_HIDDEN_SIZE,
    TTS_PAD_TOKEN_ID,
    TTS_READ_TEXT_TOKENS,
    TTS_UNIT_VOCAB_SIZE,
    TTS_WRITE_SPEECH_TOKENS,
)


OFFICIAL_REPOSITORY = "https://github.com/ictnlp/LLaMA-Omni2"
OFFICIAL_REPOSITORY_REVISION = "c8afa9061a9c2d2c1919f7293f5492d946869752"
OFFICIAL_MODEL_ID = "ICTNLP/LLaMA-Omni2-7B-Bilingual"
OFFICIAL_MODEL_REVISION = "1fecdd98c361fe50a40b58b78738e3bc690ca349"
OFFICIAL_COSY_MODEL_ID = "ICTNLP/cosy2_decoder"
OFFICIAL_COSY_MODEL_REVISION = "7ff21e8e641b00cff2e0492651d654d153b21211"


def validate_official_config(config: Any) -> None:
    expected = {
        "model_type": "omni2_speech2s_qwen2",
        "hidden_size": THINKER_HIDDEN_SIZE,
        "speech_encoder_ds_rate": 5,
        "speech_encoder_hidden_size": 1280,
        "speech_encoder_type": "whisper",
        "speech_projector_type": "linear",
        "stream_params": f"({TTS_READ_TEXT_TOKENS},{TTS_WRITE_SPEECH_TOKENS})",
        "tie_word_embeddings": False,
        "unit_vocab_size": TTS_UNIT_VOCAB_SIZE,
    }
    for name, value in expected.items():
        if getattr(config, name, None) != value:
            raise ValueError(
                f"official LLaMA-Omni2 config {name} must be {value!r}, "
                f"got {getattr(config, name, None)!r}"
            )
    speech_generator = getattr(config, "speech_generator", None)
    if not isinstance(speech_generator, dict):
        speech_generator = speech_generator.to_dict()
    for name, value in {
        "hidden_size": TTS_HIDDEN_SIZE,
        "vocab_size": TTS_EXTENDED_VOCAB_SIZE,
        "num_hidden_layers": 24,
        "eos_token_id": TTS_PAD_TOKEN_ID,
        "tie_word_embeddings": True,
    }.items():
        if speech_generator.get(name) != value:
            raise ValueError(f"official TTS config {name} changed")


def _validate_local_assets(
    source_root: str | Path,
    model_snapshot: str | Path,
    whisper_checkpoint: str | Path,
) -> tuple[Path, Path, Path]:
    from whisper.model import MultiHeadAttention

    MultiHeadAttention.use_sdpa = False
    source = Path(source_root).resolve()
    snapshot = Path(model_snapshot).resolve()
    whisper_path = Path(whisper_checkpoint).resolve()
    if not (source / "llama_omni2/model/language_model/omni2_speech2s_qwen2.py").is_file():
        raise FileNotFoundError("pinned official LLaMA-Omni2 checkout is incomplete")
    try:
        source_revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            [
                "git",
                "-C",
                str(source),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("official source must be a verifiable git checkout") from error
    if source_revision != OFFICIAL_REPOSITORY_REVISION or dirty:
        raise ValueError("official source revision is wrong or the checkout is dirty")
    if snapshot.name != OFFICIAL_MODEL_REVISION or not (snapshot / "config.json").is_file():
        raise ValueError("model snapshot must be the pinned LLaMA-Omni2 revision")
    if whisper_path.name != "large-v3.pt" or not whisper_path.is_file():
        raise ValueError("whisper_checkpoint must be the staged large-v3.pt")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return source, snapshot, whisper_path


def load_official_audio_modules(
    *,
    source_root: str | Path,
    model_snapshot: str | Path,
    whisper_checkpoint: str | Path,
    dtype: Any = None,
) -> tuple[Any, Any]:
    """Load only the released Whisper encoder and concat-5 projector.

    Stage 1 does not execute the 7B Thinker or native TTS.  Constructing the
    two official audio modules and selecting their exact tensors from the
    revision-pinned safetensor index avoids allocating irrelevant weights on
    every DDP rank while preserving the released initialization exactly.
    """

    source, snapshot, whisper_path = _validate_local_assets(
        source_root, model_snapshot, whisper_checkpoint
    )
    module = importlib.import_module("llama_omni2.model.language_model.omni2_speech2s_qwen2")
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise RuntimeError("llama_omni2 resolved outside the pinned checkout")

    from safetensors import safe_open
    from transformers import AutoConfig
    from llama_omni2.model.speech_encoder.builder import build_speech_encoder
    from llama_omni2.model.speech_projector.builder import build_speech_projector

    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    config.speech_encoder = str(whisper_path)
    validate_official_config(config)
    speech_encoder = build_speech_encoder(config)
    speech_projector = build_speech_projector(config)

    index_path = snapshot / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    owners = {
        "model.speech_encoder.": speech_encoder,
        "model.speech_projector.": speech_projector,
    }
    for prefix, owner in owners.items():
        selected = {name: shard for name, shard in weight_map.items() if name.startswith(prefix)}
        if not selected:
            raise RuntimeError(f"official snapshot contains no {prefix} tensors")
        state: dict[str, Any] = {}
        by_shard: dict[str, list[str]] = {}
        for name, shard in selected.items():
            by_shard.setdefault(shard, []).append(name)
        for shard, names in by_shard.items():
            with safe_open(snapshot / shard, framework="pt", device="cpu") as handle:
                for name in names:
                    state[name.removeprefix(prefix)] = handle.get_tensor(name)
        incompatible = owner.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"official {prefix} tensor inventory changed")
        if dtype is not None:
            owner.to(dtype=dtype)
    return speech_encoder, speech_projector


def load_official_bundle(
    *,
    source_root: str | Path,
    model_snapshot: str | Path,
    whisper_checkpoint: str | Path,
    dtype: Any = None,
) -> tuple[Any, Any]:
    """Load only from explicit local, revision-pinned directories."""

    source, snapshot, whisper_path = _validate_local_assets(
        source_root, model_snapshot, whisper_checkpoint
    )

    module = importlib.import_module("llama_omni2.model.language_model.omni2_speech2s_qwen2")
    resolved = Path(module.__file__).resolve()
    if not resolved.is_relative_to(source):
        raise RuntimeError("llama_omni2 resolved outside the pinned checkout")

    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    config.speech_encoder = str(whisper_path)
    config.tts_tokenizer = str(snapshot / "tts_tokenizer")
    validate_official_config(config)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        use_fast=False,
        local_files_only=True,
    )
    model_class = module.Omni2Speech2SQwen2ForCausalLM
    kwargs = {
        "config": config,
        "local_files_only": True,
        "attn_implementation": "sdpa",
    }
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    model = model_class.from_pretrained(snapshot, **kwargs)
    return tokenizer, model


__all__ = [
    "OFFICIAL_COSY_MODEL_ID",
    "OFFICIAL_COSY_MODEL_REVISION",
    "OFFICIAL_MODEL_ID",
    "OFFICIAL_MODEL_REVISION",
    "OFFICIAL_REPOSITORY",
    "OFFICIAL_REPOSITORY_REVISION",
    "load_official_audio_modules",
    "load_official_bundle",
    "validate_official_config",
]
