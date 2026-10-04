"""Fetch pinned upstream code and weights into the user's model cache."""

from pathlib import Path
import hashlib
import os
import subprocess
import tarfile
import tempfile
import urllib.request

from d2.hub import asset
from .official import OFFICIAL_REPOSITORY, OFFICIAL_REPOSITORY_REVISION

MATCHA_URL = "https://files.pythonhosted.org/packages/e0/86/66edcad8aaec4313ee8dcab154672b2c5896bda1bc560cbb493b35dbf737/matcha-tts-0.0.5.1.tar.gz"
MATCHA_SHA256 = "0de4341d6e46610c730ca5dc12554c4297a940fdc74604a7eab524730002fea3"


def cache_root():
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "d2"


def source_checkout(*, offline=False):
    target = cache_root() / OFFICIAL_REPOSITORY_REVISION
    if not target.exists():
        if offline:
            raise FileNotFoundError(f"Upstream source is not cached: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as temp:
            checkout = Path(temp) / "source"
            subprocess.run(
                ["git", "clone", "--no-checkout", OFFICIAL_REPOSITORY, str(checkout)], check=True
            )
            subprocess.run(
                ["git", "-C", str(checkout), "checkout", "--detach", OFFICIAL_REPOSITORY_REVISION],
                check=True,
            )
            checkout.rename(target)
    return target


def matcha_source(*, offline=False):
    target = cache_root() / "matcha-tts-0.0.5.1"
    if not target.exists():
        if offline:
            raise FileNotFoundError(f"Renderer dependency is not cached: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as temp:
            archive = Path(temp) / "source.tar.gz"
            urllib.request.urlretrieve(MATCHA_URL, archive)
            if hashlib.sha256(archive.read_bytes()).hexdigest() != MATCHA_SHA256:
                raise ValueError("Matcha source checksum does not match the pinned release")
            with tarfile.open(archive) as source:
                source.extractall(Path(temp) / "unpacked", filter="data")
            unpacked = list((Path(temp) / "unpacked").iterdir())
            if len(unpacked) != 1:
                raise ValueError("Unexpected Matcha archive layout")
            unpacked[0].rename(target)
    return target


def resolve_assets(config, *, offline=False, renderer=True):
    import whisper

    source = (
        Path(config["source_root"]) if "source_root" in config else source_checkout(offline=offline)
    )
    if "whisper_checkpoint" in config:
        whisper_path = Path(config["whisper_checkpoint"])
    else:
        directory = cache_root() / "whisper"
        whisper_path = directory / "large-v3.pt"
        if offline and not whisper_path.is_file():
            raise FileNotFoundError(f"Whisper weights are not cached: {whisper_path}")
        if not offline:
            whisper._download(whisper._MODELS["large-v3"], str(directory), False)
    result = dict(
        source_root=source,
        model_snapshot=asset(config, "base_model", offline=offline),
        whisper_checkpoint=whisper_path,
    )
    if renderer:
        result.update(
            cosy_snapshot=asset(config, "renderer", offline=offline),
            matcha_root=Path(config["matcha_root"])
            if "matcha_root" in config
            else matcha_source(offline=offline),
            voice_prompt=source / "llama_omni2/inference/prompt_en.wav",
        )
    return result
