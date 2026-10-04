"""Prepare a pinned, small VoiceBench audio subset without full shard downloads."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from common import (
    DEFAULT_SAMPLE_LIMIT,
    ROLLOUT_FRAMES,
    TASKS,
    VOICEBENCH_MANIFEST_CONTRACT,
)


DATASET_ID = "hlt-lab/voicebench"
DATASET_REVISION = "b02edcef1330480be3a11bd6f7434ac32f05ad08"
UPSTREAM_REPOSITORY = "https://github.com/MatthewCYM/VoiceBench.git"
UPSTREAM_COMMIT = "3c3b0d3a7a956f745305eb348f5e03ce7ec73dad"
DEFAULT_ROOT = Path("bench-data/voicebench")
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
MAX_CONTEXT_FRAMES = 1_125
INPUT_FRAME_SAMPLES = 1_280
INPUT_SAMPLE_RATE_HZ = 16_000

TASK_CONFIGS: dict[str, tuple[str, tuple[str, ...]]] = {
    "alpacaeval": ("alpacaeval_full", ("test",)),
    "commoneval": ("commoneval", ("test",)),
    "wildvoice": ("wildvoice", ("test",)),
    "sd-qa": (
        "sd-qa",
        (
            "aus",
            "gbr",
            "ind_n",
            "ind_s",
            "irl",
            "kenya",
            "nga",
            "nzl",
            "phl",
            "usa",
            "zaf",
        ),
    ),
    "mmsu": (
        "mmsu",
        (
            "biology",
            "business",
            "chemistry",
            "economics",
            "engineering",
            "health",
            "history",
            "law",
            "other",
            "philosophy",
            "physics",
            "psychology",
        ),
    ),
    "openbookqa": ("openbookqa", ("test",)),
    "bbh": ("bbh", ("test",)),
    "ifeval": ("ifeval", ("test",)),
    "advbench": ("advbench", ("test",)),
}


def _request_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "D2-VoiceBench/1"})
    payload: Any = None
    for attempt in range(8):
        try:
            with urlopen(request, timeout=120) as response:
                payload = json.load(response)
            break
        except (HTTPError, URLError, TimeoutError):
            if attempt == 7:
                raise
            time.sleep(min(20.0, 1.5 * (2**attempt)))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object from {url}")
    return payload


def _rows_url(config: str, split: str, *, offset: int = 0, length: int) -> str:
    query = urlencode(
        {
            "dataset": DATASET_ID,
            "config": config,
            "split": split,
            "offset": int(offset),
            "length": int(length),
        }
    )
    return f"{ROWS_ENDPOINT}?{query}"


def _fetch_split(config: str, split: str, *, length: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    total_rows: int | None = None
    while len(result) < int(length):
        page_length = min(100, int(length) - len(result))
        payload = _request_json(_rows_url(config, split, offset=len(result), length=page_length))
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"VoiceBench rows API omitted rows for {config}/{split}")
        if total_rows is None:
            total_rows = int(payload.get("num_rows_total", len(rows)))
        for item in rows:
            if not isinstance(item, Mapping) or not isinstance(item.get("row"), Mapping):
                raise ValueError(f"invalid VoiceBench API row for {config}/{split}")
            result.append({"row_idx": int(item["row_idx"]), "row": dict(item["row"])})
        if not rows or len(result) >= total_rows:
            break
    return result


def _round_robin(groups: Iterable[list[dict[str, Any]]]) -> Iterable[dict[str, Any]]:
    values = list(groups)
    for row_index in range(max((len(group) for group in values), default=0)):
        for group in values:
            if row_index < len(group):
                yield group[row_index]


def _audio_url(row: Mapping[str, Any]) -> str:
    audio = row.get("audio")
    if isinstance(audio, list) and len(audio) == 1 and isinstance(audio[0], Mapping):
        source = audio[0].get("src")
    elif isinstance(audio, Mapping):
        source = audio.get("src")
    else:
        source = None
    if not isinstance(source, str) or not source.startswith("https://"):
        raise ValueError("VoiceBench API row omitted an audio asset URL")
    if DATASET_REVISION not in source:
        raise ValueError("VoiceBench audio asset does not match the pinned revision")
    return source


def _asset_locator(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    request = Request(url, headers={"User-Agent": "D2-VoiceBench/1"})
    try:
        for attempt in range(8):
            try:
                with urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        handle.write(chunk)
                os.replace(temporary, destination)
                break
            except (HTTPError, URLError, TimeoutError):
                temporary.unlink(missing_ok=True)
                if attempt == 7:
                    raise
                time.sleep(min(20.0, 1.5 * (2**attempt)))
    finally:
        temporary.unlink(missing_ok=True)


def _audio_metadata(path: Path) -> dict[str, Any]:
    import soundfile as sf

    with sf.SoundFile(path) as handle:
        sample_rate = int(handle.samplerate)
        samples = int(handle.frames)
        channels = int(handle.channels)
    if sample_rate < 1 or samples < 1 or channels < 1:
        raise ValueError(f"VoiceBench audio must be non-empty: {path}")
    if channels != 1:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.mono.tmp")
        try:
            sf.write(
                temporary,
                mono,
                sample_rate,
                format="WAV",
                subtype="PCM_16",
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        samples = int(mono.shape[0])
        channels = 1
    resampled_samples = math.ceil(samples * INPUT_SAMPLE_RATE_HZ / sample_rate)
    source_frames = math.ceil(resampled_samples / INPUT_FRAME_SAMPLES)
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "sample_rate_hz": sample_rate,
        "samples": samples,
        "channels": channels,
        "duration_seconds": samples / sample_rate,
        "resampled_samples_16k": resampled_samples,
        "source_frames": source_frames,
    }


def _slug(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.")
    if not result:
        raise ValueError(f"cannot form a VoiceBench identity from {value!r}")
    return result


def _git_output(*args: str) -> str:
    return subprocess.run(
        args,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def prepare_upstream(path: Path) -> None:
    destination = path.expanduser().resolve()
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                UPSTREAM_REPOSITORY,
                str(destination),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--detach", UPSTREAM_COMMIT],
            check=True,
        )
    commit = _git_output("git", "-C", str(destination), "rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"VoiceBench source is at {commit}; expected {UPSTREAM_COMMIT}")
    if _git_output("git", "-C", str(destination), "status", "--porcelain"):
        raise ValueError(f"VoiceBench source has local modifications: {destination}")


def prepare_subset(
    root: str | Path,
    *,
    limit_per_task: int = DEFAULT_SAMPLE_LIMIT,
    rollout_frames: int = ROLLOUT_FRAMES,
) -> Path:
    if isinstance(limit_per_task, bool) or int(limit_per_task) < 1:
        raise ValueError("limit_per_task must be positive")
    if isinstance(rollout_frames, bool) or int(rollout_frames) < 1:
        raise ValueError("rollout_frames must be positive")
    destination_root = Path(root).expanduser().resolve()
    manifest_root = destination_root / "data" / f"diagnostic{int(limit_per_task)}"
    manifest_path = manifest_root / "manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            payload.get("contract") == VOICEBENCH_MANIFEST_CONTRACT
            and payload.get("dataset_revision") == DATASET_REVISION
            and payload.get("limit_per_task") == int(limit_per_task)
            and payload.get("rollout_frames") == int(rollout_frames)
            and payload.get("complete") is True
        ):
            return manifest_path
        raise ValueError(f"refusing to replace incompatible manifest: {manifest_path}")

    candidate_count = int(limit_per_task) + 10
    requests: list[tuple[str, str, str]] = []
    for task in TASKS:
        config, splits = TASK_CONFIGS[task]
        requests.extend((task, config, split) for split in splits)
    fetched: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_fetch_split, config, split, length=candidate_count): (
                task,
                split,
            )
            for task, config, split in requests
        }
        for future, key in futures.items():
            fetched[key] = future.result()

    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for task in TASKS:
        _, splits = TASK_CONFIGS[task]
        candidates = _round_robin(
            [[{**item, "split": split} for item in fetched[(task, split)]] for split in splits]
        )
        accepted = 0
        for item in candidates:
            if accepted == int(limit_per_task):
                break
            row = dict(item["row"])
            source_url = _audio_url(row)
            row.pop("audio", None)
            split = str(item["split"])
            source_index = int(item["row_idx"])
            benchmark_id = _slug(f"{task}--{split}--{source_index:06d}")
            relative_audio = Path("audio") / task / f"{benchmark_id}.wav"
            audio_path = manifest_root / relative_audio
            if not audio_path.is_file():
                _download(source_url, audio_path)
            metadata = _audio_metadata(audio_path)
            total_frames = int(metadata["source_frames"]) + int(rollout_frames)
            if total_frames > MAX_CONTEXT_FRAMES:
                rejected.append(
                    {
                        "task": task,
                        "split": split,
                        "row_index": source_index,
                        "benchmark_id": benchmark_id,
                        "source_frames": metadata["source_frames"],
                        "total_frames": total_frames,
                        "reason": "exceeds_fixed_context",
                    }
                )
                continue
            rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "task": task,
                    "config": TASK_CONFIGS[task][0],
                    "split": split,
                    "row_index": source_index,
                    "row": row,
                    "audio": {
                        "path": relative_audio.as_posix(),
                        "source_asset": _asset_locator(source_url),
                        **metadata,
                    },
                    "timeline": {
                        "source_frames": metadata["source_frames"],
                        "rollout_frames": int(rollout_frames),
                        "total_frames": total_frames,
                        "max_context_frames": MAX_CONTEXT_FRAMES,
                    },
                }
            )
            accepted += 1
        if accepted != int(limit_per_task):
            raise RuntimeError(
                f"only {accepted} valid VoiceBench rows found for {task}; "
                f"requested {limit_per_task}"
            )

    task_counts = Counter(str(row["task"]) for row in rows)
    payload = {
        "contract": VOICEBENCH_MANIFEST_CONTRACT,
        "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "selection": (
            "round_robin_across_official_splits_then_first_rows_fitting_the_fixed_d2_context"
        ),
        "limit_per_task": int(limit_per_task),
        "rollout_frames": int(rollout_frames),
        "max_context_frames": MAX_CONTEXT_FRAMES,
        "row_count": len(rows),
        "task_counts": dict(task_counts),
        "rejected": rejected,
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
        },
        "rows": rows,
    }
    manifest_root.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--limit-per-task", type=int, default=DEFAULT_SAMPLE_LIMIT)
    parser.add_argument("--rollout-frames", type=int, default=ROLLOUT_FRAMES)
    parser.add_argument("--skip-upstream", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.root.expanduser().resolve()
    if not args.skip_upstream:
        prepare_upstream(root / "upstream" / "VoiceBench")
    manifest = prepare_subset(
        root,
        limit_per_task=args.limit_per_task,
        rollout_frames=args.rollout_frames,
    )
    print(json.dumps({"manifest": str(manifest), "status": "complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DATASET_ID",
    "DATASET_REVISION",
    "DEFAULT_ROOT",
    "TASK_CONFIGS",
    "UPSTREAM_COMMIT",
    "UPSTREAM_REPOSITORY",
    "main",
    "prepare_subset",
    "prepare_upstream",
]
