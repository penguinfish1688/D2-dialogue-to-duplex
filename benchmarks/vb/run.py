"""Generate VoiceBench responses with the D2 streaming runtime."""

import asyncio
from collections import Counter
import json

from benchmarks.inference import arguments, run
from benchmarks.vb.common import (
    DATASET_REVISION,
    DEFAULT_SAMPLE_LIMIT,
    TASKS,
    last_audible_frame,
    response_text,
)


def samples(args):
    manifest = json.loads(args.data.read_text())
    if not manifest.get("complete") or manifest["dataset_revision"] != DATASET_REVISION:
        raise ValueError("Expected a complete manifest at the pinned VoiceBench revision")
    rows, counts = [], Counter()
    for item in manifest["rows"]:
        task = item["task"]
        if counts[task] >= args.limit:
            continue
        counts[task] += 1
        rows.append(dict(item, path=args.data.parent / item["audio"]["path"], task=task))
    if dict(counts) != dict.fromkeys(TASKS, args.limit):
        raise ValueError(f"Expected {args.limit} samples for each task: {counts}")
    return [dict(row, selection_index=index) for index, row in enumerate(rows)]


def main():
    args = arguments(__doc__, benchmark="voicebench", kv_budget=2048, limit=DEFAULT_SAMPLE_LIMIT)
    asyncio.run(
        run(
            args,
            samples(args),
            dataset_revision=DATASET_REVISION,
            tail_seconds=20,
            audio_file="output.flac",
            response_boundary=lambda wave: last_audible_frame(wave) + 1,
            extract_response=response_text,
        )
    )


if __name__ == "__main__":
    main()
