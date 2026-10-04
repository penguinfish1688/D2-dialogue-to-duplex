"""Generate time-aligned audio for Full-Duplex-Bench v1.0."""

import asyncio

from benchmarks.fdb.common import FDB_CATEGORIES
from benchmarks.inference import arguments, run


def samples(args):
    rows = []
    for category, expected in FDB_CATEGORIES.items():
        paths = sorted((args.data / category).glob("*/input.wav"), key=lambda p: int(p.parent.name))
        if len(paths) != expected:
            raise ValueError(f"Expected {expected} samples in {category}, found {len(paths)}")
        for path in paths[: args.limit or None]:
            rows.append(dict(benchmark_id=path.parent.name, task=category, path=path))
    return [dict(row, selection_index=index) for index, row in enumerate(rows)]


def main():
    args = arguments(__doc__, benchmark="fdb", kv_budget=4096, limit=0)
    asyncio.run(run(args, samples(args), dataset_revision="FDB-v1.0", copy_annotations=True))


if __name__ == "__main__":
    main()
