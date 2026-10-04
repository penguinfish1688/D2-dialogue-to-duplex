"""Run the unmodified official Full-Duplex-Bench evaluator."""

import argparse
import json
import os
from pathlib import Path
import re
import sys

from benchmarks.common import run_official, verify_upstream, write_json
from benchmarks.fdb.common import FDB_COMMIT, TASKS


def official_metrics(output):
    """Read the reported numbers verbatim; do not recompute rates or latency."""
    pairs = re.findall(r"^Average ([a-z ]+):\s+([0-9.eE+-]+)\s*$", output, re.MULTILINE)
    if not pairs:
        raise ValueError("Official evaluator did not report any metrics")
    return {name.replace(" ", "_"): float(value) for name, value in pairs}


def score(args):
    upstream = verify_upstream(args.upstream, FDB_COMMIT)
    tasks = TASKS if args.task == "all" else {args.task: TASKS[args.task]}
    if "user_interruption" in tasks and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY for official interruption quality scoring")
    root = args.run.resolve()
    report = dict(evaluator_commit=FDB_COMMIT, tasks={})
    for task, category in tasks.items():
        folder = root / category
        samples = [path for path in folder.iterdir() if path.is_dir()]
        if not samples:
            raise ValueError(f"No samples in {folder}")
        for path in samples:
            if not (path / "output.json").is_file():
                raise ValueError(f"Run official ASR first: missing {path / 'output.json'}")
        output = run_official(
            [
                sys.executable,
                str(upstream / "v1_v1.5/evaluation/evaluate.py"),
                "--task",
                task,
                "--root_dir",
                str(folder),
            ],
            root / "scores" / f"{task}.log",
        )
        report["tasks"][task] = dict(samples=len(samples), **official_metrics(output))
        if task == "user_interruption":
            report["tasks"][task]["judge"] = "gpt-4-turbo"
    write_json(root / "scores" / f"{args.task}.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--task", choices=("all", *TASKS), default="all")
    parser.add_argument("--upstream", type=Path, default=Path(__file__).parent / "official")
    score(parser.parse_args())
