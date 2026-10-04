"""Export D2 responses and run the unmodified official VoiceBench programs."""

import argparse
import json
from pathlib import Path
import sys

from benchmarks.common import run_official, verify_upstream
from benchmarks.vb.common import DEFAULT_SAMPLE_LIMIT, TASKS, VOICEBENCH_COMMIT

EVALUATORS = {
    "alpacaeval": "open",
    "commoneval": "open",
    "wildvoice": "open",
    "sd-qa": "qa",
    "mmsu": "mcq",
    "openbookqa": "mcq",
    "bbh": "bbh",
    "ifeval": "ifeval",
    "advbench": "harm",
}


def export_responses(root, limit):
    exports = root / "scores"
    exports.mkdir(parents=True, exist_ok=True)
    identity = None
    for task in TASKS:
        rows = sorted(
            (json.loads(p.read_text()) for p in (root / task).glob("*/result.json")),
            key=lambda row: row["selection_index"],
        )
        if len(rows) != limit:
            raise ValueError(f"{task}: expected {limit} responses, found {len(rows)}")
        records = []
        for row in rows:
            if identity is None:
                identity = row["identity"]
            if row["identity"] != identity:
                raise ValueError("Run mixes checkpoints or inference settings")
            records.append(
                dict(row["row"], benchmark_id=row["benchmark_id"], response=row["response"])
            )
        (exports / f"{task}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    return exports


def check_judged(source, judged):
    expected = [json.loads(line) for line in source.read_text().splitlines()]
    actual = [json.loads(line) for line in judged.read_text().splitlines()]
    if len(actual) != len(expected):
        raise ValueError(f"Incomplete official judge output: {judged}")
    for row, result in zip(expected, actual):
        if any(result.get(key) != value for key, value in row.items()):
            raise ValueError(f"Judge output belongs to different responses: {judged}")
        if len(result.get("score", [])) != 3:
            raise ValueError(f"Expected three official judge votes: {judged}")


def score(args):
    upstream = verify_upstream(args.upstream, VOICEBENCH_COMMIT)
    exports = export_responses(args.run.resolve(), args.limit)
    pending = []
    for task, evaluator in EVALUATORS.items():
        source = exports / f"{task}.jsonl"
        if evaluator in ("open", "qa"):
            judged = exports / f"result-{task}.jsonl"
            if not judged.exists() and args.judge:
                run_official(
                    [sys.executable, str(upstream / "api_judge.py"), "--src_file", source.name],
                    exports / f"{task}-judge.log",
                    cwd=exports,
                )
            if not judged.exists():
                pending.append(task)
                continue
            check_judged(source, judged)
            source = judged
        run_official(
            [
                sys.executable,
                str(upstream / "evaluate.py"),
                "--src_file",
                str(source),
                "--evaluator",
                evaluator,
            ],
            exports / f"{task}.log",
        )
    if pending:
        print(f"API judging pending for {', '.join(pending)}; rerun with --judge.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=DEFAULT_SAMPLE_LIMIT)
    parser.add_argument("--upstream", type=Path, default=Path(__file__).parent / "official")
    parser.add_argument("--judge", action="store_true", help="Run official API judging")
    score(parser.parse_args())
