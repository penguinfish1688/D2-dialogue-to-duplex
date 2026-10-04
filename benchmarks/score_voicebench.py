"""Score the fixed VoiceBench run using pinned upstream evaluators."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
from types import ModuleType

from common import TASKS, VOICEBENCH_COMMIT, write_json

DETERMINISTIC = {
    "mmsu": ("mcq", "MCQEvaluator", "acc", 1),
    "openbookqa": ("mcq", "MCQEvaluator", "acc", 1),
    "bbh": ("bbh", "BBHEvaluator", "acc", 1),
    "ifeval": ("ifeval", "IFEvaluator", "final", 100),
    "advbench": ("harm", "HarmEvaluator", "refusal_rate", 100),
}


def score(args):
    commit = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != VOICEBENCH_COMMIT:
        raise ValueError("VoiceBench source is not at the pinned revision")
    package = ModuleType("_d2_voicebench_evaluator")
    package.__path__ = [str(args.upstream / "src/evaluator")]
    sys.modules[package.__name__] = package
    judge = None
    if args.judge:
        sys.path.insert(0, str(args.upstream))
        spec = importlib.util.spec_from_file_location(
            "_d2_voicebench_judge", args.upstream / "api_judge.py"
        )
        judge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(judge)
    scores = {}
    identities = []
    for task in TASKS:
        paths = sorted((args.run / task).glob("*/result.json"))
        if len(paths) != args.limit:
            raise ValueError(f"{task}: expected {args.limit} outputs, got {len(paths)}")
        rows = sorted(
            [json.loads(p.read_text()) for p in paths], key=lambda row: row["selection_index"]
        )
        identities.extend(row["identity"] for row in rows)
        records = [
            dict(row["row"], benchmark_id=row["benchmark_id"], response=row["response"])
            for row in rows
        ]
        exports = args.run / "scores"
        exports.mkdir(exist_ok=True)
        (exports / f"{task}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        if task in DETERMINISTIC:
            module, cls, field, scale = DETERMINISTIC[task]
            evaluator = getattr(importlib.import_module(f"{package.__name__}.{module}"), cls)()
            random.seed(0)
            raw = evaluator.evaluate(records)
            scores[task] = dict(samples=len(records), score=float(raw[field]) * scale, raw=raw)
        else:
            cache = exports / f"{task}-judged"
            cache.mkdir(exist_ok=True)
            votes = []

            def judge_row(row):
                path = cache / (row["benchmark_id"] + ".json")
                if path.exists():
                    result = json.loads(path.read_text())
                    if any(result.get(k) != v for k, v in row.items()):
                        raise ValueError("Cached judge result belongs to another response")
                elif judge is not None:
                    result = judge.generate(dict(row))
                    write_json(path, result)
                else:
                    return None
                return result

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                results = list(pool.map(judge_row, records))
            for result in results:
                if result is None:
                    continue
                values = result.get("score", [])
                if len(values) != 3:
                    raise ValueError("Expected three GPT-4o-mini votes")
                if task == "sd-qa":
                    values = [str(x).strip().lower() for x in values]
                    if any(x not in ("yes", "no") for x in values):
                        raise ValueError("Invalid SD-QA judge vote")
                    votes.append(100.0 * (values.count("yes") > values.count("no")))
                else:
                    values = [float(x) for x in values]
                    if not all(1 <= x <= 5 for x in values):
                        raise ValueError("Invalid open-ended judge rating")
                    votes.append(sum(values) / 3 * 20)
            scores[task] = dict(
                samples=len(records),
                judged=len(votes),
                score=sum(votes) / len(votes) if len(votes) == len(records) else None,
            )
        print(json.dumps({task: scores[task]}), flush=True)
    if any(identity != identities[0] for identity in identities):
        raise ValueError("Run mixes model revisions or evaluation settings")
    complete = all(task["score"] is not None for task in scores.values())
    report = dict(
        status="complete" if complete else "pending_api_judge",
        tasks=scores,
        identity=identities[0],
        samples=sum(x["samples"] for x in scores.values()),
        overall=sum(x["score"] for x in scores.values()) / 9 if complete else None,
        evaluator_commit=commit,
        judge="gpt-4o-mini",
        votes_per_sample=3,
    )
    write_json(args.run / "scores/metrics.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--upstream", type=Path, default=Path("bench-data/voicebench/upstream/VoiceBench")
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent API judge requests")
    parser.add_argument(
        "--judge", action="store_true", help="Use OPENAI_API_KEY for the four API-scored tasks"
    )
    score(parser.parse_args())
