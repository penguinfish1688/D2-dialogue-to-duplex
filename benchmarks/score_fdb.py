"""Score FDB ASR transcripts with event-aligned latency and optional GPT quality."""

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import subprocess

from common import FDB_CATEGORIES, sha256, write_json

FDB_COMMIT = "3e799c45a045256f47d5f1c9cda90157e2d2ec9e"


def takes_turn(chunks):
    if not chunks:
        return False
    end = chunks[-1]["timestamp"][1]
    if end is None:
        end = chunks[-1]["timestamp"][0]
    return end - chunks[0]["timestamp"][0] >= 1 or len(chunks) > 3


def load_judge(upstream):
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != FDB_COMMIT:
        raise ValueError("FDB source is not at the pinned revision")
    path = upstream / "v1_v1.5/evaluation/eval_user_interruption.py"
    tree = ast.parse(path.read_text())
    function = next(
        x
        for x in tree.body
        if isinstance(x, ast.FunctionDef) and x.name == "eval_user_interruption"
    )
    constants = {}
    for node in function.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
    spec = importlib.util.spec_from_file_location("_d2_fdb_judge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from openai import OpenAI

    return OpenAI(), constants, module.parse_output


def score(args):
    judge = load_judge(args.upstream) if args.judge else None
    scores = {}
    identities = []
    for category, count in FDB_CATEGORIES.items():
        paths = sorted(
            (args.run / category).glob("*/result.json"), key=lambda p: int(p.parent.name)
        )
        if len(paths) != (args.limit or count):
            raise ValueError(f"Incomplete {category}: found {len(paths)} samples")
        outcomes, gaps, ratings = [], [], []
        for path in paths:
            result = json.loads(path.read_text())
            identities.append(result["identity"])
            transcript_path = path.parent / "output.json"
            transcript = json.loads(transcript_path.read_text())
            chunks = transcript["chunks"]
            take = takes_turn(chunks)
            outcomes.append(take)
            if category == "candor_pause_handling":
                continue
            interruption = category == "synthetic_user_interruption"
            annotation = json.loads(
                (
                    path.parent / ("interrupt.json" if interruption else "turn_taking.json")
                ).read_text()
            )[0]
            event_time = annotation["timestamp"][1 if interruption else 0]
            after = [chunk for chunk in chunks if chunk["timestamp"][0] >= event_time]
            if takes_turn(after):
                gaps.append(max(0.0, after[0]["timestamp"][0] - event_time))
            if not interruption or not take:
                continue
            rating_path = path.parent / "rating.json"
            identity = dict(transcript_sha256=sha256(transcript_path), annotation=annotation)
            if rating_path.exists():
                rating = json.loads(rating_path.read_text())
                if rating["identity"] != identity:
                    raise ValueError("Cached rating belongs to another transcript")
            elif judge:
                client, constants, parse = judge
                user = f"""
                - Contextual user turn: {annotation["context"]}
                - User interrupting turn: {annotation["interrupt"]}
                - AI's response: {transcript["text"]}
                """
                response = client.chat.completions.create(
                    model=constants["MODEL_NAME"],
                    seed=constants["seed"],
                    messages=[
                        {"role": "system", "content": constants["system_msg"]},
                        {"role": "user", "content": user},
                    ],
                )
                raw = response.choices[0].message.content
                rating = parse(raw + "\n")
                if not 0 <= rating.get("rating", -1) <= 5:
                    raise ValueError("Could not parse the official FDB judge response")
                rating.update(identity=identity, raw=raw)
                write_json(rating_path, rating)
            else:
                continue
            ratings.append(rating["rating"])
        scores[category] = dict(
            samples=len(outcomes),
            take_count=sum(outcomes),
            take_percent=100 * sum(outcomes) / len(outcomes),
        )
        if category == "candor_pause_handling":
            scores[category]["pause_success_percent"] = 100 - scores[category]["take_percent"]
        else:
            acoustic = sum(gaps) / len(gaps) if gaps else None
            scores[category].update(
                acoustic_seconds=acoustic,
                causal_seconds=0.08,
                response_latency_seconds=acoustic + 0.08 if acoustic is not None else None,
                latency_samples=len(gaps),
            )
        if category == "synthetic_user_interruption":
            scores[category].update(
                quality=sum(ratings) / len(ratings)
                if len(ratings) == sum(outcomes) and ratings
                else None,
                quality_samples=len(ratings),
                quality_expected=sum(outcomes),
            )
    if any(identity != identities[0] for identity in identities):
        raise ValueError("Run mixes checkpoints or evaluation settings")
    report = dict(
        tasks=scores,
        identity=identities[0],
        evaluator_commit=FDB_COMMIT,
        timing="Event-aligned acoustic gap plus 80 ms; excludes computation",
    )
    write_json(args.run / "scores/metrics.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path("bench-data/Full-Duplex-Bench"))
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Use OPENAI_API_KEY for official GPT-4o quality scoring",
    )
    score(parser.parse_args())
