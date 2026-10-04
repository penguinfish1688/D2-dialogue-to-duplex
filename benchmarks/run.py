"""Run the released Qwen runtime on VoiceBench or Full-Duplex-Bench."""

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import time

from common import DATASET_REVISION, FDB_CATEGORIES, TASKS, last_audible_frame
from common import read_benchmark_audio, response_text, sha256, write_json


def samples(args):
    rows = []
    if args.benchmark == "voicebench":
        manifest = json.loads(args.data.read_text())
        if not manifest.get("complete") or manifest["dataset_revision"] != DATASET_REVISION:
            raise ValueError("Expected a complete manifest at the pinned VoiceBench revision")
        counts = Counter()
        for item in manifest["rows"]:
            task = item["task"]
            if counts[task] >= args.limit:
                continue
            counts[task] += 1
            rows.append(dict(item, path=args.data.parent / item["audio"]["path"], task=task))
        if dict(counts) != dict.fromkeys(TASKS, args.limit):
            raise ValueError(f"Expected {args.limit} samples for each task: {counts}")
    else:
        for category, expected in FDB_CATEGORIES.items():
            files = sorted(
                (args.data / category).glob("*/input.wav"), key=lambda p: int(p.parent.name)
            )
            if len(files) != expected:
                raise ValueError(f"Expected {expected} samples in {category}, found {len(files)}")
            for path in files[: args.limit or None]:
                rows.append(dict(benchmark_id=path.parent.name, task=category, path=path))
    return [dict(row, selection_index=index) for index, row in enumerate(rows)]


async def run(args):
    import numpy as np
    import soundfile as sf
    import torch
    from d2.hub import load_release
    from d2_qwen.inference.runtime import Runtime

    if not 0 <= args.shard < args.shards:
        raise ValueError("Require 0 <= shard < shards")
    selected = samples(args)
    root, config = load_release(
        args.model, family="qwen", revision=args.revision, offline=args.offline
    )
    if config["latency_ms"] != 80:
        raise ValueError("This reproduction protocol is for the 80 ms release")
    identity = dict(
        code_revision=subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        model=args.model,
        revision=args.revision,
        d2_sha256=sha256(root / "d2.safetensors"),
        encoder_sha256=sha256(root / "encoder.safetensors"),
        seed=args.seed,
        kv_budget=args.kv_budget,
        benchmark=args.benchmark,
        limit=args.limit,
        dataset_revision=DATASET_REVISION if args.benchmark == "voicebench" else "FDB-v1.0",
        response_tail_seconds=20 if args.benchmark == "voicebench" else 0,
    )
    runtime = Runtime(str(root), kv_budget=args.kv_budget, seed=args.seed, offline=args.offline)
    started = time.perf_counter()
    await runtime.load()
    load_seconds = time.perf_counter() - started
    completed = skipped = 0
    total_wall = total_audio = 0.0
    try:
        for item in selected[args.shard :: args.shards]:
            dest = args.output / item["task"] / item["benchmark_id"]
            receipt = dest / "result.json"
            source_hash = sha256(item["path"])
            if args.benchmark == "voicebench" and source_hash != item["audio"]["sha256"]:
                raise ValueError(f"Input checksum mismatch: {item['benchmark_id']}")
            if receipt.exists():
                old = json.loads(receipt.read_text())
                if old["identity"] != identity or old["input_sha256"] != source_hash:
                    raise ValueError(f"Existing output used different settings: {dest}")
                if (
                    not (dest / old["audio_file"]).is_file()
                    or sha256(dest / old["audio_file"]) != old["audio_sha256"]
                ):
                    raise ValueError(f"Existing output audio is missing or changed: {dest}")
                skipped += 1
                continue
            pcm, wave = read_benchmark_audio(item["path"])
            if args.benchmark == "voicebench":
                boundary = last_audible_frame(wave) + 1
                pcm += bytes(20 * 32000)
            else:
                boundary = None
            source_samples = len(pcm) // 2
            if source_samples > runtime.max_context_frames * 1280:
                raise ValueError(f"{item['benchmark_id']} exceeds KV budget; increase --kv-budget")
            session = await runtime.create_session()
            padded = pcm + bytes(-len(pcm) % session.macro_bytes)
            output = bytearray()
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                for offset in range(0, len(padded), session.macro_bytes):
                    batch = await session.push_pcm16(padded[offset : offset + session.macro_bytes])
                    for frame in batch.pcm_frames:
                        output.extend(frame)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            finally:
                summary = await session.close()
            expected = round(source_samples * 1.5)
            if len(output) // 2 < expected:
                raise RuntimeError("Output shorter than the input timeline")
            output = output[: expected * 2]
            text = (
                response_text(summary, boundary, runtime.model.processor.tokenizer)
                if boundary is not None
                else "".join(
                    event["text"] for event in summary["text_trace"] if event["kind"] == "text"
                )
            )
            dest.mkdir(parents=True, exist_ok=True)
            audio_file = "output.flac" if args.benchmark == "voicebench" else "output.wav"
            sf.write(dest / audio_file, np.frombuffer(output, dtype="<i2"), 24000, subtype="PCM_16")
            if args.benchmark == "fdb":
                for annotation in item["path"].parent.glob("*.json"):
                    if annotation.name != "output.json":
                        shutil.copyfile(annotation, dest / annotation.name)
            audio_seconds = source_samples / 16000
            result = dict(
                selection_index=item["selection_index"],
                identity=identity,
                benchmark_id=item["benchmark_id"],
                task=item["task"],
                input_sha256=source_hash,
                row=item.get("row"),
                response=text,
                response_start_frame=boundary,
                audio_file=audio_file,
                audio_sha256=sha256(dest / audio_file),
                audio_seconds=audio_seconds,
                wall_seconds=elapsed,
                rtf=elapsed / audio_seconds,
                summary=summary,
            )
            write_json(receipt, result)
            completed += 1
            total_wall += elapsed
            total_audio += audio_seconds
            print(
                json.dumps(
                    dict(
                        task=item["task"],
                        id=item["benchmark_id"],
                        completed=completed,
                        selected=len(selected[args.shard :: args.shards]),
                        rtf=result["rtf"],
                        response=text,
                    )
                ),
                flush=True,
            )
    finally:
        await runtime.close()
    write_json(
        args.output / f"shard-{args.shard}.json",
        dict(
            identity=identity,
            completed=completed,
            skipped=skipped,
            selected=len(selected[args.shard :: args.shards]),
            load_seconds=load_seconds,
            streaming_wall_seconds=total_wall,
            audio_seconds=total_audio,
            rtf=total_wall / total_audio if total_audio else None,
            status="complete",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("voicebench", "fdb"))
    parser.add_argument(
        "--data", type=Path, required=True, help="VoiceBench manifest or FDB v1_0 directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="penguinfish1688/dialogue-to-duplex")
    parser.add_argument("--revision")
    parser.add_argument("--offline", action="store_true", help="Use already downloaded model files")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--kv-budget", type=int, help="Default: 2048 for VoiceBench, 4096 for FDB")
    parser.add_argument(
        "--limit", type=int, help="Samples per task; default: 200 VoiceBench, all FDB"
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.limit is None:
        args.limit = 200 if args.benchmark == "voicebench" else 0
    if args.limit < 0 or (args.benchmark == "voicebench" and args.limit == 0):
        parser.error("limit must be positive (0 means all for FDB)")
    args.kv_budget = args.kv_budget or (2048 if args.benchmark == "voicebench" else 4096)
    asyncio.run(run(args))
