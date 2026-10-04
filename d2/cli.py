"""The same small command interface for both model families."""

import argparse
import asyncio
import importlib
import json
from pathlib import Path
import time

from .hub import MODEL_IDS, load_release


def parser(family):
    p = argparse.ArgumentParser(prog="d2-" + family)
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("download", "infer", "app", "train"):
        command = commands.add_parser(name)
        command.add_argument(
            "--model",
            default=MODEL_IDS[family],
            help="Hugging Face ID or local D2 release directory",
        )
        command.add_argument("--revision", default=None)
        command.add_argument("--offline", action="store_true")
        if name != "download":
            command.add_argument("--device", default="cuda")
            command.add_argument("--seed", type=int, default=1337)
        if name in ("infer", "app"):
            command.add_argument(
                "--kv-budget", type=int, default=2048, help="Fixed Thinker KV budget in tokens"
            )
        if name == "infer":
            command.add_argument("--input", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
            command.add_argument(
                "--tail-seconds", type=float, default=8.0, help="Silence appended for the response"
            )
        if name == "app":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8000)
        if name == "train":
            command.add_argument(
                "--data", type=Path, required=True, help="Prepared sample manifest JSON"
            )
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--steps", type=int, default=3)
    return p


async def infer(runtime, args):
    from .audio import load_pcm, save_pcm

    if args.tail_seconds < 0:
        raise ValueError("tail-seconds must be nonnegative")
    pcm = load_pcm(args.input) + bytes(2 * round(args.tail_seconds * 16000))
    await runtime.load()
    session = await runtime.create_session()
    if len(pcm) > runtime.max_context_frames * runtime.native_ms * 32:
        await session.close()
        await runtime.close()
        raise ValueError("Input plus response tail exceeds the KV budget; increase --kv-budget")
    padded = pcm + bytes((-len(pcm)) % session.macro_bytes)
    audio, events = [], []
    import torch

    torch.cuda.synchronize(runtime.device)
    start = time.perf_counter()
    try:
        for offset in range(0, len(padded), session.macro_bytes):
            batch = await session.push_pcm16(padded[offset : offset + session.macro_bytes])
            audio.extend(batch.pcm_frames)
            events.extend(batch.frame_events)
        torch.cuda.synchronize(runtime.device)
        elapsed = time.perf_counter() - start
    finally:
        summary = await session.close()
        await runtime.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_pcm(args.output, b"".join(audio))
    summary.update(
        runtime.metadata(),
        wall_seconds=elapsed,
        audio_seconds=len(padded) / 32000,
        rtf=elapsed / (len(padded) / 32000),
        timing="streaming loop including first-use graph capture; excludes model loading and session initialization",
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(dict(summary=summary, events=events), indent=2)
    )
    transcript = summary.get("generated_text")
    if transcript is None:
        transcript = "".join(event["text"] for event in summary.get("text_trace", []))
    print(
        json.dumps(
            dict(
                output=str(args.output),
                text=transcript,
                wall_seconds=elapsed,
                audio_seconds=summary["audio_seconds"],
                rtf=summary["rtf"],
                kv_budget_tokens=runtime.kv_budget,
            ),
            indent=2,
        )
    )


def main(family, argv=None):
    p = parser(family)
    args = p.parse_args(argv)
    # Resolve placeholders before importing a model or allocating GPU memory.
    root, config = load_release(
        args.model, family=family, revision=args.revision, offline=args.offline
    )
    if args.command == "download":
        if family == "llama":
            from d2_llama.model.assets import resolve_assets

            resolve_assets(config, offline=args.offline)
        else:
            from .hub import asset

            asset(config, "base_model", offline=args.offline)
        print(root)
        return
    if args.command == "train":
        training = importlib.import_module("d2_" + family + ".training.sft")
        training.train(args)
        return
    runtime_class = importlib.import_module("d2_" + family + ".inference.runtime").Runtime
    runtime = runtime_class(
        args.model,
        device=args.device,
        revision=args.revision,
        offline=args.offline,
        kv_budget=args.kv_budget,
        seed=args.seed,
    )
    if args.command == "infer":
        asyncio.run(infer(runtime, args))
    else:
        import uvicorn
        from .server import create_app

        uvicorn.run(
            create_app(runtime), host=args.host, port=args.port, ws_max_size=65536, ws_max_queue=4
        )
