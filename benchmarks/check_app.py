"""Test the actual WebSocket app using microphone-sized, real-time packets."""

import argparse
import asyncio
import json
from pathlib import Path
import time

import numpy as np
import websockets
from d2.audio import load_pcm, save_pcm
from benchmarks.common import write_json


async def check(args):
    pcm = load_pcm(args.input) + bytes(round(args.tail_seconds * 32000))
    pcm += bytes(-len(pcm) % 2560)
    async with websockets.connect(args.url, max_size=2**20) as socket:
        ready = json.loads(await socket.recv())
        if ready["type"] != "ready":
            raise RuntimeError(ready)
        if len(pcm) / 32000 > ready["max_conversation_seconds"]:
            raise ValueError("Input exceeds app context limit")
        output = bytearray()
        processing = []
        acknowledged = sent = max_outstanding = 0
        start = time.perf_counter()

        async def receive():
            nonlocal acknowledged
            while acknowledged < len(pcm) // 2:
                message = await socket.recv()
                if isinstance(message, bytes):
                    output.extend(message)
                    continue
                event = json.loads(message)
                if event["type"] == "error":
                    raise RuntimeError(event["message"])
                if event["type"] == "ack":
                    acknowledged += event["samples"]
                    processing.append(event["processing_seconds"])

        receiver = asyncio.create_task(receive())
        try:
            for offset in range(0, len(pcm), 1280):
                if receiver.done():
                    await receiver
                await asyncio.sleep(max(0, start + (offset + 1280) / 32000 - time.perf_counter()))
                packet = pcm[offset : offset + 1280]
                await socket.send(packet)
                sent += len(packet) // 2
                max_outstanding = max(max_outstanding, sent - acknowledged)
            await asyncio.wait_for(receiver, timeout=120)
            wall = time.perf_counter() - start
            await socket.send(json.dumps(dict(type="stop")))
        finally:
            if not receiver.done():
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
        duration = len(pcm) / 32000
        result = dict(
            metadata=ready,
            input_seconds=duration,
            paced_wall_seconds=wall,
            processing_seconds=sum(processing),
            rtf=sum(processing) / duration,
            maximum_backlog_seconds=max_outstanding / 16000,
            output_seconds=len(output) / 48000,
            p95_packet_processing_seconds=float(np.quantile(processing, 0.95)),
        )
        result["passed"] = (
            result["rtf"] < 1
            and result["maximum_backlog_seconds"] <= 2
            and len(output) == len(pcm) * 3 // 2
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_pcm(args.output, output)
        write_json(args.output.with_suffix(".json"), result)
        print(json.dumps(result, indent=2), flush=True)
        if not result["passed"]:
            raise RuntimeError(
                "App did not sustain real-time processing within the browser backlog limit"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://localhost:8000/stream")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("app-check.wav"))
    parser.add_argument("--tail-seconds", type=float, default=20)
    args = parser.parse_args()
    if args.tail_seconds < 0:
        parser.error("tail-seconds must be nonnegative")
    asyncio.run(check(args))
