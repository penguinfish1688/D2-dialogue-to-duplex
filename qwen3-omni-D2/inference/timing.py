from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
import time
from typing import Any, Iterator

import torch


@dataclass(frozen=True)
class _PendingCudaMeasurement:
    name: str
    sequence_step: int | None
    device_index: int
    start: torch.cuda.Event
    end: torch.cuda.Event


class ForwardTimings:
    """Collect component forward times without synchronizing the decode loop."""

    def __init__(self) -> None:
        self._samples_ms: dict[str, list[float]] = defaultdict(list)
        self._sample_steps: dict[str, list[int | None]] = defaultdict(list)
        self._pending_cuda: list[_PendingCudaMeasurement] = []
        self._pause_depth = 0
        self._sequence_step: int | None = None

    @contextmanager
    def sequence_step(self, step: int) -> Iterator[None]:
        if int(step) < 0:
            raise ValueError("sequence timing step must be non-negative")
        previous = self._sequence_step
        self._sequence_step = int(step)
        try:
            yield
        finally:
            self._sequence_step = previous

    @contextmanager
    def measure(self, name: str, device: torch.device) -> Iterator[None]:
        if self._pause_depth:
            yield
            return
        target = torch.device(device)
        if target.type != "cuda" or not torch.cuda.is_available():
            started = time.perf_counter()
            try:
                yield
            finally:
                self._samples_ms[str(name)].append((time.perf_counter() - started) * 1000.0)
                self._sample_steps[str(name)].append(self._sequence_step)
            return

        device_index = target.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(device_index):
            start.record(torch.cuda.current_stream(device_index))
        try:
            yield
        finally:
            with torch.cuda.device(device_index):
                end.record(torch.cuda.current_stream(device_index))
            self._pending_cuda.append(
                _PendingCudaMeasurement(
                    name=str(name),
                    sequence_step=self._sequence_step,
                    device_index=int(device_index),
                    start=start,
                    end=end,
                )
            )

    @contextmanager
    def paused(self) -> Iterator[None]:
        self._pause_depth += 1
        try:
            yield
        finally:
            self._pause_depth -= 1

    def _resolve_cuda(self) -> None:
        if not self._pending_cuda:
            return
        for device_index in sorted(
            {measurement.device_index for measurement in self._pending_cuda}
        ):
            torch.cuda.synchronize(device_index)
        for measurement in self._pending_cuda:
            self._samples_ms[measurement.name].append(
                float(measurement.start.elapsed_time(measurement.end))
            )
            self._sample_steps[measurement.name].append(measurement.sequence_step)
        self._pending_cuda.clear()

    def sample_records(self) -> list[dict[str, float | int | str | None]]:
        self._resolve_cuda()
        records: list[dict[str, float | int | str | None]] = []
        for name in sorted(self._samples_ms):
            values = self._samples_ms[name]
            steps = self._sample_steps[name]
            if len(values) != len(steps):
                raise RuntimeError(f"Timing sample metadata is misaligned for {name!r}")
            records.extend(
                {
                    "component": name,
                    "sequence_step": step,
                    "milliseconds": float(value),
                }
                for value, step in zip(values, steps, strict=True)
            )
        return records

    def summary(self) -> dict[str, Any]:
        self._resolve_cuda()
        components: dict[str, dict[str, float | int]] = {}
        for name in sorted(self._samples_ms):
            samples = self._samples_ms[name]
            if not samples:
                continue
            total_ms = float(sum(samples))
            components[name] = {
                "calls": len(samples),
                "total_ms": total_ms,
                "average_ms": total_ms / len(samples),
                "min_ms": float(min(samples)),
                "max_ms": float(max(samples)),
            }
        return {
            "unit": "milliseconds",
            "method": "cuda_events_with_cpu_perf_counter_fallback",
            "diagnostic_forwards_excluded": True,
            "components": components,
        }


def measure_forward(
    timings: ForwardTimings | None,
    name: str,
    device: torch.device,
) -> AbstractContextManager[None]:
    if timings is None:
        return nullcontext()
    return timings.measure(name, device)


def aggregate_forward_timing_summaries(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    combined: dict[str, dict[str, float | int]] = {}
    for summary in summaries:
        for name, stats in summary.get("components", {}).items():
            calls = int(stats["calls"])
            total_ms = float(stats["total_ms"])
            if name not in combined:
                combined[name] = {
                    "calls": calls,
                    "total_ms": total_ms,
                    "min_ms": float(stats["min_ms"]),
                    "max_ms": float(stats["max_ms"]),
                }
                continue
            target = combined[name]
            target["calls"] = int(target["calls"]) + calls
            target["total_ms"] = float(target["total_ms"]) + total_ms
            target["min_ms"] = min(float(target["min_ms"]), float(stats["min_ms"]))
            target["max_ms"] = max(float(target["max_ms"]), float(stats["max_ms"]))
    for stats in combined.values():
        stats["average_ms"] = float(stats["total_ms"]) / int(stats["calls"])
    return {
        "unit": "milliseconds",
        "method": "cuda_events_with_cpu_perf_counter_fallback",
        "diagnostic_forwards_excluded": True,
        "rows": len(summaries),
        "components": dict(sorted(combined.items())),
    }


__all__ = [
    "ForwardTimings",
    "aggregate_forward_timing_summaries",
    "measure_forward",
]
