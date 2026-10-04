"""Small, shared configuration contract for the Qwen3 D2 series.

The native D2 clock is fixed at 12.5 Hz.  A model variant is therefore
identified completely by the number of native 80 ms frames in one macro
unit; there is no latency-specific model code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


NATIVE_FRAME_MS = 80
NATIVE_AUDIO_HZ = 12.5
ALLOWED_MACRO_FRAMES = (1, 2, 4, 8, 13)
ALLOWED_LATENCIES_MS = tuple(frames * NATIVE_FRAME_MS for frames in ALLOWED_MACRO_FRAMES)
DEFAULT_XL_CHUNK_TARGET_MS = 90_000


def _plain_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return value


def frames_for_latency(latency_ms: int) -> int:
    """Return native frames per macro unit for a supported D2 latency."""

    latency_ms = _plain_int(latency_ms, "latency_ms")
    if latency_ms not in ALLOWED_LATENCIES_MS:
        allowed = ", ".join(str(value) for value in ALLOWED_LATENCIES_MS)
        raise ValueError(f"latency_ms must be one of ({allowed}), got {latency_ms}")
    return latency_ms // NATIVE_FRAME_MS


def latency_for_frames(frames_per_unit: int) -> int:
    """Return latency in milliseconds for a supported macro-unit size."""

    frames_per_unit = _plain_int(frames_per_unit, "frames_per_unit")
    if frames_per_unit not in ALLOWED_MACRO_FRAMES:
        allowed = ", ".join(str(value) for value in ALLOWED_MACRO_FRAMES)
        raise ValueError(f"frames_per_unit must be one of ({allowed}), got {frames_per_unit}")
    return frames_per_unit * NATIVE_FRAME_MS


def align_chunk_frames(frame_count: int, frames_per_unit: int) -> int:
    """Round ``frame_count`` down without cutting a macro unit.

    This operation deliberately never rounds up: an XL chunk may be a little
    shorter than its requested wall-clock duration, but can never leak part of
    the following D2 unit into the current optimizer step.
    """

    frame_count = _plain_int(frame_count, "frame_count")
    frames_per_unit = _plain_int(frames_per_unit, "frames_per_unit")
    if frame_count < 1:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if frames_per_unit not in ALLOWED_MACRO_FRAMES:
        latency_for_frames(frames_per_unit)  # raises the canonical error
    aligned = frame_count - frame_count % frames_per_unit
    if aligned < frames_per_unit:
        raise ValueError(
            f"frame_count={frame_count} is shorter than one {frames_per_unit}-frame macro unit"
        )
    return aligned


def validate_chunk_frames(frame_count: int, frames_per_unit: int) -> int:
    """Validate and return a positive, macro-unit-aligned frame count."""

    frame_count = _plain_int(frame_count, "frame_count")
    frames_per_unit = _plain_int(frames_per_unit, "frames_per_unit")
    latency_for_frames(frames_per_unit)
    if frame_count < 1 or frame_count % frames_per_unit:
        raise ValueError(
            f"frame_count={frame_count} must be a positive multiple of "
            f"frames_per_unit={frames_per_unit}"
        )
    return frame_count


def xl_chunk_frames(
    frames_per_unit: int,
    target_ms: int = DEFAULT_XL_CHUNK_TARGET_MS,
) -> int:
    """Return the largest complete-unit XL chunk no longer than ``target_ms``."""

    target_ms = _plain_int(target_ms, "target_ms")
    if target_ms < NATIVE_FRAME_MS:
        raise ValueError(f"target_ms must be at least {NATIVE_FRAME_MS}, got {target_ms}")
    native_frames = target_ms // NATIVE_FRAME_MS
    return align_chunk_frames(native_frames, frames_per_unit)


@dataclass(frozen=True)
class D2Config:
    """The data-layout parameters shared by every D2 Qwen3 variant."""

    latency_ms: int = 80
    # Extra causal Code2Wav/TTS uncertainty, quantized to the native clock.
    # Keep this explicit until the final streaming TTS latency is measured.
    tts_uncertainty_frames: int = 0
    xl_chunk_target_ms: int = DEFAULT_XL_CHUNK_TARGET_MS

    def __post_init__(self) -> None:
        frames_for_latency(self.latency_ms)
        uncertainty = _plain_int(self.tts_uncertainty_frames, "tts_uncertainty_frames")
        if uncertainty < 0:
            raise ValueError(f"tts_uncertainty_frames must be non-negative, got {uncertainty}")
        _plain_int(self.xl_chunk_target_ms, "xl_chunk_target_ms")
        # Also proves that the requested duration holds at least one unit.
        xl_chunk_frames(self.frames_per_unit, self.xl_chunk_target_ms)

    @property
    def frames_per_unit(self) -> int:
        return frames_for_latency(self.latency_ms)

    @property
    def tokens_per_unit(self) -> int:
        # Environment audio, delayed self audio, and AR text input.
        return 3 * self.frames_per_unit

    @property
    def self_audio_delay_frames(self) -> int:
        return self.frames_per_unit + self.tts_uncertainty_frames

    @property
    def xl_chunk_frames(self) -> int:
        return xl_chunk_frames(self.frames_per_unit, self.xl_chunk_target_ms)

    @property
    def xl_chunk_units(self) -> int:
        return self.xl_chunk_frames // self.frames_per_unit

    @property
    def xl_chunk_ms(self) -> int:
        return self.xl_chunk_frames * NATIVE_FRAME_MS

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "D2Config":
        """Construct from a YAML/JSON-like mapping without owning a parser."""

        known = {"latency_ms", "tts_uncertainty_frames", "xl_chunk_target_ms"}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown D2 config keys: {sorted(unknown)}")
        return cls(**dict(values))

    def as_dict(self) -> dict[str, int]:
        """Return only source fields, so the result round-trips via from_mapping."""

        return {
            "latency_ms": self.latency_ms,
            "tts_uncertainty_frames": self.tts_uncertainty_frames,
            "xl_chunk_target_ms": self.xl_chunk_target_ms,
        }


__all__ = [
    "ALLOWED_LATENCIES_MS",
    "ALLOWED_MACRO_FRAMES",
    "D2Config",
    "DEFAULT_XL_CHUNK_TARGET_MS",
    "NATIVE_AUDIO_HZ",
    "NATIVE_FRAME_MS",
    "align_chunk_frames",
    "frames_for_latency",
    "latency_for_frames",
    "validate_chunk_frames",
    "xl_chunk_frames",
]
