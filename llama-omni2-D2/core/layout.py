"""Canonical D2 Thinker layout for LLaMA-Omni2.

Every macro is serialized as ``environment[k] | delayed_self[k] | text[k]``.
The Qwen2.5 backbone then uses its ordinary one-dimensional causal attention.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

from d2_llama.core.constants import (
    INPUT_SAMPLES_PER_FRAME,
    latency_for_frames,
    self_audio_delay_frames,
)


T = TypeVar("T")

ENVIRONMENT_AUDIO = "environment_audio"
SELF_AUDIO = "delayed_self_audio"
TEXT = "text"
LANES = (ENVIRONMENT_AUDIO, SELF_AUDIO, TEXT)
ATTENTION_POLICY = "llama_omni2_envk_selfk_textk_serialized_causal_v1"
SELF_AUDIO_DELAY_POLICY = "thinker_macro_plus_native_write10_release_v1"


def _plain_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _k(value: int) -> int:
    value = _plain_int(value, "frames_per_unit")
    latency_for_frames(value)
    return value


def _positive_multiple(length: int, width: int, name: str) -> int:
    if length < 1 or length % width:
        raise ValueError(f"{name} length={length} must be a positive multiple of {width}")
    return length // width


def shift_text_inputs(
    text_targets: Sequence[T],
    frames_per_unit: int,
    bos: T,
    previous_final: T | None = None,
) -> list[T]:
    """Apply one global AR shift; never restart at macro boundaries."""

    k = _k(frames_per_unit)
    targets = list(text_targets)
    _positive_multiple(len(targets), k, "text_targets")
    return [bos if previous_final is None else previous_final, *targets[:-1]]


def delay_self_audio(
    self_audio: Sequence[T],
    frames_per_unit: int,
    *,
    samples_per_frame: int = INPUT_SAMPLES_PER_FRAME,
    fill: T | None = None,
) -> list[T | None]:
    """Right-shift clean self PCM by the deployment-visible delay.

    The shift is ``k`` Thinker frames plus four 100-ms frames for the native
    Write-10 TTS release.  Values normally represent individual PCM samples.
    """

    k = _k(frames_per_unit)
    samples_per_frame = _plain_int(samples_per_frame, "samples_per_frame")
    if samples_per_frame < 1:
        raise ValueError("samples_per_frame must be positive")
    values = list(self_audio)
    unit_samples = k * samples_per_frame
    _positive_multiple(len(values), unit_samples, "self_audio")
    delay = self_audio_delay_frames(k) * samples_per_frame
    if delay >= len(values):
        return [fill] * len(values)
    return [fill] * delay + values[:-delay]


def pack_units(
    environment_audio: Sequence[Any],
    delayed_self_audio: Sequence[Any],
    text_inputs: Sequence[Any],
    frames_per_unit: int,
) -> list[Any]:
    """Serialize complete units as ``env[k] | delayed-self[k] | text[k]``."""

    k = _k(frames_per_unit)
    text = list(text_inputs)
    units = _positive_multiple(len(text), k, "text_inputs")
    expected = units * k
    environment = list(environment_audio)
    own = list(delayed_self_audio)
    if len(environment) != expected or len(own) != expected:
        raise ValueError("environment, delayed-self, and text lanes must have equal frame counts")
    packed: list[Any] = []
    for start in range(0, expected, k):
        stop = start + k
        packed.extend(environment[start:stop])
        packed.extend(own[start:stop])
        packed.extend(text[start:stop])
    return packed


def position_ids(
    unit_count: int,
    frames_per_unit: int,
    unit_offset: int = 0,
) -> tuple[int, ...]:
    unit_count = _plain_int(unit_count, "unit_count")
    unit_offset = _plain_int(unit_offset, "unit_offset")
    k = _k(frames_per_unit)
    if unit_count < 1 or unit_offset < 0:
        raise ValueError("unit_count must be positive and unit_offset non-negative")
    width = 3 * k
    start = unit_offset * width
    return tuple(range(start, start + unit_count * width))


def attention_mask(
    unit_count: int,
    frames_per_unit: int,
    memory_tokens: int = 0,
) -> tuple[tuple[bool, ...], ...]:
    """Inclusive causal visibility for the serialized current sequence."""

    unit_count = _plain_int(unit_count, "unit_count")
    memory_tokens = _plain_int(memory_tokens, "memory_tokens")
    k = _k(frames_per_unit)
    if unit_count < 1 or memory_tokens < 0:
        raise ValueError("unit_count must be positive and memory_tokens non-negative")
    current = unit_count * 3 * k
    return tuple(
        (True,) * memory_tokens + tuple(key <= query for key in range(current))
        for query in range(current)
    )


def lane_slot_indices(
    lane: str,
    unit_count: int,
    frames_per_unit: int,
    unit_offset: int = 0,
) -> tuple[int, ...]:
    if lane not in LANES:
        raise ValueError(f"lane must be one of {LANES}")
    unit_count = _plain_int(unit_count, "unit_count")
    unit_offset = _plain_int(unit_offset, "unit_offset")
    k = _k(frames_per_unit)
    if unit_count < 1 or unit_offset < 0:
        raise ValueError("unit_count must be positive and unit_offset non-negative")
    lane_offset = LANES.index(lane) * k
    return tuple(
        unit * 3 * k + lane_offset + local
        for unit in range(unit_offset, unit_offset + unit_count)
        for local in range(k)
    )


def text_slot_indices(
    unit_count: int,
    frames_per_unit: int,
    unit_offset: int = 0,
) -> tuple[int, ...]:
    return lane_slot_indices(TEXT, unit_count, frames_per_unit, unit_offset)


__all__ = [
    "ATTENTION_POLICY",
    "ENVIRONMENT_AUDIO",
    "LANES",
    "SELF_AUDIO",
    "SELF_AUDIO_DELAY_POLICY",
    "TEXT",
    "attention_mask",
    "delay_self_audio",
    "lane_slot_indices",
    "pack_units",
    "position_ids",
    "shift_text_inputs",
    "text_slot_indices",
]
