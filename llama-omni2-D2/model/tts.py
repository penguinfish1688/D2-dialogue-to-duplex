"""Native LLaMA-Omni2 Read-3/Write-10 followed by a SEP/audio-only tail.

This module is intentionally independent of Torch.  Data preparation, model
teacher forcing, and runtime all consume the same explicit slot plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, TypeVar

from d2_llama.core.constants import (
    TTS_EXTENDED_VOCAB_SIZE,
    TTS_READ_TEXT_TOKENS,
    TTS_UNIT_TOKEN_OFFSET,
    TTS_UNIT_VOCAB_SIZE,
    TTS_WRITE_SPEECH_TOKENS,
    ceil_div,
)


T = TypeVar("T")
SlotKind = Literal["condition", "separator", "speech"]


@dataclass(frozen=True, slots=True)
class ScheduleSlot:
    """One input position in the native TTS decoder sequence."""

    kind: SlotKind
    source_index: int
    block_index: int
    index_in_block: int


def raw_unit_to_tts_token_id(unit: int) -> int:
    """Map a raw CosyVoice2 FSQ code (0..6560) to the TTS vocabulary."""

    if isinstance(unit, bool) or not isinstance(unit, int):
        raise TypeError("speech unit must be an integer")
    if not 0 <= unit < TTS_UNIT_VOCAB_SIZE:
        raise ValueError(f"speech unit must be in [0,{TTS_UNIT_VOCAB_SIZE})")
    result = TTS_UNIT_TOKEN_OFFSET + unit
    if result >= TTS_EXTENDED_VOCAB_SIZE:
        raise AssertionError("speech-unit token mapping exceeds TTS vocabulary")
    return result


def tts_token_id_to_raw_unit(token_id: int) -> int:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise TypeError("TTS token ID must be an integer")
    unit = token_id - TTS_UNIT_TOKEN_OFFSET
    if not 0 <= unit < TTS_UNIT_VOCAB_SIZE:
        raise ValueError("TTS token ID is not a CosyVoice2 unit token")
    return unit


def maximum_condition_count(speech_token_count: int) -> int:
    """Maximum conditions, including text-end, before the final audio EOS.

    ``speech_token_count`` includes EOS. At least that final token must remain
    for the SEP tail after all nonfinal Read-3/Write-10 groups.
    """

    if isinstance(speech_token_count, bool) or not isinstance(speech_token_count, int):
        raise TypeError("speech_token_count must be an integer")
    if speech_token_count < 1:
        raise ValueError("speech_token_count must be positive")
    return ceil_div(speech_token_count, TTS_WRITE_SPEECH_TOKENS) * TTS_READ_TEXT_TOKENS


def build_read_write_schedule(
    condition_count: int,
    speech_token_count: int,
) -> tuple[ScheduleSlot, ...]:
    """Interleave nonfinal 3:10 groups, then 1..3 conditions, SEP, and audio.

    The last condition is the native text-end token, fused with the first D2
    PAD prediction's hidden state. EOS must be in the contiguous final audio
    tail; it cannot occur inside a nonfinal Write-10 group.
    """

    for value, name in (
        (condition_count, "condition_count"),
        (speech_token_count, "speech_token_count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if speech_token_count < 1:
        raise ValueError("speech_token_count must be positive")
    if condition_count < 1:
        raise ValueError("condition_count must include the final text-end condition")
    nonfinal_blocks = (condition_count - 1) // TTS_READ_TEXT_TOKENS
    nonfinal_speech = nonfinal_blocks * TTS_WRITE_SPEECH_TOKENS
    if speech_token_count <= nonfinal_speech:
        raise ValueError("audio EOS would occur before native text finalization")

    slots: list[ScheduleSlot] = []
    condition_index = 0
    speech_index = 0
    for block in range(nonfinal_blocks):
        for local in range(TTS_READ_TEXT_TOKENS):
            slots.append(ScheduleSlot("condition", condition_index, block, local))
            condition_index += 1
        for local in range(TTS_WRITE_SPEECH_TOKENS):
            slots.append(ScheduleSlot("speech", speech_index, block, local))
            speech_index += 1
    final_conditions = condition_count - condition_index
    for local in range(final_conditions):
        slots.append(ScheduleSlot("condition", condition_index, nonfinal_blocks, local))
        condition_index += 1
    slots.append(ScheduleSlot("separator", 0, nonfinal_blocks, final_conditions))
    for local in range(speech_token_count - speech_index):
        slots.append(ScheduleSlot("speech", speech_index, nonfinal_blocks, local))
        speech_index += 1
    if condition_index != condition_count or speech_index != speech_token_count:
        raise AssertionError("internal Read-3/Write-10 scheduling error")
    return tuple(slots)


def interleave_values(
    conditions: Sequence[T],
    speech_tokens: Sequence[T],
    *,
    separator_value: T,
) -> list[T]:
    schedule = build_read_write_schedule(len(conditions), len(speech_tokens))
    return [
        conditions[slot.source_index]
        if slot.kind == "condition"
        else separator_value
        if slot.kind == "separator"
        else speech_tokens[slot.source_index]
        for slot in schedule
    ]


def speech_label_positions(
    condition_count: int,
    speech_token_count: int,
) -> tuple[int, ...]:
    """Positions whose labels participate in standard shifted causal CE."""

    return tuple(
        index
        for index, slot in enumerate(build_read_write_schedule(condition_count, speech_token_count))
        if slot.kind == "speech"
    )


__all__ = [
    "ScheduleSlot",
    "build_read_write_schedule",
    "interleave_values",
    "raw_unit_to_tts_token_id",
    "maximum_condition_count",
    "speech_label_positions",
    "tts_token_id_to_raw_unit",
]
