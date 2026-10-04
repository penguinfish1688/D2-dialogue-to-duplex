"""Shared macro-unit inference *layout* loop for every D2 latency.

The runner owns ordering and recurrent bookkeeping, while callbacks own Qwen
sampling and cache objects.  This keeps the research contract testable without
loading the 30B model and leaves sampling policy outside the data layout.

This module deliberately contains only the small ordering primitive.  The
executable adapter in :mod:`d2_qwen.inference.runtime` supplies the official
response transitions, layer-24/Talker context, codec BOS/EOS handling, and
16-codebook sampling through these callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from d2_qwen.model.config import latency_for_frames


@dataclass(frozen=True)
class MacroInferenceState:
    """State carried between complete D2 macro units."""

    thinker_kv: Any = None
    talker_kv: Any = None
    previous_text: Any = None
    self_audio_history: Any = None
    unit_index: int = 0
    previous_codec_frame: Any = None
    frames_per_unit: int | None = None


@dataclass(frozen=True)
class AudioPrefillResult:
    thinker_kv: Any
    next_self_audio_history: Any


@dataclass(frozen=True)
class TextStepResult:
    generated_text: Any
    thinker_kv: Any
    talker_context: Any


@dataclass(frozen=True)
class TalkerStepResult:
    codec_frame: Any
    talker_kv: Any
    # The physical/logical frame emitted now is not always the shifted Talker
    # input for the next clock.  INTERRUPT and spontaneous codec EOS close the
    # response, so the official policy resets that recurrence to silence.
    next_codec_input: Any


@dataclass(frozen=True)
class MacroUnitOutput:
    text_tokens: tuple[Any, ...]
    codec_frames: tuple[Any, ...]
    state: MacroInferenceState


class AudioPrefillCallback(Protocol):
    def __call__(
        self,
        *,
        environment_audio: Any,
        self_audio: Any,
        previous_self_audio: Any,
        unit_index: int,
        thinker_kv: Any,
    ) -> AudioPrefillResult: ...


class TextStepCallback(Protocol):
    def __call__(
        self,
        *,
        text_index: int,
        text_input: Any,
        unit_index: int,
        thinker_kv: Any,
    ) -> TextStepResult: ...


class TalkerStepCallback(Protocol):
    def __call__(
        self,
        *,
        frame_index: int,
        generated_text: Any,
        talker_context: Any,
        codec_input: Any,
        unit_index: int,
        talker_kv: Any,
    ) -> TalkerStepResult: ...


def run_macro_unit(
    *,
    frames_per_unit: int,
    initial_text: Any,
    environment_audio: Any,
    self_audio: Any,
    state: MacroInferenceState | None,
    audio_prefill: AudioPrefillCallback,
    text_step: TextStepCallback,
    talker_step: TalkerStepCallback,
) -> MacroUnitOutput:
    """Generate one complete macro unit with a fixed, latency-independent loop.

    Call order is exactly ``audio``, then ``text[j]``, ``talker[j]`` for
    ``j=0..k-1``.  The audio callback is responsible for using the model's
    serialized-causal ``2k`` prefill; text and Talker callbacks advance their
    own KV objects one native frame at a time.  The first text input is the
    configured initial token (D2 uses PAD) for a new stream or the previous
    unit's final generated token, and every later input is the preceding
    generated token.  Likewise, ``codec_input`` is the
    preceding complete codec input (``None`` only for the first stream frame,
    where the callback supplies Qwen's silence/BOS frame).  A callback may
    override that recurrence through ``next_codec_input``; the deployed policy
    uses silence after INTERRUPT or generated codec EOS.  Keeping that value in
    :class:`MacroInferenceState` makes a resumed stream self-contained; Talker
    KV alone does not contain the shifted input for the next call.
    """

    latency_for_frames(frames_per_unit)  # authoritative {1,2,4,8,13} check
    state = state or MacroInferenceState()
    if isinstance(state.unit_index, bool) or not isinstance(state.unit_index, int):
        raise TypeError("unit_index must be an integer")
    if state.unit_index < 0:
        raise ValueError("unit_index must be non-negative")
    if state.frames_per_unit is not None:
        latency_for_frames(state.frames_per_unit)
        if state.frames_per_unit != frames_per_unit:
            raise ValueError(
                "inference state belongs to a different macro-unit size: "
                f"{state.frames_per_unit} != {frames_per_unit}"
            )
    elif state.unit_index or any(
        value is not None
        for value in (
            state.thinker_kv,
            state.talker_kv,
            state.previous_text,
            state.self_audio_history,
            state.previous_codec_frame,
        )
    ):
        raise ValueError("non-empty inference state must record frames_per_unit")

    prefill = audio_prefill(
        environment_audio=environment_audio,
        self_audio=self_audio,
        previous_self_audio=state.self_audio_history,
        unit_index=int(state.unit_index),
        thinker_kv=state.thinker_kv,
    )
    thinker_kv = prefill.thinker_kv
    talker_kv = state.talker_kv
    text_input = initial_text if state.previous_text is None else state.previous_text
    codec_input = state.previous_codec_frame
    generated: list[Any] = []
    codec: list[Any] = []
    for index in range(frames_per_unit):
        text_result = text_step(
            text_index=index,
            text_input=text_input,
            unit_index=int(state.unit_index),
            thinker_kv=thinker_kv,
        )
        thinker_kv = text_result.thinker_kv
        generated.append(text_result.generated_text)
        text_input = text_result.generated_text

        talker_result = talker_step(
            frame_index=index,
            generated_text=text_result.generated_text,
            talker_context=text_result.talker_context,
            codec_input=codec_input,
            unit_index=int(state.unit_index),
            talker_kv=talker_kv,
        )
        talker_kv = talker_result.talker_kv
        codec.append(talker_result.codec_frame)
        codec_input = talker_result.next_codec_input

    next_state = MacroInferenceState(
        thinker_kv=thinker_kv,
        talker_kv=talker_kv,
        previous_text=generated[-1],
        self_audio_history=prefill.next_self_audio_history,
        unit_index=int(state.unit_index) + 1,
        previous_codec_frame=codec_input,
        frames_per_unit=frames_per_unit,
    )
    return MacroUnitOutput(
        text_tokens=tuple(generated),
        codec_frames=tuple(codec),
        state=next_state,
    )


__all__ = [
    "AudioPrefillCallback",
    "AudioPrefillResult",
    "MacroInferenceState",
    "MacroUnitOutput",
    "TalkerStepCallback",
    "TalkerStepResult",
    "TextStepCallback",
    "TextStepResult",
    "run_macro_unit",
]
