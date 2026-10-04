from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ResponseTransition:
    before: bool
    use_text_context: bool
    emit_audio: bool
    after: bool
    talker_before: bool | None = None
    talker_after: bool | None = None


def response_transition(
    *,
    responding: bool,
    text_done: bool,
    text_id: int,
    pad_id: int,
    assistant_start_id: int,
    assistant_end_id: int,
    talker_responding: bool | None = None,
) -> tuple[ResponseTransition, bool]:
    before = bool(responding)
    token = int(text_id)
    talker_before = bool(before if talker_responding is None else talker_responding)
    response_now = token == int(assistant_start_id)
    interrupt_now = token == int(assistant_end_id)
    if response_now and before:
        raise ValueError("Nested RESPONSE during async inference")
    after = (before or response_now) and not interrupt_now
    talker_after = (talker_before or response_now) and not interrupt_now
    text_done_after = bool(text_done)
    if response_now:
        text_done_after = False
    elif before and token == int(pad_id):
        text_done_after = True
    if not after:
        text_done_after = True
    return (
        ResponseTransition(
            before=before,
            use_text_context=talker_before or response_now,
            emit_audio=talker_before and not interrupt_now,
            after=after,
            talker_before=talker_before,
            talker_after=talker_after,
        ),
        text_done_after,
    )
