"""Dependency-injected D2 runtime with atomic interruption semantics.

The concrete loader owns model calls.  This module owns causal ordering,
Read-3/Write-10 buffering, fixed-rate playback, and played-self feedback.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import sys
from typing import Any, Protocol, Sequence

from d2_llama.core.constants import (
    INPUT_SAMPLES_PER_FRAME,
    OUTPUT_SAMPLE_RATE,
    TTS_READ_TEXT_TOKENS,
    TTS_RELEASE_LATENCY_FRAMES,
    TTS_TEXT_END_TOKEN_ID,
    TTS_WRITE_SPEECH_TOKENS,
    latency_for_frames,
)


OUTPUT_SAMPLES_PER_FRAME = OUTPUT_SAMPLE_RATE // 10


def _pcm(payload: bytes | bytearray | memoryview, name: str) -> bytes:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be PCM16 bytes")
    result = bytes(payload)
    if len(result) % 2:
        raise ValueError(f"{name} contains a partial PCM16 sample")
    return result


class PCMQueue:
    """Generated-but-unplayed 24-kHz PCM."""

    def __init__(self, payload: bytes = b"") -> None:
        self._buffer = bytearray(_pcm(payload, "queued PCM"))

    @property
    def pending_samples(self) -> int:
        return len(self._buffer) // 2

    @property
    def payload(self) -> bytes:
        return bytes(self._buffer)

    def push(self, payload: bytes) -> None:
        self._buffer.extend(_pcm(payload, "rendered PCM"))

    def clear(self) -> None:
        self._buffer.clear()

    def pop_exact_or_silence(self, sample_count: int) -> bytes:
        if isinstance(sample_count, bool) or not isinstance(sample_count, int):
            raise TypeError("sample_count must be an integer")
        if sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        byte_count = 2 * sample_count
        available = min(byte_count, len(self._buffer))
        result = bytes(self._buffer[:available])
        del self._buffer[:available]
        return result + bytes(byte_count - available)


def resample_feedback_24k_to_16k(payload: bytes) -> bytes:
    """Exact-count dependency-free linear resampling for feedback framing."""

    source_bytes = _pcm(payload, "published PCM")
    source = array("h")
    source.frombytes(source_bytes)
    if sys.byteorder != "little":
        source.byteswap()
    if len(source) % 3:
        raise ValueError("24-kHz PCM sample count must be divisible by three")
    target = array("h")
    for index in range(len(source) * 2 // 3):
        numerator = 3 * index
        left = numerator // 2
        if numerator % 2 == 0 or left + 1 == len(source):
            value = int(source[left])
        else:
            value = round((int(source[left]) + int(source[left + 1])) / 2)
        target.append(max(-32768, min(32767, value)))
    if sys.byteorder != "little":
        target.byteswap()
    return target.tobytes()


@dataclass(frozen=True, slots=True)
class AudioEncoding:
    embeddings: Any
    state: Any


@dataclass(frozen=True, slots=True)
class ThinkerPrefill:
    state: Any


@dataclass(frozen=True, slots=True)
class ThinkerStep:
    token_id: int
    hidden_state: Any
    state: Any


@dataclass(frozen=True, slots=True)
class TTSWrite:
    speech_units: tuple[int, ...]
    state: Any
    finished: bool = False


@dataclass(frozen=True, slots=True)
class RenderedChunk:
    pcm_s16le_24k: bytes
    state: Any


@dataclass(frozen=True, slots=True)
class RuntimeState:
    frames_per_unit: int | None = None
    unit_index: int = 0
    environment_encoder: Any = None
    self_encoder: Any = None
    thinker: Any = None
    tts: Any = None
    renderer: Any = None
    response_open: bool = False
    text_finished: bool = False
    tts_finished: bool = False
    tts_tail_ticks: int = 0
    tts_tail_generated_tokens: int = 0
    tts_token_limit_reached: bool = False
    playback_finished: bool = False
    release_frame: int | None = None
    pending_conditions: tuple[tuple[int, Any], ...] = ()
    playback_pcm_s16le_24k: bytes = b""
    previous_played_self_s16le_16k: bytes | None = None
    previous_text_token: int | None = None


@dataclass(frozen=True, slots=True)
class RuntimeOutput:
    text_tokens: tuple[int, ...]
    played_pcm_s16le_24k: bytes
    cleared_playback_samples: int
    state: RuntimeState


class EncodeAudio(Protocol):
    def __call__(
        self, *, pcm_s16le_16k: bytes, state: Any, frames_per_unit: int, lane: str
    ) -> AudioEncoding: ...


class PrefillThinker(Protocol):
    def __call__(
        self, *, environment: Any, delayed_self: Any, state: Any, unit_index: int
    ) -> ThinkerPrefill: ...


class DecodeThinker(Protocol):
    def __call__(
        self, *, text_input: int, state: Any, unit_index: int, text_index: int
    ) -> ThinkerStep: ...


class WriteTTS(Protocol):
    def __call__(
        self,
        *,
        conditions: Sequence[tuple[int, Any]],
        count: int | None,
        final: bool,
        state: Any,
    ) -> TTSWrite:
        """Generate up to ``count`` units; ``finished`` means audio EOS.

        ``final=True`` fuses the last 1–3 conditions, ending in native text
        end, then appends SEP exactly once. Later audio-only writes pass no
        conditions and ``final=False``; final does not request audio EOS.
        """
        ...


class RenderSpeech(Protocol):
    def __call__(self, *, speech_units: Sequence[int], final: bool, state: Any) -> RenderedChunk:
        """Render incrementally; final=True must flush flow/vocoder caches."""
        ...


def run_macro_unit(
    *,
    frames_per_unit: int,
    initial_text_token: int,
    response_token_id: int,
    interrupt_token_id: int,
    pad_token_id: int,
    environment_pcm_s16le_16k: bytes,
    state: RuntimeState | None,
    encode_audio: EncodeAudio,
    prefill_thinker: PrefillThinker,
    decode_thinker: DecodeThinker,
    write_tts: WriteTTS,
    render_speech: RenderSpeech,
    max_tail_tokens: int = 1024,
) -> RuntimeOutput:
    """Execute one macro and clear every unpublished byte on INTERRUPT.

    INTERRUPTION is not represented only by asking TTS for EOS.  The pending
    3-text condition group, TTS state, renderer state, and playback FIFO are
    all invalidated before PCM is published for the interrupting text slot.
    """

    latency_for_frames(frames_per_unit)
    if (
        isinstance(max_tail_tokens, bool)
        or not isinstance(max_tail_tokens, int)
        or max_tail_tokens < 1
    ):
        raise ValueError("native TTS tail token limit must be a positive integer")
    k = frames_per_unit
    current = state or RuntimeState()
    if current.frames_per_unit not in (None, k):
        raise ValueError("runtime state belongs to another latency")
    environment_pcm = _pcm(environment_pcm_s16le_16k, "environment PCM")
    expected_input_bytes = 2 * k * INPUT_SAMPLES_PER_FRAME
    if len(environment_pcm) != expected_input_bytes:
        raise ValueError("environment PCM must contain exactly one macro")
    own_pcm = current.previous_played_self_s16le_16k or bytes(expected_input_bytes)
    if len(own_pcm) != expected_input_bytes:
        raise ValueError("played-self feedback must contain exactly one macro")

    environment = encode_audio(
        pcm_s16le_16k=environment_pcm,
        state=current.environment_encoder,
        frames_per_unit=k,
        lane="environment",
    )
    own = encode_audio(
        pcm_s16le_16k=own_pcm,
        state=current.self_encoder,
        frames_per_unit=k,
        lane="self",
    )
    prefill = prefill_thinker(
        environment=environment.embeddings,
        delayed_self=own.embeddings,
        state=current.thinker,
        unit_index=current.unit_index,
    )

    queue = PCMQueue(current.playback_pcm_s16le_24k)
    conditions = list(current.pending_conditions)
    thinker_state = prefill.state
    tts_state = current.tts
    renderer_state = current.renderer
    response_open = current.response_open
    text_finished = current.text_finished
    tts_finished = current.tts_finished
    tail_ticks = current.tts_tail_ticks
    tail_generated_tokens = current.tts_tail_generated_tokens
    token_limit_reached = current.tts_token_limit_reached
    playback_finished = current.playback_finished
    release_frame = current.release_frame
    text_input = (
        initial_text_token if current.previous_text_token is None else current.previous_text_token
    )
    tokens: list[int] = []
    played: list[bytes] = []
    cleared = 0

    for text_index in range(k):
        frame = current.unit_index * k + text_index
        decoded = decode_thinker(
            text_input=text_input,
            state=thinker_state,
            unit_index=current.unit_index,
            text_index=text_index,
        )
        token = int(decoded.token_id)
        tokens.append(token)
        text_input = token
        thinker_state = decoded.state

        if token == interrupt_token_id:
            cleared += queue.pending_samples
            queue.clear()
            conditions.clear()
            tts_state = None
            renderer_state = None
            response_open = False
            text_finished = False
            tts_finished = True
            tail_ticks = 0
            tail_generated_tokens = 0
            token_limit_reached = False
            playback_finished = False
            release_frame = None
        elif token == response_token_id:
            # A new response never inherits a partial condition block.
            cleared += queue.pending_samples
            queue.clear()
            conditions.clear()
            tts_state = None
            renderer_state = None
            response_open = True
            text_finished = False
            tts_finished = False
            tail_ticks = 0
            tail_generated_tokens = 0
            token_limit_reached = False
            playback_finished = False
            release_frame = None
        elif response_open and not tts_finished:
            if release_frame is None:
                release_frame = frame + TTS_RELEASE_LATENCY_FRAMES
            final_group = token == pad_token_id and not text_finished
            should_write = False
            if final_group:
                text_finished = True
                conditions.append((TTS_TEXT_END_TOKEN_ID, decoded.hidden_state))
                tail_ticks = 0
                should_write = True
            elif text_finished:
                if token != pad_token_id:
                    raise RuntimeError("Thinker emitted lexical text after response PAD")
                tail_ticks += 1
                should_write = tail_ticks == TTS_READ_TEXT_TOKENS
            else:
                conditions.append((token, decoded.hidden_state))
                should_write = len(conditions) == TTS_READ_TEXT_TOKENS
            if should_write:
                count = TTS_WRITE_SPEECH_TOKENS
                if text_finished:
                    count = min(count, max_tail_tokens - tail_generated_tokens)
                written = write_tts(
                    conditions=tuple(conditions),
                    count=count,
                    final=final_group,
                    state=tts_state,
                )
                if len(written.speech_units) > count:
                    raise RuntimeError("TTS emitted more units than the requested burst")
                if not written.finished and len(written.speech_units) != count:
                    raise RuntimeError("nonterminal TTS write did not emit the requested burst")
                tts_state = written.state
                conditions.clear()
                tail_ticks = 0
                tts_finished = bool(written.finished)
                if text_finished:
                    tail_generated_tokens += len(written.speech_units) + int(written.finished)
                    if not tts_finished and tail_generated_tokens >= max_tail_tokens:
                        token_limit_reached = True
                        tts_finished = True
                if written.speech_units or tts_finished:
                    rendered = render_speech(
                        speech_units=written.speech_units,
                        final=tts_finished,
                        state=renderer_state,
                    )
                    renderer_state = rendered.state
                    queue.push(rendered.pcm_s16le_24k)
                if tts_finished:
                    text_finished = True

        # Clearing occurs before this pop, so the interrupting clock cannot
        # publish or feed back audio predicted for the cancelled response.
        if release_frame is not None and frame >= release_frame:
            played.append(queue.pop_exact_or_silence(OUTPUT_SAMPLES_PER_FRAME))
        else:
            played.append(bytes(2 * OUTPUT_SAMPLES_PER_FRAME))
        if tts_finished and response_open and not queue.pending_samples:
            response_open = False
            playback_finished = True

    played_pcm = b"".join(played)
    feedback = resample_feedback_24k_to_16k(played_pcm)
    if len(feedback) != expected_input_bytes:
        raise AssertionError("feedback resampler changed the D2 macro duration")
    next_state = RuntimeState(
        frames_per_unit=k,
        unit_index=current.unit_index + 1,
        environment_encoder=environment.state,
        self_encoder=own.state,
        thinker=thinker_state,
        tts=tts_state,
        renderer=renderer_state,
        response_open=response_open,
        text_finished=text_finished,
        tts_finished=tts_finished,
        tts_tail_ticks=tail_ticks,
        tts_tail_generated_tokens=tail_generated_tokens,
        tts_token_limit_reached=token_limit_reached,
        playback_finished=playback_finished,
        release_frame=release_frame,
        pending_conditions=tuple(conditions),
        playback_pcm_s16le_24k=queue.payload,
        previous_played_self_s16le_16k=feedback,
        previous_text_token=tokens[-1],
    )
    return RuntimeOutput(tuple(tokens), played_pcm, cleared, next_state)


__all__ = [
    "AudioEncoding",
    "PCMQueue",
    "RenderedChunk",
    "RuntimeOutput",
    "RuntimeState",
    "TTSWrite",
    "ThinkerPrefill",
    "ThinkerStep",
    "resample_feedback_24k_to_16k",
    "run_macro_unit",
]
