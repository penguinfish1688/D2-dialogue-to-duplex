"""One timing contract for every LLaMA-Omni2 D2 latency.

The released model uses Whisper-large-v3 followed by a five-row adaptor, so
the D2 Thinker clock is 10 Hz.  Its native speech generator is deliberately
kept at Read-3/Write-10 and its CosyVoice2 units remain at 25 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


NATIVE_FRAME_MS = 100
NATIVE_AUDIO_HZ = 10
INPUT_SAMPLE_RATE = 16_000
INPUT_SAMPLES_PER_FRAME = 1_600

MEL_CLOCK_HZ = 100
MEL_FRAMES_PER_FRAME = 10
WHISPER_CLOCK_HZ = 50
WHISPER_ROWS_PER_FRAME = 5
WHISPER_WIDTH = 1_280
WHISPER_MAX_ROWS = 1_500
ADAPTOR_GROUP_ROWS = 5
ADAPTOR_HIDDEN_SIZE = 2_048
THINKER_HIDDEN_SIZE = 3_584
THINKER_BASE_VOCAB_SIZE = 152_064

TTS_READ_TEXT_TOKENS = 3
TTS_WRITE_SPEECH_TOKENS = 10
TTS_HIDDEN_SIZE = 896
TTS_UNIT_VOCAB_SIZE = 6_561
TTS_TEXT_VOCAB_SIZE = 151_666
TTS_PAD_TOKEN_ID = 151_643
TTS_EOS_TOKEN_ID = 151_643
TTS_AUDIO_EOS_LOSS_WEIGHT = 3.0
STAGE3_TRAINABILITY_CONTRACT = "frozen_tts_bridge_attention_mlp_lora_v1"
TTS_BRIDGE_TRAINABILITY = "frozen_released_input_proj_and_gate"
TTS_LORA_RANK = 128
TTS_LORA_ALPHA = 256
TTS_LORA_DROPOUT = 0.0
TTS_LORA_TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
TTS_TEXT_END_TOKEN_ID = 151_645
TTS_SEPARATOR_TOKEN_ID = 151_665
TTS_UNIT_TOKEN_OFFSET = 151_666
TTS_EXTENDED_VOCAB_SIZE = 158_227
TTS_SPEECH_TOKEN_HZ = 25
TTS_SPEECH_TOKEN_MS = 40
TTS_SPEECH_SAMPLES_PER_TOKEN = INPUT_SAMPLE_RATE // TTS_SPEECH_TOKEN_HZ
TTS_AUDIO_START_BOUNDARY = "clean_rms_minus60dbfs_first_25hz_audible_unit_as_audio_slot_zero"
TTS_TARGET_CONTRACT = "native_sep_audio_tail_v1"
TTS_TARGET_LAYOUT = "native_read3_write10_sep_audio_tail"
TTS_TEXT_END_CONDITION = "first_duplex_pad_prediction_hidden_plus_native_text_end"
SELF_AUDIO_BOUNDARY = "clean_rms_minus60dbfs_first_through_last_25hz_audible_units"
DUPLEX_CONTROL_DELAY_FRAMES = 4
DUPLEX_CONTROL_BOUNDARY = "online_clean_user_rms_first_last_native_frames_plus4_v2"
LEGACY_DUPLEX_CONTROL_BOUNDARY = "clean_user_rms_first_last_100ms_frames_plus8_v1"
DUPLEX_SOURCE_SELECTION = "manifest_accepted_rows_first_complete_human_gpt_audio_pair_v1"
TTS_MEL_HZ = 50
TTS_TOKEN_MEL_RATIO = 2
TTS_FLOW_PRELOOKAHEAD_TOKENS = 3
TTS_HIFT_CACHE_MEL_FRAMES = 8
OUTPUT_SAMPLE_RATE = 24_000

# The official scheduler releases speech only after a complete Write-10 block.
# Ten 25-Hz units cover 400 ms.  This is the deterministic scheduling delay
# used by offline D2 alignment; measured GPU execution time is intentionally
# not baked into a checkpoint contract.
TTS_RELEASE_LATENCY_MS = TTS_WRITE_SPEECH_TOKENS * 1_000 // TTS_SPEECH_TOKEN_HZ
TTS_RELEASE_LATENCY_FRAMES = TTS_RELEASE_LATENCY_MS // NATIVE_FRAME_MS

# These two official renderer lookaheads live inside the Write-10 release
# block.  They are recorded for validation and documentation, not added again.
TTS_FLOW_PRELOOKAHEAD_MS = TTS_FLOW_PRELOOKAHEAD_TOKENS * 1_000 // TTS_SPEECH_TOKEN_HZ
TTS_HIFT_CACHE_MS = TTS_HIFT_CACHE_MEL_FRAMES * 1_000 // TTS_MEL_HZ
TTS_RENDERER_INTERNAL_HOLD_MS = TTS_FLOW_PRELOOKAHEAD_MS + TTS_HIFT_CACHE_MS

ALLOWED_MACRO_FRAMES = (1, 2, 4, 8, 13)
ALLOWED_LATENCIES_MS = tuple(k * NATIVE_FRAME_MS for k in ALLOWED_MACRO_FRAMES)
PRODUCTION_MACRO_FRAMES = (1, 2, 4, 8)
PRODUCTION_LATENCIES_MS = tuple(k * NATIVE_FRAME_MS for k in PRODUCTION_MACRO_FRAMES)
DEFAULT_XL_CHUNK_TARGET_MS = 90_000


def _plain_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def ceil_div(numerator: int, denominator: int) -> int:
    numerator = _plain_int(numerator, "numerator")
    denominator = _plain_int(denominator, "denominator")
    if numerator < 0 or denominator <= 0:
        raise ValueError("ceil_div requires numerator >= 0 and denominator > 0")
    return (numerator + denominator - 1) // denominator


def frames_for_latency(latency_ms: int) -> int:
    latency_ms = _plain_int(latency_ms, "latency_ms")
    if latency_ms not in ALLOWED_LATENCIES_MS:
        raise ValueError(f"latency_ms must be one of {ALLOWED_LATENCIES_MS}, got {latency_ms}")
    return latency_ms // NATIVE_FRAME_MS


def latency_for_frames(frames_per_unit: int) -> int:
    frames_per_unit = _plain_int(frames_per_unit, "frames_per_unit")
    if frames_per_unit not in ALLOWED_MACRO_FRAMES:
        raise ValueError(
            f"frames_per_unit must be one of {ALLOWED_MACRO_FRAMES}, got {frames_per_unit}"
        )
    return frames_per_unit * NATIVE_FRAME_MS


def align_chunk_frames(frame_count: int, frames_per_unit: int) -> int:
    frame_count = _plain_int(frame_count, "frame_count")
    frames_per_unit = _plain_int(frames_per_unit, "frames_per_unit")
    latency_for_frames(frames_per_unit)
    if frame_count < frames_per_unit:
        raise ValueError("chunk is shorter than one complete macro unit")
    return frame_count - frame_count % frames_per_unit


def xl_chunk_frames(
    frames_per_unit: int,
    target_ms: int = DEFAULT_XL_CHUNK_TARGET_MS,
) -> int:
    target_ms = _plain_int(target_ms, "target_ms")
    if target_ms < NATIVE_FRAME_MS:
        raise ValueError(f"target_ms must be at least {NATIVE_FRAME_MS}")
    return align_chunk_frames(target_ms // NATIVE_FRAME_MS, frames_per_unit)


def tts_release_latency_frames() -> int:
    """Return the fixed Write-10 release delay on the 100-ms D2 clock."""

    return ceil_div(TTS_RELEASE_LATENCY_MS, NATIVE_FRAME_MS)


def self_audio_delay_frames(frames_per_unit: int) -> int:
    """Offline right shift = one Thinker macro + one TTS release block."""

    latency_for_frames(frames_per_unit)
    return frames_per_unit + tts_release_latency_frames()


@dataclass(frozen=True, slots=True)
class D2Config:
    """Validated latency-specific D2 geometry."""

    latency_ms: int = NATIVE_FRAME_MS
    xl_chunk_target_ms: int = DEFAULT_XL_CHUNK_TARGET_MS

    def __post_init__(self) -> None:
        frames_for_latency(self.latency_ms)
        _plain_int(self.xl_chunk_target_ms, "xl_chunk_target_ms")
        xl_chunk_frames(self.frames_per_unit, self.xl_chunk_target_ms)

    @property
    def frames_per_unit(self) -> int:
        return frames_for_latency(self.latency_ms)

    @property
    def tokens_per_unit(self) -> int:
        return 3 * self.frames_per_unit

    @property
    def mel_frames_per_unit(self) -> int:
        return MEL_FRAMES_PER_FRAME * self.frames_per_unit

    @property
    def whisper_rows_per_unit(self) -> int:
        return WHISPER_ROWS_PER_FRAME * self.frames_per_unit

    @property
    def self_audio_delay_frames(self) -> int:
        return self_audio_delay_frames(self.frames_per_unit)

    @property
    def self_audio_delay_samples(self) -> int:
        return self.self_audio_delay_frames * INPUT_SAMPLES_PER_FRAME

    @property
    def self_audio_delay_ms(self) -> int:
        return self.self_audio_delay_frames * NATIVE_FRAME_MS

    @property
    def thinker_chunk_latency_ms(self) -> int:
        return self.latency_ms

    @property
    def tts_release_latency_ms(self) -> int:
        return TTS_RELEASE_LATENCY_MS

    @property
    def xl_chunk_frames(self) -> int:
        return xl_chunk_frames(self.frames_per_unit, self.xl_chunk_target_ms)

    @property
    def xl_chunk_units(self) -> int:
        return self.xl_chunk_frames // self.frames_per_unit

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "D2Config":
        if not isinstance(values, Mapping):
            raise TypeError("D2 config must be a mapping")
        unknown = set(values) - {"latency_ms", "xl_chunk_target_ms"}
        if unknown:
            raise ValueError(f"unknown D2 config fields: {sorted(unknown)}")
        return cls(**dict(values))

    def as_dict(self) -> dict[str, int]:
        return {
            "latency_ms": self.latency_ms,
            "xl_chunk_target_ms": self.xl_chunk_target_ms,
        }


assert TTS_RELEASE_LATENCY_MS == 400
assert TTS_RELEASE_LATENCY_FRAMES == 4
assert TTS_RENDERER_INTERNAL_HOLD_MS == 280
assert TTS_UNIT_TOKEN_OFFSET + TTS_UNIT_VOCAB_SIZE == TTS_EXTENDED_VOCAB_SIZE
assert ADAPTOR_GROUP_ROWS == WHISPER_ROWS_PER_FRAME


__all__ = [name for name in globals() if name.isupper()] + [
    "D2Config",
    "align_chunk_frames",
    "ceil_div",
    "frames_for_latency",
    "latency_for_frames",
    "self_audio_delay_frames",
    "tts_release_latency_frames",
    "xl_chunk_frames",
]
