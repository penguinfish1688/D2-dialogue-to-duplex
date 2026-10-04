from __future__ import annotations
from dataclasses import dataclass
from .config import frames_for_latency, latency_for_frames

NATIVE_MEL_FRAMES = 8


MACRO_CONV_WINDOW_MEL_FRAMES = 104


OFFICIAL_ATTENTION_TOKENS = 104


MACRO_AUT_RUNTIME_FORMAT = "d2.encoder.v1"


MACRO_AUT_FRONTEND_POLICY = "macro_window_last_k"


MACRO_AUT_FRONTEND_POLICY_VERSION = 1


def _plain_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class MacroAuTContract:
    """Latency and bounded-context parameters for one D2 AuT variant."""

    frames_per_unit: int
    attention_window_tokens: int = OFFICIAL_ATTENTION_TOKENS
    mel_frames_per_token: int = NATIVE_MEL_FRAMES
    conv_window_mel_frames: int = MACRO_CONV_WINDOW_MEL_FRAMES

    def __post_init__(self) -> None:
        latency_for_frames(self.frames_per_unit)
        for name in ("attention_window_tokens", "mel_frames_per_token", "conv_window_mel_frames"):
            value = _plain_int(getattr(self, name), name)
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.frames_per_unit > self.attention_window_tokens:
            raise ValueError("a macro unit cannot exceed the Transformer attention window")
        if self.mel_frames_per_token != NATIVE_MEL_FRAMES:
            raise ValueError(
                f"D2 keeps the 12.5 Hz clock: mel_frames_per_token must be {NATIVE_MEL_FRAMES}"
            )
        if self.conv_window_mel_frames != MACRO_CONV_WINDOW_MEL_FRAMES:
            raise ValueError(
                f"D2 uses the 104-mel macro frontend window, got {self.conv_window_mel_frames}"
            )

    @classmethod
    def from_latency(
        cls, latency_ms: int, *, attention_window_tokens: int = OFFICIAL_ATTENTION_TOKENS
    ) -> "MacroAuTContract":
        return cls(
            frames_per_unit=frames_for_latency(latency_ms),
            attention_window_tokens=attention_window_tokens,
        )

    @property
    def latency_ms(self) -> int:
        return latency_for_frames(self.frames_per_unit)

    @property
    def macro_mel_frames(self) -> int:
        return self.frames_per_unit * self.mel_frames_per_token

    @property
    def max_past_tokens(self) -> int:
        """Maximum KV tokens retained before processing the next unit."""
        return self.attention_window_tokens - self.frames_per_unit

    def as_dict(self) -> dict[str, int | str]:
        return {
            "format": MACRO_AUT_RUNTIME_FORMAT,
            "frontend_policy": MACRO_AUT_FRONTEND_POLICY,
            "frontend_policy_version": MACRO_AUT_FRONTEND_POLICY_VERSION,
            "latency_ms": self.latency_ms,
            "frames_per_unit": self.frames_per_unit,
            "native_frame_ms": 80,
            "mel_frames_per_token": self.mel_frames_per_token,
            "macro_mel_frames": self.macro_mel_frames,
            "conv_window_mel_frames": self.conv_window_mel_frames,
            "attention_window_tokens": self.attention_window_tokens,
            "max_past_tokens": self.max_past_tokens,
        }
