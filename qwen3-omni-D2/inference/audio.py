from __future__ import annotations
from fractions import Fraction
import torch
from d2_qwen.model.constants import AUDIO_HZ, CODEC_HZ

OMNI_CODE2WAV_CAUSAL_TAIL_SAMPLES_24K = 555


GLOBAL_CODEC_DELAY_FRAMES = 1


GLOBAL_CODEC_DELAY_SAMPLES_16K = round(16000 * GLOBAL_CODEC_DELAY_FRAMES / AUDIO_HZ)


def _round_positive_fraction(value: Fraction) -> int:
    if value < 0:
        raise ValueError(f"Expected a non-negative value, got {value}")
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def mapped_playback_latency_samples(
    raw_samples: int,
    *,
    output_sample_rate: int,
    codec_sample_rate: int = 24000,
    codec_hz: float = 12.5,
    duplex_hz: float = AUDIO_HZ,
) -> int:
    """Output samples before the fixed clock map can touch unpadded audio."""
    raw_samples = int(raw_samples)
    if raw_samples < 0 or int(output_sample_rate) < 1:
        raise ValueError(
            f"Invalid playback latency inputs: raw_samples={raw_samples}, "
            f"output_sample_rate={output_sample_rate}"
        )
    if int(output_sample_rate) == int(codec_sample_rate) and Fraction(str(codec_hz)) == Fraction(
        str(duplex_hz)
    ):
        return raw_samples
    source_step = (
        Fraction(int(codec_sample_rate), int(output_sample_rate))
        * Fraction(str(duplex_hz))
        / Fraction(str(codec_hz))
    )
    if raw_samples == 0:
        return 0
    numerator = raw_samples * int(source_step.denominator)
    denominator = int(source_step.numerator)
    # The resampler uses source position ``i*step - 1``.  A padded source
    # sample at index P first contributes when i*step > P (strictly), because
    # at equality it is the right interpolation endpoint with zero weight.
    return numerator // denominator + 1


OMNI_CODE2WAV_PLAYBACK_DELAY_SAMPLES_16K = mapped_playback_latency_samples(
    OMNI_CODE2WAV_CAUSAL_TAIL_SAMPLES_24K,
    output_sample_rate=16000,
)


SELF_AUDIO_PLAYBACK_DELAY_SAMPLES_16K = (
    OMNI_CODE2WAV_PLAYBACK_DELAY_SAMPLES_16K + GLOBAL_CODEC_DELAY_SAMPLES_16K
)


def code2wav_output_samples(
    code_frames: int,
    *,
    upsampling_ratios: tuple[int, ...] = (2, 2),
    upsample_rates: tuple[int, ...] = (8, 5, 4, 3),
) -> int:
    """Exact released Code2Wav output length for a causal code prefix."""
    samples = int(code_frames)
    if samples < 1:
        return 0
    for factor in upsampling_ratios:
        samples *= int(factor)
    for factor in upsample_rates:
        samples = int(factor) * (samples - 1)
    return samples


def apply_self_playback_delay(
    wav_16k: torch.Tensor,
    *,
    delay_samples: int = SELF_AUDIO_PLAYBACK_DELAY_SAMPLES_16K,
) -> torch.Tensor:
    """Apply one global fixed-length causal playback delay to each waveform."""
    delay_samples = int(delay_samples)
    if delay_samples < 0:
        raise ValueError(f"delay_samples must be non-negative, got {delay_samples}")
    if wav_16k.ndim < 1:
        raise ValueError(f"Expected waveform with a sample axis, got {tuple(wav_16k.shape)}")
    if delay_samples == 0:
        return wav_16k.clone()
    delayed = torch.zeros_like(wav_16k)
    if delay_samples < int(wav_16k.shape[-1]):
        delayed[..., delay_samples:] = wav_16k[..., :-delay_samples]
    return delayed.contiguous()


def apply_delayed_frame_activity_mask(
    wav: torch.Tensor,
    frame_activity: torch.Tensor | list[bool] | tuple[bool, ...],
    *,
    output_sample_rate: int,
    playback_tail_samples_24k: int = OMNI_CODE2WAV_CAUSAL_TAIL_SAMPLES_24K,
    audio_hz: float = AUDIO_HZ,
) -> torch.Tensor:
    """Hard-zero Code2Wav outside the causally delayed speech timeline."""
    values = wav.detach().float()
    squeeze = values.ndim == 1
    if squeeze:
        values = values.unsqueeze(0)
    if values.ndim != 2:
        raise ValueError(f"Expected waveform [N] or [B,N], got {tuple(wav.shape)}")
    activity = torch.as_tensor(frame_activity, dtype=torch.bool, device=values.device)
    if activity.ndim == 1:
        activity = activity.unsqueeze(0)
    if activity.ndim != 2 or int(activity.shape[0]) != int(values.shape[0]):
        raise ValueError(
            f"Activity shape {tuple(activity.shape)} does not match waveform batch "
            f"{tuple(values.shape)}"
        )
    frames = int(activity.shape[1])
    sample_rate = int(output_sample_rate)
    expected = int(round(frames / float(audio_hz) * sample_rate))
    if int(values.shape[1]) != expected:
        raise ValueError(
            f"Activity timeline has {expected} samples for {frames} frames, "
            f"waveform has {values.shape[1]}"
        )
    base = torch.zeros_like(values, dtype=torch.bool)
    for frame_index in range(frames):
        start = int(round(frame_index / float(audio_hz) * sample_rate))
        end = int(round((frame_index + 1) / float(audio_hz) * sample_rate))
        base[:, start:end] = activity[:, frame_index : frame_index + 1]
    delay = mapped_playback_latency_samples(
        int(playback_tail_samples_24k),
        output_sample_rate=sample_rate,
    )
    delayed = torch.zeros_like(base)
    if delay < int(base.shape[1]):
        delayed[:, delay:] = base[:, :-delay] if delay else base
    masked = (values * delayed.to(dtype=values.dtype)).contiguous()
    return masked.squeeze(0) if squeeze else masked


def align_codec_wav_to_duplex_clock(
    wav: torch.Tensor,
    *,
    code_frames: int,
    output_sample_rate: int,
    codec_sample_rate: int = 24000,
    codec_hz: float = CODEC_HZ,
    duplex_hz: float = AUDIO_HZ,
) -> torch.Tensor:
    """Causally convert native-clock codec audio to an output sample rate.

    Duplex and Mimi normally share the same 12.5 Hz clock, so this performs no
    clock rescaling. A fixed, prefix-stable grid is still required when the
    codec waveform is converted from 24 kHz to (for example) 16 kHz feedback.
    """
    wav = wav.detach().float().flatten().contiguous()
    code_frames = int(code_frames)
    output_sample_rate = int(output_sample_rate)
    codec_sample_rate = int(codec_sample_rate)
    if code_frames < 0:
        raise ValueError(f"code_frames must be non-negative, got {code_frames}")
    if code_frames == 0:
        return wav.new_empty((0,))
    if wav.numel() < 2:
        raise ValueError(f"Codec waveform is too short to resample: {wav.numel()} samples")
    if output_sample_rate < 1 or codec_sample_rate < 1 or codec_hz <= 0.0 or duplex_hz <= 0.0:
        raise ValueError(
            "Invalid codec/Duplex clocks: "
            f"output_sample_rate={output_sample_rate} codec_sample_rate={codec_sample_rate} "
            f"codec_hz={codec_hz} duplex_hz={duplex_hz}"
        )

    codec_clock = Fraction(str(codec_hz))
    duplex_clock = Fraction(str(duplex_hz))
    target_length = _round_positive_fraction(
        Fraction(code_frames * output_sample_rate, 1) / duplex_clock
    )
    if output_sample_rate == codec_sample_rate and codec_clock == duplex_clock:
        if int(wav.numel()) < target_length:
            raise ValueError(f"Codec waveform has {wav.numel()} samples; need {target_length}")
        return wav[:target_length].clone().contiguous()
    # Raw codec samples consumed per output sample on the shared frame clock.
    source_step = Fraction(codec_sample_rate, output_sample_rate) * duplex_clock / codec_clock
    step_num = int(source_step.numerator)
    step_den = int(source_step.denominator)

    output_indices = torch.arange(target_length, device=wav.device, dtype=torch.long)
    # Subtract one raw sample so the right interpolation endpoint is finalized
    # even at a rounded-up logical frame boundary.
    source_numerators = (output_indices * step_num - step_den).clamp_min_(0)
    left = torch.div(source_numerators, step_den, rounding_mode="floor")
    remainder = torch.remainder(source_numerators, step_den)
    right = left + 1
    if int(right.max().item()) >= int(wav.numel()):
        raise ValueError(
            "Codec waveform does not contain enough finalized samples for the requested Duplex boundary: "
            f"codes={code_frames} raw_samples={wav.numel()} target_samples={target_length} "
            f"last_source_index={int(right.max().item())}"
        )
    weight = remainder.to(dtype=wav.dtype) / float(step_den)
    return (wav[left] * (1.0 - weight) + wav[right] * weight).contiguous()
