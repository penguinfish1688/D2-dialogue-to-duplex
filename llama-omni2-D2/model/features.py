"""Strictly causal Whisper-large-v3 log-mel features for Stage 1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import torch

from d2_llama.core.constants import INPUT_SAMPLE_RATE, MEL_CLOCK_HZ


WHISPER_N_FFT = 400
WHISPER_HOP_SAMPLES = INPUT_SAMPLE_RATE // MEL_CLOCK_HZ
WHISPER_MEL_BINS = 128
CAUSAL_STFT_LEFT_CONTEXT = WHISPER_N_FFT - WHISPER_HOP_SAMPLES
CAUSAL_FEATURE_FORMAT = "d2.llama_omni2.causal_whisper_features.v1"


@dataclass
class CausalWhisperFeatureState:
    """Minimal state needed to reproduce full-prefix causal log-mels."""

    waveform_tail: torch.Tensor | None = None
    running_logmax: torch.Tensor | None = None


def _mono_float(value: torch.Tensor | Any) -> torch.Tensor:
    waveform = value if torch.is_tensor(value) else torch.as_tensor(value)
    if waveform.ndim == 2 and int(waveform.shape[0]) == 1:
        waveform = waveform[0]
    if waveform.ndim != 1 or not int(waveform.numel()):
        raise ValueError("causal Whisper features require non-empty mono audio")
    return waveform.detach().to(device="cpu", dtype=torch.float32).contiguous()


class CausalWhisperFeatureExtractor:
    """Emit one past-only mel row for each complete 10-ms waveform hop.

    The 400-sample analysis window is shifted into the past by left-padding
    240 samples and using ``center=False``.  Dynamic-range normalization uses
    a running maximum, so an earlier row never depends on a later sample.
    """

    def __init__(self, mel_filters: torch.Tensor | None = None) -> None:
        if mel_filters is None:
            try:
                from whisper.audio import mel_filters as load_filters
            except ImportError as error:
                raise RuntimeError("openai-whisper is required for Stage 1") from error
            mel_filters = load_filters("cpu", WHISPER_MEL_BINS)
        filters = torch.as_tensor(mel_filters).detach().float().cpu().contiguous()
        expected = (WHISPER_MEL_BINS, WHISPER_N_FFT // 2 + 1)
        if tuple(filters.shape) != expected:
            raise ValueError(f"Whisper mel filters must have shape {expected}")
        if not bool(torch.isfinite(filters).all()) or bool(filters.lt(0).any()):
            raise ValueError("Whisper mel filters must be finite and non-negative")
        self.mel_filters = filters

    @property
    def contract(self) -> dict[str, Any]:
        raw = self.mel_filters.view(torch.uint8).numpy().tobytes()
        return {
            "format": CAUSAL_FEATURE_FORMAT,
            "sample_rate_hz": INPUT_SAMPLE_RATE,
            "mel_clock_hz": MEL_CLOCK_HZ,
            "n_fft": WHISPER_N_FFT,
            "hop_samples": WHISPER_HOP_SAMPLES,
            "mel_bins": WHISPER_MEL_BINS,
            "center": False,
            "left_context_samples": CAUSAL_STFT_LEFT_CONTEXT,
            "normalization": "causal_running_logmax_v1",
            "mel_filter_sha256": hashlib.sha256(raw).hexdigest(),
        }

    def extract(self, waveform: torch.Tensor | Any) -> torch.Tensor:
        samples = _mono_float(waveform)
        complete = int(samples.numel()) // WHISPER_HOP_SAMPLES * WHISPER_HOP_SAMPLES
        if complete < WHISPER_HOP_SAMPLES:
            raise ValueError("waveform is shorter than one complete mel hop")
        samples = samples[:complete]
        analysis = torch.cat((torch.zeros(CAUSAL_STFT_LEFT_CONTEXT), samples), dim=0)
        spectrum = torch.stft(
            analysis,
            n_fft=WHISPER_N_FFT,
            hop_length=WHISPER_HOP_SAMPLES,
            win_length=WHISPER_N_FFT,
            window=torch.hann_window(WHISPER_N_FFT, periodic=True),
            center=False,
            return_complex=True,
        )
        magnitudes = spectrum.abs().square()
        log_mel = (self.mel_filters @ magnitudes).clamp_min(1.0e-10).log10()
        running_max = log_mel.amax(dim=0).cummax(dim=0).values
        log_mel = torch.maximum(log_mel, running_max.unsqueeze(0) - 8.0)
        result = (log_mel + 4.0) / 4.0
        expected_frames = complete // WHISPER_HOP_SAMPLES
        if tuple(result.shape) != (WHISPER_MEL_BINS, expected_frames):
            raise RuntimeError("causal Whisper feature clock changed")
        return result.unsqueeze(0)

    def extract_streaming(
        self,
        waveform: torch.Tensor | Any,
        state: CausalWhisperFeatureState | None = None,
    ) -> tuple[torch.Tensor, CausalWhisperFeatureState]:
        """Extract a complete chunk with exact full-prefix normalization.

        The STFT retains its 240-sample left context and the scalar running
        log-magnitude maximum across calls. Concatenating outputs from equal
        chunks therefore matches :meth:`extract` on the full waveform, apart
        from possible float32 one-ULP differences between STFT calls.
        """

        samples = _mono_float(waveform)
        if int(samples.numel()) % WHISPER_HOP_SAMPLES:
            raise ValueError("streaming waveform must contain complete mel hops")
        current = CausalWhisperFeatureState() if state is None else state
        if current.waveform_tail is None:
            tail = torch.zeros(CAUSAL_STFT_LEFT_CONTEXT)
        else:
            tail = _mono_float(current.waveform_tail)
            if int(tail.numel()) != CAUSAL_STFT_LEFT_CONTEXT:
                raise ValueError("streaming feature waveform tail has changed size")
        analysis = torch.cat((tail, samples), dim=0)
        spectrum = torch.stft(
            analysis,
            n_fft=WHISPER_N_FFT,
            hop_length=WHISPER_HOP_SAMPLES,
            win_length=WHISPER_N_FFT,
            window=torch.hann_window(WHISPER_N_FFT, periodic=True),
            center=False,
            return_complex=True,
        )
        log_mel = (self.mel_filters @ spectrum.abs().square()).clamp_min(1.0e-10).log10()
        frame_maxima = log_mel.amax(dim=0)
        if current.running_logmax is not None:
            previous = torch.as_tensor(current.running_logmax).detach().float().cpu()
            if previous.numel() != 1 or not bool(torch.isfinite(previous).all()):
                raise ValueError("streaming feature running maximum is invalid")
            frame_maxima = torch.maximum(frame_maxima, previous.reshape(()))
        running_max = frame_maxima.cummax(dim=0).values
        log_mel = torch.maximum(log_mel, running_max.unsqueeze(0) - 8.0)
        result = (log_mel + 4.0) / 4.0
        expected_frames = int(samples.numel()) // WHISPER_HOP_SAMPLES
        if tuple(result.shape) != (WHISPER_MEL_BINS, expected_frames):
            raise RuntimeError("streaming causal Whisper feature clock changed")
        next_state = CausalWhisperFeatureState(
            waveform_tail=analysis[-CAUSAL_STFT_LEFT_CONTEXT:].detach().contiguous(),
            running_logmax=running_max[-1].detach().reshape(()),
        )
        return result.unsqueeze(0), next_state


__all__ = [
    "CAUSAL_FEATURE_FORMAT",
    "CAUSAL_STFT_LEFT_CONTEXT",
    "CausalWhisperFeatureState",
    "CausalWhisperFeatureExtractor",
    "WHISPER_HOP_SAMPLES",
    "WHISPER_MEL_BINS",
    "WHISPER_N_FFT",
]
