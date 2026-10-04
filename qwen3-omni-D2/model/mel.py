from __future__ import annotations
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F


def _mono_float_tensor(audio: np.ndarray | torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(audio, dtype=torch.float32)
    if values.ndim == 2:
        # Audio datasets conventionally use [samples, channels]. A [1, samples]
        # tensor is also accepted for the single-item inference path.
        if values.shape[0] == 1:
            values = values[0]
        else:
            values = values.mean(dim=-1)
    if values.ndim != 1:
        raise ValueError(f"Expected mono waveform, got shape={tuple(values.shape)}")
    return values.contiguous()


class CausalLogMelStream:
    """Incremental causal log-mel frontend with duration-independent state."""

    def __init__(
        self,
        feature_extractor: Any,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.sample_rate = int(feature_extractor.sampling_rate)
        self.n_fft = int(feature_extractor.n_fft)
        self.hop_length = int(feature_extractor.hop_length)
        self.dither = float(getattr(feature_extractor, "dither", 0.0))
        if self.sample_rate != 16000:
            raise ValueError(
                f"Causal Qwen mel contract requires 16 kHz audio, got {self.sample_rate}"
            )
        if self.n_fft < self.hop_length:
            raise ValueError(
                f"Expected n_fft >= hop_length, got n_fft={self.n_fft} hop={self.hop_length}"
            )
        if self.dither != 0.0:
            raise ValueError("Causal prefix parity requires feature-extractor dither=0")

        self.device = torch.device(device)
        filters = np.asarray(feature_extractor.mel_filters)
        self.mel_bins = int(getattr(feature_extractor, "feature_size", filters.shape[1]))
        self._mel_filters = torch.as_tensor(
            filters,
            device=self.device,
            dtype=torch.float32,
        )
        self._window = torch.hann_window(
            self.n_fft,
            device=self.device,
            dtype=torch.float32,
        )
        self._left_context = self.n_fft - self.hop_length
        self._history = torch.zeros(
            self._left_context,
            device=self.device,
            dtype=torch.float32,
        )
        self._pending = torch.empty(0, device=self.device, dtype=torch.float32)
        self._running_peak = torch.tensor(
            float("-inf"),
            device=self.device,
            dtype=torch.float32,
        )
        self.samples_received = 0
        self.emitted_frames = 0

    @property
    def retained_audio_samples(self) -> int:
        return int(self._history.numel() + self._pending.numel())

    @property
    def max_retained_audio_samples(self) -> int:
        return self.n_fft - 1

    @torch.no_grad()
    def push(self, audio: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Consume new waveform samples and return newly finalized mel frames."""

        waveform = _mono_float_tensor(audio).to(device=self.device)
        self.samples_received += int(waveform.numel())
        available = torch.cat((self._pending, waveform), dim=0)
        frame_count = int(available.numel()) // self.hop_length
        if frame_count == 0:
            self._pending = available.contiguous()
            return available.new_empty((self.mel_bins, 0))

        complete_samples = frame_count * self.hop_length
        complete = available[:complete_samples]
        self._pending = available[complete_samples:].contiguous()
        stft_input = torch.cat((self._history, complete), dim=0)
        stft = torch.stft(
            stft_input,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self._window,
            center=False,
            return_complex=True,
        )
        if int(stft.shape[-1]) != frame_count:
            raise RuntimeError(
                f"Streaming causal STFT frame mismatch: expected={frame_count} "
                f"actual={stft.shape[-1]}"
            )
        if self._left_context:
            self._history = stft_input[-self._left_context :].contiguous()
        else:
            self._history = stft_input.new_empty(0)

        magnitudes = stft.abs().square()
        mel_spec = self._mel_filters.T @ magnitudes.float()
        raw_log_spec = mel_spec.clamp_min(1e-10).log10()
        frame_peak = raw_log_spec.amax(dim=0)
        running_peak = torch.cummax(
            torch.cat((self._running_peak.view(1), frame_peak), dim=0),
            dim=0,
        ).values[1:]
        self._running_peak = running_peak[-1].detach()
        log_spec = torch.maximum(raw_log_spec, running_peak.unsqueeze(0) - 8.0)
        self.emitted_frames += frame_count
        return ((log_spec + 4.0) / 4.0).contiguous()


def causal_log_mel(
    audio: np.ndarray | torch.Tensor,
    feature_extractor: Any,
) -> torch.Tensor:
    """Return prefix-stable Qwen log-mel features as ``[mel, frames]``.

    The official Whisper frontend has two noncausal operations: a centered
    STFT and utterance-wide dynamic-range clipping. This contract replaces
    them with:

    * an end-aligned Hann window that only uses samples received so far; and
    * a cumulative peak for each emitted frame, so an emitted frame never
      changes when louder audio arrives later.

    One feature frame is finalized per complete ``hop_length`` samples. The
    left pad of ``n_fft - hop_length`` samples keeps the official
    ``floor(num_samples / hop_length)`` frame count without right padding.
    """

    sample_rate = int(feature_extractor.sampling_rate)
    n_fft = int(feature_extractor.n_fft)
    hop_length = int(feature_extractor.hop_length)
    dither = float(getattr(feature_extractor, "dither", 0.0))
    if sample_rate != 16000:
        raise ValueError(f"Causal Qwen mel contract requires 16 kHz audio, got {sample_rate}")
    if n_fft < hop_length:
        raise ValueError(f"Expected n_fft >= hop_length, got n_fft={n_fft} hop={hop_length}")
    if dither != 0.0:
        raise ValueError("Causal prefix parity requires feature-extractor dither=0")

    waveform = _mono_float_tensor(audio)
    frame_count = int(waveform.numel()) // hop_length
    mel_bins = int(
        getattr(
            feature_extractor, "feature_size", np.asarray(feature_extractor.mel_filters).shape[1]
        )
    )
    if frame_count == 0:
        return waveform.new_empty((mel_bins, 0))

    # A partial hop cannot finalize another feature and must not influence any
    # already-emitted frame.
    waveform = waveform[: frame_count * hop_length]
    waveform = F.pad(waveform, (n_fft - hop_length, 0))
    window = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=False,
        return_complex=True,
    )
    if int(stft.shape[-1]) != frame_count:
        raise RuntimeError(
            f"Causal STFT frame mismatch: expected={frame_count} actual={stft.shape[-1]}"
        )

    magnitudes = stft.abs().square()
    mel_filters = torch.as_tensor(
        np.asarray(feature_extractor.mel_filters),
        device=waveform.device,
        dtype=torch.float32,
    )
    mel_spec = mel_filters.T @ magnitudes.float()
    raw_log_spec = mel_spec.clamp_min(1e-10).log10()

    # This is the causal counterpart of Whisper's max(log_spec,
    # utterance_max - 8). At frame t it uses only peaks from frames <= t.
    frame_peak = raw_log_spec.amax(dim=0)
    running_peak = torch.cummax(frame_peak, dim=0).values
    log_spec = torch.maximum(raw_log_spec, running_peak.unsqueeze(0) - 8.0)
    return ((log_spec + 4.0) / 4.0).contiguous()
