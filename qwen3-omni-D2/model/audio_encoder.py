from __future__ import annotations
import torch
from torch import nn

from .mel import CausalLogMelStream
from .encoder_base import CausalAutStreamState
from .encoder import MacroAuTEncoder, MacroAuTStreamState


class AudioStream:
    """Bounded inference state for one causal AuT waveform stream."""

    def __init__(self, model: "AudioEncoder", *, device: torch.device) -> None:
        self.model = model
        self.device = torch.device(device)
        self.frontend = CausalLogMelStream(model.feature_extractor, device="cpu")
        self.encoder_state: CausalAutStreamState = model.encoder.new_stream_state()

    @property
    def samples_received(self) -> int:
        return self.frontend.samples_received

    @property
    def emitted_tokens(self) -> int:
        return self.encoder_state.emitted_tokens

    @property
    def latest_token_index(self) -> int | None:
        return self.encoder_state.latest_token_index

    @property
    def latest_output(self) -> torch.Tensor | None:
        return self.encoder_state.latest_output

    @torch.no_grad()
    def push_audio(self, samples: torch.Tensor) -> torch.Tensor:
        new_mel = self.frontend.push(samples)
        if new_mel.shape[1] == 0:
            return torch.empty(
                (0, self.model.output_dim),
                device=self.device,
                dtype=self.model.encoder.dtype,
            )
        return self.model.encoder.stream_uniform_mel(
            new_mel.to(device=self.device, dtype=self.model.encoder.dtype),
            self.encoder_state,
        )

    def state_summary(self) -> dict[str, int | None]:
        return {
            "samples_received": self.samples_received,
            "emitted_tokens": self.emitted_tokens,
            "latest_token_index": self.latest_token_index,
            "retained_frontend_audio_samples": self.frontend.retained_audio_samples,
            "max_frontend_audio_samples": self.frontend.max_retained_audio_samples,
            "mel_chunk_frames": self.encoder_state.mel_chunk_frames,
            "max_mel_chunk_frames": self.encoder_state.max_mel_frames,
            "transformer_cache_tokens": self.encoder_state.cached_tokens,
            "allocated_transformer_cache_tokens": (self.encoder_state.allocated_cache_tokens),
            "max_transformer_cache_tokens": self.encoder_state.max_kv_tokens,
            "attention_window_tokens": self.model.encoder.window_tokens,
        }


class AudioEncoder(nn.Module):
    def __init__(self, tower, feature_extractor, *, latency_ms, encoder_state=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = MacroAuTEncoder(tower, latency_ms=latency_ms, encoder_state=encoder_state)
        self.training_lora_metadata = self.encoder.sft_training_metadata

    @property
    def output_dim(self):
        return int(self.encoder.audio_tower.config.output_dim)

    @property
    def training_lora_enabled(self):
        return self.training_lora_metadata is not None

    def push_stream_batch(self, streams, waveforms):
        return [stream.push_audio(wav) for stream, wav in zip(streams, waveforms, strict=True)]

    """Drop-in ``FrozenCausalAut`` runtime with an explicit latency contract.

    Production inference freezes the distilled runtime.  SFT may request a
    fresh rank-32 attention adapter plus the existing rank-32 MLP adapters and
    full convolution frontend; those weights belong only to the SFT
    checkpoint, never to the distilled runtime payload.
    """

    def prepare(self, device: torch.device | str, *, trainable: bool = False) -> "AudioEncoder":
        self.encoder.to(torch.device(device))
        if trainable:
            if self.encoder.sft_training_metadata is None:
                raise RuntimeError("D2 AuT SFT topology is not configured")
            self.training_lora_metadata = dict(self.encoder.sft_training_metadata)
            self.encoder.enable_sft_training_parameters()
            self.encoder.configure_sft_training_mode()
        else:
            self.training_lora_metadata = None
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
            self.encoder.eval()
        return self

    def enable_training_lora_parameters(self) -> None:
        if self.encoder.sft_training_metadata is None:
            raise RuntimeError("D2 AuT SFT topology is not configured")
        self.training_lora_metadata = dict(self.encoder.sft_training_metadata)
        self.encoder.enable_sft_training_parameters()

    enable_sft_parameters = enable_training_lora_parameters

    def configure_training_lora_mode(self) -> None:
        self.encoder.configure_sft_training_mode()

    configure_sft_mode = configure_training_lora_mode

    def new_stream(self, device: torch.device) -> AudioStream:
        self.encoder.to(device)
        self.encoder.eval()
        return AudioStream(self, device=device)

    def encode_trainable_mel_chunk(
        self, input_features: torch.Tensor, states: list[MacroAuTStreamState]
    ) -> torch.Tensor:
        """Encode a uniform mel chunk with one detached state per batch row.

        Feature extraction is intentionally outside this method: SFT can cache
        prefix-stable causal mel once, while this in-graph call contains every
        trainable encoder operation and only bounded 104-mel/104-token state.
        """
        if not self.training_lora_enabled:
            raise RuntimeError(
                "Differentiable macro AuT encoding requires audio_encoder_sft.enabled"
            )
        if input_features.ndim != 3:
            raise ValueError(
                f"expected SFT features [batch,mel,frames], got {tuple(input_features.shape)}"
            )
        if int(input_features.shape[0]) < 1 or len(states) != int(input_features.shape[0]):
            raise ValueError("SFT AuT requires one state per non-empty batch row")
        device = next(self.encoder.parameters()).device
        features = input_features.to(device=device, dtype=self.encoder.dtype)
        self.encoder.configure_sft_training_mode()
        outputs = [
            self.encoder.forward_sft_mel_chunk(row, state)
            for row, state in zip(features, states, strict=True)
        ]
        lengths = {int(output.shape[0]) for output in outputs}
        if len(lengths) != 1:
            raise RuntimeError("uniform SFT mel rows emitted different lengths")
        return torch.stack(outputs, dim=0).float()
