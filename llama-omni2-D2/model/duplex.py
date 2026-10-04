"""Stage-3 D2 forward path using the released LLaMA-Omni2 TTS module."""

from __future__ import annotations


import torch
from torch import nn

from d2_llama.core.constants import (
    TTS_EOS_TOKEN_ID,
    TTS_SEPARATOR_TOKEN_ID,
    TTS_TEXT_END_TOKEN_ID,
)
from d2_llama.core.controls import TrainableControlRows
from d2_llama.core.layout import text_slot_indices


def pack_d2_embeddings(
    environment: torch.Tensor,
    delayed_self: torch.Tensor,
    text: torch.Tensor,
    *,
    frames_per_unit: int,
) -> torch.Tensor:
    """Pack tensor lanes as env[k], self[k], text[k] for every macro."""

    if not all(value.ndim == 3 for value in (environment, delayed_self, text)):
        raise ValueError("D2 embedding lanes must be [batch,time,hidden]")
    if environment.shape != delayed_self.shape or environment.shape != text.shape:
        raise ValueError("D2 embedding lane shapes must be identical")
    batch, frames, hidden = environment.shape
    if frames < 1 or frames % frames_per_unit:
        raise ValueError("D2 frame count must contain complete macro units")
    units = frames // frames_per_unit
    lanes = [
        value.reshape(batch, units, frames_per_unit, hidden)
        for value in (environment, delayed_self, text)
    ]
    return torch.cat(lanes, dim=2).reshape(batch, units * 3 * frames_per_unit, hidden)


class LlamaOmni2D2Model(nn.Module):
    """Thin D2 wrapper; native speech generator weights are never replaced."""

    def __init__(
        self, official_model: nn.Module, audio_tower: nn.Module, *, thinker_alignment: bool = False
    ) -> None:
        super().__init__()
        self.thinker_alignment = bool(thinker_alignment)
        self.official_model = official_model
        self.audio_tower = audio_tower
        self.thinker = official_model.get_model()
        self.speech_generator = official_model.get_speech_generator()
        if self.speech_generator is None:
            raise ValueError("official model does not contain its speech generator")
        self.tts_eos_token_id = int(self.speech_generator.tokenizer.eos_token_id)
        if self.tts_eos_token_id != TTS_EOS_TOKEN_ID:
            raise ValueError("official TTS EOS token changed")
        self.tts_text_end_token_id = self.speech_generator.tokenizer.convert_tokens_to_ids(
            "<|im_end|>"
        )
        self.tts_separator_token_id = self.speech_generator.tokenizer.convert_tokens_to_ids("<sep>")
        if self.tts_text_end_token_id != TTS_TEXT_END_TOKEN_ID:
            raise ValueError("official TTS text-end token changed")
        if self.tts_separator_token_id != TTS_SEPARATOR_TOKEN_ID:
            raise ValueError("official TTS separator token changed")
        self.control_rows: TrainableControlRows | None = None
        self.stream_type_embedding: nn.Embedding | None = None
        if thinker_alignment:
            reference = self.thinker.embed_tokens.weight
            self.stream_type_embedding = nn.Embedding(
                2, reference.shape[1], device=reference.device, dtype=torch.float32
            )
            nn.init.zeros_(self.stream_type_embedding.weight)

    def pack_thinker_embeddings(
        self,
        environment: torch.Tensor,
        delayed_self: torch.Tensor,
        text: torch.Tensor,
        *,
        frames_per_unit: int,
    ) -> torch.Tensor:
        """Apply Qwen's self=0/env=1 audio-only lane identities, then pack."""
        if self.stream_type_embedding is not None:
            streams = self.stream_type_embedding.weight
            environment = environment + streams[1].to(environment).view(1, 1, -1)
            delayed_self = delayed_self + streams[0].to(delayed_self).view(1, 1, -1)
        return pack_d2_embeddings(environment, delayed_self, text, frames_per_unit=frames_per_unit)

    def encode_audio_lanes(
        self, environment_features: torch.Tensor, delayed_self_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        environment = self.audio_tower(environment_features).embeddings
        own = self.audio_tower(delayed_self_features).embeddings
        return (environment, own)


def gather_text_hidden(
    packed_hidden: torch.Tensor,
    *,
    frame_count: int,
    frames_per_unit: int,
) -> torch.Tensor:
    units = frame_count // frames_per_unit
    indices = torch.tensor(
        text_slot_indices(units, frames_per_unit),
        device=packed_hidden.device,
        dtype=torch.long,
    )
    return packed_hidden.index_select(1, indices)
