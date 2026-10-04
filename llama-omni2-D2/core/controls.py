"""Compact D2 control rows without resizing either released vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from d2_llama.core.constants import THINKER_BASE_VOCAB_SIZE, TTS_PAD_TOKEN_ID


RESPONSE_TOKEN = "<|duplex_response|>"
INTERRUPT_TOKEN = "<|duplex_interrupt|>"
PAD_TOKEN = "<|duplex_pad|>"
CONTROL_TOKENS = (RESPONSE_TOKEN, INTERRUPT_TOKEN, PAD_TOKEN)
RESERVED_ROW_TOKEN_PREFIX = "<|d2_reserved_model_row_"


@dataclass(frozen=True, slots=True)
class ControlTokenIds:
    response: int
    interrupt: int
    pad: int

    @property
    def values(self) -> tuple[int, int, int]:
        return self.response, self.interrupt, self.pad


def install_control_tokens(tokenizer: Any, official_model: nn.Module) -> ControlTokenIds:
    """Append exactly three tokenizer IDs while leaving model tables frozen.

    TrainableControlRows handles these IDs outside the released embedding and
    LM head. This avoids optimizer states for two 152k-by-3584 matrices.
    """

    input_rows = official_model.get_input_embeddings()
    output_rows = official_model.get_output_embeddings()
    if int(input_rows.num_embeddings) != THINKER_BASE_VOCAB_SIZE:
        raise ValueError("released Thinker input vocabulary geometry changed")
    if int(output_rows.out_features) != THINKER_BASE_VOCAB_SIZE:
        raise ValueError("released Thinker output vocabulary geometry changed")

    vocabulary = tokenizer.get_vocab()
    present = [token in vocabulary for token in CONTROL_TOKENS]
    if any(present) and not all(present):
        raise ValueError("D2 control tokens are only partially installed")
    if not all(present):
        tokenizer_rows = len(tokenizer)
        if tokenizer_rows > THINKER_BASE_VOCAB_SIZE:
            raise ValueError("released tokenizer exceeds the Thinker embedding table")
        # The released checkpoint deliberately has 152,064 model rows but its
        # tokenizer exposes only 151,666 IDs.  Give the 398 reserved model
        # rows stable, unreachable-by-normal-tokenization names before adding
        # D2 controls.  These placeholders use the already-frozen official
        # rows; they do not allocate trainable embeddings or resize the model.
        gap = THINKER_BASE_VOCAB_SIZE - tokenizer_rows
        if gap:
            add_tokens = getattr(tokenizer, "add_tokens", None)
            if not callable(add_tokens):
                raise TypeError("released tokenizer cannot name reserved model rows")
            placeholders = [
                f"{RESERVED_ROW_TOKEN_PREFIX}{token_id}|>"
                for token_id in range(tokenizer_rows, THINKER_BASE_VOCAB_SIZE)
            ]
            if int(add_tokens(placeholders, special_tokens=True)) != gap:
                raise RuntimeError("tokenizer did not expose every reserved model row")
        if len(tokenizer) != THINKER_BASE_VOCAB_SIZE:
            raise RuntimeError("tokenizer/model reserved-row bridge is incomplete")
        added = tokenizer.add_special_tokens({"additional_special_tokens": list(CONTROL_TOKENS)})
        if added != len(CONTROL_TOKENS):
            raise RuntimeError("tokenizer did not append exactly three D2 controls")

    ids = tuple(int(tokenizer.convert_tokens_to_ids(token)) for token in CONTROL_TOKENS)
    expected = tuple(range(THINKER_BASE_VOCAB_SIZE, THINKER_BASE_VOCAB_SIZE + 3))
    if ids != expected or len(tokenizer) != THINKER_BASE_VOCAB_SIZE + 3:
        raise ValueError(f"D2 control IDs must be contiguous appended rows {expected}, got {ids}")
    return ControlTokenIds(*ids)


def control_initialization_token_id(tokenizer: Any) -> int:
    """Use Qwen's one-token SPACE row to initialize all three controls."""

    encoded = tokenizer.encode(" ", add_special_tokens=False)
    if len(encoded) != 1 or not 0 <= int(encoded[0]) < THINKER_BASE_VOCAB_SIZE:
        raise ValueError("tokenizer must encode SPACE as one released token")
    return int(encoded[0])


class TrainableControlRows(nn.Module):
    """Three small FP32 input/output rows over frozen released tables."""

    def __init__(
        self,
        *,
        controls: ControlTokenIds,
        base_embedding: nn.Embedding,
        base_head: nn.Linear,
        initialization_token_id: int,
    ) -> None:
        super().__init__()
        if controls.values != tuple(range(THINKER_BASE_VOCAB_SIZE, THINKER_BASE_VOCAB_SIZE + 3)):
            raise ValueError("control IDs do not match compact row order")
        source = int(initialization_token_id)
        if not 0 <= source < THINKER_BASE_VOCAB_SIZE:
            raise ValueError("control initialization row is outside released vocabulary")
        if base_embedding.weight.ndim != 2 or base_head.weight.ndim != 2:
            raise ValueError("released input/output weights must be matrices")
        if base_embedding.weight.shape[1] != base_head.weight.shape[1]:
            raise ValueError("released input/output hidden sizes differ")

        hidden = int(base_embedding.weight.shape[1])
        self.controls = controls
        self.initialization_token_id = source
        self.input_rows = nn.Embedding(
            3,
            hidden,
            device=base_embedding.weight.device,
            dtype=torch.float32,
        )
        self.output_rows = nn.Linear(
            hidden,
            3,
            bias=False,
            device=base_head.weight.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            input_source = base_embedding.weight[source].detach().float()
            output_source = base_head.weight[source].detach().float()
            self.input_rows.weight.copy_(input_source.unsqueeze(0).expand(3, -1))
            self.output_rows.weight.copy_(output_source.unsqueeze(0).expand(3, -1))

    def embed(self, base_embedding: nn.Embedding, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype != torch.long:
            raise TypeError("Thinker token IDs must be torch.long")
        control = token_ids >= THINKER_BASE_VOCAB_SIZE
        if bool((token_ids < 0).any()) or bool((token_ids > self.controls.pad).any()):
            raise ValueError("Thinker token ID is outside released+D2 vocabulary")
        safe_ids = torch.where(
            control,
            torch.full_like(token_ids, self.initialization_token_id),
            token_ids,
        )
        base = base_embedding(safe_ids)
        indices = (token_ids - THINKER_BASE_VOCAB_SIZE).clamp(0, 2)
        added = self.input_rows(indices.to(self.input_rows.weight.device)).to(
            device=base.device,
            dtype=base.dtype,
        )
        return torch.where(control.to(base.device).unsqueeze(-1), added, base)

    def logits(self, base_head: nn.Linear, hidden: torch.Tensor) -> torch.Tensor:
        base = base_head(hidden.to(base_head.weight.device))
        controls = F.linear(
            hidden.to(
                device=self.output_rows.weight.device,
                dtype=self.output_rows.weight.dtype,
            ),
            self.output_rows.weight,
        ).to(device=base.device, dtype=base.dtype)
        return torch.cat((base, controls), dim=-1)


def map_thinker_token_to_tts(token_id: int, controls: ControlTokenIds) -> int:
    """Translate only D2 PAD; RESPONSE/INTERRUPT cannot condition TTS."""

    token_id = int(token_id)
    if token_id in (controls.response, controls.interrupt):
        raise ValueError("RESPONSE and INTERRUPT are not TTS conditions")
    return TTS_PAD_TOKEN_ID if token_id == controls.pad else token_id


__all__ = [
    "CONTROL_TOKENS",
    "INTERRUPT_TOKEN",
    "PAD_TOKEN",
    "RESPONSE_TOKEN",
    "TTS_PAD_TOKEN_ID",
    "ControlTokenIds",
    "TrainableControlRows",
    "control_initialization_token_id",
    "install_control_tokens",
    "map_thinker_token_to_tts",
]
