from __future__ import annotations
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from .constants import DUPLEX_PAD_TOKEN


def mask_async_text_logits(
    logits: torch.Tensor,
    *,
    speech_open: torch.Tensor,
    text_done: torch.Tensor,
    previous_text_id: torch.Tensor,
    pad_id: int,
    assistant_start_id: int,
    assistant_end_id: int,
    special_ids: tuple[int, ...] | list[int] = (),
) -> torch.Tensor:
    """Apply the INTERRUPT/RESPONSE async text state machine."""

    state_shape = logits.shape[:-1]
    for name, value in (
        ("speech_open", speech_open),
        ("text_done", text_done),
        ("previous_text_id", previous_text_id),
    ):
        if tuple(value.shape) != tuple(state_shape):
            raise ValueError(
                f"{name} shape {tuple(value.shape)} != logits prefix {tuple(state_shape)}"
            )
    vocab_size = int(logits.shape[-1])
    control_ids = (
        int(pad_id),
        int(assistant_start_id),
        int(assistant_end_id),
    )
    if len(set(control_ids)) != 3 or min(control_ids) < 0 or max(control_ids) >= vocab_size:
        raise ValueError(f"PAD/RESPONSE/INTERRUPT IDs {control_ids} do not fit vocab {vocab_size}")

    speech_open = speech_open.to(device=logits.device, dtype=torch.bool)
    text_done = text_done.to(device=logits.device, dtype=torch.bool)
    previous_text_id = previous_text_id.to(device=logits.device)
    just_started = speech_open & (previous_text_id == int(assistant_start_id))
    lexical_allowed = speech_open & ~text_done
    allowed = lexical_allowed.unsqueeze(-1).expand_as(logits).clone()
    invalid_special_ids = sorted(set(int(token_id) for token_id in special_ids))
    if invalid_special_ids:
        if min(invalid_special_ids) < 0 or max(invalid_special_ids) >= vocab_size:
            raise ValueError(
                f"Special token IDs {invalid_special_ids} do not fit vocab {vocab_size}"
            )
        allowed[..., invalid_special_ids] = False
    allowed[..., int(pad_id)] = (~speech_open) | (speech_open & ~just_started)
    allowed[..., int(assistant_start_id)] = ~speech_open
    allowed[..., int(assistant_end_id)] = True
    return logits.masked_fill(~allowed, float("-inf"))


def module_device(module: nn.Module, fallback: torch.device | None = None) -> torch.device:
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return fallback or torch.device("cpu")


def embed_on(
    embedding: nn.Module,
    ids: torch.Tensor,
    *,
    target_device: torch.device | None = None,
) -> torch.Tensor:
    out = embedding(ids.to(device=module_device(embedding, ids.device)))
    if target_device is not None:
        out = out.to(device=target_device)
    return out


class TrainableTextControlRow(nn.Module):
    """One trainable PAD row over the frozen official Qwen vocabulary."""

    def __init__(
        self,
        *,
        pad_id: int,
        space_input_row: torch.Tensor,
        space_output_row: torch.Tensor,
    ) -> None:
        super().__init__()
        if space_input_row.ndim != 1 or space_output_row.ndim != 1:
            raise ValueError("SPACE initialization rows must be vectors")
        if tuple(space_input_row.shape) != tuple(space_output_row.shape):
            raise ValueError(
                "Qwen input/output SPACE rows have different shapes: "
                f"{tuple(space_input_row.shape)} vs {tuple(space_output_row.shape)}"
            )
        hidden_size = int(space_input_row.numel())
        self.pad_id = int(pad_id)
        self.input_rows = nn.Embedding(1, hidden_size, dtype=torch.float32)
        self.output_rows = nn.Linear(hidden_size, 1, bias=False, dtype=torch.float32)
        with torch.no_grad():
            input_init = space_input_row.detach().float().cpu().view(1, -1)
            output_init = space_output_row.detach().float().cpu().view(1, -1)
            self.input_rows.weight.copy_(input_init)
            self.output_rows.weight.copy_(output_init)

    def embed(self, base_embedding: nn.Module, ids: torch.Tensor) -> torch.Tensor:
        base = embed_on(base_embedding, ids)
        is_pad = ids == int(self.pad_id)
        control = self.input_rows(torch.zeros_like(ids, device=self.input_rows.weight.device)).to(
            device=base.device, dtype=base.dtype
        )
        return torch.where(
            is_pad.to(device=base.device).unsqueeze(-1),
            control,
            base,
        )

    def logits(self, base_head: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        base_device = module_device(base_head, hidden.device)
        base_logits = base_head(hidden.to(device=base_device))
        if self.pad_id >= int(base_logits.shape[-1]):
            raise ValueError(
                f"PAD ID {self.pad_id} exceeds vocabulary size {base_logits.shape[-1]}"
            )
        control_hidden = hidden.to(
            device=self.output_rows.weight.device,
            dtype=self.output_rows.weight.dtype,
        )
        control_logits = self.output_rows(control_hidden).to(
            device=base_logits.device,
            dtype=base_logits.dtype,
        )
        control_id = torch.tensor(
            [self.pad_id],
            device=base_logits.device,
            dtype=torch.long,
        )
        view_shape = (1,) * (control_logits.ndim - 1) + (1,)
        scatter_index = control_id.view(view_shape).expand_as(control_logits)
        return base_logits.scatter(-1, scatter_index, control_logits)


class TrainableAudioControlRows(nn.Module):
    """Trainable input and output rows for RESPONSE and INTERRUPT."""

    def __init__(
        self,
        *,
        assistant_start_id: int,
        assistant_end_id: int,
        start_input_row: torch.Tensor,
        end_input_row: torch.Tensor,
        start_output_row: torch.Tensor,
        end_output_row: torch.Tensor,
    ) -> None:
        super().__init__()
        if int(assistant_start_id) == int(assistant_end_id):
            raise ValueError("AUDIO_START and AUDIO_END IDs must be distinct")
        rows = (start_input_row, end_input_row, start_output_row, end_output_row)
        if any((row.ndim != 1 for row in rows)):
            raise ValueError("Audio-control initialization rows must be vectors")
        shapes = {tuple(row.shape) for row in rows}
        if len(shapes) != 1:
            raise ValueError(
                f"Audio-control input/output initialization rows have different shapes: {sorted(shapes)}"
            )
        hidden_size = int(start_input_row.numel())
        self.assistant_start_id = int(assistant_start_id)
        self.assistant_end_id = int(assistant_end_id)
        self.input_rows = nn.Embedding(2, hidden_size, dtype=torch.float32)
        self.output_rows = nn.Linear(hidden_size, 2, bias=False, dtype=torch.float32)
        initial_input = (
            torch.stack((start_input_row.detach(), end_input_row.detach()), dim=0).float().cpu()
        )
        initial_output = (
            torch.stack((start_output_row.detach(), end_output_row.detach()), dim=0).float().cpu()
        )
        with torch.no_grad():
            self.input_rows.weight.copy_(initial_input)
            self.output_rows.weight.copy_(initial_output)
        self.register_buffer("_official_input_rows", initial_input.clone(), persistent=False)
        self.register_buffer("_official_output_rows", initial_output.clone(), persistent=False)

    def embed(self, base: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        if tuple(base.shape[:-1]) != tuple(ids.shape):
            raise ValueError("Audio-control IDs do not align with base embeddings")
        is_start = ids == self.assistant_start_id
        is_end = ids == self.assistant_end_id
        control_mask = is_start | is_end
        control_index = is_end.to(dtype=torch.long)
        delta_weight = self.input_rows.weight - self._official_input_rows
        delta = F.embedding(control_index.to(device=delta_weight.device), delta_weight).to(
            device=base.device, dtype=base.dtype
        )
        return base + torch.where(
            control_mask.to(device=base.device).unsqueeze(-1), delta, torch.zeros_like(delta)
        )

    def logits(self, base_logits: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if tuple(base_logits.shape[:-1]) != tuple(hidden.shape[:-1]):
            raise ValueError("Audio-control hidden states do not align with logits")
        if max(self.assistant_start_id, self.assistant_end_id) >= int(base_logits.shape[-1]):
            raise ValueError(
                f"AUDIO_START/AUDIO_END IDs exceed vocabulary size {base_logits.shape[-1]}"
            )
        delta_weight = self.output_rows.weight - self._official_output_rows
        control_hidden = hidden.to(device=delta_weight.device, dtype=delta_weight.dtype)
        delta_logits = F.linear(control_hidden, delta_weight).to(
            device=base_logits.device, dtype=base_logits.dtype
        )
        control_ids = torch.tensor(
            [self.assistant_start_id, self.assistant_end_id],
            device=base_logits.device,
            dtype=torch.long,
        )
        view_shape = (1,) * (delta_logits.ndim - 1) + (2,)
        scatter_index = control_ids.view(view_shape).expand_as(delta_logits)
        return base_logits.scatter_add(-1, scatter_index, delta_logits)


def ensure_duplex_control_token(tokenizer: Any, *, vocab_size: int) -> int:
    existing_specials = list(getattr(tokenizer, "additional_special_tokens", ()) or ())
    merged_specials = list(existing_specials)
    if DUPLEX_PAD_TOKEN not in merged_specials:
        merged_specials.append(DUPLEX_PAD_TOKEN)
    try:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": merged_specials},
            replace_additional_special_tokens=True,
        )
    except TypeError:
        tokenizer.add_special_tokens({"additional_special_tokens": merged_specials})
    pad_id = int(tokenizer.convert_tokens_to_ids(DUPLEX_PAD_TOKEN))
    if pad_id < 0 or pad_id >= int(vocab_size):
        raise ValueError(
            "Custom PAD ID does not fit the pretrained Qwen vocabulary: "
            f"id={pad_id} vocab_size={vocab_size}"
        )
    return pad_id
