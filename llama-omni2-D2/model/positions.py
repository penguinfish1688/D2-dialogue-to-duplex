"""Original native serial RoPE and matching training/inference XL boundaries."""

from __future__ import annotations

import torch

from d2_llama.core.constants import align_chunk_frames
from d2_llama.prompts import normalize_task_kind, task_prompt_ids


THINKER_POSITION_POLICY = "native_serial_env_self_text_rope_v1"
THINKER_MEMORY_FRAMES = 900
THINKER_CHUNK_FRAMES = 900
THINKER_INFERENCE_POLICY = "absolute_rope_90s_history_90s_chunk_kv_v1"


def thinker_chunk_frames(frames_per_unit):
    """Use the same whole-macro 90-second target as the training dataloader."""
    return align_chunk_frames(THINKER_CHUNK_FRAMES, frames_per_unit)


def thinker_position_ids(
    *, batch_size, frame_count, frames_per_unit, frame_offset, device, prompt_tokens=0
):
    """Keep every packed token's original monotonically increasing position.

    For k=1, frame f is [3*f, 3*f+1, 3*f+2] for env/self/text. The absolute
    cursor survives XL cache eviction; a new conversation starts at zero.
    Native TTS retains its independent ordinary serial position sequence.
    """
    if (
        batch_size < 1
        or frame_count < 1
        or frames_per_unit < 1
        or prompt_tokens < 0
        or frame_count % frames_per_unit
        or frame_offset < 0
        or frame_offset % frames_per_unit
    ):
        raise ValueError("Thinker positions require complete, aligned macro units")
    local = torch.arange(3 * frame_count, device=device, dtype=torch.long)
    positions = local + 3 * frame_offset + prompt_tokens
    return positions.unsqueeze(0).expand(batch_size, -1).contiguous()


def prepare_thinker_prompted_inputs(
    model, packed, past, *, frame_offset, frames_per_unit, task_kind=None
):
    """Embed one prefix with the first chunk, preserving its training graph.

    Prompts use only released text embeddings and have no stream identities.
    The returned prefix count excludes them from every text/TTS readout. A
    prefix can age out of ordinary XL memory, but is never reinserted.
    """
    ids = task_prompt_ids(model, task_kind)
    prompt_tokens = len(ids)
    metadata = {}
    if prompt_tokens:
        if not getattr(model, "thinker_alignment", False):
            raise ValueError("task prompts require aligned serial Thinker positions")
        metadata = {
            "task_kind": normalize_task_kind(task_kind),
            "prompt_tokens": prompt_tokens,
            "task_prompt_sha256": model.task_prompt_contract["sha256"],
        }
        if past is not None and any(past.get(key) != value for key, value in metadata.items()):
            raise ValueError("Thinker cache task system prompt identity changed")
        if (past is None) != (frame_offset == 0):
            raise ValueError("task system prompt must be inserted exactly at conversation start")
    elif past is not None and past.get("prompt_tokens", 0):
        raise ValueError("prompted Thinker cache cannot resume with task prompts disabled")
    if past is not None and int(past["offset"]) != prompt_tokens + 3 * frame_offset:
        raise ValueError("Thinker XL cache cursor disagrees with frame clock")
    frame_count = packed.shape[1] // 3
    positions = thinker_position_ids(
        batch_size=packed.shape[0],
        frame_count=frame_count,
        frames_per_unit=frames_per_unit,
        frame_offset=frame_offset,
        device=packed.device,
        prompt_tokens=prompt_tokens,
    )
    prefix_count = prompt_tokens if ids and past is None else 0
    if prefix_count:
        token_ids = torch.tensor(ids, dtype=torch.long, device=packed.device).unsqueeze(0)
        prefix = model.thinker.embed_tokens(token_ids).expand(packed.shape[0], -1, -1)
        packed = torch.cat((prefix.to(dtype=packed.dtype), packed), dim=1)
        prefix_positions = torch.arange(prefix_count, device=packed.device, dtype=torch.long)
        positions = torch.cat(
            (prefix_positions.unsqueeze(0).expand(packed.shape[0], -1), positions), dim=1
        )
    return packed, positions, prefix_count, metadata
