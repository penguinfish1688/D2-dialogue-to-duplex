"""The native CosyVoice chunk mask without a GPU operation per row."""

import torch


def subsequent_chunk_mask(size, chunk_size, num_left_chunks=-1, device=torch.device("cpu")):
    index = torch.arange(size, device=device)
    chunk = index // chunk_size
    visible = index[None, :] < ((chunk + 1) * chunk_size)[:, None]
    if num_left_chunks >= 0:
        first = ((chunk - num_left_chunks) * chunk_size).clamp_min(0)
        visible &= index[None, :] >= first[:, None]
    return visible.contiguous()
