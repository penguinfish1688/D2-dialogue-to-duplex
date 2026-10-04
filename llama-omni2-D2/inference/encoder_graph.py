"""Block-causal Whisper with fixed KV storage and native segment resets."""

import torch
import torch.nn.functional as F


class EncoderGraph:
    @torch.no_grad()
    def __init__(self, tower):
        self.tower = tower
        self.encoder = tower.speech_encoder
        self.rows = tower.macro_rows
        self.capacity = tower.segment_rows
        self.device = next(tower.parameters()).device
        self.dtype = torch.bfloat16
        self.features = torch.zeros(
            1, 128, tower.macro_mel_frames, device=self.device, dtype=self.dtype
        )
        self.tail = torch.zeros(1, 128, 2, device=self.device, dtype=self.dtype)
        self.slots = torch.arange(self.rows, device=self.device)
        self.mask = torch.zeros(
            1, 1, self.rows, self.capacity, device=self.device, dtype=torch.bool
        )
        self.mask[..., : self.rows] = True
        self.keys = []
        self.values = []
        for block in self.encoder.blocks:
            shape = (1, block.attn.n_head, self.capacity, 1280 // block.attn.n_head)
            self.keys.append(torch.zeros(shape, device=self.device, dtype=self.dtype))
            self.values.append(torch.zeros(shape, device=self.device, dtype=self.dtype))
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._forward()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = self._forward()
        self.reset()

    def _forward(self):
        hidden, next_tail = self.tower._frontend(self.features, self.tail)
        self.tail.copy_(next_tail)
        hidden = hidden + self.encoder.positional_embedding.index_select(0, self.slots).to(hidden)
        for i, block in enumerate(self.encoder.blocks):
            values = block.attn_ln(hidden)
            attn = block.attn

            def split(value):
                return value.reshape(1, self.rows, attn.n_head, -1).transpose(1, 2)

            query = split(attn.query(values))
            key = split(attn.key(values))
            value = split(attn.value(values))
            self.keys[i].index_copy_(2, self.slots, key)
            self.values[i].index_copy_(2, self.slots, value)
            attended = F.scaled_dot_product_attention(
                query,
                self.keys[i],
                self.values[i],
                attn_mask=self.mask,
                dropout_p=0,
                is_causal=False,
            )
            hidden = hidden + attn.out(
                attended.transpose(1, 2).reshape(1, self.rows, 1280).contiguous()
            )
            hidden = hidden + block.mlp(block.mlp_ln(hidden))
        return self.tower.speech_projector(self.encoder.ln_post(hidden))

    def reset(self):
        self.length = 0
        self.tail.zero_()
        for value in self.keys + self.values:
            value.zero_()

    @torch.no_grad()
    def push(self, features):
        if features.shape != self.features.shape:
            raise ValueError("Whisper graph expects one complete interaction interval")
        if self.length == self.capacity:
            self.length = 0
            # The original encoder carries its two-mel convolution tail across
            # segment boundaries, but resets positions and attention history.
            for value in self.keys + self.values:
                value.zero_()
        self.features.copy_(features)
        self.slots.copy_(torch.arange(self.length, self.length + self.rows, device=self.device))
        self.mask.zero_()
        self.mask[..., : self.length + self.rows] = True
        self.graph.replay()
        self.length += self.rows
        return self.output.clone()
