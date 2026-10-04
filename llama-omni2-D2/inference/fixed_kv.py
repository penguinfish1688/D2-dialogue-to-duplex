from __future__ import annotations
import torch
import torch.nn.functional as F


def talker_kv_capacity(text_tokens, tail_tokens=1024):
    # Include the forced text-end condition, SEP, and a final partial text burst.
    bound = text_tokens + 2 + (text_tokens // 3 + 1) * 10 + tail_tokens
    return 1 << (bound - 1).bit_length()


def attention_buckets(capacity):
    return tuple(
        sorted({capacity, *(2**i for i in range(7, capacity.bit_length()) if 2**i < capacity)})
    )


class FixedKVDecoder:
    """One frozen decoder, one fixed batch/step shape, stable backing buffers."""

    def __init__(self, decoder, *, batch_size, token_count, capacity, capture=True, storage=None):
        if decoder.training or any(p.requires_grad for p in decoder.parameters()):
            raise ValueError("fixed KV decoder requires frozen eval weights")
        if min(batch_size, token_count, capacity) < 1 or token_count > capacity:
            raise ValueError("invalid fixed KV dimensions")
        self.decoder = decoder
        self.batch_size, self.token_count, self.capacity = batch_size, token_count, capacity
        weight = decoder.embed_tokens.weight
        self.device, self.dtype = weight.device, weight.dtype
        self.capture = capture
        if capture and self.device.type != "cuda":
            raise ValueError("CUDA graph capture requires CUDA")
        self.inputs = weight.new_zeros(batch_size, token_count, weight.shape[1])
        attention = decoder.layers[0].self_attn
        self.slots = torch.zeros(token_count, dtype=torch.long, device=self.device)
        self.key_indices = torch.arange(capacity, device=self.device)
        self.mask = weight.new_zeros(1, 1, token_count, capacity)
        self.cos = weight.new_zeros(1, 1, token_count, attention.head_dim)
        self.sin = self.cos.clone()
        self.keys, self.values = ([], []) if storage is None else storage
        for layer in decoder.layers:
            attn = layer.self_attn
            if (
                attn.head_dim != attention.head_dim
                or attn.rotary_emb.base != attention.rotary_emb.base
                or not torch.equal(attn.rotary_emb.inv_freq, attention.rotary_emb.inv_freq)
            ):
                raise ValueError("fixed graph requires identical native RoPE tables across layers")
            shape = (batch_size, attn.num_key_value_heads, capacity, attn.head_dim)
            if storage is None:
                self.keys.append(weight.new_zeros(shape))
                self.values.append(weight.new_zeros(shape))
        self.length = 0
        self.epoch = 0
        self.buckets = attention_buckets(capacity)
        self.graphs = {}
        self.validate = False
        self.validation_errors = {}
        self.replays = 0

    def reset(self):
        self.length = 0
        self.epoch += 1
        for key, value in zip(self.keys, self.values, strict=True):
            key.zero_()
            value.zero_()

    def import_prefill(self, layers):
        if len(layers) != len(self.keys):
            raise ValueError("prefill KV layer count changed")
        count = layers[0][0].shape[-2]
        batch = layers[0][0].shape[0]
        if not 0 < count <= self.capacity or not 0 < batch <= self.batch_size:
            raise ValueError("prefill exceeds fixed KV capacity/batch")
        self.reset()
        for index, (key, value) in enumerate(layers):
            expected = (batch, self.keys[index].shape[1], count, self.keys[index].shape[-1])
            if key.shape != expected or value.shape != expected:
                raise ValueError("prefill KV shapes differ")
            self.keys[index][:batch, :, :count].copy_(key)
            self.values[index][:batch, :, :count].copy_(value)
        self.length = count

    def export(self, rows=None):
        # Views expose valid lengths to native HF bookkeeping, never resize
        # or replace the fixed storage used inside the captured graph.
        if rows is None:
            return tuple(
                (k[:, :, : self.length], v[:, :, : self.length])
                for k, v in zip(self.keys, self.values, strict=True)
            )
        indices = torch.tensor(rows, device=self.device)
        return tuple(
            (
                k.index_select(0, indices)[:, :, : self.length],
                v.index_select(0, indices)[:, :, : self.length],
            )
            for k, v in zip(self.keys, self.values, strict=True)
        )

    def keep_last(self, count):
        if count < 1:
            raise ValueError("KV retention must be positive")
        if self.length <= count:
            return
        start = self.length - count
        for key, value in zip(self.keys, self.values, strict=True):
            key[:, :, :count].copy_(key[:, :, start : self.length].clone())
            value[:, :, :count].copy_(value[:, :, start : self.length].clone())
        self.length = count

    def _forward(self, limit):
        from transformers.models.qwen2.modeling_qwen2 import repeat_kv, rotate_half

        mask = self.mask[..., :limit]
        mask.copy_(
            torch.where(
                self.key_indices[None, :limit] <= self.slots[:, None],
                0.0,
                torch.finfo(self.dtype).min,
            )[None, None]
        )
        hidden = self.inputs
        batch, count = self.batch_size, self.token_count
        for index, layer in enumerate(self.decoder.layers):
            value = layer.input_layernorm(hidden)
            attn = layer.self_attn
            q = attn.q_proj(value).view(batch, count, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = (
                attn.k_proj(value)
                .view(batch, count, attn.num_key_value_heads, attn.head_dim)
                .transpose(1, 2)
            )
            v = (
                attn.v_proj(value)
                .view(batch, count, attn.num_key_value_heads, attn.head_dim)
                .transpose(1, 2)
            )
            q = q * self.cos + rotate_half(q) * self.sin
            k = k * self.cos + rotate_half(k) * self.sin
            self.keys[index].index_copy_(2, self.slots, k)
            self.values[index].index_copy_(2, self.slots, v)
            attended = F.scaled_dot_product_attention(
                q.contiguous(),
                repeat_kv(self.keys[index][:, :, :limit], attn.num_key_value_groups).contiguous(),
                repeat_kv(self.values[index][:, :, :limit], attn.num_key_value_groups).contiguous(),
                attn_mask=mask,
                dropout_p=0.0,
            )
            hidden = hidden + attn.o_proj(attended.transpose(1, 2).reshape(batch, count, -1))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return self.decoder.norm(hidden)

    def step(self, inputs, *, absolute_offset, rows=None):
        if torch.is_grad_enabled():
            raise RuntimeError("fixed KV decode is inference-only")
        rows = tuple(range(inputs.shape[0])) if rows is None else tuple(rows)
        if (
            inputs.shape != (len(rows), self.token_count, self.inputs.shape[-1])
            or not rows
            or len(set(rows)) != len(rows)
            or min(rows) < 0
            or max(rows) >= self.batch_size
        ):
            raise ValueError("invalid fixed KV decode rows/shape")
        if absolute_offset < 0 or self.length + self.token_count > self.capacity:
            raise ValueError("fixed KV capacity exceeded or negative RoPE position")
        indices = torch.tensor(rows, device=self.device)
        self.inputs.zero_()
        self.inputs.index_copy_(0, indices, inputs)
        self.slots.copy_(
            torch.arange(self.length, self.length + self.token_count, device=self.device)
        )
        rotary = self.decoder.layers[0].self_attn.rotary_emb
        cos, sin = rotary(self.inputs, seq_len=absolute_offset + self.token_count)
        self.cos.copy_(cos[absolute_offset : absolute_offset + self.token_count][None, None])
        self.sin.copy_(sin[absolute_offset : absolute_offset + self.token_count][None, None])
        limit = next(size for size in self.buckets if size >= self.length + self.token_count)
        if self.capture:
            if limit not in self.graphs:
                with torch.cuda.device(self.device):
                    stream = torch.cuda.Stream(device=self.device)
                    stream.wait_stream(torch.cuda.current_stream(self.device))
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            self._forward(limit)
                    torch.cuda.current_stream(self.device).wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        graph_output = self._forward(limit)
                    self.graphs[limit] = graph, graph_output
            graph, output = self.graphs[limit]
            graph.replay()
            self.replays += 1
            if self.validate and (self.validate == "all" or limit not in self.validation_errors):
                expected = self._forward(limit)
                error = float((output.float() - expected.float()).abs().max())
                self.validation_errors[limit] = max(error, self.validation_errors.get(limit, 0))
                torch.testing.assert_close(output, expected, atol=0, rtol=0)
        else:
            output = self._forward(limit)
        self.length += self.token_count
        return output.index_select(0, indices).clone()


class GraphDecoder:
    """Captured prefill and serial decode share one fixed KV allocation."""

    def __init__(self, decoder, capacity, *, capture=True):
        self.decoder, self.capacity, self.capture = decoder, capacity, capture
        self.shapes = {}
        self.length = 0
        self.storage = None

    def reset(self):
        self.length = 0
        if self.storage is not None:
            for tensors in self.storage:
                for tensor in tensors:
                    tensor.zero_()

    def step(self, inputs):
        if self.length + inputs.shape[1] > self.capacity:
            raise ValueError("Fixed KV budget exhausted; start a new conversation")
        outputs = []
        # Native TTS may supply a long prefix. Reuse the same bounded shapes.
        for values in inputs.split(32, dim=1):
            count = values.shape[1]
            if count not in self.shapes:
                core = FixedKVDecoder(
                    self.decoder,
                    batch_size=1,
                    token_count=count,
                    capacity=self.capacity,
                    capture=self.capture,
                    storage=self.storage,
                )
                self.storage = core.keys, core.values
                self.shapes[count] = core
            core = self.shapes[count]
            core.length = self.length
            outputs.append(core.step(values, absolute_offset=self.length))
            self.length += count
        return torch.cat(outputs, dim=1)

    def export(self):
        return tuple(
            (k[:, :, : self.length], v[:, :, : self.length])
            for k, v in zip(*self.storage, strict=True)
        )


class FixedTalkerForward:
    """Use the native sampling loop with captured prefix and decode forwards."""

    def __init__(self, decoder, *, capacity, capture=True):
        self.core = GraphDecoder(decoder, capacity, capture=capture)
        self.history = None
        self.last_cache = None

    def reset(self):
        self.core.reset()
        self.history = None
        self.last_cache = None

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        from transformers.modeling_outputs import BaseModelOutputWithPast

        if output_attentions or output_hidden_states or use_cache is False:
            raise ValueError("Fixed Talker requires cached native inference")
        if attention_mask is not None and not bool((attention_mask == 1).all()):
            raise ValueError("Fixed Talker requires an unpadded prefix")
        values = (
            self.core.decoder.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        seen = (
            past_key_values.get_seq_length()
            if hasattr(past_key_values, "get_seq_length")
            else past_key_values[0][0].shape[-2]
            if past_key_values
            else 0
        )
        if seen == 0:
            reused = (
                self.history is not None
                and values.shape[1] > self.history.shape[1]
                and torch.equal(values[:, : self.history.shape[1]], self.history)
            )
            if reused:
                values = values[:, self.history.shape[1] :]
            else:
                self.reset()
        elif past_key_values is not self.last_cache or seen != self.core.length:
            raise ValueError("Talker cache does not belong to this response")
        output = self.core.step(values)
        self.history = (
            values.detach().clone()
            if self.history is None
            else torch.cat((self.history, values.detach()), dim=1)
        )
        self.last_cache = self.core.export()
        if return_dict is False:
            return output, self.last_cache
        return BaseModelOutputWithPast(last_hidden_state=output, past_key_values=self.last_cache)
