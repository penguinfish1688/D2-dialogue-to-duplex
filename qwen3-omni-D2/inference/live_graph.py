"""Native PyTorch CUDA graphs for causal D2 audio input and waveform output.

The AuT graph keeps exactly 103 past tokens at 80 ms after eager startup.
It never receives a future waveform frame. The waveform graph uses the native
Qwen modules and a fixed transformer KV capacity, without an inference engine.
"""

from __future__ import annotations

from types import SimpleNamespace
import torch
from torch.nn import functional as F

from .cuda_graph import (
    split_dtype_static_cache,
    _set_cache_cursor,
    D2ThinkerCudaGraphs,
    _last_hidden,
)
from .waveform_graph import Code2WavDecoderGraph
from .streaming import inference_autocast


class CausalEncoderGraph:
    @torch.no_grad()
    def __init__(self, encoder):
        self.encoder = encoder
        self.tower = encoder.audio_tower
        self.k = encoder.frames_per_unit
        self.past = encoder.max_past_tokens
        self.device = next(encoder.parameters()).device
        self.dtype = encoder.dtype
        self.window = torch.zeros((128, 104), device=self.device, dtype=self.dtype)
        self.keys, self.values = [], []
        for layer in self.tower.layers:
            attn = layer.self_attn
            shape = (
                (1, attn.num_heads, self.past, attn.q_proj.out_features // attn.num_heads)
                if hasattr(attn.q_proj, "out_features")
                else (1, attn.num_heads, self.past, attn.q_proj.base.out_features // attn.num_heads)
            )
            self.keys.append(torch.zeros(shape, device=self.device, dtype=self.dtype))
            self.values.append(torch.zeros(shape, device=self.device, dtype=self.dtype))
        self.graph = torch.cuda.CUDAGraph()
        self.reset()
        with inference_autocast(self.device, self.dtype):
            self._forward()
        self.reset()
        torch.cuda.synchronize(self.device)
        with (
            inference_autocast(self.device, self.dtype),
            torch.cuda.graph(self.graph, capture_error_mode="thread_local"),
        ):
            self.output = self._forward()
        self.reset()
        self.parity = self.validate()

    def _forward(self):
        hidden = self.encoder._conv_macro_window_batch([self.window])
        for index, layer in enumerate(self.tower.layers):
            residual = hidden
            normalized = layer.self_attn_layer_norm(hidden)
            attn = layer.self_attn
            query = (
                attn.q_proj(normalized).reshape(self.k, attn.num_heads, -1).transpose(0, 1)[None]
            )
            key = attn.k_proj(normalized).reshape(self.k, attn.num_heads, -1).transpose(0, 1)[None]
            value = (
                attn.v_proj(normalized).reshape(self.k, attn.num_heads, -1).transpose(0, 1)[None]
            )
            key = torch.cat((self.keys[index], key), dim=-2)
            value = torch.cat((self.values[index], value), dim=-2)
            attended = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0, is_causal=False
            )
            self.keys[index].copy_(key[..., -self.past :, :])
            self.values[index].copy_(value[..., -self.past :, :])
            hidden = residual + attn.out_proj(
                attended.squeeze(0).transpose(0, 1).reshape(self.k, -1).contiguous()
            )
            residual = hidden
            hidden = layer.final_layer_norm(hidden)
            hidden = residual + layer.fc2(layer.activation_fn(layer.fc1(hidden)))
        hidden = self.tower.ln_post(hidden)
        return self.tower.proj2(self.tower.act(self.tower.proj1(hidden)))

    @torch.no_grad()
    def reset(self):
        self.window.zero_()
        for value in self.keys + self.values:
            value.zero_()
        self.frames = 0
        self.startup_state = self.encoder.new_stream_state()

    @torch.no_grad()
    def push(self, mel):
        assert tuple(mel.shape) == (128, 8 * self.k)
        self.window[:, : -8 * self.k].copy_(self.window[:, 8 * self.k :].clone())
        self.window[:, -8 * self.k :].copy_(mel)
        # Preserve the native attention shape while history fills. Padding
        # startup keys changes BF16 SDPA rounding across the encoder stack.
        # Once full, both paths use exactly the same bounded 104-token shape.
        if self.startup_state is not None:
            with inference_autocast(self.device, self.dtype):
                output = self.encoder.stream_uniform_mel(mel, self.startup_state)
            self.frames += self.k
            if self.startup_state.cached_tokens == self.past:
                for index, state in enumerate(self.startup_state.layer_states):
                    self.keys[index].copy_(state.key)
                    self.values[index].copy_(state.value)
                self.startup_state = None
            return output
        self.graph.replay()
        self.frames += self.k
        return self.output.clone()

    @torch.no_grad()
    def validate(self):
        # Cover empty, partially filled, full and evicted history with the
        # restored checkpoint, comparing against the original causal runtime.
        state = self.encoder.new_stream_state()
        self.reset()
        maximum = 0.0
        count = self.past // self.k + 3
        generator = torch.Generator(device=self.device).manual_seed(567)
        for index in range(count):
            mel = torch.randn(
                (128, 8 * self.k), device=self.device, dtype=self.dtype, generator=generator
            )
            with inference_autocast(self.device, self.dtype):
                expected = self.encoder.stream_uniform_mel(mel, state)
            actual = self.push(mel)
            error = float((actual.float() - expected.float()).abs().max().item())
            maximum = max(maximum, error)
            torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.02)
        self.reset()
        result = {"frames": count * self.k, "maximum_absolute_error": maximum, "passed": True}
        print(f"[live graph] causal encoder eager parity: {result}", flush=True)
        return result


class GraphAudioStream:
    def __init__(self, original, graph):
        self.original = original
        self.frontend = original.frontend
        self.graph = graph
        graph.reset()

    def push_audio(self, pcm):
        return self.graph.push(
            self.frontend.push(pcm).to(device=self.graph.device, dtype=self.graph.dtype)
        )

    def state_summary(self):
        count = self.graph.frames
        return {
            "samples_received": self.frontend.samples_received,
            "emitted_tokens": count,
            "latest_token_index": count - 1 if count else None,
            "retained_frontend_audio_samples": self.frontend.retained_audio_samples,
            "max_frontend_audio_samples": self.frontend.max_retained_audio_samples,
            "mel_chunk_frames": min(8 * count, 104),
            "max_mel_chunk_frames": min(8 * count, 104),
            "transformer_cache_tokens": min(count, self.graph.past),
            "allocated_transformer_cache_tokens": self.graph.past,
            "max_transformer_cache_tokens": min(count, self.graph.past),
            "attention_window_tokens": self.graph.past + self.graph.k,
            "cuda_graph": True,
            "eager_parity": self.graph.parity,
        }


class WaveformGraphs:
    code2wav_graph_enabled = True

    @torch.no_grad()
    def __init__(self, model, capacity):
        self.code2wav = model.qwen.code2wav
        self.model = self.code2wav.pre_transformer
        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype
        self.capacity = capacity
        self.input = torch.zeros(
            (1, 1, self.model.config.hidden_size), device=self.device, dtype=self.dtype
        )
        self.mask = torch.zeros((1, 1, 1, capacity), device=self.device, dtype=torch.bool)
        self.position = torch.zeros((1, 1), device=self.device, dtype=torch.long)
        self.cache = split_dtype_static_cache(layers=len(self.model.layers), max_cache_len=capacity)
        self.mask[..., 0] = True
        with inference_autocast(self.device, self.dtype):
            self._forward()
        self.cache.reset()
        self.graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(self.device)
        with (
            inference_autocast(self.device, self.dtype),
            torch.cuda.graph(self.graph, capture_error_mode="thread_local"),
        ):
            self.output = self._forward()
        # Cover all startup convolution lengths as well as the steady 26-frame window.
        self.decoders = {
            n: Code2WavDecoderGraph(self.code2wav, window_frames=n, dtype=self.dtype)
            for n in range(1, 27)
        }
        self.reset()
        self.parity = self.validate(model)

    def _forward(self):
        return self.model(
            inputs_embeds=self.input,
            attention_mask=self.mask,
            position_ids=self.position,
            cache_position=self.position.view(-1),
            past_key_values=self.cache,
            use_cache=True,
        )

    @torch.no_grad()
    def reset(self):
        self.cache.reset()
        _set_cache_cursor(self.cache, 0)
        self.frames = 0

    @torch.no_grad()
    def code2wav_transformer_forward(self, hidden, *, past_key_values, frame_index):
        if frame_index != self.frames or frame_index >= self.capacity:
            raise RuntimeError("Code2Wav fixed KV cursor/capacity violated")
        if past_key_values is not None and past_key_values is not self.cache:
            raise RuntimeError("Code2Wav cache ownership changed")
        self.input.copy_(hidden)
        self.position.fill_(frame_index)
        self.mask.zero_()
        self.mask[..., : frame_index + 1] = True
        self.graph.replay()
        self.frames += 1
        return SimpleNamespace(
            last_hidden_state=self.output.last_hidden_state, past_key_values=self.cache
        )

    def code2wav_decoder_forward(self, transformed):
        return self.decoders[transformed.shape[1]].replay(transformed)

    @torch.no_grad()
    def validate(self, wrapper):
        from .streaming import IncrementalCode2WavRenderer

        graphs = self

        class EagerBackend:
            code2wav_graph_enabled = True

            def code2wav_transformer_forward(self, hidden, *, past_key_values, frame_index):
                graphs.input.copy_(hidden)
                graphs.position.fill_(frame_index)
                graphs.mask.zero_()
                graphs.mask[..., : frame_index + 1] = True
                with inference_autocast(graphs.device, graphs.dtype):
                    return graphs._forward()

            def code2wav_decoder_forward(self, transformed):
                decoder = graphs.decoders[transformed.shape[1]]
                decoder.transformed.copy_(transformed)
                with inference_autocast(graphs.device, graphs.dtype):
                    return decoder._forward()

        options = dict(
            left_context_frames=25,
            parity_atol=float("inf"),
            parity_mean_atol=float("inf"),
            verify_reference=False,
            capture_components=False,
            retain_raw_prefix=False,
        )
        codes = [(wrapper.codec_silence_frame.detach().cpu().long() + i) % 2048 for i in range(28)]
        reference = IncrementalCode2WavRenderer(wrapper, forward_backend=EagerBackend(), **options)
        expected = [reference.append_frame(code) for code in codes]
        self.reset()
        captured = IncrementalCode2WavRenderer(wrapper, forward_backend=self, **options)
        maximum = maximum_rmse = 0.0
        for code, target in zip(codes, expected, strict=True):
            actual = captured.append_frame(code)
            torch.testing.assert_close(actual, target, atol=0.04, rtol=0.02)
            maximum = max(maximum, float((actual - target).abs().max()))
            maximum_rmse = max(maximum_rmse, float((actual - target).square().mean().sqrt()))
        if maximum_rmse > 0.01:
            raise RuntimeError(f"Code2Wav graph RMS error too large: {maximum_rmse}")
        self.reset()
        result = {
            "frames": 28,
            "reference": "eager with identical static cache, masks and autocast",
            "maximum_absolute_error": maximum,
            "maximum_rmse": maximum_rmse,
            "atol": 0.04,
            "rtol": 0.02,
            "passed": True,
        }
        print(f"[capture] waveform eager/graph parity: {result}", flush=True)
        return result


class LiveAudioGraphs:
    def __init__(self, model, capacity):
        self.encoders = [CausalEncoderGraph(model.audio_encoder.encoder) for _ in range(2)]
        self.waveform = WaveformGraphs(model, capacity)

    def reset(self, env_stream, self_stream):
        self.waveform.reset()
        return tuple(
            GraphAudioStream(original, graph)
            for original, graph in zip((env_stream, self_stream), self.encoders, strict=True)
        )


class FusedNativeThinker(D2ThinkerCudaGraphs):
    """Stage audio, then replay [environment, delayed self, previous text].

    The previous text is known before this frame starts. A triangular mask
    makes this equivalent to the two original calls, with no new dependency.
    """

    @torch.no_grad()
    def __init__(self, model, *, accepted_hidden_layer, capacity, dtype):
        if capacity.frames_per_unit != 1:
            raise ValueError("live fused Thinker currently requires 80-ms D2")
        self.model, self.capacity, self.dtype = model, capacity, dtype
        self.device = next(model.parameters()).device
        self.accepted_hidden_layer = accepted_hidden_layer
        self.cache = split_dtype_static_cache(
            layers=len(model.layers), max_cache_len=capacity.thinker_tokens
        )
        self.inputs = torch.zeros((1, 3, model.config.hidden_size), device=self.device, dtype=dtype)
        self.mask = torch.zeros(
            (1, 1, 3, capacity.thinker_tokens), device=self.device, dtype=torch.bool
        )
        self.mask[..., :3] = torch.ones((3, 3), device=self.device, dtype=torch.bool).tril()
        self.positions = (
            torch.arange(3, device=self.device)[None, None].expand(4, 1, -1).contiguous()
        )
        self.cache_positions = torch.arange(3, device=self.device)
        self.graph = torch.cuda.CUDAGraph()
        self.audio_replays = self.text_replays = 0
        with inference_autocast(self.device, dtype):
            self._forward()
        self.cache.reset()
        captured = {}

        def hook(_module, _inputs, output):
            captured["context"] = _last_hidden(output)

        handle = model.layers[accepted_hidden_layer - 1].register_forward_hook(hook)
        try:
            torch.cuda.synchronize(self.device)
            with (
                inference_autocast(self.device, dtype),
                torch.cuda.graph(self.graph, capture_error_mode="thread_local"),
            ):
                self.output = self._forward()
            self.context = captured["context"]
        finally:
            handle.remove()
        self.reset()
        self.parity = self.validate()

    def _forward(self):
        return self.model(
            inputs_embeds=self.inputs,
            attention_mask=self.mask,
            position_ids=self.positions,
            cache_position=self.cache_positions,
            past_key_values=self.cache,
            use_cache=True,
            output_hidden_states=False,
        )

    def replay_audio(self, *, inputs_embeds, attention_mask, position_ids, cached_tokens):
        if cached_tokens != self._tokens:
            raise RuntimeError("fused Thinker audio cursor differs")
        self.inputs[:, :2].copy_(inputs_embeds)
        self.positions[:, :, :2].copy_(position_ids)
        self.mask.zero_()
        self.mask[:, :, :2, : attention_mask.shape[-1]].copy_(attention_mask)
        self.audio_replays += 1
        return SimpleNamespace(past_key_values=self.cache)

    def replay_text(self, *, inputs_embeds, attention_mask, position_ids, cached_tokens):
        if cached_tokens != self._tokens + 2 or cached_tokens + 1 > self.capacity.thinker_tokens:
            raise RuntimeError("fused Thinker text cursor/capacity differs")
        self.inputs[:, 2:].copy_(inputs_embeds)
        self.positions[:, :, 2:].copy_(position_ids)
        self.cache_positions.copy_(torch.arange(self._tokens, self._tokens + 3, device=self.device))
        self.mask[:, :, 2:, : attention_mask.shape[-1]].copy_(attention_mask)
        self.graph.replay()
        self._tokens += 3
        self.text_replays += 1
        hidden = [None] * (self.accepted_hidden_layer + 1)
        hidden[-1] = self.context[:, -1:]
        return SimpleNamespace(
            last_hidden_state=self.output.last_hidden_state[:, -1:],
            hidden_states=tuple(hidden),
            past_key_values=self.cache,
        )

    @torch.no_grad()
    def validate(self):
        self.reset()
        generator = torch.Generator(device=self.device).manual_seed(42)
        values = (
            torch.randn(
                (1, 6, self.model.config.hidden_size),
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            )
            * 0.1
        )
        expected = []
        with inference_autocast(self.device, self.dtype):
            for start, width in ((0, 2), (2, 1), (3, 2), (5, 1)):
                order = torch.arange(start, start + width, device=self.device)
                mask = (
                    torch.arange(self.capacity.thinker_tokens, device=self.device)[None]
                    <= order[:, None]
                )[None, None]
                result = self.model(
                    inputs_embeds=values[:, start : start + width],
                    attention_mask=mask,
                    position_ids=order[None, None].expand(4, 1, -1),
                    cache_position=order,
                    past_key_values=self.cache,
                    use_cache=True,
                    output_hidden_states=True,
                )
                if width == 1:
                    expected.append(
                        (
                            result.last_hidden_state.clone(),
                            result.hidden_states[self.accepted_hidden_layer].clone(),
                        )
                    )
        self.reset()
        fused_expected = []
        with inference_autocast(self.device, self.dtype):
            for index in range(2):
                start = index * 3
                order = torch.arange(start, start + 3, device=self.device)
                mask = (
                    torch.arange(self.capacity.thinker_tokens, device=self.device)[None]
                    <= order[:, None]
                )[None, None]
                result = self.model(
                    inputs_embeds=values[:, start : start + 3],
                    attention_mask=mask,
                    position_ids=order[None, None].expand(4, 1, -1),
                    cache_position=order,
                    past_key_values=self.cache,
                    use_cache=True,
                    output_hidden_states=True,
                )
                fused_expected.append(
                    (
                        result.last_hidden_state[:, -1:].clone(),
                        result.hidden_states[self.accepted_hidden_layer][:, -1:].clone(),
                    )
                )
        self.reset()
        maximum = 0.0
        for index in range(2):
            start = index * 3
            order = torch.arange(start, start + 3, device=self.device)
            mask = (
                torch.arange(self.capacity.thinker_tokens, device=self.device)[None]
                <= order[:, None]
            )[None, None]
            positions = order[None, None].expand(4, 1, -1)
            self.replay_audio(
                inputs_embeds=values[:, start : start + 2],
                attention_mask=mask[:, :, :2],
                position_ids=positions[:, :, :2],
                cached_tokens=start,
            )
            actual = self.replay_text(
                inputs_embeds=values[:, start + 2 : start + 3],
                attention_mask=mask[:, :, 2:],
                position_ids=positions[:, :, 2:],
                cached_tokens=start + 2,
            )
            for name, got, ref, fused_ref in zip(
                ("final", "context"),
                (actual.last_hidden_state, actual.hidden_states[-1]),
                expected[index],
                fused_expected[index],
                strict=True,
            ):
                print(
                    f"[live graph] fused parity frame={index} layer={name} graph_vs_fused_eager={float((got.float() - fused_ref.float()).abs().max())} fused_eager_vs_split={float((fused_ref.float() - ref.float()).abs().max())} cosine={float(F.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0))}",
                    flush=True,
                )
                torch.testing.assert_close(got, fused_ref, atol=0.04, rtol=0.02)
                maximum = max(maximum, float((got.float() - ref.float()).abs().max().item()))
                # BF16 changes reduction kernels when the query batch changes;
                # near-tied MoE routes can then differ. Compare graph correctness
                # against eager execution with the SAME fused shape above, and
                # record the split-shape difference instead of hiding it.
        self.reset()
        # Changing the final (future relative to audio) token must leave both
        # audio KV entries unchanged at EVERY layer of the fused forward.
        self.inputs.copy_(values[:, :3])
        self.positions.copy_(torch.arange(3, device=self.device)[None, None].expand(4, 1, -1))
        self.mask.zero_()
        self.mask[..., :3] = torch.ones((3, 3), device=self.device, dtype=torch.bool).tril()
        self.graph.replay()
        prefix = [
            (layer.keys[..., :2, :].clone(), layer.values[..., :2, :].clone())
            for layer in self.cache.layers
        ]
        self.reset()
        self.inputs[:, 2].add_(3.0)
        self.graph.replay()
        for layer, (key, value) in zip(self.cache.layers, prefix, strict=True):
            torch.testing.assert_close(layer.keys[..., :2, :], key, atol=0, rtol=0)
            torch.testing.assert_close(layer.values[..., :2, :], value, atol=0, rtol=0)
        self.reset()
        result = {
            "frames": 2,
            "split_bf16_maximum_absolute_difference": maximum,
            "graph_vs_fused_eager_passed": True,
            "future_token_prefix_invariance_passed": True,
            "passed": True,
        }
        print(f"[live graph] fused Thinker versus split causal forwards: {result}", flush=True)
        return result
