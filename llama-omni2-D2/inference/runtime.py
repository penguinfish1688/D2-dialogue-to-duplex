"""Native streaming inference with bounded caches and captured decoder forwards."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import numpy as np
import torch

from d2.hub import load_release
from d2_llama.model.assets import resolve_assets
from d2_llama.model.loading import load_model
from d2_llama.model.features import CausalWhisperFeatureExtractor
from d2_llama.core.constants import TTS_PAD_TOKEN_ID
from d2_llama.prompts import task_prompt_ids
from .fixed_kv import GraphDecoder, FixedTalkerForward, talker_kv_capacity
from .encoder_graph import EncoderGraph
from .native import (
    _NativeRenderer,
    _NativeTTSSession,
    _feedback_16k,
    _protocol_argmax,
    _token_label,
)
from .session import (
    AudioEncoding,
    RenderedChunk,
    RuntimeState,
    TTSWrite,
    ThinkerPrefill,
    ThinkerStep,
    run_macro_unit,
)


def decode_pcm(payload):
    return torch.from_numpy(np.frombuffer(payload, dtype="<i2").astype(np.float32)) / 32768


def encode_pcm(wave):
    values = wave.detach().float().cpu().flatten()
    if not torch.isfinite(values).all():
        raise FloatingPointError("Renderer produced non-finite audio")
    return (
        values.mul(32768)
        .round()
        .clamp(-32768, 32767)
        .to(torch.int16)
        .numpy()
        .astype("<i2")
        .tobytes()
    )


@dataclass(frozen=True)
class LiveBatch:
    pcm_frames: tuple[bytes, ...]
    frame_events: tuple[dict, ...]


class Runtime:
    input_rate = 16000
    output_rate = 24000
    native_ms = 100

    def __init__(
        self, source, *, device="cuda", revision=None, offline=False, kv_budget=2048, seed=1337
    ):
        if type(kv_budget) is not int or kv_budget < 128:
            raise ValueError("KV budget must be an integer of at least 128 tokens")
        self.source, self.device, self.revision, self.offline = (
            source,
            torch.device(device),
            revision,
            offline,
        )
        self.kv_budget, self.seed = kv_budget, seed
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="d2-inference")
        self._loaded = False
        self._active_live_session = False

    async def _execute(self, callback, *args):
        def execute():
            # Native generate creates inference tensors. Keep cache creation,
            # reuse, and reset in the same mode across conversations.
            with torch.inference_mode():
                return callback(*args)

        return await asyncio.get_running_loop().run_in_executor(self._executor, execute)

    async def load(self):
        if not self._loaded:
            await self._execute(self._load)
            self._loaded = True

    def _load(self):
        load_release(self.source, family="llama", revision=self.revision, offline=self.offline)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("D2 inference requires an NVIDIA CUDA GPU")
        torch.cuda.set_device(self.device.index if self.device.index is not None else 0)
        torch.manual_seed(self.seed)
        self.model, self.config = load_model(
            self.source, device=self.device, revision=self.revision, offline=self.offline
        )
        self.tokenizer = self.model.tokenizer
        self.controls = self.model.control_rows.controls
        self.latency_ms = self.config["latency_ms"]
        self.k = self.latency_ms // 100
        self.prompt = task_prompt_ids(self.model, "conversation")
        self.max_context_frames = ((self.kv_budget - len(self.prompt)) // (3 * self.k)) * self.k
        self.extractor = CausalWhisperFeatureExtractor()
        self.encoder_graphs = [EncoderGraph(self.model.audio_tower) for _ in range(2)]
        assets = resolve_assets(self.config, offline=self.offline)
        self.renderer = _NativeRenderer(
            assets["source_root"],
            assets["cosy_snapshot"],
            assets["voice_prompt"],
            matcha_root=assets["matcha_root"],
        )
        self.thinker = GraphDecoder(self.model.thinker, self.kv_budget)
        self.talker = FixedTalkerForward(
            self.model.speech_generator.model.model, capacity=talker_kv_capacity(192)
        )
        self.model.speech_generator.model.model.forward = self.talker

    def metadata(self):
        return dict(
            family="llama",
            latency_ms=self.latency_ms,
            input_sample_rate=16000,
            output_sample_rate=24000,
            kv_budget_tokens=self.kv_budget,
            speech_kv_budget_tokens=self.talker.core.capacity,
            max_conversation_seconds=self.max_context_frames / 10,
            compute_dtype="bfloat16",
            waveform_dtype="float32",
            cuda_graphs=dict(
                prefill=True, serial_decode=True, audio_encoder=True, waveform_estimator=True
            ),
            control_tokens=dict(
                response=self.controls.response,
                interrupt=self.controls.interrupt,
                pad=self.controls.pad,
            ),
        )

    async def create_session(self):
        await self.load()
        return await self._execute(self._create_session)

    def _create_session(self):
        if self._active_live_session:
            raise RuntimeError("A conversation is already active")
        torch.manual_seed(self.seed)
        self.thinker.reset()
        self.talker.reset()
        for graph in self.encoder_graphs:
            graph.reset()
        prefix = torch.tensor([self.prompt], device=self.device, dtype=torch.long)
        self.thinker.step(self.model.thinker.embed_tokens(prefix))
        session = LiveSession(self)
        self._active_live_session = True
        return session

    async def close(self):
        self._executor.shutdown(wait=True)


class LiveSession:
    def __init__(self, owner):
        self.owner = owner
        self.buffer = bytearray()
        self.state = RuntimeState()
        self.frame = 0
        self.closed = False
        self.tokens = []
        self.response_tokens = 0
        self.summary = None

    @property
    def macro_bytes(self):
        return self.owner.k * 3200

    async def push_pcm16(self, payload):
        if not isinstance(payload, (bytes, bytearray, memoryview)) or len(payload) % 2:
            raise ValueError("Expected complete PCM16 samples")
        return await self.owner._execute(self._push_sync, bytes(payload))

    def _encode_lane(self, *, pcm_s16le_16k, state, frames_per_unit, lane):
        r = self.owner
        feature_state, aut_state, embedding, frame, history = (
            (None, None, None, 0, b"") if state is None else state
        )
        if lane == "self":
            history = (history + pcm_s16le_16k)[-self.macro_bytes :]
        if frame % r.k == 0:
            source = (
                self.macro_source
                if lane == "environment"
                else bytes(self.macro_bytes - len(history)) + history
            )
            features, feature_state = r.extractor.extract_streaming(
                decode_pcm(source), feature_state
            )
            graph = r.encoder_graphs[0 if lane == "environment" else 1]
            embedding = graph.push(features.to(device=r.device, dtype=torch.bfloat16))
            if embedding.shape[1] != r.k:
                raise ValueError("Audio encoder clock changed")
        return AudioEncoding(embedding, (feature_state, aut_state, embedding, frame + 1, history))

    def _decode(self, *, text_input, state, unit_index, text_index, protocol):
        r = self.owner
        environment, own = state
        text = r.model.control_rows.embed(
            r.model.thinker.embed_tokens, torch.tensor([[text_input]], device=r.device)
        )
        if self.frame % r.k == 0:
            packed = r.model.pack_thinker_embeddings(
                environment, own, text.expand(-1, r.k, -1), frames_per_unit=r.k
            )[:, : 2 * r.k + 1]
        else:
            packed = text
        hidden = r.thinker.step(packed)[:, -1, :]
        logits = r.model.control_rows.logits(r.model.official_model.lm_head, hidden)
        token = _protocol_argmax(
            logits,
            tokenizer=r.tokenizer,
            controls=r.controls,
            response_open=protocol.response_open,
            text_finished=protocol.text_finished,
        )
        if token in (r.controls.interrupt, r.controls.response):
            r.renderer.discard(protocol.renderer)
            r.talker.reset()
            self.response_tokens = 0
        elif protocol.response_open and not protocol.text_finished and token != r.controls.pad:
            self.response_tokens += 1
        return ThinkerStep(token, hidden, (environment, own))

    def _push_sync(self, payload):
        if self.closed:
            raise RuntimeError("Conversation is closed")
        self.buffer.extend(payload)
        r = self.owner
        frames, events = [], []
        while len(self.buffer) >= self.macro_bytes:
            if self.frame + r.k > r.max_context_frames:
                raise ValueError("Fixed KV budget exhausted; start a new conversation")
            self.macro_source = bytes(self.buffer[: self.macro_bytes])
            del self.buffer[: self.macro_bytes]
            for slot in range(r.k):
                old = self.state

                def write_tts(*, conditions, count, final, state):
                    tts = state or _NativeTTSSession(r.model.speech_generator, max_tail_tokens=1024)
                    units = tts.write(conditions, final=final)
                    return TTSWrite(units, tts, tts.finished)

                def render_speech(*, speech_units, final, state):
                    wave, next_state = r.renderer.render(speech_units, final=final, session=state)
                    return RenderedChunk(encode_pcm(wave), next_state)

                output = run_macro_unit(
                    frames_per_unit=1,
                    initial_text_token=TTS_PAD_TOKEN_ID,
                    response_token_id=r.controls.response,
                    interrupt_token_id=r.controls.interrupt,
                    pad_token_id=r.controls.pad,
                    environment_pcm_s16le_16k=self.macro_source[slot * 3200 : (slot + 1) * 3200],
                    state=old,
                    encode_audio=self._encode_lane,
                    prefill_thinker=lambda **kw: ThinkerPrefill(
                        (kw["environment"], kw["delayed_self"])
                    ),
                    decode_thinker=lambda **kw: self._decode(**kw, protocol=old),
                    write_tts=write_tts,
                    render_speech=render_speech,
                    max_tail_tokens=1024,
                )
                self.state = replace(
                    output.state,
                    previous_played_self_s16le_16k=encode_pcm(
                        _feedback_16k(decode_pcm(output.played_pcm_s16le_24k))
                    ),
                )
                if self.response_tokens >= 192 and not self.state.tts_finished:
                    flushed = b""
                    if self.state.renderer is not None:
                        wave, _ = r.renderer.render((), final=True, session=self.state.renderer)
                        flushed = encode_pcm(wave)
                    queue = self.state.playback_pcm_s16le_24k + flushed
                    self.state = replace(
                        self.state,
                        pending_conditions=(),
                        tts_finished=True,
                        text_finished=True,
                        renderer=None,
                        playback_pcm_s16le_24k=queue,
                        response_open=bool(queue),
                    )
                token = output.text_tokens[0]
                delta = (
                    r.tokenizer.decode([token], skip_special_tokens=True)
                    if token not in r.controls.values
                    else ""
                )
                self.tokens.append(token)
                events.append(
                    dict(
                        frame=self.frame,
                        token_id=token,
                        token=_token_label(r.tokenizer, token, r.controls),
                        text_delta=delta,
                        interrupt=token == r.controls.interrupt,
                        response=token == r.controls.response,
                    )
                )
                frames.append(output.played_pcm_s16le_24k)
                self.frame += 1
        return LiveBatch(tuple(frames), tuple(events))

    async def close(self):
        return await self.owner._execute(self._close_sync)

    def _close_sync(self):
        if not self.closed:
            self.owner.renderer.discard(self.state.renderer)
            self.summary = dict(
                processed_frames=self.frame,
                seed=self.owner.seed,
                dropped_partial_input_samples=len(self.buffer) // 2,
                generated_text=self.owner.tokenizer.decode(self.tokens, skip_special_tokens=True),
            )
            self.buffer.clear()
            self.closed = True
            self.owner._active_live_session = False
        return dict(self.summary)
