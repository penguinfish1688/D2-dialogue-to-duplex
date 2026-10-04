"""One model, one fixed cache allocation, and independent streaming sessions."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid

import torch

from d2.prompts import CONVERSATION_SYSTEM_PROMPT, encode_system_prompt
from d2_qwen.model.loading import load_model
from .cuda_graph import D2CudaGraphBackend
from .live_graph import LiveAudioGraphs
from .session import D2LiveSession, SamplingConfig, _MacroSession


class Runtime:
    input_rate = 16000
    output_rate = 24000
    native_ms = 80

    def __init__(
        self, source, *, device="cuda", revision=None, offline=False, kv_budget=2048, seed=1337
    ):
        if type(kv_budget) is not int or kv_budget < 128:
            raise ValueError("kv_budget must be an integer of at least 128 tokens")
        self.source, self.device = source, torch.device(device)
        self.revision, self.offline, self.kv_budget, self.seed = revision, offline, kv_budget, seed
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="d2-inference")
        self._active_live_session = False
        self._loaded = False

    async def _execute(self, callback, *args):
        def execute():
            with torch.no_grad():
                return callback(*args)

        return await asyncio.get_running_loop().run_in_executor(self._executor, execute)

    async def load(self):
        if not self._loaded:
            await self._execute(self._load)
            self._loaded = True

    @torch.no_grad()
    def _load(self):
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("D2 inference requires an NVIDIA CUDA GPU")
        torch.cuda.set_device(self.device.index if self.device.index is not None else 0)
        torch.manual_seed(self.seed)
        self.model, self.config = load_model(
            self.source, device=self.device, revision=self.revision, offline=self.offline
        )
        self.latency_ms = self.config["latency_ms"]
        self.k = self.latency_ms // self.native_ms
        self.prompt = encode_system_prompt(
            self.model.processor.tokenizer, CONVERSATION_SYSTEM_PROMPT
        )
        self.capacity_frames = (self.kv_budget // (3 * self.k)) * self.k
        self.max_context_frames = (
            (3 * self.capacity_frames - len(self.prompt)) // (3 * self.k)
        ) * self.k
        if self.max_context_frames < self.k:
            raise ValueError(
                "KV budget cannot hold the prompt and one complete interaction interval"
            )
        self.graphs = D2CudaGraphBackend(
            self.model,
            frames_per_unit=self.k,
            capacity_frames=self.capacity_frames,
            fuse_audio_text=self.k == 1,
        )
        self.audio_graphs = LiveAudioGraphs(self.model, self.capacity_frames)

    def metadata(self):
        return {
            "family": "qwen",
            "latency_ms": self.latency_ms,
            "input_sample_rate": self.input_rate,
            "output_sample_rate": self.output_rate,
            "kv_budget_tokens": self.kv_budget,
            "compute_dtype": "bfloat16",
            "max_conversation_seconds": self.max_context_frames * self.native_ms / 1000,
            "cuda_graphs": self.graphs.metadata(),
            "system_prompt": CONVERSATION_SYSTEM_PROMPT,
            "control_tokens": {
                "response": self.model.response_id,
                "interrupt": self.model.interrupt_id,
                "pad": self.model.pad_id,
            },
        }

    async def create_session(self):
        await self.load()
        return await self._execute(self._create_session)

    @torch.no_grad()
    def _create_session(self):
        if self._active_live_session:
            raise RuntimeError("A conversation is already active")
        torch.manual_seed(self.seed)
        session = _MacroSession(
            model=self.model,
            device=self.device,
            frames_per_unit=self.k,
            max_context_frames=self.max_context_frames,
            kv_cache_capacity_frames=self.capacity_frames,
            kv_cache_capacity_effective_ms=self.capacity_frames * self.native_ms,
            require_cuda_graphs=True,
            cuda_graph_backend=self.graphs,
            sampling=SamplingConfig(),
            checkpoint_metadata={"format": self.config["format"], "model": self.source},
            logical_frames=self.max_context_frames,
            session_id=uuid.uuid4().hex,
            system_prompt_kind="conversation",
            system_prompt_input_ids=self.prompt,
            live_audio_graphs=self.audio_graphs,
        )
        self._active_live_session = True
        return D2LiveSession(self, session, seed=self.seed)

    async def close(self):
        self._executor.shutdown(wait=True)
