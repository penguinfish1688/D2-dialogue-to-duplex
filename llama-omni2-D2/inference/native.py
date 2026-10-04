from __future__ import annotations
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import Any, Sequence, Mapping
from d2_llama.core.constants import (
    INPUT_SAMPLES_PER_FRAME,
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    TTS_EOS_TOKEN_ID,
    TTS_PAD_TOKEN_ID,
    TTS_READ_TEXT_TOKENS,
    TTS_SEPARATOR_TOKEN_ID,
    TTS_TEXT_END_TOKEN_ID,
    TTS_TEXT_VOCAB_SIZE,
    TTS_UNIT_TOKEN_OFFSET,
    TTS_UNIT_VOCAB_SIZE,
    TTS_WRITE_SPEECH_TOKENS,
)


OUTPUT_SAMPLES_PER_FRAME = OUTPUT_SAMPLE_RATE // 10


MAX_NATIVE_TAIL_TOKENS = 1024


def _load_audio(path: Path, *, sample_rate: int) -> torch.Tensor:
    try:
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError("soundfile is required for D2 inference") from error
    values, source_rate = sf.read(str(path), dtype="float32", always_2d=False)
    waveform = torch.as_tensor(values).float()
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)
    if waveform.ndim != 1 or not waveform.numel():
        raise ValueError(f"inference audio is empty or non-mono: {path}")
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError(f"inference audio is non-finite: {path}")
    if int(source_rate) != int(sample_rate):
        target = max(1, round(waveform.numel() * sample_rate / int(source_rate)))
        waveform = F.interpolate(
            waveform.view(1, 1, -1),
            size=target,
            mode="linear",
            align_corners=False,
        ).view(-1)
    return waveform.contiguous()


def _discard_released_audio_copy(official_model: torch.nn.Module) -> None:
    owner = official_model.get_model()
    owner.speech_encoder = None
    owner.speech_projector = None


class _NativeTTSSession:
    """Native Read-3/Write-10 prefix, final text+SEP, then audio-only writes."""

    def __init__(
        self, speech_generator: torch.nn.Module, *, max_tail_tokens: int = MAX_NATIVE_TAIL_TOKENS
    ) -> None:
        if (
            isinstance(max_tail_tokens, bool)
            or not isinstance(max_tail_tokens, int)
            or max_tail_tokens < 1
        ):
            raise ValueError("native TTS tail token limit must be a positive integer")
        self.generator = speech_generator
        tokenizer = self.generator.tokenizer
        if (
            int(tokenizer.eos_token_id) != TTS_EOS_TOKEN_ID
            or int(tokenizer.convert_tokens_to_ids("<sep>")) != TTS_SEPARATOR_TOKEN_ID
            or int(tokenizer.convert_tokens_to_ids("<|im_end|>")) != TTS_TEXT_END_TOKEN_ID
        ):
            raise RuntimeError("released TTS text-end, separator, or audio EOS identity changed")
        self.prefix: torch.Tensor | None = None
        self.text_finished = False
        self.finished = False
        self.invalid_token_ids: list[int] = []
        self.missing_eos = False
        self.limit_reached = False
        self.max_tail_tokens = max_tail_tokens
        self.tail_generated_tokens = 0

    @torch.inference_mode()
    def write(
        self,
        conditions: Sequence[tuple[int, torch.Tensor]],
        *,
        final: bool = False,
    ) -> tuple[int, ...]:
        if self.finished:
            raise RuntimeError("native TTS session was already finalized")
        if self.text_finished:
            if conditions or final:
                raise ValueError(
                    "native TTS cannot receive conditions or a second SEP after text end"
                )
        elif final:
            if not 1 <= len(conditions) <= TTS_READ_TEXT_TOKENS:
                raise ValueError("native final TTS group requires one to three conditions")
            if int(conditions[-1][0]) != TTS_TEXT_END_TOKEN_ID:
                raise ValueError("native final TTS condition must be released text end")
        elif len(conditions) != TTS_READ_TEXT_TOKENS:
            raise ValueError("nonfinal native TTS write requires exactly three conditions")
        lexical = conditions[:-1] if final else conditions
        for token, _ in lexical:
            if not 0 <= int(token) < TTS_TEXT_VOCAB_SIZE or int(token) in (
                TTS_PAD_TOKEN_ID,
                TTS_TEXT_END_TOKEN_ID,
                TTS_SEPARATOR_TOKEN_ID,
            ):
                raise ValueError("native TTS lexical conditions contain a control/ending token")
        device = next(self.generator.parameters()).device
        embedding = self.generator.model.get_input_embeddings()
        if conditions:
            ids = torch.tensor(
                [int(token) for token, _ in conditions], dtype=torch.long, device=device
            )
            hidden = torch.cat(
                [value.reshape(1, -1).to(device=device) for _, value in conditions], dim=0
            )
            fused = self.generator.fusion(self.generator.input_proj(hidden), embedding(ids))
            self.prefix = fused if self.prefix is None else torch.cat((self.prefix, fused), dim=0)
        if final:
            separator = embedding(
                torch.tensor([TTS_SEPARATOR_TOKEN_ID], dtype=torch.long, device=device)
            )
            self.prefix = torch.cat((self.prefix, separator), dim=0)
            self.text_finished = True
        if self.prefix is None:
            raise RuntimeError("native TTS cannot generate without a conditioned prefix")
        count = TTS_WRITE_SPEECH_TOKENS
        if self.text_finished:
            count = min(count, self.max_tail_tokens - self.tail_generated_tokens)
        generated = self.generator.model.generate(
            inputs_embeds=self.prefix.unsqueeze(0),
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            num_beams=1,
            max_new_tokens=count,
            pad_token_id=int(self.generator.tokenizer.pad_token_id),
            eos_token_id=int(self.generator.tokenizer.eos_token_id),
        )[0]
        raw: list[int] = []
        eos = int(self.generator.tokenizer.eos_token_id)
        consumed = 0
        if generated.ndim != 1 or not 1 <= generated.numel() <= count:
            raise RuntimeError("native TTS generate returned an invalid burst length")
        for value in generated.detach().cpu().tolist():
            consumed += 1
            token = int(value)
            if token == eos:
                self.finished = True
                break
            if TTS_UNIT_TOKEN_OFFSET <= token < TTS_UNIT_TOKEN_OFFSET + TTS_UNIT_VOCAB_SIZE:
                raw.append(token - TTS_UNIT_TOKEN_OFFSET)
            else:
                self.invalid_token_ids.append(token)
                self.missing_eos = True
                self.finished = True
                break
        self.prefix = torch.cat((self.prefix, embedding(generated[:consumed])), dim=0)
        if self.text_finished:
            self.tail_generated_tokens += consumed
        if not self.finished and len(raw) != count:
            self.missing_eos = True
            self.finished = True
        if (
            not self.finished
            and self.text_finished
            and self.tail_generated_tokens >= self.max_tail_tokens
        ):
            self.limit_reached = True
            self.missing_eos = True
            self.finished = True
        return tuple(raw)


class _NativeRenderer:
    def __init__(
        self, source_root: Path, model_dir: Path, voice_prompt: Path, *, matcha_root: Path
    ) -> None:
        import logging
        import sys
        import types

        source = str(source_root)
        if source not in sys.path:
            sys.path.insert(0, source)
        if not (matcha_root / "matcha/models/components/flow_matching.py").is_file():
            raise FileNotFoundError("staged Matcha-TTS 0.0.5.1 renderer dependency is unavailable")
        matcha = str(matcha_root)
        if matcha not in sys.path:
            sys.path.insert(0, matcha)
        # Matcha 0.0.5.1 eagerly imports its training-only Hydra/Lightning
        # utilities from package __init__.  The renderer only needs its
        # logger helper; provide that tiny API without loading the incompatible
        # Python-3.12 training stack or changing any model computation.
        if "matcha.utils.pylogger" not in sys.modules:
            utils_module = types.ModuleType("matcha.utils")
            utils_module.__path__ = [str(matcha_root / "matcha" / "utils")]
            logger_module = types.ModuleType("matcha.utils.pylogger")
            logger_module.get_pylogger = logging.getLogger
            sys.modules["matcha.utils"] = utils_module
            sys.modules["matcha.utils.pylogger"] = logger_module
        import numpy as np
        import onnxruntime
        from hyperpyyaml import load_hyperpyyaml

        renderer_config = Path(__file__).resolve().parents[1] / "configs" / "cosy2_renderer.yaml"
        with renderer_config.open("r", encoding="utf-8") as handle:
            config = load_hyperpyyaml(handle)
        if int(config["sample_rate"]) != OUTPUT_SAMPLE_RATE:
            raise RuntimeError("CosyVoice2 output sample rate changed")
        self.device = torch.device("cuda")
        self.flow = config["flow"]
        self.flow.load_state_dict(
            torch.load(model_dir / "flow.pt", map_location="cpu"), strict=True
        )
        self.flow.to(self.device).eval()
        self.flow.decoder.fp16 = False
        # Keep the native mask's values and visibility, with a vectorized
        # construction shared by the renderer's encoder and diffusion steps.
        from cosyvoice.utils import mask
        from .mask import subsequent_chunk_mask

        mask.subsequent_chunk_mask = subsequent_chunk_mask
        self.hift = config["hift"]
        hift_state = {
            name.replace("generator.", ""): value
            for name, value in torch.load(model_dir / "hift.pt", map_location="cpu").items()
        }
        self.hift.load_state_dict(hift_state, strict=True)
        self.hift.to(self.device).eval()
        self.flow.encoder.static_chunk_size = 2 * self.flow.input_frame_rate
        self.flow.decoder.estimator.static_chunk_size = (
            2 * self.flow.input_frame_rate * self.flow.token_mel_ratio
        )
        from .renderer_graph import EstimatorGraph

        self.flow.decoder.forward_estimator = EstimatorGraph(self.flow.decoder.forward_estimator)
        self.mel_cache_len = 8
        self.source_cache_len = self.mel_cache_len * 480
        self.speech_window = np.hamming(2 * self.source_cache_len)
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        self.campplus = onnxruntime.InferenceSession(
            str(model_dir / "campplus.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        prompt = _load_audio(voice_prompt, sample_rate=INPUT_SAMPLE_RATE)
        self.prompt = prompt.unsqueeze(0)

    def new_session(self) -> dict[str, Any]:
        import torchaudio.compliance.kaldi as kaldi

        features = kaldi.fbank(
            self.prompt,
            num_mel_bins=80,
            dither=0,
            sample_frequency=INPUT_SAMPLE_RATE,
        )
        features = features - features.mean(dim=0, keepdim=True)
        embedding = self.campplus.run(
            None,
            {self.campplus.get_inputs()[0].name: features.unsqueeze(0).cpu().numpy()},
        )[0]
        return {
            "embedding": torch.as_tensor(embedding, device=self.device).reshape(1, -1),
            "token_offset": 0,
            "generated_tokens": None,
            "hift_cache": None,
        }

    def discard(self, session: Mapping[str, Any] | None) -> None:
        if isinstance(session, dict):
            session.clear()

    @staticmethod
    def _fade_in_out(
        fade_in: torch.Tensor,
        fade_out: torch.Tensor,
        window: Any,
    ) -> torch.Tensor:
        device = fade_in.device
        fade_in = fade_in.cpu()
        fade_out = fade_out.cpu()
        overlap = int(window.shape[0] // 2)
        fade_in[..., :overlap] = (
            fade_in[..., :overlap] * window[:overlap] + fade_out[..., -overlap:] * window[overlap:]
        )
        return fade_in.to(device)

    def _token_to_wav(
        self,
        tokens: torch.Tensor,
        *,
        token_offset: int,
        final: bool,
        session: dict[str, Any],
    ) -> torch.Tensor:
        prompt_token = torch.zeros(1, 0, dtype=torch.int32, device=self.device)
        prompt_features = torch.zeros(1, 0, 80, device=self.device)
        token_batch = tokens.unsqueeze(0).to(self.device)
        mel, _ = self.flow.inference(
            token=token_batch,
            token_len=torch.tensor([token_batch.shape[1]], dtype=torch.int32, device=self.device),
            prompt_token=prompt_token,
            prompt_token_len=torch.tensor([0], dtype=torch.int32, device=self.device),
            prompt_feat=prompt_features,
            prompt_feat_len=torch.tensor([0], dtype=torch.int32, device=self.device),
            embedding=session["embedding"],
            finalize=bool(final),
        )
        mel = mel[:, :, int(token_offset) * self.flow.token_mel_ratio :]
        cache = session["hift_cache"]
        if cache is None:
            cache_source = torch.zeros(1, 1, 0)
        else:
            mel = torch.cat((cache["mel"], mel), dim=2)
            cache_source = cache["source"]
        if final:
            speech, _ = self.hift.inference(speech_feat=mel, cache_source=cache_source)
            if cache is not None:
                speech = self._fade_in_out(speech, cache["speech"], self.speech_window)
            return speech

        speech, source = self.hift.inference(speech_feat=mel, cache_source=cache_source)
        if cache is not None:
            speech = self._fade_in_out(speech, cache["speech"], self.speech_window)
        session["hift_cache"] = {
            "mel": mel[:, :, -self.mel_cache_len :],
            "source": source[:, :, -self.source_cache_len :],
            "speech": speech[:, -self.source_cache_len :],
        }
        return speech[:, : -self.source_cache_len]

    @torch.inference_mode()
    def render(
        self,
        units: Sequence[int],
        *,
        final: bool,
        session: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        if session is None:
            if not units:
                return torch.empty(0), None
            session = self.new_session()
        chunk = torch.tensor(tuple(int(value) for value in units), dtype=torch.long)
        if session["generated_tokens"] is None:
            session["generated_tokens"] = chunk
        else:
            session["generated_tokens"] = torch.cat((session["generated_tokens"], chunk), dim=0)
        generated = session["generated_tokens"]
        waveform = self._token_to_wav(
            generated,
            token_offset=int(session["token_offset"]),
            final=bool(final),
            session=session,
        )
        session["token_offset"] = (
            int(generated.numel())
            if final
            else int(generated.numel()) - int(self.flow.pre_lookahead_len)
        )
        result = waveform.detach().float().cpu().flatten().contiguous()
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("CosyVoice2 renderer emitted non-finite PCM")
        if final:
            self.discard(session)
            session = None
        return result, session


def _feedback_16k(played_24k: torch.Tensor) -> torch.Tensor:
    if int(played_24k.numel()) != OUTPUT_SAMPLES_PER_FRAME:
        raise ValueError("played output must contain exactly one 100-ms frame")
    return (
        F.interpolate(
            played_24k.view(1, 1, -1),
            size=INPUT_SAMPLES_PER_FRAME,
            mode="linear",
            align_corners=False,
        )
        .view(-1)
        .contiguous()
    )


def _token_label(tokenizer: Any, token: int, controls: Any) -> str:
    if token == controls.response:
        return "<|duplex_response|>"
    if token == controls.interrupt:
        return "<|duplex_interrupt|>"
    if token == controls.pad:
        return "<|duplex_pad|>"
    if 0 <= token < len(tokenizer):
        return tokenizer.decode([token], skip_special_tokens=False)
    return f"<invalid:{token}>"


def _protocol_argmax(
    logits: torch.Tensor,
    *,
    tokenizer: Any,
    controls: Any,
    response_open: bool,
    text_finished: bool = False,
) -> int:
    """Greedy selection over tokens valid in the current D2 state."""

    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("D2 inference logits must be [1,vocabulary]")
    scores = logits[0].float()
    allowed = torch.zeros_like(scores, dtype=torch.bool)
    if response_open and text_finished:
        allowed[controls.pad] = True
        allowed[controls.interrupt] = True
    elif response_open:
        allowed[:TTS_TEXT_VOCAB_SIZE] = True
        for token in tokenizer.all_special_ids:
            if 0 <= int(token) < TTS_TEXT_VOCAB_SIZE:
                allowed[int(token)] = False
        allowed[controls.interrupt] = True
        allowed[controls.pad] = True
    else:
        for token in controls.values:
            allowed[token] = True
    return int(scores.masked_fill(~allowed, float("-inf")).argmax().item())
