from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
import torch
from torch import nn
from .constants import CODEC_SILENCE_FRAME
from .lora import LoRALinear, LoRAConfig, apply_lora_to_suffixes, freeze_module
from .utils import (
    TrainableTextControlRow,
    TrainableAudioControlRows,
)
from .config import D2Config, validate_chunk_frames
from .base import (
    QwenDuplexModel as _ReferenceQwenDuplexModel,
    QwenDuplexTransformerXLState,
    shift_right_text,
    weighted_text_ce_terms,
)
from .encoder import SFT_ENCODER_PARAMETERS

MEL_FRAMES_PER_AUDIO_FRAME = 8

QWEN_MACRO_ATTENTION_POLICY = "serialized_env_self_text_causal"
QWEN_MACRO_ATTENTION_POLICY_VERSION = 1


ATTENTION_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)


D2_SFT_TRAINABLE_PARAMETERS = 266_286_976


def _optimizer_group_name(name: str) -> str:
    """Classify one D2 trainable into its unique learning-rate group."""

    if name.startswith("qwen.thinker.model.") and ".lora_" in name:
        return "thinker_lora"
    if (
        name.startswith(("qwen.talker.text_projection.", "qwen.talker.hidden_projection."))
        and ".lora_" in name
    ):
        return "talker_context_lora"
    if name.startswith("qwen.talker.code_predictor.model.") and ".lora_" in name:
        return "code_predictor_lora"
    if name.startswith("qwen.talker.model.layers.") and (
        ".lora_" in name or ".fresh_lora_" in name
    ):
        return "talker_lora"
    if (
        name.startswith(("text_control_rows.", "audio_control_rows.", "stream_type_embedding."))
        or name == "self_silence_embed"
    ):
        return "control_and_stream_rows"
    if ".codec_embedding." in name:
        return "codec_embeddings"
    if name.startswith("qwen.talker.codec_head.") or ".lm_head." in name:
        return "codec_output_heads"
    return "small_parts"


def pack_macro_triplets(
    environment_audio: torch.Tensor,
    delayed_self_audio: torch.Tensor,
    text_inputs: torch.Tensor,
    frames_per_unit: int,
) -> torch.Tensor:
    """Pack ``env[k] | self[k] | text[k]`` for every macro unit."""

    if environment_audio.shape != delayed_self_audio.shape:
        raise ValueError(
            "environment/self embedding shapes differ: "
            f"{tuple(environment_audio.shape)} != {tuple(delayed_self_audio.shape)}"
        )
    if environment_audio.ndim != 3 or text_inputs.ndim != 3:
        raise ValueError("macro streams must be rank-3 [batch,frames,hidden]")
    if environment_audio.shape != text_inputs.shape:
        raise ValueError(
            "audio/text embedding shapes differ: "
            f"{tuple(environment_audio.shape)} != {tuple(text_inputs.shape)}"
        )
    batch, frames, hidden = map(int, environment_audio.shape)
    validate_chunk_frames(frames, int(frames_per_unit))
    units = frames // int(frames_per_unit)
    shape = (batch, units, int(frames_per_unit), hidden)
    return torch.cat(
        (
            environment_audio.reshape(shape),
            delayed_self_audio.reshape(shape),
            text_inputs.reshape(shape),
        ),
        dim=2,
    ).reshape(batch, frames * 3, hidden)


def select_macro_text_slots(
    hidden_states: torch.Tensor,
    frames_per_unit: int,
) -> torch.Tensor:
    """Recover the ``k`` text readouts from every packed macro unit."""

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must be [batch,tokens,hidden]")
    batch, tokens, hidden = map(int, hidden_states.shape)
    width = 3 * int(frames_per_unit)
    if tokens < 1 or tokens % width:
        raise ValueError(f"packed token length {tokens} must be a multiple of {width}")
    units = tokens // width
    return hidden_states.reshape(batch, units, width, hidden)[
        :, :, 2 * int(frames_per_unit) :, :
    ].reshape(batch, units * int(frames_per_unit), hidden)


def prepend_system_prompt_to_macro_layout(
    *,
    prompt_embeddings: torch.Tensor,
    macro_embeddings: torch.Tensor,
    macro_positions: torch.Tensor,
    macro_attention: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepend ordinary causal text tokens to one sample's first macro chunk."""

    if prompt_embeddings.ndim != 3 or macro_embeddings.ndim != 3:
        raise ValueError("prompt and macro embeddings must be [batch,tokens,hidden]")
    if int(prompt_embeddings.shape[0]) != int(macro_embeddings.shape[0]) or int(
        prompt_embeddings.shape[2]
    ) != int(macro_embeddings.shape[2]):
        raise ValueError("prompt and macro embedding batch/hidden shapes differ")
    batch, prompt_tokens, _hidden = map(int, prompt_embeddings.shape)
    macro_tokens = int(macro_embeddings.shape[1])
    if prompt_tokens < 1:
        raise ValueError("system prompt must contain at least one token")
    if tuple(macro_positions.shape) != (4, batch, macro_tokens):
        raise ValueError("macro position IDs do not match macro embeddings")
    if tuple(macro_attention.shape) != (batch, 1, macro_tokens, macro_tokens):
        raise ValueError("first-chunk macro attention must have no cached prefix")

    device = macro_embeddings.device
    prompt_order = torch.arange(prompt_tokens, device=device, dtype=torch.long)
    prompt_positions = prompt_order[None, None, :].expand(4, batch, -1)
    positions = torch.cat(
        (prompt_positions, macro_positions + prompt_tokens),
        dim=2,
    ).contiguous()

    prompt_causal = (
        torch.ones(
            (prompt_tokens, prompt_tokens),
            dtype=torch.bool,
            device=device,
        )
        .tril()[None, :, :]
        .expand(batch, -1, -1)
    )
    prompt_rows = torch.cat(
        (
            prompt_causal,
            torch.zeros(
                (batch, prompt_tokens, macro_tokens),
                dtype=torch.bool,
                device=device,
            ),
        ),
        dim=2,
    )
    macro_rows = torch.cat(
        (
            torch.ones(
                (batch, macro_tokens, prompt_tokens),
                dtype=torch.bool,
                device=device,
            ),
            macro_attention[:, 0],
        ),
        dim=2,
    )
    attention = torch.cat((prompt_rows, macro_rows), dim=1)[:, None].contiguous()
    embeddings = torch.cat((prompt_embeddings, macro_embeddings), dim=1)
    return embeddings, positions, attention


def macro_position_ids(
    *,
    batch_size: int,
    frame_count: int,
    frames_per_unit: int,
    frame_offset: int,
    device: torch.device,
) -> torch.Tensor:
    """Build Qwen's 4-row token/TM-RoPE positions for macro layout."""

    frames_per_unit = int(frames_per_unit)
    validate_chunk_frames(int(frame_count), frames_per_unit)
    if int(frame_offset) < 0 or int(frame_offset) % frames_per_unit:
        raise ValueError(f"frame_offset={frame_offset} must align to {frames_per_unit} frames")
    units = int(frame_count) // frames_per_unit
    unit_offset = int(frame_offset) // frames_per_unit
    width = 3 * frames_per_unit
    local = torch.arange(units * width, device=device, dtype=torch.long)
    unit = torch.div(local, width, rounding_mode="floor") + unit_offset
    slot = local.remainder(width)
    token_order = local + unit_offset * width
    frame_position = unit * frames_per_unit + slot.remainder(frames_per_unit)
    temporal = torch.where(slot < 2 * frames_per_unit, frame_position, token_order)
    rows = torch.stack((token_order, temporal, temporal, temporal), dim=0)
    return rows[:, None, :].expand(-1, int(batch_size), -1).contiguous()


def macro_attention_mask(
    *,
    frame_mask: torch.Tensor,
    frames_per_unit: int,
    cached_tokens: int = 0,
    valid_cached_tokens: int | None = None,
) -> torch.Tensor:
    """Return the 4-D boolean D2 mask (``True`` means attend).

    Cached memory and completed units are causal history.  The current unit is
    causal in its serialized ``env[k] | self[k] | text[k]`` order.  Text can
    therefore use the complete current audio prefix without allowing an audio
    query to see a future audio token.
    """

    if frame_mask.ndim != 2:
        raise ValueError("frame_mask must be [batch,frames]")
    batch, frames = map(int, frame_mask.shape)
    frames_per_unit = int(frames_per_unit)
    validate_chunk_frames(frames, frames_per_unit)
    cached_tokens = int(cached_tokens)
    if cached_tokens < 0:
        raise ValueError("cached_tokens must be non-negative")
    if valid_cached_tokens is None:
        valid_cached_tokens = cached_tokens
    valid_cached_tokens = int(valid_cached_tokens)
    if not 0 <= valid_cached_tokens <= cached_tokens:
        raise ValueError(
            f"valid_cached_tokens must lie inside the cache: {valid_cached_tokens}/{cached_tokens}"
        )

    units = frames // frames_per_unit
    width = 3 * frames_per_unit
    query_tokens = units * width
    positions = torch.arange(query_tokens, device=frame_mask.device)
    query_unit = torch.div(positions, width, rounding_mode="floor")[:, None]
    key_unit = torch.div(positions, width, rounding_mode="floor")[None, :]
    query_slot = positions.remainder(width)[:, None]
    key_slot = positions.remainder(width)[None, :]
    same_unit = query_unit == key_unit
    current_allowed = (key_unit < query_unit) | (same_unit & (key_slot <= query_slot))
    current_allowed = current_allowed[None, :, :].expand(batch, -1, -1)

    validity = frame_mask.to(dtype=torch.bool).reshape(batch, units, frames_per_unit)
    current_key_valid = torch.cat((validity, validity, validity), dim=2).reshape(
        batch, query_tokens
    )
    current_allowed = current_allowed & current_key_valid[:, None, :]

    if cached_tokens:
        past_valid = torch.zeros((batch, cached_tokens), dtype=torch.bool, device=frame_mask.device)
        if valid_cached_tokens:
            past_valid[:, -valid_cached_tokens:] = True
        allowed = torch.cat(
            (
                past_valid[:, None, :].expand(-1, query_tokens, -1),
                current_allowed,
            ),
            dim=-1,
        )
    else:
        allowed = current_allowed

    # Padded queries do not contribute to loss, but SDPA must not receive an
    # entirely masked row.  Its own current-token diagonal is harmless.
    empty = ~allowed.any(dim=-1)
    if bool(empty.any().item()):
        diagonal = torch.zeros_like(allowed)
        diagonal[:, positions, cached_tokens + positions] = True
        allowed = allowed | (empty[:, :, None] & diagonal)
    return allowed[:, None, :, :].contiguous()


def _rolling_self_history(
    self_audio: torch.Tensor,
    *,
    delay_frames: int,
    silence: torch.Tensor,
    previous: torch.Tensor | None,
) -> torch.Tensor:
    if self_audio.ndim != 3:
        raise ValueError("self_audio must be [batch,frames,hidden]")
    batch, _frames, hidden = map(int, self_audio.shape)
    if delay_frames < 1:
        raise ValueError("delay_frames must be positive")
    if previous is None:
        prefix = (
            silence.to(device=self_audio.device, dtype=self_audio.dtype)
            .view(1, 1, hidden)
            .expand(batch, delay_frames, hidden)
        )
    else:
        expected = (batch, delay_frames, hidden)
        if tuple(previous.shape) != expected:
            raise ValueError(f"previous self history shape {tuple(previous.shape)} != {expected}")
        # Keep the trainable silence row in every continuation-chunk graph.
        # Its numerical contribution is intentionally zero once real history is
        # available, but DDP with find_unused_parameters=False still requires
        # the parameter to participate in every backward pass.
        silence_dependency = silence.to(
            device=self_audio.device,
            dtype=self_audio.dtype,
        ).view(1, 1, hidden)
        prefix = (
            previous.to(
                device=self_audio.device,
                dtype=self_audio.dtype,
            )
            + silence_dependency * 0.0
        )
    return torch.cat((prefix, self_audio), dim=1)[:, -delay_frames:, :]


@dataclass(frozen=True)
class MacroAudioPrefill:
    """Inputs for the one-call, causal ``2k`` audio prefill."""

    inputs_embeds: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    next_self_history: torch.Tensor


@dataclass(frozen=True)
class MacroTextStep:
    """Inputs for one of the ``k`` sequential AR text calls."""

    inputs_embeds: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    text_index: int


TALKER_TARGETS = ATTENTION_TARGETS + (
    "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
)
CODE_PREDICTOR_TARGETS = ATTENTION_TARGETS + ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def d2_lora_defaults() -> dict[str, dict[str, Any]]:
    """Return fresh dictionaries for D2's rank-128 trainable adapters."""
    common = {"rank": 128, "alpha": 256.0, "dropout": 0.0}
    return {
        "thinker_lora": {
            **common,
            "target_suffixes": list(ATTENTION_TARGETS),
            "expected_wrapped_module_count": 192,
        },
        "talker_lora": {
            **common,
            "target_suffixes": list(TALKER_TARGETS),
            "expected_wrapped_module_count": 140,
        },
        "talker_context_lora": {
            **common,
            "target_suffixes": ["linear_fc1", "linear_fc2"],
            "expected_wrapped_module_count": 4,
        },
        "code_predictor_lora": {
            **common,
            "target_suffixes": list(CODE_PREDICTOR_TARGETS),
            "expected_wrapped_module_count": 35,
        },
    }


class D2Qwen3OmniModel(_ReferenceQwenDuplexModel):
    def __init__(self, qwen, processor, audio_encoder, *, latency_ms):
        nn.Module.__init__(self)
        self.qwen, self.processor, self.audio_encoder = qwen, processor, audio_encoder
        self._compute_dtype = torch.bfloat16
        self.d2_config = D2Config(latency_ms=latency_ms)
        self._control_signal_contract = "interrupt_response_v1"
        self.space_id = int(processor.tokenizer.encode(" ", add_special_tokens=False)[0])
        from .utils import ensure_duplex_control_token

        self.pad_id = ensure_duplex_control_token(
            processor.tokenizer, vocab_size=qwen.config.thinker_config.text_config.vocab_size
        )
        self.assistant_start_id = self.response_id = 151669
        self.assistant_end_id = self.interrupt_id = 151670
        self.pad_weight = self.inside_response_pad_weight = 0.05
        self.assistant_start_weight = self.assistant_end_weight = 20.0
        self.interrupt_end_overwritten_text_weight = 20.0
        self.response_text_weight_policy = "split_pad_by_response_span"
        self.text_loss_weight = self.codec_loss_weight = 1.0
        self.talker_codec_eos_weight = 20.0
        self.talker_codec_bos_id = int(qwen.config.talker_config.codec_bos_id)
        self.talker_codec_eos_id = int(qwen.config.talker_config.codec_eos_token_id)
        self.talker_codec_vocab_size = int(qwen.config.talker_config.text_config.vocab_size)
        self.talker_speaker_id = int(qwen.config.talker_config.speaker_id["chelsie"])
        self.accept_hidden_layer = int(qwen.config.talker_config.accept_hidden_layer)
        self.text_tokenizer_size = len(processor.tokenizer)
        self.forbidden_text_special_ids = tuple(
            sorted(
                set(processor.tokenizer.all_special_ids)
                - {self.pad_id, self.response_id, self.interrupt_id}
            )
        )
        defaults = d2_lora_defaults()
        freeze_module(self.qwen)
        for module, key in (
            (qwen.thinker.model, "thinker_lora"),
            (qwen.talker.model, "talker_lora"),
            (qwen.talker.text_projection, "talker_context_lora"),
            (qwen.talker.hidden_projection, "talker_context_lora"),
            (qwen.talker.code_predictor.model, "code_predictor_lora"),
        ):
            config = dict(defaults[key])
            config.pop("expected_wrapped_module_count", None)
            apply_lora_to_suffixes(module, LoRAConfig.from_mapping(config))
        for name, module in qwen.talker.model.named_modules():
            if isinstance(module, LoRALinear) and ".mlp.shared_expert." in name:
                module.enable_fresh_adapter(rank=128, alpha=256.0, dropout=0.0)
        hidden = qwen.config.thinker_config.text_config.hidden_size
        self.stream_type_embedding = nn.Embedding(2, hidden, dtype=torch.float32)
        nn.init.zeros_(self.stream_type_embedding.weight)
        emb, head = qwen.thinker.get_input_embeddings(), qwen.thinker.lm_head
        self.text_control_rows = TrainableTextControlRow(
            pad_id=self.pad_id,
            space_input_row=emb.weight[self.space_id],
            space_output_row=head.weight[self.space_id],
        )
        self.audio_control_rows = TrainableAudioControlRows(
            assistant_start_id=self.response_id,
            assistant_end_id=self.interrupt_id,
            start_input_row=emb.weight[self.response_id],
            end_input_row=emb.weight[self.interrupt_id],
            start_output_row=head.weight[self.response_id],
            end_output_row=head.weight[self.interrupt_id],
        )
        self.self_silence_embed = nn.Parameter(torch.zeros(hidden, dtype=torch.float32))
        self.register_buffer(
            "codec_silence_frame", torch.tensor(CODEC_SILENCE_FRAME, dtype=torch.long)
        )
        self.register_buffer("codec_codebook_weights", torch.ones(16, dtype=torch.float32))
        self._configure_d2_training_policy()

    def configure_training_modes(self):
        self.qwen.thinker.eval()
        self.qwen.talker.eval()
        self.qwen.talker.code_predictor.train(self.training)
        self.audio_encoder.encoder.configure_sft_training_mode()

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:
            self.configure_training_modes()
        return self

    """One implementation shared by the 80/160/320/640/1040 ms series."""

    @property
    def frames_per_unit(self) -> int:
        return self.d2_config.frames_per_unit

    @property
    def self_audio_delay_frames(self) -> int:
        return self.d2_config.self_audio_delay_frames

    def _configure_d2_training_policy(self) -> None:
        """Enable the D2 training parameter groups."""
        freeze_module(self)
        self._frozen_carry_names = set()
        self.train_codec_interface = True
        self.train_codec0_interface = True
        self.train_codec0_head = True
        self.train_code_predictor_heads = tuple(range(15))
        self.train_full_thinker_attention = False
        self.train_full_talker_attention = False
        self.train_full_talker_shared_expert = False
        self.train_full_code_predictor = False
        self.train_talker_context_lora = True
        self.freeze_talker = False
        self._enable_code_predictor_checkpointing()
        thinker_count = 0
        for name, module in self.qwen.thinker.model.named_modules():
            if isinstance(module, LoRALinear) and any(
                (name.endswith(target) for target in ATTENTION_TARGETS)
            ):
                self._enable_standard_adapter(module, name)
                thinker_count += 1
        if thinker_count != 192:
            raise RuntimeError(f"Expected 192 Thinker attention LoRAs, got {thinker_count}")
        talker_attention_count = 0
        talker_shared_count = 0
        for name, module in self.qwen.talker.model.named_modules():
            if not isinstance(module, LoRALinear):
                continue
            if ".self_attn." in name and any(
                (name.endswith(target) for target in ATTENTION_TARGETS)
            ):
                self._enable_standard_adapter(module, name)
                talker_attention_count += 1
            elif ".mlp.shared_expert." in name:
                module.disable_zero_adapter()
                self._enable_fresh_adapter(module, name)
                talker_shared_count += 1
        if talker_attention_count != 80 or talker_shared_count != 60:
            raise RuntimeError(
                f"Talker LoRA inventory mismatch: attention={talker_attention_count}/80 shared={talker_shared_count}/60"
            )
        context_count = self._enable_matching_standard_adapters(
            self.qwen.talker, prefixes=("text_projection.", "hidden_projection.")
        )
        predictor_count = self._enable_matching_standard_adapters(
            self.qwen.talker.code_predictor.model
        )
        if context_count != 4 or predictor_count != 35:
            raise RuntimeError(
                f"Talker adapter inventory mismatch: context={context_count}/4 code_predictor={predictor_count}/35"
            )
        for module in (
            self.qwen.talker.model.codec_embedding,
            self.qwen.talker.codec_head,
            self.qwen.talker.code_predictor.model.codec_embedding,
            self.qwen.talker.code_predictor.lm_head,
        ):
            module.to(dtype=torch.float32)
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        for module in (self.stream_type_embedding, self.text_control_rows, self.audio_control_rows):
            module.to(dtype=torch.float32)
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.self_silence_embed.data = self.self_silence_embed.data.float()
        self.self_silence_embed.requires_grad_(True)
        for name, parameter in self.named_parameters():
            small = (
                name.endswith((".q_norm.weight", ".k_norm.weight"))
                or name.endswith(".mlp.shared_expert_gate.weight")
                or (
                    name.startswith(
                        ("qwen.talker.text_projection.", "qwen.talker.hidden_projection.")
                    )
                    and name.endswith(".base.bias")
                )
            )
            if small:
                parameter.requires_grad_(True)
        if self.audio_encoder.training_lora_enabled:
            self.audio_encoder.enable_sft_parameters()
            encoder_trainable = sum(
                (
                    parameter.numel()
                    for parameter in self.audio_encoder.parameters()
                    if parameter.requires_grad
                )
            )
            if encoder_trainable != SFT_ENCODER_PARAMETERS:
                raise RuntimeError(
                    f"D2 SFT audio-encoder inventory changed: {encoder_trainable} != {SFT_ENCODER_PARAMETERS}"
                )
        else:
            freeze_module(self.audio_encoder)
        invalid = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
            and (
                ".mlp.experts." in name
                or (name.endswith(".mlp.gate.weight") and "shared_expert_gate" not in name)
            )
        ]
        if invalid:
            raise RuntimeError(f"D2 routed MoE parameters became trainable: {invalid[:10]}")
        self.optimizer_parameter_groups()

    def _enable_code_predictor_checkpointing(self) -> None:
        predictor = self.qwen.talker.code_predictor
        enable = getattr(predictor, "gradient_checkpointing_enable", None)
        if not callable(enable):
            raise RuntimeError("Qwen CodePredictor cannot enable checkpointing")
        enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        layers = tuple(predictor.model.layers)
        if not layers or not all(
            (bool(getattr(layer, "gradient_checkpointing", False)) for layer in layers)
        ):
            raise RuntimeError("Qwen CodePredictor checkpointing did not activate")

    @staticmethod
    def _enable_standard_adapter(module: LoRALinear, name: str) -> None:
        if module.rank != 128 or module.alpha != 256.0 or module.dropout.p != 0.0:
            raise RuntimeError(f"D2 adapter {name} is not rank-128/alpha-256/dropout-0")
        for parameter in module.lora_a.parameters():
            parameter.requires_grad_(True)
        for parameter in module.lora_b.parameters():
            parameter.requires_grad_(True)

    @staticmethod
    def _enable_fresh_adapter(module: LoRALinear, name: str) -> None:
        if module.fresh_lora_a is None or module.fresh_lora_b is None:
            raise RuntimeError(f"D2 shared adapter is missing: {name}")
        if (
            int(module.fresh_lora_a.out_features) != 128
            or module.fresh_scaling != 2.0
            or module.fresh_dropout.p != 0.0
        ):
            raise RuntimeError(f"D2 fresh shared adapter has the wrong topology: {name}")
        for parameter in module.fresh_lora_a.parameters():
            parameter.requires_grad_(True)
        for parameter in module.fresh_lora_b.parameters():
            parameter.requires_grad_(True)

    def _enable_matching_standard_adapters(
        self, root: nn.Module, *, prefixes: tuple[str, ...] = ()
    ) -> int:
        count = 0
        for name, module in root.named_modules():
            if not isinstance(module, LoRALinear):
                continue
            if prefixes and (not any((name.startswith(prefix) for prefix in prefixes))):
                continue
            self._enable_standard_adapter(module, name)
            count += 1
        return count

    def optimizer_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Return the eleven LR groups used by the fine-tuning loop."""
        groups: dict[str, list[nn.Parameter]] = {
            "thinker_lora": [],
            "talker_lora": [],
            "talker_context_lora": [],
            "code_predictor_lora": [],
            "control_and_stream_rows": [],
            "codec_embeddings": [],
            "codec_output_heads": [],
            "small_parts": [],
            "audio_frontend_full": [],
            "audio_attention_lora": [],
            "audio_non_attention_lora": [],
        }
        encoder_parameters = {
            id(dict(self.audio_encoder.encoder.named_parameters())[name]): group
            for group, names in self.audio_encoder.encoder.sft_parameter_group_names().items()
            for name in names
        }
        assigned: set[int] = set()
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            encoder_group = encoder_parameters.get(id(parameter))
            group = (
                {
                    "frontend_full": "audio_frontend_full",
                    "attention_lora": "audio_attention_lora",
                    "non_attention_lora": "audio_non_attention_lora",
                }[encoder_group]
                if encoder_group is not None
                else _optimizer_group_name(name)
            )
            if id(parameter) in assigned:
                raise RuntimeError(f"Trainable parameter was grouped twice: {name}")
            groups[group].append(parameter)
            assigned.add(id(parameter))
        expected = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        if assigned != expected:
            raise RuntimeError("Optimizer groups do not exactly cover D2 trainables")
        audio_count = sum(
            (
                parameter.numel()
                for name in (
                    "audio_frontend_full",
                    "audio_attention_lora",
                    "audio_non_attention_lora",
                )
                for parameter in groups[name]
            )
        )
        expected_audio = SFT_ENCODER_PARAMETERS if self.audio_encoder.training_lora_enabled else 0
        if audio_count != expected_audio:
            raise RuntimeError(
                f"audio encoder groups have {audio_count} parameters, expected {expected_audio}"
            )
        total = sum((parameter.numel() for values in groups.values() for parameter in values))
        expected_total = D2_SFT_TRAINABLE_PARAMETERS - (
            0 if self.audio_encoder.training_lora_enabled else SFT_ENCODER_PARAMETERS
        )
        if total != expected_total:
            raise RuntimeError(
                f"D2 SFT has {total} trainable parameters, expected {expected_total}"
            )
        return groups

    def _shift_right_audio(
        self, self_audio: torch.Tensor, *, previous_self_audio: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Delay representations by one complete macro unit.

        The existing async data path separately applies the measured 371-sample
        Code2Wav playback shift to raw self waveform.  This method adds only the
        representation-level ``k``-frame delay (plus the explicit future knob).
        """
        history = _rolling_self_history(
            self_audio.new_empty((int(self_audio.shape[0]), 0, int(self_audio.shape[2]))),
            delay_frames=self.self_audio_delay_frames,
            silence=self.self_silence_embed,
            previous=previous_self_audio,
        )
        return torch.cat((history, self_audio), dim=1)[:, : self_audio.shape[1], :]

    def _run_thinker(
        self,
        *,
        text_target: torch.Tensor,
        env_audio: torch.Tensor,
        self_audio: torch.Tensor,
        frame_mask: torch.Tensor,
        response_mask: torch.Tensor,
        interrupt_end_weight_class: torch.Tensor | None,
        dtype: torch.dtype,
        past_key_values: Any = None,
        frame_offset: int = 0,
        previous_text_target: torch.Tensor | None = None,
        previous_self_audio: torch.Tensor | None = None,
        valid_memory_frames: int | None = None,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Any,
    ]:
        batch_size, sequence_length = map(int, text_target.shape)
        validate_chunk_frames(sequence_length, self.frames_per_unit)
        if int(frame_offset) % self.frames_per_unit:
            raise ValueError("Thinker frame_offset cuts a D2 macro unit")
        text_input = shift_right_text(
            text_target, self.pad_id, previous_text_target=previous_text_target
        )
        self_audio_input = self._shift_right_audio(
            self_audio, previous_self_audio=previous_self_audio
        )
        text_embedding = self.embed_text(text_input).to(dtype=dtype)
        embedding_device = text_embedding.device
        stream_embedding = self.stream_type_embedding(
            torch.arange(2, device=self.stream_type_embedding.weight.device, dtype=torch.long)
        ).to(device=embedding_device, dtype=dtype)
        thinker_inputs = pack_macro_triplets(
            env_audio.to(device=embedding_device, dtype=dtype) + stream_embedding[1].view(1, 1, -1),
            self_audio_input.to(device=embedding_device, dtype=dtype)
            + stream_embedding[0].view(1, 1, -1),
            text_embedding,
            self.frames_per_unit,
        )
        position_ids = macro_position_ids(
            batch_size=batch_size,
            frame_count=sequence_length,
            frames_per_unit=self.frames_per_unit,
            frame_offset=int(frame_offset),
            device=embedding_device,
        )
        cached_tokens = (
            int(past_key_values.get_seq_length())
            if past_key_values is not None and hasattr(past_key_values, "get_seq_length")
            else 0
        )
        valid_cached_tokens = (
            cached_tokens if valid_memory_frames is None else int(valid_memory_frames) * 3
        )
        attention_mask = macro_attention_mask(
            frame_mask=frame_mask.to(device=embedding_device),
            frames_per_unit=self.frames_per_unit,
            cached_tokens=cached_tokens,
            valid_cached_tokens=valid_cached_tokens,
        )
        prompt_tokens = 0
        system_prompt_input_ids = getattr(self, "_d2_system_prompt_input_ids", None)
        if system_prompt_input_ids is not None:
            if int(frame_offset) != 0 or cached_tokens != 0:
                raise ValueError("system prompt may be inserted only at sample start")
            if (
                system_prompt_input_ids.ndim != 2
                or int(system_prompt_input_ids.shape[0]) != batch_size
            ):
                raise ValueError("system prompt IDs must be [batch,prompt_tokens]")
            prompt_embedding = self.embed_text(
                system_prompt_input_ids.to(device=embedding_device)
            ).to(dtype=dtype)
            prompt_tokens = int(prompt_embedding.shape[1])
            thinker_inputs, position_ids, attention_mask = prepend_system_prompt_to_macro_layout(
                prompt_embeddings=prompt_embedding,
                macro_embeddings=thinker_inputs,
                macro_positions=position_ids,
                macro_attention=attention_mask,
            )
        captured_layer24: dict[str, torch.Tensor] = {}

        def capture_layer24(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(hidden):
                raise RuntimeError("Thinker layer-24 hook returned a non-tensor")
            captured_layer24["hidden"] = hidden

        handle = self.qwen.thinker.model.layers[self.accept_hidden_layer - 1].register_forward_hook(
            capture_layer24
        )
        try:
            thinker_outputs = self.qwen.thinker.model(
                inputs_embeds=thinker_inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=False,
            )
        finally:
            handle.remove()
        if "hidden" not in captured_layer24:
            raise RuntimeError("Thinker layer-24 hook did not run")
        layer24 = select_macro_text_slots(
            captured_layer24["hidden"][:, prompt_tokens:], self.frames_per_unit
        )
        final_hidden = select_macro_text_slots(
            thinker_outputs.last_hidden_state[:, prompt_tokens:], self.frames_per_unit
        )
        text_logits = self.text_logits(final_hidden)
        event_control_logits = None
        text_loss, numerator, denominator = weighted_text_ce_terms(
            text_logits,
            text_target.to(device=text_logits.device),
            frame_mask.to(device=text_logits.device),
            response_mask.to(device=text_logits.device),
            self.pad_id,
            self.pad_weight,
            self.assistant_start_id,
            self.assistant_start_weight,
            self.assistant_end_id,
            self.assistant_end_weight,
            self.response_text_weight_policy,
            self.inside_response_pad_weight,
            None,
        )
        return (
            layer24,
            text_logits,
            event_control_logits,
            text_loss,
            numerator,
            denominator,
            thinker_outputs.past_key_values,
        )

    def forward_chunk(
        self,
        *,
        env_audio: torch.Tensor,
        self_audio: torch.Tensor,
        text_target: torch.Tensor,
        codec_target: torch.Tensor,
        frame_mask: torch.Tensor,
        assistant_mask: torch.Tensor | None = None,
        interrupt_end_weight_class: torch.Tensor | None = None,
        speaker_ids: torch.Tensor | None = None,
        state: QwenDuplexTransformerXLState | None = None,
        memory_frames: int | None = None,
        system_prompt_input_ids: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], QwenDuplexTransformerXLState]:
        frames = int(text_target.shape[1])
        validate_chunk_frames(frames, self.frames_per_unit)
        if memory_frames is not None and int(memory_frames) != 0:
            validate_chunk_frames(int(memory_frames), self.frames_per_unit)
        if state is None:
            state = QwenDuplexTransformerXLState()
        if int(state.frame_offset) % self.frames_per_unit:
            raise ValueError("Transformer-XL state cuts a D2 macro unit")
        previous = state.previous_self_audio
        if hasattr(self, "_d2_system_prompt_input_ids"):
            raise RuntimeError("nested D2 system-prompt forward is unsupported")
        self._d2_system_prompt_input_ids = system_prompt_input_ids
        try:
            outputs, next_state = super().forward_chunk(
                env_audio=env_audio,
                self_audio=self_audio,
                text_target=text_target,
                codec_target=codec_target,
                frame_mask=frame_mask,
                assistant_mask=assistant_mask,
                interrupt_end_weight_class=interrupt_end_weight_class,
                speaker_ids=speaker_ids,
                state=state,
                memory_frames=memory_frames,
            )
        finally:
            del self._d2_system_prompt_input_ids
        next_state.previous_self_audio = _rolling_self_history(
            self_audio,
            delay_frames=self.self_audio_delay_frames,
            silence=self.self_silence_embed,
            previous=previous,
        ).detach()
        return (outputs, next_state.detach_())

    def _encode_sft_audio_chunk(
        self,
        env_mel: torch.Tensor,
        self_mel: torch.Tensor,
        *,
        state: QwenDuplexTransformerXLState,
        start_frame: int,
        frame_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one XL chunk with bounded detached AuT recurrence."""
        start_frame = int(start_frame)
        frame_count = int(frame_count)
        if start_frame < 0 or start_frame % self.frames_per_unit:
            raise ValueError("SFT audio_start_frame cuts a macro unit")
        validate_chunk_frames(frame_count, self.frames_per_unit)
        if int(state.frame_offset) != start_frame:
            raise ValueError(
                f"SFT AuT recurrence is out of sync with Transformer-XL: audio_start={start_frame} frame_offset={state.frame_offset}"
            )
        expected_mel_frames = frame_count * MEL_FRAMES_PER_AUDIO_FRAME
        if env_mel.shape != self_mel.shape:
            raise ValueError("SFT environment/self mel chunks differ in shape")
        batch_size = int(env_mel.shape[0]) if env_mel.ndim == 3 else 0
        if env_mel.ndim != 3 or batch_size < 1 or int(env_mel.shape[-1]) != expected_mel_frames:
            raise ValueError(
                f"SFT mel input must cover one complete local XL batch: shape={tuple(env_mel.shape)} expected_frames={expected_mel_frames}"
            )
        env_state = getattr(state, "d2_env_aut_state", None)
        self_state = getattr(state, "d2_self_aut_state", None)
        if start_frame == 0:
            if env_state is not None or self_state is not None:
                raise RuntimeError("new SFT sample inherited stale AuT state")
            env_states = [self.audio_encoder.encoder.new_stream_state() for _ in range(batch_size)]
            self_states = [self.audio_encoder.encoder.new_stream_state() for _ in range(batch_size)]
            env_state = env_states[0] if batch_size == 1 else env_states
            self_state = self_states[0] if batch_size == 1 else self_states
            state.d2_env_aut_state = env_state
            state.d2_self_aut_state = self_state
        elif env_state is None or self_state is None:
            raise RuntimeError("SFT continuation is missing bounded AuT state")
        env_states = list(env_state) if isinstance(env_state, (tuple, list)) else [env_state]
        self_states = list(self_state) if isinstance(self_state, (tuple, list)) else [self_state]
        if len(env_states) != batch_size or len(self_states) != batch_size:
            raise RuntimeError("SFT AuT state batch size changed between chunks")
        output = self.audio_encoder.encode_trainable_mel_chunk(
            torch.cat((env_mel, self_mel), dim=0), [*env_states, *self_states]
        )
        expected = (2 * batch_size, frame_count)
        if tuple(output.shape[:2]) != expected:
            raise RuntimeError(f"SFT AuT output {tuple(output.shape[:2])} != {expected}")
        return (output[:batch_size], output[batch_size:])

    def forward(
        self,
        *,
        env_audio: torch.Tensor | None = None,
        self_audio: torch.Tensor | None = None,
        env_mel: torch.Tensor | None = None,
        self_mel: torch.Tensor | None = None,
        audio_start_frame: int = 0,
        text_target: torch.Tensor,
        codec_target: torch.Tensor,
        frame_mask: torch.Tensor,
        assistant_mask: torch.Tensor | None = None,
        interrupt_end_weight_class: torch.Tensor | None = None,
        speaker_ids: torch.Tensor | None = None,
        state: QwenDuplexTransformerXLState | None = None,
        memory_frames: int | None = None,
        system_prompt_input_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], QwenDuplexTransformerXLState]:
        raw_audio = env_mel is not None or self_mel is not None
        embedded_audio = env_audio is not None or self_audio is not None
        if raw_audio == embedded_audio:
            raise ValueError("provide exactly one of SFT mel chunks or encoded AuT features")
        if raw_audio:
            if env_mel is None or self_mel is None:
                raise ValueError("both SFT mel lanes are required")
            if state is None:
                state = QwenDuplexTransformerXLState()
            env_audio, self_audio = self._encode_sft_audio_chunk(
                env_mel,
                self_mel,
                state=state,
                start_frame=int(audio_start_frame),
                frame_count=int(text_target.shape[1]),
            )
        if env_audio is None or self_audio is None:
            raise RuntimeError("D2 audio preparation produced no features")
        if state is not None or memory_frames is not None:
            return self.forward_chunk(
                env_audio=env_audio,
                self_audio=self_audio,
                text_target=text_target,
                codec_target=codec_target,
                frame_mask=frame_mask,
                assistant_mask=assistant_mask,
                interrupt_end_weight_class=interrupt_end_weight_class,
                speaker_ids=speaker_ids,
                state=state,
                memory_frames=memory_frames,
                system_prompt_input_ids=system_prompt_input_ids,
            )
        if system_prompt_input_ids is not None:
            raise ValueError("system prompts require Transformer-XL chunked forward")
        return super().forward(
            env_audio=env_audio,
            self_audio=self_audio,
            text_target=text_target,
            codec_target=codec_target,
            frame_mask=frame_mask,
            assistant_mask=assistant_mask,
            interrupt_end_weight_class=interrupt_end_weight_class,
            speaker_ids=speaker_ids,
        )

    def checkpoint_parameter_names(self) -> set[str]:
        """D2 checkpoints contain trainables only, never official/INT8 weights."""
        return {name for name, parameter in self.named_parameters() if parameter.requires_grad}

    def checkpoint_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def load_checkpoint_state_dict(self, state_dict: Mapping[str, torch.Tensor], **_: Any) -> None:
        parameters = dict(self.named_parameters())
        expected = self.checkpoint_parameter_names()
        provided = set(state_dict)
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        mismatched = sorted(
            (
                (name, tuple(state_dict[name].shape), tuple(parameters[name].shape))
                for name in expected & provided
                if tuple(state_dict[name].shape) != tuple(parameters[name].shape)
            )
        )
        if missing or unexpected or mismatched:
            raise RuntimeError(
                f"D2 checkpoint inventory mismatch: missing={missing[:10]} unexpected={unexpected[:10]} shape={mismatched[:10]}"
            )
        for name in sorted(expected):
            source = state_dict[name]
            if not torch.isfinite(source).all():
                raise RuntimeError(f"D2 checkpoint tensor is non-finite: {name}")
            parameters[name].copy_(
                source.to(device=parameters[name].device, dtype=parameters[name].dtype)
            )

    @torch.no_grad()
    def inference_audio_prefill(
        self,
        *,
        environment_audio: torch.Tensor,
        self_audio: torch.Tensor,
        previous_self_audio: torch.Tensor | None,
        unit_offset: int,
        cached_tokens: int = 0,
        valid_cached_tokens: int | None = None,
        position_offset_tokens: int = 0,
    ) -> MacroAudioPrefill:
        """Build the single causal ``2k``-audio inference call."""
        k = self.frames_per_unit
        if environment_audio.shape != self_audio.shape:
            raise ValueError("inference environment/self shapes differ")
        if environment_audio.ndim != 3 or int(environment_audio.shape[1]) != k:
            raise ValueError(f"inference audio unit must be [batch,{k},hidden]")
        delayed = self._shift_right_audio(self_audio, previous_self_audio=previous_self_audio)
        dtype = self.compute_dtype
        device = self.stream_type_embedding.weight.device
        stream = self.stream_type_embedding(torch.arange(2, device=device, dtype=torch.long)).to(
            dtype=dtype
        )
        env = environment_audio.to(device=device, dtype=dtype) + stream[1].view(1, 1, -1)
        own = delayed.to(device=device, dtype=dtype) + stream[0].view(1, 1, -1)
        inputs = torch.cat((env, own), dim=1)
        full_positions = macro_position_ids(
            batch_size=int(inputs.shape[0]),
            frame_count=k,
            frames_per_unit=k,
            frame_offset=int(unit_offset) * k,
            device=device,
        ) + int(position_offset_tokens)
        current = (
            torch.ones((2 * k, 2 * k), dtype=torch.bool, device=device)
            .tril()[None, :, :]
            .expand(int(inputs.shape[0]), -1, -1)
        )
        if valid_cached_tokens is None:
            valid_cached_tokens = int(cached_tokens)
        past = torch.zeros(
            (int(inputs.shape[0]), int(cached_tokens)), dtype=torch.bool, device=device
        )
        if int(valid_cached_tokens):
            past[:, -int(valid_cached_tokens) :] = True
        mask = torch.cat((past[:, None, :].expand(-1, 2 * k, -1), current), dim=2)[
            :, None, :, :
        ].contiguous()
        history = _rolling_self_history(
            self_audio,
            delay_frames=self.self_audio_delay_frames,
            silence=self.self_silence_embed,
            previous=previous_self_audio,
        ).detach()
        return MacroAudioPrefill(
            inputs_embeds=inputs,
            position_ids=full_positions[..., : 2 * k],
            attention_mask=mask,
            next_self_history=history,
        )

    @torch.no_grad()
    def inference_text_step(
        self,
        *,
        previous_text_ids: torch.Tensor,
        text_index: int,
        unit_offset: int,
        cached_tokens: int,
        valid_cached_tokens: int | None = None,
        position_offset_tokens: int = 0,
    ) -> MacroTextStep:
        """Build one AR text call after the macro audio prefill."""
        k = self.frames_per_unit
        text_index = int(text_index)
        if not 0 <= text_index < k:
            raise ValueError(f"text_index must be in [0,{k}), got {text_index}")
        ids = previous_text_ids
        if ids.ndim == 1:
            ids = ids[:, None]
        if ids.ndim != 2 or int(ids.shape[1]) != 1:
            raise ValueError("previous_text_ids must be [batch] or [batch,1]")
        embeds = self.embed_text(ids).to(dtype=self.compute_dtype)
        device = embeds.device
        cached_tokens = int(cached_tokens)
        if valid_cached_tokens is None:
            valid_cached_tokens = cached_tokens
        mask = torch.zeros(
            (int(ids.shape[0]), 1, 1, cached_tokens + 1), dtype=torch.bool, device=device
        )
        if int(valid_cached_tokens):
            mask[..., cached_tokens - int(valid_cached_tokens) : cached_tokens] = True
        mask[..., -1] = True
        absolute = int(position_offset_tokens) + int(unit_offset) * 3 * k + 2 * k + text_index
        positions = torch.full((4, int(ids.shape[0]), 1), absolute, dtype=torch.long, device=device)
        return MacroTextStep(
            inputs_embeds=embeds, position_ids=positions, attention_mask=mask, text_index=text_index
        )

    def inference_contract(self) -> dict[str, Any]:
        """Exact shared-inference orchestration contract."""
        return {
            "frames_per_unit": self.frames_per_unit,
            "stages": [
                "inference_audio_prefill_once_for_causal_2k_audio_tokens",
                "inference_text_step_k_times_sequentially_carrying_thinker_kv",
                "run_talker_and_code_predictor_k_times_sequentially_from_captured_text_and_layer24",
            ],
            "thinker_call": "qwen.thinker.model(use_cache=True, output_hidden_states=True); text readout is last_hidden_state[:,-1], Talker outside-context is hidden_states[accept_hidden_layer][:,-1]",
            "cache_compaction": "after the unit keep 3*memory_frames Thinker tokens and memory_frames+1 Talker tokens",
            "text_input": "PAD_or_previous_unit_final_then_previous_generated_token",
            "self_audio_delay_frames": self.self_audio_delay_frames,
            "qwen_attention_policy": QWEN_MACRO_ATTENTION_POLICY,
            "qwen_attention_policy_version": QWEN_MACRO_ATTENTION_POLICY_VERSION,
            "raw_code2wav_shift": "371_samples_owned_by_async_audio_loader",
        }
