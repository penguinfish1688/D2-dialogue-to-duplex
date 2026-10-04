from __future__ import annotations
from dataclasses import dataclass, fields, is_dataclass
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from d2_llama.core.constants import TTS_EOS_TOKEN_ID, TTS_TEXT_END_TOKEN_ID
from d2_llama.model.duplex import LlamaOmni2D2Model, gather_text_hidden
from d2_llama.training.objectives import sft_prediction_counts
from d2_llama.model.positions import prepare_thinker_prompted_inputs


def tensor_tree(value, device=None):
    if torch.is_tensor(value):
        return value.detach().to(device=device or value.device).contiguous()
    if isinstance(value, dict):
        return {key: tensor_tree(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)((tensor_tree(item, device) for item in value))
    if is_dataclass(value):
        return type(value)(
            **{
                field.name: tensor_tree(getattr(value, field.name), device)
                for field in fields(value)
            }
        )
    return value


def xl_decoder(
    decoder, inputs, state=None, *, memory_tokens, checkpoint_layers=True, position_ids=None
):
    """Native Qwen2 attention/MLP, absolute RoPE, detached past-only KV memory.

    The released Transformers 4.43 attention sizes its rotary table from
    cached length, not absolute position. Explicit native rotary application
    here preserves absolute positions after bounded-cache eviction. No model
    weights or attention semantics change; tests compare against full native
    Qwen2 forward and gradients before truncation.
    """
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    state = state or {"offset": 0, "layers": []}
    offset, history = (int(state["offset"]), state["layers"])
    batch, count, _ = inputs.shape
    past = 0 if not history else history[0][0].shape[-2]
    if memory_tokens < 1:
        raise ValueError("XL memory must retain at least one token")
    positions = (
        torch.arange(offset, offset + count, device=inputs.device).unsqueeze(0)
        if position_ids is None
        else position_ids.to(device=inputs.device)
    )
    if positions.shape not in ((1, count), (batch, count)) or positions.dtype != torch.long:
        raise ValueError("XL rotary position IDs must be int64 [1|batch,tokens]")
    if bool((positions < 0).any()):
        raise ValueError("XL rotary positions must be non-negative")
    rotary_length = max(offset + count, int(positions.max()) + 1)
    allowed = (
        torch.arange(past + count, device=inputs.device)[None, :]
        <= past + torch.arange(count, device=inputs.device)[:, None]
    )
    mask = inputs.new_zeros((1, 1, count, past + count)).masked_fill(
        ~allowed[None, None], torch.finfo(inputs.dtype).min
    )
    next_layers = []
    hidden = inputs
    for index, layer in enumerate(decoder.layers):
        old = history[index] if history else None

        def run(value, layer=layer, old=old):
            residual = value
            value = layer.input_layernorm(value)
            attention = layer.self_attn
            q = (
                attention.q_proj(value)
                .view(batch, count, attention.num_heads, attention.head_dim)
                .transpose(1, 2)
            )
            k = (
                attention.k_proj(value)
                .view(batch, count, attention.num_key_value_heads, attention.head_dim)
                .transpose(1, 2)
            )
            v = (
                attention.v_proj(value)
                .view(batch, count, attention.num_key_value_heads, attention.head_dim)
                .transpose(1, 2)
            )
            cos, sin = attention.rotary_emb(v, seq_len=rotary_length)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, positions)
            if old is not None:
                k, v = (torch.cat((old[0], k), dim=-2), torch.cat((old[1], v), dim=-2))
            attended = F.scaled_dot_product_attention(
                q.contiguous(),
                repeat_kv(k, attention.num_key_value_groups).contiguous(),
                repeat_kv(v, attention.num_key_value_groups).contiguous(),
                attn_mask=mask,
                dropout_p=attention.attention_dropout if layer.training else 0.0,
            )
            value = residual + attention.o_proj(attended.transpose(1, 2).reshape(batch, count, -1))
            value = value + layer.mlp(layer.post_attention_layernorm(value))
            return (value, k[..., -memory_tokens:, :], v[..., -memory_tokens:, :])

        if checkpoint_layers and torch.is_grad_enabled():
            hidden, keys, values = checkpoint(run, hidden, use_reentrant=False)
        else:
            hidden, keys, values = run(hidden)
        next_layers.append((keys.detach().contiguous(), values.detach().contiguous()))
    return (decoder.norm(hidden), {"offset": offset + count, "layers": next_layers})


def weighted_head_terms(hidden, labels, weights, head):
    """Bound the vocabulary projection's activation memory, preserving CE."""
    numerator = hidden.sum() * 0
    predictions = []
    flat = hidden.reshape(-1, hidden.shape[-1])
    labels, weights = (labels.flatten(), weights.flatten().float())
    for start in range(0, flat.shape[0], 32):
        stop = start + 32
        target, weight = (labels[start:stop], weights[start:stop])

        def block(value, target=target, weight=weight):
            logits = head(value).float()
            losses = F.cross_entropy(logits, target, ignore_index=-100, reduction="none")
            return ((losses * weight).sum(), logits.detach().argmax(-1))

        loss, prediction = checkpoint(block, flat[start:stop], use_reentrant=False)
        numerator = numerator + loss
        predictions.append(prediction)
    return (numerator, weights.sum(), torch.cat(predictions))


def distributed_ratio(numerator, denominator):
    """Global weighted mean and the corresponding DDP-scaled local gradient."""
    import torch.distributed as dist

    world = dist.get_world_size() if dist.is_initialized() else 1
    totals = torch.stack((numerator.detach().float(), denominator.detach().float()))
    if world > 1:
        dist.all_reduce(totals)
    denominator_global = totals[1].clamp_min(1)
    differentiable = numerator * world / denominator_global
    mean = totals[0] / denominator_global
    return differentiable + (mean - differentiable.detach())


@dataclass
class DynamicLosses:
    loss: torch.Tensor
    text_loss: torch.Tensor
    tts_loss: torch.Tensor
    prediction_counts: dict[str, int]


class DynamicD2Model(LlamaOmni2D2Model):
    def __init__(self, official_model, audio_tower, *, thinker_alignment=True):
        super().__init__(official_model, audio_tower, thinker_alignment=thinker_alignment)
        self.xl_state = None
        self.task_prompt_contract = None

    def forward(
        self,
        *,
        environment_features,
        delayed_self_features,
        text_input_ids,
        text_target_ids,
        text_weights,
        episodes,
        start_frame,
        frames_per_unit,
        reset,
        memory_frames,
        audio_eos_loss_weight=3.0,
        task_kind=None,
    ):
        if reset:
            self.xl_state = {
                "thinker": None,
                "environment": self.audio_tower.new_state(),
                "self": self.audio_tower.new_state(),
                "tts": {},
                "conditions": {},
            }
        if self.xl_state is None:
            raise ValueError("XL continuation omitted its initial state")
        state = self.xl_state
        frames = text_target_ids.shape[1]
        end_frame = start_frame + frames
        env = self.audio_tower(environment_features, state["environment"]).embeddings
        own = self.audio_tower(delayed_self_features, state["self"]).embeddings
        text = self.control_rows.embed(self.thinker.embed_tokens, text_input_ids)
        packed = self.pack_thinker_embeddings(env, own, text, frames_per_unit=frames_per_unit)
        prefix_count, metadata = (0, {})
        packed, positions, prefix_count, metadata = prepare_thinker_prompted_inputs(
            self,
            packed,
            state["thinker"],
            frame_offset=start_frame,
            frames_per_unit=frames_per_unit,
            task_kind=task_kind,
        )
        output, state["thinker"] = xl_decoder(
            self.thinker,
            packed,
            state["thinker"],
            memory_tokens=3 * memory_frames,
            position_ids=positions,
        )
        state["thinker"].update(metadata)
        output = output[:, prefix_count:]
        hidden = gather_text_hidden(output, frame_count=frames, frames_per_unit=frames_per_unit)
        text_num, text_den, text_pred = weighted_head_terms(
            hidden,
            text_target_ids,
            text_weights,
            lambda value: self.control_rows.logits(self.official_model.lm_head, value),
        )
        tts_num, tts_den = (hidden.sum() * 0, hidden.new_zeros((), dtype=torch.float32))
        predictions, targets = ([], [])
        embedding = self.speech_generator.model.get_input_embeddings()
        for episode in episodes:
            slots = episode.slots()
            selected = [i for i, slot in enumerate(slots) if start_frame <= slot[0] < end_frame]
            for clock, kind, token, frame, _ in slots:
                if (
                    kind in ("condition", "interrupt_condition")
                    and start_frame <= frame < end_frame
                    and (clock >= end_frame)
                ):
                    state["conditions"][episode.identity, frame] = hidden[
                        :, frame - start_frame : frame - start_frame + 1
                    ].detach()
            if not selected:
                continue
            first, stop = (selected[0], selected[-1] + 1)
            cache = state["tts"].get(episode.identity)
            if first != (0 if cache is None else cache["offset"]):
                raise ValueError("TTS XL prefix has a gap or was replayed")
            values, labels = ([], [])
            for index in selected:
                clock, kind, token, frame, _ = slots[index]
                ids = torch.tensor([[token]], dtype=torch.long, device=hidden.device)
                token_embedding = embedding(ids)
                if kind in ("condition", "interrupt_condition"):
                    if frame < start_frame:
                        condition_hidden = state["conditions"].pop((episode.identity, frame))
                    else:
                        condition_hidden = hidden[:, frame - start_frame : frame - start_frame + 1]
                        expected = (
                            self.control_rows.controls.interrupt
                            if kind == "interrupt_condition"
                            else self.control_rows.controls.pad
                            if token == TTS_TEXT_END_TOKEN_ID
                            else token
                        )
                        if int(text_target_ids[0, frame - start_frame]) != expected:
                            raise ValueError(
                                "dynamic TTS condition does not match its Thinker target"
                            )
                    token_embedding = self.speech_generator.fusion(
                        self.speech_generator.input_proj(condition_hidden), token_embedding
                    )
                values.append(token_embedding)
                labels.append(slots[index + 1][4] if index + 1 < len(slots) else -100)
            tts_hidden, next_cache = xl_decoder(
                self.speech_generator.model.model,
                torch.cat(values, 1),
                cache,
                memory_tokens=4 * memory_frames,
            )
            if stop < len(slots):
                state["tts"][episode.identity] = next_cache
            else:
                state["tts"].pop(episode.identity, None)
            target = torch.tensor(labels, device=hidden.device, dtype=torch.long)
            weight = (target != -100).float()
            weight[target == TTS_EOS_TOKEN_ID] = audio_eos_loss_weight
            num, den, pred = weighted_head_terms(
                tts_hidden, target, weight, self.speech_generator.model.lm_head
            )
            tts_num, tts_den = (tts_num + num, tts_den + den)
            predictions.append(pred)
            targets.append(target)
        text_loss, tts_loss = (
            distributed_ratio(text_num, text_den),
            distributed_ratio(tts_num, tts_den),
        )
        empty = torch.empty(0, device=hidden.device, dtype=torch.long)
        tts_pred = torch.cat(predictions) if predictions else empty
        tts_target = torch.cat(targets) if targets else empty
        active = text_weights.flatten() > 0
        counts = sft_prediction_counts(
            text_predictions=text_pred[active],
            text_targets=text_target_ids.flatten()[active],
            tts_code_predictions=tts_pred,
            tts_code_targets=tts_target,
            tts_code_metric_mask=tts_target >= 151666,
            controls=self.control_rows.controls,
        )
        return DynamicLosses(text_loss + tts_loss, text_loss, tts_loss, counts)
