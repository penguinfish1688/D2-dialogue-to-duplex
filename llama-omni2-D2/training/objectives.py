from __future__ import annotations
import torch
from d2_llama.core.controls import ControlTokenIds
from d2_llama.core.constants import (
    TTS_EOS_TOKEN_ID,
    TTS_TEXT_VOCAB_SIZE,
    TTS_UNIT_TOKEN_OFFSET,
    TTS_UNIT_VOCAB_SIZE,
)

PREDICTION_COUNT_NAMES = (
    "lexical_correct",
    "lexical_total",
    "assistant_tts_code_correct",
    "assistant_tts_code_total",
    "audio_eos_true_positive",
    "audio_eos_predicted_positive",
    "audio_eos_target_positive",
    "audio_supervised_positions",
    "interrupt_true_positive",
    "interrupt_predicted_positive",
    "interrupt_target_positive",
    "response_true_positive",
    "response_predicted_positive",
    "response_target_positive",
    "text_supervised_positions",
)


def sft_prediction_counts(
    *,
    text_predictions: torch.Tensor,
    text_targets: torch.Tensor,
    tts_code_predictions: torch.Tensor,
    tts_code_targets: torch.Tensor,
    tts_code_metric_mask: torch.Tensor,
    controls: ControlTokenIds,
) -> dict[str, int]:
    """Count exact, unweighted Stage-3 predictions for global reduction.

    TTS predictions and targets must already use the one-position causal shift
    applied by the released Hugging Face causal-LM loss.
    """

    if text_predictions.shape != text_targets.shape:
        raise ValueError("text predictions and targets must have equal shapes")
    if tts_code_predictions.shape != tts_code_targets.shape:
        raise ValueError("shifted TTS predictions and targets must have equal shapes")
    if tts_code_metric_mask.shape != tts_code_targets.shape:
        raise ValueError("shifted TTS code metric mask must match targets")
    if tts_code_metric_mask.dtype != torch.bool:
        raise TypeError("shifted TTS code metric mask must be boolean")
    if text_predictions.dtype != torch.long or text_targets.dtype != torch.long:
        raise TypeError("text predictions and targets must be torch.long")
    if tts_code_predictions.dtype != torch.long or tts_code_targets.dtype != torch.long:
        raise TypeError("TTS predictions and targets must be torch.long")

    lexical = (text_targets >= 0) & (text_targets < TTS_TEXT_VOCAB_SIZE)
    control = (
        (text_targets == controls.response)
        | (text_targets == controls.interrupt)
        | (text_targets == controls.pad)
    )
    if not bool((lexical | control).all()):
        raise ValueError("text metric targets are neither lexical IDs nor D2 controls")

    native_unit = (tts_code_targets >= TTS_UNIT_TOKEN_OFFSET) & (
        tts_code_targets < TTS_UNIT_TOKEN_OFFSET + TTS_UNIT_VOCAB_SIZE
    )
    audio_eos_target = tts_code_targets == TTS_EOS_TOKEN_ID
    ignored = tts_code_targets == -100
    if not bool((native_unit | audio_eos_target | ignored).all()):
        raise ValueError("TTS metric target is not a unit, audio EOS, or IGNORE")
    if bool((tts_code_metric_mask & ~native_unit).any()):
        raise ValueError("assistant code metric mask includes a non-unit target")
    audio_supervised = ~ignored
    audio_eos_prediction = (tts_code_predictions == TTS_EOS_TOKEN_ID) & audio_supervised

    interrupt_target = text_targets == controls.interrupt
    interrupt_prediction = text_predictions == controls.interrupt
    response_target = text_targets == controls.response
    response_prediction = text_predictions == controls.response

    values = torch.stack(
        (
            ((text_predictions == text_targets) & lexical).sum(),
            lexical.sum(),
            ((tts_code_predictions == tts_code_targets) & tts_code_metric_mask).sum(),
            tts_code_metric_mask.sum(),
            (audio_eos_prediction & audio_eos_target).sum(),
            audio_eos_prediction.sum(),
            audio_eos_target.sum(),
            audio_supervised.sum(),
            (interrupt_prediction & interrupt_target).sum(),
            interrupt_prediction.sum(),
            interrupt_target.sum(),
            (response_prediction & response_target).sum(),
            response_prediction.sum(),
            response_target.sum(),
            torch.tensor(text_targets.numel(), device=text_targets.device),
        )
    ).to(dtype=torch.int64)
    return {
        name: int(value)
        for name, value in zip(
            PREDICTION_COUNT_NAMES,
            values.detach().cpu().tolist(),
            strict=True,
        )
    }
