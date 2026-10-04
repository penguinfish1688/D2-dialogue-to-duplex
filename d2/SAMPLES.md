# Prepared samples

The fine-tuning command consumes native-clock tensors rather than a dataset-specific research loader. This keeps dataset acquisition, annotation, and mixing outside the model runtime. Use `safetensors.torch.save_file` to write tensors and point to them from a JSON manifest:

```json
{
  "format": "d2.samples.v1",
  "family": "qwen",
  "latency_ms": 80,
  "samples": [
    {"tensors": "sample-0001.safetensors", "task": "conversation"}
  ]
}
```

Paths are relative to the manifest. `family` is `qwen` or `llama`; `latency_ms` must match the release. `task` is `conversation` by default or `avqa`. Each uses its hardcoded training prompt. Samples contain no system-prefix tokens or prefix loss targets.

Each sample is a complete timeline beginning at frame zero, with an integral number of interaction intervals. The small trainer cycles through samples in manifest order and performs one optimizer update per sample. It resets recurrence at each sample. It does not splice unrelated samples into one conversation or resume partial timelines.

## Qwen

Let `B` be batch size and `T` be the number of native 80-ms frames. `T` must be divisible by `k = latency_ms / 80`, and no larger than the model's macro-aligned 90-second training chunk.

| Tensor | Shape | Type |
| --- | --- | --- |
| `env_mel` | `[B, 128, 8*T]` | float32 |
| `self_mel` | `[B, 128, 8*T]` | float32 |
| `text_target` | `[B, T]` | int64 |
| `codec_target` | `[B, T, 16]` | int64 |
| `frame_mask` | `[B, T]` | bool |
| `assistant_mask` | `[B, T]` | bool |
| `interrupt_end_weight_class` (optional) | `[B, T]` | int64 |
| `speaker_ids` (optional) | `[B]` | int64 |

Extract mel features with `d2_qwen.model.mel.causal_log_mel(waveform_16k, model.processor.feature_extractor)`. Keep environment and assistant waveforms on the same timeline. The assistant lane includes the original Code2Wav playback delay of 371 samples at 16 kHz; the model adds the interaction-interval delay to the encoded self lane. Do not apply that macro delay twice.

Text targets contain lexical tokenizer IDs, RESPONSE 151669, INTERRUPT 151670, and the appended PAD ID exposed by `model.pad_id`. The model performs the teacher-forcing text shift itself. `assistant_mask` marks the actual assistant-audio span; `frame_mask` marks valid supervised frames.

Speech targets use the first 16 native Mimi codebooks at 12.5 Hz (ordinary codes 0–2047), with the checkpoint's codec BOS/EOS IDs in codebook zero at response boundaries. Non-speech frames use the canonical silence frame from `d2_qwen.model.constants.CODEC_SILENCE_FRAME`. Preserve the original audio/control alignment when preparing targets; tokenizing a transcript alone does not create a duplex sample.

## LLaMA-Omni2

Use `family: "llama"`. Let `T` be the number of native 100-ms frames and `k = latency_ms / 100`. The small trainer accepts batch size one, `T` divisible by `k`, and at most the macro-aligned 90-second chunk.

| Tensor | Shape | Type |
| --- | --- | --- |
| `environment_features` | `[1, 128, 10*T]` | float32 |
| `delayed_self_features` | `[1, 128, 10*T]` | float32 |
| `text_input_ids` | `[1, T]` | int64 |
| `text_target_ids` | `[1, T]` | int64 |
| `text_weights` | `[1, T]` | float32 |

Extract features with `d2_llama.model.features.CausalWhisperFeatureExtractor().extract(waveform_16k)`. Before feature extraction, the clean assistant lane is delayed by `(k + 4) * 1600` samples: the interaction interval plus the native Write-10 release delay. The model consumes this already delayed lane directly.

`text_input_ids` begins with native ID 151643; the rest is the previous frame's target. Targets contain lexical IDs and appended RESPONSE/INTERRUPT/PAD IDs 152064/152065/152066. Text weights are 20 for either control, 0.05 for PAD, and 1 for lexical tokens; use zero only for unsupervised frames.

Each manifest row also contains an `episodes` list. An episode records a response's text and native speech units:

```json
{
  "tensors": "sample-0001.safetensors",
  "task": "conversation",
  "episodes": [
    {
      "identity": "response-0",
      "first_frame": 20,
      "lexical": [9707],
      "audio_ids": [151700, 151800, 151643],
      "stop_frame": null
    }
  ]
}
```

The values above illustrate the schema, not a trainable audio/text example. `first_frame` is the first lexical prediction frame, following RESPONSE. `lexical` must equal the corresponding text targets. `audio_ids` contains native CosyVoice2 FSQ IDs offset by 151666, ending in speech EOS 151643. `stop_frame` optionally marks an interruption. The episode scheduler inserts text-end 151645, separator 151665, and interruption EOS in the original Read-3/Write-10 order; do not insert those conditions yourself into `lexical`.
