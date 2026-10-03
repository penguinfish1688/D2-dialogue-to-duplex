# D2

**Dialogue-to-Duplex** adapts pretrained turn-based speech models for
simultaneous listening and speaking. The goal is to preserve their speech
understanding and general capabilities while enabling natural turn-taking and
interruption handling.

**Code release in preparation.** This repository currently contains only the
project skeleton. Training, inference, apps, and model weights are not available
yet.

## How it works

D2 treats dialogue as a sequence of short interaction intervals. During each
interval, the model receives new user audio and its own audio from the previous
interval, then generates the next text and speech outputs. Listening continues
while the model speaks.

The conversion has two stages:

1. **Streaming encoder distillation.** Adapt the original audio encoder to
   process incoming chunks, using the unchanged encoder as a teacher. The
   student can attend within the current chunk and to past chunks, but cannot
   see future audio.
2. **Duplex fine-tuning.** Place user and assistant speech on a shared timeline
   containing silence, overlap, and interruptions. Train the model to produce
   speech and control when to respond or yield, following the same temporal
   order used at inference.

The interaction interval is configurable, allowing the tradeoff between
responsiveness and general capability to be studied across both backbones.

## Structure

```text
.
├── qwen3-omni-D2/
│   ├── configs/       Training and inference configurations
│   ├── model/         Qwen3-Omni duplex model and streaming encoder
│   ├── training/      Encoder distillation and duplex fine-tuning
│   ├── inference/     Streaming inference and playback
│   └── app/           Interactive speech demo
├── llama-omni2-D2/
│   ├── configs/       Training and inference configurations
│   ├── model/         LLaMA-Omni2 duplex model and streaming encoder
│   ├── training/      Encoder distillation and duplex fine-tuning
│   ├── inference/     Streaming inference and playback
│   └── app/           Interactive speech demo
├── data/              Shared dataset preparation
└── evaluation/        Benchmark runners and scoring
```

These directories are placeholders for the planned release. Each model will
keep one implementation across its granularity settings. Its app will use the
same streaming inference code as its command-line runner.

## Planned models

| Model | Interaction granularity |
| --- | --- |
| [Qwen3-Omni D2](qwen3-omni-D2/) | 80, 160, 320, 640, 1040 ms |
| [LLaMA-Omni2 D2](llama-omni2-D2/) | 100, 200, 400, 800 ms |

These are configuration targets for the release, not measured response latencies.

## Release plan

- [x] Publish the repository skeleton.
- [ ] Add model code, inference, setup instructions, and checkpoint loading.
- [ ] Add an interactive app for each model.
- [ ] Add data preparation, encoder distillation, and duplex training recipes.
- [ ] Add Full-Duplex-Bench and VoiceBench evaluation and reproducible results.

Datasets, weights, generated audio, caches, and experiment logs stay outside Git.
Download instructions and dependency attribution will accompany the code release.
