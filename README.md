# Dialogue-to-Duplex

[Hugging Face](https://huggingface.co/penguinfish1688/dialogue-to-duplex) · [Quick Start](#quick-start) · [Fine-tuning](#fine-tuning) · [Evaluation](benchmarks/README.md)

**Dialogue-to-Duplex (D2)** turns pretrained speech models into full-duplex conversational models that listen and speak at the same time. D2 learns when to respond, continue listening, and yield to interruptions.

D2 combines causal audio encoder distillation with dialogue fine-tuning, while keeping each model's native speech generator. This repository provides inference, a browser app, and fine-tuning for Qwen3-Omni and LLaMA-Omni2.

![Dialogue-to-Duplex architecture](https://huggingface.co/penguinfish1688/dialogue-to-duplex/resolve/main/assets/d2-overview.png)

## Models

**Currently, D2 releases only the Qwen3-Omni 80 ms checkpoint.**

| Model | Checkpoint | Documentation |
| --- | --- | --- |
| Qwen3-Omni D2 · 80 ms | [Hugging Face](https://huggingface.co/penguinfish1688/dialogue-to-duplex) | [Qwen3-Omni D2](qwen3-omni-D2/README.md) |
| LLaMA-Omni2 D2 | Not yet released | [LLaMA-Omni2 D2](llama-omni2-D2/README.md) |

The 80 ms interval is the model's audio processing step; response latency also depends on generation and computation.

## Quick Start

### Installation

Use Linux, Python 3.12, and an NVIDIA GPU with a compatible CUDA driver. For Qwen, we recommend an **RTX PRO 6000 Blackwell (96 GB)** and **96 GB of system RAM**. Allow approximately **73 GB** for model weights, plus space for dependencies and caches.

Install `git`, `ffmpeg`, `libsndfile`, a C++ compiler, and Python 3.12 development headers through your system package manager, then run:

```bash
git clone https://github.com/penguinfish1688/D2-dialogue-to-duplex.git
cd D2-dialogue-to-duplex
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -c qwen3-omni-D2/configs/requirements.txt -e '.[qwen]'
```

### Browser Demo

```bash
d2-qwen app
```

The checkpoint and backbone download automatically from Hugging Face; no login is required. The first startup also compiles kernels and can take several minutes. Once `Uvicorn running` appears, open [http://localhost:8000](http://localhost:8000), allow microphone access, and use headphones.

For a remote GPU, run this on your computer before opening the same URL:

```bash
ssh -L 8000:localhost:8000 user@gpu-host
```

The app serves one conversation at a time. The default context supports about 54 seconds; start a new conversation or increase it with `d2-qwen app --kv-budget 4096`.

### Audio File Inference

```bash
d2-qwen infer --input question.wav --output response.wav --tail-seconds 20
```

This saves the spoken response to `response.wav` and a transcript and event trace to `response.json`. The response tail gives the model time to finish speaking after the input ends.

Both the app and file inference use BF16, a fixed KV cache, and CUDA graphs for prefill and serial decoding. See the [Qwen guide](qwen3-omni-D2/README.md) for runtime options and the [LLaMA guide](llama-omni2-D2/README.md) for its separate environment.

## Fine-tuning

Prepare your data using the [sample format](d2/SAMPLES.md), then fine-tune the released checkpoint:

```bash
d2-qwen train --data samples.json --output my-d2 --steps 2000
```

Load the resulting checkpoint with the same commands:

```bash
d2-qwen app --model my-d2
```

See the model guides for training settings and trainable parameters.

## Evaluation

See the [evaluation guide](benchmarks/README.md) to run VoiceBench and Full-Duplex-Bench. Results and figures are available on the [model card](https://huggingface.co/penguinfish1688/dialogue-to-duplex#research-results).

## Acknowledgments

D2 builds on [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni), [LLaMA-Omni2](https://github.com/ictnlp/LLaMA-Omni2), [Transformers](https://github.com/huggingface/transformers), [Whisper](https://github.com/openai/whisper), [CosyVoice](https://github.com/FunAudioLLM/CosyVoice), and [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS). Please follow the licenses of the underlying models and dependencies.
