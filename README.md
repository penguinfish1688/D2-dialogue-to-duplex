# Dialogue-to-Duplex

[Paper](https://arxiv.org/html/2610.17124v1) · [Demo](https://d2-duplex-demo.github.io/) · [Hugging Face](https://huggingface.co/penguinfish1688/dialogue-to-duplex) · [Quick Start](#quick-start) · [Fine-tuning](#fine-tuning) · [Evaluation](benchmarks/README.md)

**Dialogue-to-Duplex (D2)** turns pretrained speech models into full-duplex spoken language models that listen and speak at the same time. D2 learns when to respond, continue listening, and yield to interruptions. D2 achieves strong performance on VoiceBench and Full-Duplex-Bench.

D2 combines causal audio encoder distillation with dialogue fine-tuning, while keeping each model's native speech generator. This repository provides inference, a browser app, and fine-tuning for Qwen3-Omni and LLaMA-Omni2.

![Dialogue-to-Duplex architecture](https://huggingface.co/penguinfish1688/dialogue-to-duplex/resolve/main/assets/d2-overview.png)

## Models

All nine main checkpoint variants from Tables 1 and 2 of the paper are available. Each release contains the benchmark model at update 10,000 and its matching streaming encoder.

| Model | Chunk durations (ms) | Hugging Face revisions |
| --- | --- | --- |
| Qwen3-Omni-D2 | [80](https://huggingface.co/penguinfish1688/dialogue-to-duplex/tree/qwen-80ms), [160](https://huggingface.co/penguinfish1688/dialogue-to-duplex/tree/qwen-160ms), [320](https://huggingface.co/penguinfish1688/dialogue-to-duplex/tree/qwen-320ms), [640](https://huggingface.co/penguinfish1688/dialogue-to-duplex/tree/qwen-640ms), [1040](https://huggingface.co/penguinfish1688/dialogue-to-duplex/tree/qwen-1040ms) | `qwen-80ms`, `qwen-160ms`, `qwen-320ms`, `qwen-640ms`, `qwen-1040ms` |
| LLaMA-Omni 2-D2 | [100](https://huggingface.co/penguinfish1688/dialogue-to-duplex/tree/llama-100ms), [200](https://huggingface.co/penguinfish1688/dialogue-to-duplex/tree/llama-200ms), [400](https://huggingface.co/penguinfish1688/dialogue-to-duplex/tree/llama-400ms), [800](https://huggingface.co/penguinfish1688/dialogue-to-duplex/tree/llama-800ms) | `llama-100ms`, `llama-200ms`, `llama-400ms`, `llama-800ms` |

`main` remains the Qwen 80-ms default. Each named revision contains one variant, so downloading a revision does not download the other models. Use `--revision` to select a chunk duration, or pass a commit SHA to pin the exact release. The interval is the audio processing chunk duration; response latency also depends on generation and computation.

See the [Qwen guide](qwen3-omni-D2/README.md) or [LLaMA-Omni 2 guide](llama-omni2-D2/README.md) for model-specific installation and commands.

## Quick Start

### Installation

Use Linux, Python 3.12, and an NVIDIA GPU with **driver 580 or newer**. The pinned PyTorch build uses CUDA 13; see [NVIDIA's driver requirements](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). For Qwen, we recommend an **RTX PRO 6000 Blackwell (96 GB)** and **96 GB of system RAM**. Reserve **at least 100 GB of free disk space**: approximately 73 GB for model weights, plus the Python environment, downloads, and compiler caches.

Install `git`, `ffmpeg`, `libsndfile`, a C++ compiler, and Python 3.12 with development headers and `venv` support. On Ubuntu 24.04:

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg libsndfile1 build-essential python3.12-dev python3.12-venv
```

Then install D2 in its own environment:

```bash
git clone https://github.com/penguinfish1688/D2-dialogue-to-duplex.git
cd D2-dialogue-to-duplex
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c qwen3-omni-D2/configs/requirements.txt -e '.[qwen]'
```

### Browser Demo

```bash
d2-qwen app
```

The checkpoint and backbone download automatically from Hugging Face; no login is required. If your home directory has limited space, set `HF_HOME` to a larger disk before launching. The first startup also compiles kernels and can take several minutes. Once `Uvicorn running` appears, open [http://localhost:8000](http://localhost:8000), use headphones, click **Start conversation**, and allow microphone access. Wait for **Listening**, then speak.

In a new terminal, run `source .venv/bin/activate` from the repository directory before starting the app again.

For a remote GPU, run this on your computer before opening the same URL:

```bash
ssh -L 8000:localhost:8000 user@gpu-host
```

The app serves one conversation at a time. The default context supports about 54 seconds; start a new conversation or increase it with `d2-qwen app --kv-budget 4096`.

To measure real-time performance on your GPU, see the [app performance check](benchmarks/README.md#app-performance).

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

## Research Results

The following figures and table summarize D2 research results across models and interaction intervals. The nine main D2 variants are available through the revisions listed above.

### VoiceBench

![VoiceBench performance across interaction granularities](https://huggingface.co/penguinfish1688/dialogue-to-duplex/resolve/main/assets/voicebench.png)

The Qwen3-Omni 80 ms model achieves an overall VoiceBench score of **70.07**. The horizontal axis shows interaction intervals.

### Full-Duplex-Bench

| Model | D2 interval (ms) | Quality / 5 ↑ | Turn taking (%) ↑ | Turn latency (s) ↓ | Interruption (%) ↑ | Interruption latency (s) ↓ | Pause success (%) ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Qwen3-Omni-D2** | **80** | **4.56** | **99.2** | **0.521** | **99.5** | **0.403** | **21.3** |
| Qwen3-Omni-D2 | 160 | 4.67 | 100 | 0.539 | 99.5 | 0.432 | 19.0 |
| Qwen3-Omni-D2 | 320 | 4.61 | 100 | 0.710 | 99.5 | 0.571 | 25.5 |
| Qwen3-Omni-D2 | 640 | 4.62 | 99.2 | 1.01 | 99.5 | 0.928 | 55.1 |
| Qwen3-Omni-D2 | 1040 | 4.84 | 100 | 1.38 | 99.5 | 1.30 | 61.6 |
| LLaMA-Omni2-D2 | 100 | 3.33 | 100 | 0.746 | 99.5 | 0.621 | 74.5 |
| LLaMA-Omni2-D2 | 200 | 3.62 | 99.2 | 0.825 | 100 | 0.710 | 78.2 |
| LLaMA-Omni2-D2 | 400 | 3.69 | 100 | 1.02 | 99.0 | 0.894 | 69.4 |
| LLaMA-Omni2-D2 | 800 | 4.03 | 98.3 | 1.45 | 99.0 | 1.32 | 70.8 |
| BayLing-Duplex | — | 3.55 | 85.7 | 5.57 | 100 | 5.19 | 72.2 |
| Freeze-Omni | — | 3.82 | 34.5 | 0.288 | 98.5 | 0.349 | 69.4 |
| MiniCPM-o 4.5 | — | 4.45 | 75.6 | 1.92 | 94.5 | 1.89 | 91.2 |
| PersonaPlex | — | 4.43 | 99.2 | 0.346 | 95.0 | 0.294 | 31.9 |
| Moshi | — | 3.40 | 100 | 0.615 | 89.5 | 1.23 | 0.0 |
| Nemotron VoiceChat | — | 3.97 | 91.6 | 0.584 | 99.0 | 0.638 | 60.2 |

Quality uses GPT-4o ratings on a 0–5 scale. Turn taking and interruption measure successful responses to user events; pause handling measures whether the model waits through a user's pause. Latencies are event-aligned acoustic gaps plus the interaction interval, excluding computation time.

See the [evaluation guide](benchmarks/README.md) to run the benchmarks.

## Acknowledgments

D2 builds on [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni), [LLaMA-Omni2](https://github.com/ictnlp/LLaMA-Omni2), [Transformers](https://github.com/huggingface/transformers), [Whisper](https://github.com/openai/whisper), [CosyVoice](https://github.com/FunAudioLLM/CosyVoice), and [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS). Please follow the licenses of the underlying models and dependencies.
