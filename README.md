# Dialogue-to-Duplex

**Currently, D2 releases only the Qwen3-Omni 80 ms checkpoint.**

**Dialogue-to-Duplex** adapts pretrained speech models to listen and speak at the same time. D2 learns when to respond, continue listening, and stop speaking after an interruption.

The conversion has two stages: distill a causal audio encoder from the original encoder, then fine-tune the speech model on dialogue timelines containing silence, overlap, and interruptions. At each interaction interval, the model reads user audio and its previously played audio, then produces text and speech. Each model keeps its native audio codec and speech generator.

## Start here

Use Linux, Python 3.12, an NVIDIA CUDA GPU, and a compatible driver. The full Qwen model needs substantial GPU memory; validation uses an RTX Pro 6000 with 96 GB. Install `git`, `ffmpeg`, and `libsndfile` through your system package manager. Qwen also needs a C++ compiler and Python 3.12 development headers for its native PyTorch kernels.

```bash
git clone https://github.com/penguinfish1688/D2-dialogue-to-duplex.git
cd D2-dialogue-to-duplex
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install **one** model per environment:

```bash
pip install -c qwen3-omni-D2/configs/requirements.txt -e '.[qwen]'
# Or, in a separate environment:
pip install -c llama-omni2-D2/configs/requirements.txt -e '.[llama]'
```

Their Transformers versions differ. Keep the environments separate.

The constraint files pin transitive dependencies as well as model libraries.

| Model | Interaction intervals | Instructions |
| --- | --- | --- |
| Qwen3-Omni D2 | 80, 160, 320, 640, 1040 ms | [Qwen README](qwen3-omni-D2/README.md) |
| LLaMA-Omni2 D2 | 100, 200, 400, 800 ms | [LLaMA README](llama-omni2-D2/README.md) |

The interval is part of the trained checkpoint; changing a command-line option cannot turn one checkpoint into another interval.

## Checkpoints

[Dialogue-to-Duplex](https://huggingface.co/penguinfish1688/dialogue-to-duplex) (Qwen3-Omni, 80 ms) is public and is the default for `d2-qwen`. No Hugging Face login is needed. LLaMA weights are not published yet; `d2-llama` requires a local release with `--model`.

```bash
d2-qwen download
d2-qwen infer --input question.wav --output response.wav --tail-seconds 20
d2-qwen app
```

The first download needs approximately **73 GB** for the Qwen backbone and D2 weights, plus space for dependencies and caches. Set `HF_HOME` before running to choose the cache location. Downloads are reused across commands. Each D2 release contains `d2.json`, `d2.safetensors`, and `encoder.safetensors`; the loader checks parameter names and shapes and downloads the pinned backbone automatically. Use `--revision COMMIT` to pin a D2 release, `--offline` (or `HF_HUB_OFFLINE=1`) to require cached assets, or `--model PATH_OR_HF_ID` for another release.

Open `http://localhost:8000` for the microphone app. For a remote GPU, run `ssh -L 8000:localhost:8000 user@gpu-host` on your computer, then open that same localhost URL. Use headphones. LLaMA uses the same commands with `d2-llama`. The app and WAV command share the inference implementation. Output is mono 24-kHz PCM. The WAV command also writes a JSON transcript/event trace and measured real-time factor (RTF). App acknowledgments report server processing time for measuring sustained streaming RTF; RTF below 1 means faster than real time.

Inference uses native PyTorch, a fixed KV-cache allocation, CUDA graphs for decoder prefill and serial decoding, and the trained control protocol. The default Thinker budget is **2048 tokens**. Conversations stop when that budget is exhausted; start a new conversation or increase `--kv-budget`. The maximum duration, including the system prompt, is reported by the runtime. File inference appends eight seconds of silence by default; use `--tail-seconds` for a longer response.

The system prompt is hardcoded, matching InstructS2S training:

> You are a helpful spoken conversational assistant. Respond naturally when the user finishes speaking.

It is inserted once, before the first audio interval. There is no app setting that changes it.

## Fine-tuning

Use a local or downloaded release and a manifest of prepared samples:

```bash
d2-qwen train --model MODEL --data samples.json --output my-d2 --steps 3
# Or:
d2-llama train --model MODEL --data samples.json --output my-d2 --steps 3
```

This is a small single-GPU fine-tuning loop. It uses the original model losses, parameter groups, BF16 compute, FP32 trainable weights, AdamW, and gradient clipping. It writes a new release directory that can be passed directly to `infer`, `app`, or `train`. Each invocation starts a new optimizer; optimizer-state resume and distributed orchestration are outside this interface.

See [sample format](d2/SAMPLES.md) and the model READMEs for tensor shapes and training defaults. Data, model weights, generated audio, and run logs belong outside Git.

## Validation

The 80-ms Qwen and 100-ms LLaMA paths were tested on RTX Pro 6000 GPUs using separate Python 3.12 environments and local D2 releases. Both completed three optimizer updates, saved and reloaded their checkpoints, and generated coherent spoken answers verified with Whisper ASR. The app's WebSocket path was exercised with real model output and 40-ms microphone-sized packets, including repeated conversations. Physical browser microphone/speaker operation has not been tested.

At update 1, Qwen's finite training loss was **0.46766442** and LLaMA's was **2.21675825**. Separate comparisons matched the original training forward losses bit for bit at the same weights and samples over three updates. These comparisons do not establish an independently trained, bitwise-identical trajectory from scratch.

On a 4.25-second question followed by 20 seconds of silence, warm streaming RTF was approximately **0.76 for Qwen** and **0.75 for LLaMA**, with a 2048-token Thinker budget. RTF is wall time divided by the complete input-plus-silence timeline; model loading and session initialization are excluded. **The 0.6 RTF target has not been reached.** These are sample smoke measurements, not a quality or throughput benchmark across all interaction intervals. Inference uses BF16 kernels and is not promised to be bitwise deterministic; the seed controls sampling.

The wheel builds with its configs and browser assets. Dependency checks and CPU tests pass in both model environments. To run the CPU checks, install `.[test]` alongside your model extra and run `python -m pytest`; tests for the other model's Transformers version are skipped.

## Repository

```text
d2/                 Shared checkpoint loading, PCM I/O, training loop, browser UI
qwen3-omni-D2/       Qwen model/, inference/, training/, app/, configs/
llama-omni2-D2/      LLaMA model/, inference/, training/, app/, configs/
tests/              Checkpoint, timing, cache, and app checks
```

Each model directory is an installable Python package (`d2_qwen` or `d2_llama`). There are no dependencies on a private checkout or cluster directory.

Upstream projects: [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni), [LLaMA-Omni2](https://github.com/ictnlp/LLaMA-Omni2), [Transformers](https://github.com/huggingface/transformers), [Whisper](https://github.com/openai/whisper), [CosyVoice](https://github.com/FunAudioLLM/CosyVoice), and [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS). Their code and model licenses apply to those dependencies.

## Benchmarks

See [benchmarks/README.md](benchmarks/README.md) for the fixed VoiceBench selection (200 examples per task), Full-Duplex-Bench, and a paced app test. Reported research scores and fresh public-release measurements are kept separate.
