# Qwen3-Omni D2

A causal audio encoder, the native Qwen Thinker, Talker, 16-codebook predictor, and Code2Wav renderer. All interaction intervals retain the native **80-ms** text/audio clock.

## Install and run

From the repository root, in a Python 3.12 environment:

```bash
pip install -c qwen3-omni-D2/configs/requirements.txt -e '.[qwen]'
d2-qwen download
d2-qwen infer --input question.wav --output response.wav --tail-seconds 20
d2-qwen app
# Equivalent app entry point:
python -m d2_qwen.app
```

The default is the public [80 ms checkpoint](https://huggingface.co/penguinfish1688/dialogue-to-duplex). No login is required. Pass `--model PATH_OR_HF_ID` to select another release. Downloading the backbone and D2 weights needs about 73 GB of disk space; set `HF_HOME` to select the cache directory. Use `--revision COMMIT` to pin a hosted D2 release and `--offline` to require cached assets. Dependencies are pinned to PyTorch 2.11 and Transformers 5.13.

The Dialogue-to-Duplex app listens on localhost:8000. Use headphones. For a remote GPU, forward port 8000 over SSH; microphone access works on localhost. Only one conversation can use a model instance at a time.

## Runtime

- Input: mono PCM16 at 16 kHz. Output: mono PCM16 at 24 kHz.
- An interval contains `k = latency_ms / 80` native frames: `k ∈ {1,2,4,8,13}`.
- The causal encoder uses a 104-mel convolution window and at most 104 attention tokens.
- Thinker input order is `[environment × k, delayed self × k, previous text × k]`. Text is decoded serially after the audio prefill. Self audio comes from played output, delayed by one interaction interval.
- The default fixed Thinker budget is 2048 tokens, including the hardcoded system prefix. The runtime reports the exact conversation limit. Talker and waveform caches are bounded for the same duration.
- Native decoder operations use CUDA graphs. Encoder startup keeps the original attention shape while its history fills; steady-state encoder work uses a graph with the full bounded window.
- Text is greedy. Codec sampling uses temperature 0.7, top-k 50, top-p 1.0, repetition penalty 1.1; residual codebooks use temperature 0.7, top-k 50, top-p 0.8. Default seed: 1337.
- RESPONSE and INTERRUPT retain IDs 151669 and 151670. The appended PAD row, audio timing, codec BOS/EOS, and native Chelsie speaker are unchanged.

Inference uses BF16 model computation and PyTorch-compiled native `grouped_mm` experts. This avoids copying each selected expert's weights and supports CUDA graph capture. Install a C++ compiler and Python 3.12 development headers; the first model load compiles and caches kernels and can take several minutes. Training uses native `eager` to keep expert activations memory-efficient. First-use graph capture within the streaming loop is included in the WAV command's reported RTF.

## Fine-tune

```bash
d2-qwen train --data samples.json --output my-qwen-d2 --steps 3
```

Samples use the [shared manifest format](../d2/SAMPLES.md). The model has **266,286,976** trainable parameters: rank-128 Thinker/Talker/context/CodePredictor adapters, codec interfaces and small parameters, control/stream rows, and the current encoder SFT parameters. Routed experts remain frozen. Encoder SFT trains its convolution frontend, rank-32 attention adapters, and existing rank-32 non-attention adapters.

Loss is text CE plus codec CE. PAD weight is 0.05, RESPONSE/INTERRUPT weights are 20, codec EOS weight is 20, and the 16 codebook weights are 1. Optimizer groups and their peak/minimum rates are in [configs/train.json](configs/train.json). Warmup lasts 200 updates, followed by cosine decay through update 2000. AdamW uses betas `(0.9, 0.999)`, epsilon `1e-8`, no weight decay, and gradient-norm clipping at 1.0.

The release manifest schema is illustrated in [configs/release.json](configs/release.json). `encoder.safetensors` contains the distilled encoder trainables; `d2.safetensors` contains the complete current SFT trainable inventory. No optimizer, dataset, or research-run metadata is required for inference.

[Benchmark reproduction and paced app test](../benchmarks/README.md). The app sends processing time in each acknowledgment so streaming RTF can be measured independently of microphone pacing.
