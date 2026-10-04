# LLaMA-Omni2 D2

LLaMA-Omni2 D2 checkpoints are not released yet. Currently, D2 releases only the [Qwen3-Omni 80 ms checkpoint](https://huggingface.co/penguinfish1688/dialogue-to-duplex).

A causal Whisper-large-v3 encoder, the released LLaMA-Omni2 Qwen2 Thinker, its native speech generator, and CosyVoice2 waveform rendering. Every interval retains the native **100-ms** text/playback clock and **25-Hz** speech-unit clock.

## Install and run

From the repository root, in its own Python 3.12 environment:

```bash
pip install -c llama-omni2-D2/configs/requirements.txt -e '.[llama]'
d2-llama download --model MODEL
d2-llama infer --model MODEL --input question.wav --output response.wav
d2-llama app --model MODEL
# Equivalent app entry point:
python -m d2_llama.app --model MODEL
```

`MODEL` is a local release directory or an eventual Hugging Face ID. The default `HF_ORG/LLaMA-Omni2-D2` is a placeholder. Use `--revision COMMIT` to pin a hosted D2 release and `--offline` to require cached assets. This environment uses Transformers 4.43.4 and must be separate from Qwen's environment.

The downloader also fetches the pinned LLaMA-Omni2 source, native model, Whisper weights, CosyVoice2 renderer, and checksum-verified Matcha renderer dependency. No manual private checkout is needed. Upstream code is cached under `HF_HOME/d2`; the released English voice prompt is used.

The app listens on localhost:8000. Use headphones. For a remote GPU, forward port 8000 over SSH. Only one conversation can use a model instance at a time.

## Runtime

- Input: mono PCM16 at 16 kHz. Output: mono PCM16 at 24 kHz.
- An interval contains `k = latency_ms / 100` native frames: `k ∈ {1,2,4,8}`.
- Whisper sees the current complete macro and past macros, with the original centered convolution tail and macro-aligned segments of up to 30 seconds. Its captured inference path caches past keys and values instead of recomputing previous macros.
- Thinker input order is `[environment × k, delayed self × k, previous text × k]`, with native serial rotary positions and the hardcoded system prompt inserted once.
- Thinker and speech-decoder prefill and serial decode share fixed KV allocations and use CUDA graphs. The default Thinker budget is 2048 tokens; the reported conversation limit includes the prefix.
- The native speech generator reads three text conditions and writes ten speech units. PAD finishes the text conditions; native EOS finishes speech. INTERRUPT clears unplayed audio and response state while retaining listening history.
- Thinker text is greedy. The native speech generator samples with temperature 1.0 and top-p 1.0. Default seed: 1337. A response is bounded to 192 lexical tokens and 1024 tail tokens.
- Self feedback is the PCM actually published for playback, resampled to 16 kHz with the original linear interpolation rule.

Thinker, speech-decoder, and encoder computation use BF16. Inference stores the encoder's linear/convolution weights in BF16 after merging adapters, avoiding repeated casts; training retains its FP32 masters. The native CosyVoice2 waveform renderer retains FP32. Its chunk mask uses the same attention visibility as upstream with vectorized construction. A CUDA graph replays its native estimator across the ten diffusion steps, retaining at most four input shapes and checking each captured shape against eager execution. First-use graph capture is included in the WAV command's reported RTF.

## Fine-tune

```bash
d2-llama train --model MODEL --data samples.json --output my-llama-d2 --steps 2000
```

Samples use the [shared manifest format](../d2/SAMPLES.md). The current model has **180,607,488** trainable parameters: rank-128 Thinker attention and speech-decoder attention/MLP adapters, rank-32 encoder attention/MLP/projector adapters, encoder convolution weights, control rows, and two stream rows. The released speech-conditioning projection and fusion gate remain frozen.

Loss is weighted Thinker CE plus native speech-token CE. PAD weight is 0.05; RESPONSE/INTERRUPT weights are 20; speech EOS weight is 3. [configs/train.json](configs/train.json) contains the exact learning rates and optimizer schedule: 200-update warmup, cosine decay through update 2000 to 10% of peak, AdamW betas `(0.9, 0.999)`, epsilon `1e-8`, no weight decay, and gradient-norm clipping at 1.0.

The release manifest schema is illustrated in [configs/release.json](configs/release.json). `encoder.safetensors` contains the full distilled encoder state, including learned dense attention; `d2.safetensors` contains the current SFT trainable inventory. Tensor precision is preserved when loading the encoder before SFT adapters are installed.
