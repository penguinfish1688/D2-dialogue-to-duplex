# Qwen3-Omni 80 ms public-release validation

Completed on **2026-10-04** using the released update-10,000 weights and the public inference runtime. A fresh clone and Python environment successfully downloaded the backbone and D2 bundle anonymously into an empty model cache. All **1,800 VoiceBench** examples and **535 Full-Duplex-Bench** examples completed generation and scoring.

VoiceBench scored **69.05**, compared with **70.07** in the historical run (-1.02 points). FDB quality was **4.58/5**. The observed results are close to the historical evaluation; individual responses and aggregate scores differ. Pause handling remains weak.

[Commands and protocol](README.md) · [Machine-readable results](results.json)

## VoiceBench

Exactly 200 examples per task at the pinned dataset revision. Scores are on a 0–100 scale; the three open-ended ratings are multiplied by 20. The overall score is the mean of all nine task scores. These fixed subsets are not an official full leaderboard submission.

| Task | Historical run | Public release run |
| --- | ---: | ---: |
| AlpacaEval | 85.97 | 87.57 |
| CommonEval | 69.47 | 70.47 |
| WildVoice | 71.53 | 70.87 |
| SD-QA | 64.50 | 62.00 |
| MMSU | 64.50 | 61.00 |
| OpenBookQA | 81.50 | 81.00 |
| BBH | 64.50 | 60.50 |
| IFEval | 37.19 | 34.04 |
| AdvBench | 91.50 | 94.00 |
| **Overall** | **70.07** | **69.05** |

AlpacaEval, CommonEval, WildVoice and SD-QA received fresh GPT-4o-mini judgments with three votes per example. The other five tasks use the pinned upstream evaluators. As a separate scorer check, the public scorer reproduced all nine historical scores exactly when given the original saved responses and votes.

## Full-Duplex-Bench

The reported protocol contains 119 Candor turn-taking, 200 synthetic interruption and 216 Candor pause examples. All 535 generated waveforms were transcribed with the pinned upstream Parakeet script from a fresh ASR environment. The public run has 198 qualifying interruption responses, each judged with the official GPT-4o prompt.

| Metric | Historical run | Public release run |
| --- | ---: | ---: |
| Quality / 5 | 4.56 | 4.58 |
| Turn-taking success | 118/119 (99.16%) | 118/119 (99.16%) |
| Turn latency | 0.521 s | 0.545 s |
| Interruption success | 199/200 (99.50%) | 198/200 (99.00%) |
| Interruption latency | 0.403 s | 0.392 s |
| Pause success | 46/216 (21.30%) | 47/216 (21.76%) |

These latencies are event-aligned acoustic gaps plus 80 ms and exclude device computation. RTF below measures computation separately.

## Real-time performance

All GPU runs used **RTX PRO 6000 Blackwell Server Edition (96 GB)** GPUs, BF16 inference and native CUDA graphs. The full benchmark used one CPU core and a 96 GiB host-memory allocation per GPU. Python was 3.12.15, PyTorch 2.11.0 and Transformers 5.13.0.

| Test | Fixed Thinker KV budget | Measured RTF |
| --- | ---: | ---: |
| VoiceBench, all 1,800 examples | 2048 tokens | 0.593 |
| FDB, all 535 examples | 4096 tokens | 0.753 |
| Browser conversation 1, 28.04 s acknowledged | 2048 tokens | 0.638 |
| Browser conversation 2, 28.04 s acknowledged | 2048 tokens | 0.637 |
| Fresh-install paced WebSocket, first conversation | 2048 tokens | 0.657 |
| Fresh-install paced WebSocket, second conversation | 2048 tokens | 0.653 |

Benchmark RTF is total streaming wall time divided by total input-timeline duration. VoiceBench includes 20 seconds of response silence per example; FDB adds none. First-use graph capture during streaming is included. Loading, kernel compilation during initialization, session setup, queueing, transcription and scoring are excluded.

The browser test used Chromium 153 with a prerecorded microphone input and the app's actual AudioWorklet, WebSocket and playback path. Both conversations produced a relevant answer, nonzero audio and no browser errors; maximum input backlog was **120 ms**. Session kernels were prewarmed. Physical microphone/speaker operation was not tested. The browser meets RTF below 1; its measured RTF remains above 0.6.

## Reproduction details

- Inference source: [`6668d3b`](https://github.com/penguinfish1688/D2-dialogue-to-duplex/tree/6668d3bd7ca638c0e38d8a708d04d1da3e5ffc00).
- D2 bundle: [`fa595c6`](https://huggingface.co/penguinfish1688/dialogue-to-duplex/tree/fa595c6cc9556e57480101448409f5386d2bc3ae); the manifest pins the upstream backbone as well.
- VoiceBench data: `b02edcef1330480be3a11bd6f7434ac32f05ad08`; evaluator: `3c3b0d3a7a956f745305eb348f5e03ce7ec73dad`.
- FDB v1.0 evaluator: `3e799c45a045256f47d5f1c9cda90157e2d2ec9e`; quality judge: `gpt-4o-2024-08-06`, seed 0.
- Public inference resets seed **1337 per example**, independently of sharding and resume order. Historical runs advanced worker-local seeds and used a larger fixed KV allocation. BF16 kernels, sampled speech, ASR and API judging can change results; this comparison does not establish bitwise equivalence.

Checkpoint hashes and full-precision aggregate metrics are in [results.json](results.json). Generation, transcription and judging records were saved and checksum-verified before archiving. Rerun with `--revision fa595c6cc9556e57480101448409f5386d2bc3ae` to select the same released weights.
