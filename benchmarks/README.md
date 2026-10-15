# Evaluation

Each benchmark has its own directory, instructions, and official repository:

| Benchmark | Guide | Official code |
| --- | --- | --- |
| Full-Duplex-Bench | [fdb/](fdb/README.md) | [Full-Duplex-Bench](fdb/official) |
| VoiceBench | [vb/](vb/README.md) | [VoiceBench](vb/official) |

The official repositories are pinned Git submodules. Initialize them from the D2 repository root:

```bash
git submodule update --init
```

D2 generates responses in each benchmark's expected format. Scoring runs the unmodified official programs, including their prompts, judge models, and metric calculations.

## App Performance

Start `d2-qwen app`, then run this in another terminal:

```bash
python -m benchmarks.check_app --input question.wav --output app-check.wav
```

Use a short question recording; the check appends 20 seconds of silence for the response. The recording and response tail must fit within the app's conversation limit (about 54 seconds by default).

The check streams audio at microphone speed and saves `app-check.wav` and `app-check.json`. Look for `"passed": true`: this requires **RTF < 1**, no more than two seconds of input backlog, and the expected output audio duration. RTF is processing time divided by input duration; it excludes model loading and session initialization. The paced wall time stays close to the audio duration because the input arrives in real time.

### Verified configuration

On October 15, 2026, a fresh Python 3.12 environment using the pinned dependencies passed on an **RTX PRO 6000 Blackwell Server Edition (96 GB)** with NVIDIA driver 590.48.01 and 96 GB of host RAM. The app used its default 80 ms checkpoint and 2048-token KV budget, with one CPU core and `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`.

| Check (two sessions each) | Input duration per session | RTF | Maximum input backlog |
| --- | --- | --- | --- |
| Paced WebSocket stream | 24.8 s | 0.583–0.584 | 0.08 s |
| Chromium browser, recorded microphone input | 49.0 s | 0.591–0.598 | 0.12–0.48 s |

The browser sessions exercised microphone capture, audio playback, transcript updates, and stop/restart, with three recorded questions per session and no browser or server errors. Model loading and compilation took about 130 seconds with an empty compiler cache; backbone weights were already cached. These are measurements for this configuration and input, rather than a guarantee for every GPU or conversation.
