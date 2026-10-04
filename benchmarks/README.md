# Evaluation

Evaluate the Qwen3-Omni 80 ms checkpoint on VoiceBench and Full-Duplex-Bench. Run these commands from the repository root with the [Qwen environment](../README.md#installation) activated.

## Setup

```bash
pip install -r benchmarks/requirements.txt
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng
```

Set `OPENAI_API_KEY` in your environment to enable API-based scoring with `--judge`. Judging incurs API charges; completed judgments are cached so scoring can resume.

## VoiceBench

Prepare the audio and evaluator, then generate responses:

```bash
python benchmarks/prepare_voicebench.py --root bench-data/voicebench
python benchmarks/run.py voicebench \
  --data bench-data/voicebench/data/diagnostic200/manifest.json \
  --output results/voicebench
```

Score the responses:

```bash
python benchmarks/score_voicebench.py --run results/voicebench --judge
```

Scores are saved to `results/voicebench/scores/metrics.json`. Omit `--judge` to run only local scoring. Use each script's `--help` for dataset and inference options.

## Full-Duplex-Bench

Download the [v1.0 dataset](https://github.com/DanielLin94144/Full-Duplex-Bench/tree/3e799c45a045256f47d5f1c9cda90157e2d2ec9e/v1_v1.5/dataset), following its access instructions and terms, then generate responses:

```bash
python benchmarks/run.py fdb --data /path/to/v1_0 --output results/fdb
```

Transcribe the generated speech using the upstream ASR tool. Its dependencies require a separate environment:

```bash
git clone https://github.com/DanielLin94144/Full-Duplex-Bench.git bench-data/Full-Duplex-Bench
git -C bench-data/Full-Duplex-Bench checkout 3e799c45a045256f47d5f1c9cda90157e2d2ec9e
python3.12 -m venv .venv-asr
.venv-asr/bin/python -m pip install --upgrade pip
.venv-asr/bin/python -m pip install -r benchmarks/asr-requirements.txt
.venv-asr/bin/python bench-data/Full-Duplex-Bench/v1_v1.5/get_transcript/asr.py \
  --root_dir results/fdb/synthetic_user_interruption --task user_interruption
for category in candor_turn_taking candor_pause_handling; do
  .venv-asr/bin/python bench-data/Full-Duplex-Bench/v1_v1.5/get_transcript/asr.py \
    --root_dir "results/fdb/$category"
done
```

Score the transcribed responses with the Qwen environment:

```bash
python benchmarks/score_fdb.py --run results/fdb --judge
```

Scores are saved to `results/fdb/scores/metrics.json`. Omit `--judge` to compute interaction metrics without API-based response quality scoring.

## App Performance

Start the app with `d2-qwen app`, then run this in another terminal:

```bash
python benchmarks/check_app.py --input question.wav --output app-check.wav
```

This streams audio at microphone speed and saves the response and timing results. RTF is processing time divided by input duration; values below 1 mean faster than real time.

## Options

- **Resume:** rerun the same command to skip completed samples. Use a new output directory when changing model or inference settings.
- **Multiple GPUs:** launch one process per GPU with `--shard I --shards N`, sharing the output directory.
- **Offline:** run `d2-qwen download` first, then add `--offline` to inference commands.
- **Checkpoint:** select a release with `--model PATH_OR_HF_ID` and optionally `--revision COMMIT`.
