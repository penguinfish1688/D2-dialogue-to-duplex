# Full-Duplex-Bench

Run these commands from the D2 repository root with the [Qwen environment](../../README.md#installation) activated.

## Setup

```bash
git submodule update --init benchmarks/fdb/official
pip install -r benchmarks/fdb/requirements.txt
```

The [official repository](https://github.com/DanielLin94144/Full-Duplex-Bench/tree/3e799c45a045256f47d5f1c9cda90157e2d2ec9e) provides the ASR and evaluation code. Download the [v1.0 dataset](https://github.com/DanielLin94144/Full-Duplex-Bench/tree/3e799c45a045256f47d5f1c9cda90157e2d2ec9e/v1_v1.5/dataset), following its access instructions.

## Generate Responses

```bash
python -m benchmarks.fdb.run --data /path/to/v1_0 --output results/fdb
```

This generates time-aligned `output.wav` files and copies the annotations into the official directory layout. Use `--help` for checkpoint, cache, and sharding options.

## Transcribe

Install ASR dependencies in a separate environment, then run the official transcription script:

```bash
python3.12 -m venv .venv-asr
.venv-asr/bin/python -m pip install --upgrade pip
.venv-asr/bin/python -m pip install -r benchmarks/fdb/asr-requirements.txt
.venv-asr/bin/python benchmarks/fdb/official/v1_v1.5/get_transcript/asr.py \
  --root_dir results/fdb/synthetic_user_interruption --task user_interruption
for category in candor_turn_taking candor_pause_handling; do
  .venv-asr/bin/python benchmarks/fdb/official/v1_v1.5/get_transcript/asr.py \
    --root_dir "results/fdb/$category"
done
```

## Score

Set `OPENAI_API_KEY` in your environment, then use the Qwen environment:

```bash
python -m benchmarks.fdb.score --run results/fdb
```

This calls [`official/v1_v1.5/evaluation/evaluate.py`](https://github.com/DanielLin94144/Full-Duplex-Bench/blob/3e799c45a045256f47d5f1c9cda90157e2d2ec9e/v1_v1.5/evaluation/evaluate.py) for smooth turn taking, user interruption, and pause handling. The official evaluator uses `gpt-4-turbo` for interruption quality; API judging incurs charges. Rerunning interruption scoring makes new judge requests.

Official output is saved under `results/fdb/scores/`, with the reported numbers in `all.json`. Rates are fractions from 0 to 1; latency is the official acoustic delay in seconds, without an added interaction interval or computation time. Pause handling reports turn-over rate, where lower is better.

To run an individual task:

```bash
python -m benchmarks.fdb.score --run results/fdb --task user_interruption
```

The same evaluation can be called directly:

```bash
python benchmarks/fdb/official/v1_v1.5/evaluation/evaluate.py \
  --task user_interruption --root_dir results/fdb/synthetic_user_interruption
```
