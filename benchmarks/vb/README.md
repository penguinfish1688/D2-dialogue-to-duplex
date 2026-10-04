# VoiceBench

Run these commands from the D2 repository root with the [Qwen environment](../../README.md#installation) activated.

## Setup

```bash
git submodule update --init benchmarks/vb/official
pip install -r benchmarks/vb/requirements.txt
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng
```

## Generate Responses

Prepare the audio, then run inference:

```bash
python -m benchmarks.vb.prepare --root bench-data/vb
python -m benchmarks.vb.run \
  --data bench-data/vb/data/diagnostic200/manifest.json \
  --output results/vb
```

Use `--help` for dataset selection, checkpoint, cache, and sharding options.

## Score

Set `OPENAI_API_KEY` in your environment, then run:

```bash
python -m benchmarks.vb.score --run results/vb --judge
```

The adapter exports response JSONL files, calls the official [`api_judge.py`](https://github.com/MatthewCYM/VoiceBench/blob/3c3b0d3a7a956f745305eb348f5e03ce7ec73dad/api_judge.py) for API-scored tasks, then calls [`evaluate.py`](https://github.com/MatthewCYM/VoiceBench/blob/3c3b0d3a7a956f745305eb348f5e03ce7ec73dad/evaluate.py) for each task. Both programs run directly from the unmodified [VoiceBench repository](https://github.com/MatthewCYM/VoiceBench/tree/3c3b0d3a7a956f745305eb348f5e03ce7ec73dad).

The official judge uses `gpt-4o-mini` and incurs API charges. Completed judge files are reused when they match the responses. Omit `--judge` to score local tasks and any existing judge outputs.

Response files, judge outputs, and evaluator logs are saved under `results/vb/scores/`. Open-ended ratings use a 1–5 scale; QA and multiple-choice results use percentages; IFEval and refusal rates use a 0–1 scale, following the official evaluators.

For example, to call the official programs directly on an exported task:

```bash
D2_REPO="$PWD"
cd results/vb/scores
python "$D2_REPO/benchmarks/vb/official/api_judge.py" --src_file alpacaeval.jsonl
python "$D2_REPO/benchmarks/vb/official/evaluate.py" \
  --src_file result-alpacaeval.jsonl --evaluator open
```
