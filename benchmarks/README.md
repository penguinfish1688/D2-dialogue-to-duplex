# Reproduce the Qwen 80 ms release

Run these commands from the repository root after the Qwen installation. They use the same public inference runtime as the app. Set `HF_HOME` to a disk with room for the model. No research repository is needed.

The HF model card's scores are the existing research results. A fresh release run must be scored before claiming that those results reproduce. Sampling, BF16 kernels, and API judges can change individual answers.

## VoiceBench: 200 examples per task

```bash
python benchmarks/prepare_voicebench.py --root bench-data/voicebench --limit-per-task 200
python benchmarks/run.py voicebench \
  --data bench-data/voicebench/data/diagnostic200/manifest.json \
  --output results/voicebench
```

Preparation reports metadata and audio progress, downloads only the selected audio, and clones the pinned upstream evaluator. The dataset revision is `b02edcef1330480be3a11bd6f7434ac32f05ad08`; the evaluator commit is `3c3b0d3a7a956f745305eb348f5e03ce7ec73dad`. SD-QA accents and MMSU domains are selected round-robin; the remaining tasks use their first 200 rows. AlpacaEval uses `alpacaeval_full`.

Each prompt receives 20 seconds of response silence. Scoring uses model text after a RESPONSE event following the final audible user frame. The original benchmark's linear resampling, PCM rounding, and -60 dBFS audible-boundary detector are preserved. Audio is saved losslessly as FLAC to save disk space.

```bash
pip install -r benchmarks/requirements.txt
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng
python benchmarks/score_voicebench.py --run results/voicebench
# Set OPENAI_API_KEY in your shell for the four API-scored tasks, then:
python benchmarks/score_voicebench.py --run results/voicebench --judge
```

The official GPT-4o-mini judge uses three votes for AlpacaEval, CommonEval, WildVoice, and SD-QA. API judging incurs charges. The other five tasks use the pinned upstream evaluators. Missing API scores remain pending; they are never estimated. The overall score is the mean of all nine 0–100 task scores; open-ended 1–5 ratings are multiplied by 20. Results are written to `results/voicebench/scores/metrics.json`.

## Full-Duplex-Bench

Download the [upstream v1.0 data](https://github.com/DanielLin94144/Full-Duplex-Bench/tree/3e799c45a045256f47d5f1c9cda90157e2d2ec9e/v1_v1.5/dataset) and follow its dataset terms. This protocol evaluates the three subsets used for the reported D2 results: 119 Candor turn-taking, 200 synthetic user-interruption, and 216 Candor pause-handling samples.

```bash
python benchmarks/run.py fdb --data /path/to/v1_0 --output results/fdb
```

FDB uses a fixed 4096-token KV budget so the longest input (94 seconds) fits without truncation. It preserves each input's duration, adds no response tail, and saves `output.wav` plus annotations in the upstream directory layout. Each `result.json` also includes generated dialogue text, the control trace, and timing; FDB scoring uses the ASR transcript of the waveform. VoiceBench and the app default to 2048 tokens. Every sample is checked against the allocated budget before inference.

Use the pinned [upstream ASR and evaluation scripts](https://github.com/DanielLin94144/Full-Duplex-Bench/tree/3e799c45a045256f47d5f1c9cda90157e2d2ec9e/v1_v1.5) to transcribe and score the output. ASR requires its separate NeMo environment and `nvidia/parakeet-tdt-0.6b-v2`; install the upstream ASR requirements separately from Qwen. Interruption transcription crops at the annotated interruption end. The model card's latency convention is the event-aligned acoustic gap plus 80 ms, excluding device computation time.

After transcription, compute the table metrics and optional GPT quality:

```bash
git clone https://github.com/DanielLin94144/Full-Duplex-Bench.git bench-data/Full-Duplex-Bench
git -C bench-data/Full-Duplex-Bench checkout 3e799c45a045256f47d5f1c9cda90157e2d2ec9e
python benchmarks/score_fdb.py --run results/fdb
# With OPENAI_API_KEY set:
python benchmarks/score_fdb.py --run results/fdb --judge
```

Quality judging uses the pinned upstream prompt, `gpt-4o-2024-08-06`, and seed 0, and incurs API charges. Each sample's rating is saved so interrupted scoring can resume. Turn-taking requires at least one second of transcribed response or more than three words; the latency calculation uses qualifying words after the annotated user event. Missing quality ratings remain unreported.

## App: sustained real-time streaming

Start the app with `d2-qwen app`, then run this in another terminal:

```bash
python benchmarks/check_app.py --input question.wav --output app-check.wav
```

The check connects to the real WebSocket server, sends 40 ms packets at microphone speed, receives generated audio, and checks RTF below 1, output duration, and the browser's two-second backlog limit. It writes timing and audio artifacts. RTF is server processing time divided by input duration; paced wall time is reported separately. Model loading and session setup happen before microphone streaming. This automated transport check does not replace listening through a physical microphone and speakers.

## Resume or use multiple GPUs

Inference resumes completed samples only when checkpoint hashes, inputs, and settings match. Use a new output directory for a different configuration. Add `--offline` after `d2-qwen download` to use only cached model files. Add `--revision HF_COMMIT` to pin the release and `--shard I --shards N` to run N independent one-GPU processes into the same output directory. Sharding does not change the selected samples. For a short smoke, use `--limit 1`; full VoiceBench requires 200 per task, and full FDB uses the default without a limit.
