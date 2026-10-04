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

This streams audio at microphone speed and saves the response and timing results. RTF is processing time divided by input duration; values below 1 mean faster than real time.
