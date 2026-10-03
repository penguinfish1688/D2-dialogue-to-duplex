# LLaMA-Omni2 D2

Placeholder for the LLaMA-Omni2 implementation of D2.

Planned configurations: **100, 200, 400, and 800 ms** interaction granularity,
sharing one implementation with a 100-ms native clock and native speech synthesis.

- `configs/`: model, data, training, and inference settings.
- `model/`: streaming audio encoder, duplex model, and native speech generator.
- `training/`: audio encoder distillation, then duplex fine-tuning.
- `inference/`: streaming sessions, speech generation, and playback.
- `app/`: microphone/speaker demo using the streaming inference code.

Code, dependencies, launch commands, and checkpoint instructions will be added
with the implementation. Shared data preparation and evaluation belong in
[`../data/`](../data/) and [`../evaluation/`](../evaluation/).
