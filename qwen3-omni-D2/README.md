# Qwen3-Omni D2

Placeholder for the Qwen3-Omni implementation of D2.

Planned configurations: **80, 160, 320, 640, and 1040 ms** interaction
granularity, sharing one implementation with an 80-ms native clock.

- `configs/`: model, data, training, and inference settings.
- `model/`: streaming audio encoder and duplex model.
- `training/`: audio encoder distillation, then duplex fine-tuning.
- `inference/`: streaming sessions, speech generation, and playback.
- `app/`: microphone/speaker demo using the streaming inference code.

Code, dependencies, launch commands, and checkpoint instructions will be added
with the implementation. Shared data preparation and evaluation belong in
[`../data/`](../data/) and [`../evaluation/`](../evaluation/).
