"""Fine-tune native Thinker and TTS losses on complete short timelines."""

from d2.training import run
from d2_llama.data.episode import Episode
from d2_llama.model.loading import load_model, training_config
from d2_llama.model.trainability import build_stage3_optimizer, build_stage3_scheduler


def configure_optimizer(model):
    config = training_config()
    optimizer = build_stage3_optimizer(model.trainability, config)
    return optimizer, build_stage3_scheduler(optimizer, config)


def forward(model, tensors, row):
    required = {
        "environment_features",
        "delayed_self_features",
        "text_input_ids",
        "text_target_ids",
        "text_weights",
    }
    if set(tensors) != required:
        raise ValueError(f"Llama sample requires exactly {sorted(required)}")
    k = model.audio_tower.frames_per_unit
    frames = tensors["text_target_ids"].shape[1]
    if frames % k or frames > 900 // k * k or tensors["text_target_ids"].shape[0] != 1:
        raise ValueError(
            "Sample must be one complete, macro-aligned timeline of at most 90 seconds and batch size 1"
        )
    episodes = [
        Episode(
            identity=e["identity"],
            first_frame=e["first_frame"],
            lexical=tuple(e["lexical"]),
            audio_ids=tuple(e["audio_ids"]),
            stop_frame=e.get("stop_frame"),
        )
        for e in row["episodes"]
    ]
    result = model(
        **tensors,
        episodes=episodes,
        start_frame=0,
        frames_per_unit=k,
        reset=True,
        memory_frames=900 // k * k,
        task_kind=row.get("task", "conversation"),
    )
    return result.loss, dict(text_loss=result.text_loss, tts_loss=result.tts_loss)


def train(args):
    run(
        args,
        family="llama",
        load_model=load_model,
        configure_optimizer=configure_optimizer,
        forward=forward,
    )
