"""VoiceBench sample selection and D2 response extraction."""

TASKS = (
    "alpacaeval",
    "commoneval",
    "wildvoice",
    "sd-qa",
    "mmsu",
    "openbookqa",
    "bbh",
    "ifeval",
    "advbench",
)
DATASET_REVISION = "b02edcef1330480be3a11bd6f7434ac32f05ad08"
VOICEBENCH_COMMIT = "3c3b0d3a7a956f745305eb348f5e03ce7ec73dad"
DEFAULT_SAMPLE_LIMIT = 200
ROLLOUT_FRAMES = 250
VOICEBENCH_MANIFEST_CONTRACT = "d2_voicebench_fixed_subset_manifest_v1"


def response_text(summary, start_frame, tokenizer):
    """Use only the response beginning after the last audible user frame."""
    started = False
    tokens = []
    for event in summary["text_trace"]:
        kind = event["kind"]
        if not started:
            started = kind == "response" and event["frame"] >= start_frame
        elif kind == "interrupt":
            break
        elif kind == "text":
            tokens.append(event["token_id"])
    return tokenizer.decode(tokens, skip_special_tokens=True) if tokens else ""


def last_audible_frame(wave):
    """Last 80 ms frame in a >=20 ms run above -60 dBFS, before PCM rounding."""
    import torch

    blocks = torch.nn.functional.pad(wave, (0, -len(wave) % 160)).view(-1, 160)
    active = (blocks.double().square().mean(1) >= 1e-6).tolist()
    run = 0
    last = None
    for index, value in enumerate(active):
        run = run + 1 if value else 0
        if run >= 2:
            last = min(len(wave), (index + 1) * 160) - 1
    if last is None:
        raise ValueError("Input has no qualifying audible run")
    return last // 1280
