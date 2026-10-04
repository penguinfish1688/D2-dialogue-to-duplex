import importlib.util
from pathlib import Path

import torch


def common():
    path = Path(__file__).resolve().parents[1] / "benchmarks/common.py"
    spec = importlib.util.spec_from_file_location("benchmark_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audible_boundary_ignores_an_isolated_late_spike():
    wave = torch.zeros(3200)
    wave[160:480] = 0.1  # Two consecutive 10 ms blocks qualify.
    wave[2880:3040] = 0.2  # One isolated block must not move the boundary.
    assert common().last_audible_frame(wave) == 0


def test_response_excludes_early_answer_and_stops_at_interruption():
    trace = [
        dict(kind="response", frame=2),
        dict(kind="text", frame=3, token_id=7),
        dict(kind="interrupt", frame=4),
        dict(kind="response", frame=8),
        dict(kind="text", frame=9, token_id=11),
        dict(kind="interrupt", frame=10),
        dict(kind="response", frame=12),
        dict(kind="text", frame=13, token_id=99),
    ]

    class Tokenizer:
        def decode(self, ids, **kwargs):
            assert ids == [11]
            return "answer"

    assert common().response_text(dict(text_trace=trace), 5, Tokenizer()) == "answer"
