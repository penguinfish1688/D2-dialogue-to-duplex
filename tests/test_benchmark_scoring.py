import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.common import verify_upstream
from benchmarks.fdb.common import FDB_COMMIT
from benchmarks.fdb.score import official_metrics
from benchmarks.vb.common import TASKS
from benchmarks.vb.score import check_judged, export_responses


def test_official_fdb_counts_early_speech_without_changing_latency(tmp_path):
    """The adapter must preserve upstream's early-speech and clamping rules."""
    official = Path(__file__).resolve().parents[1] / "benchmarks/fdb/official"
    if not (official / ".git").exists():
        pytest.skip("Initialize benchmark submodules to run official scorer integration")
    pytest.importorskip("dotenv")
    pytest.importorskip("openai")
    verify_upstream(official, FDB_COMMIT)
    root = tmp_path / "candor_turn_taking"
    for index, start in enumerate([1.0, 3.0]):
        folder = root / str(index)
        folder.mkdir(parents=True)
        (folder / "turn_taking.json").write_text(json.dumps([dict(timestamp=[2.0, 2.0])]))
        (folder / "output.json").write_text(
            json.dumps(
                dict(text="Hello", chunks=[dict(text="Hello", timestamp=[start, start + 2])])
            )
        )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.fdb.score",
            "--run",
            str(tmp_path),
            "--task",
            "smooth_turn_taking",
        ],
        cwd=official.parents[2],
        capture_output=True,
        text=True,
        check=True,
    )
    metrics = official_metrics(result.stdout)
    assert metrics == {"take_turn": 1.0, "latency": 0.5}
    assert (
        json.loads((tmp_path / "scores/smooth_turn_taking.json").read_text())["tasks"][
            "smooth_turn_taking"
        ]["latency"]
        == 0.5
    )


def test_voicebench_exports_responses_and_rejects_stale_judgments(tmp_path):
    for index, task in enumerate(TASKS):
        folder = tmp_path / task / "sample"
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(
            json.dumps(
                dict(
                    selection_index=index,
                    identity={"model": "test"},
                    benchmark_id="sample",
                    row={"prompt": "What color is the sky?"},
                    response="It is blue.",
                )
            )
        )
    exports = export_responses(tmp_path, 1)
    source = exports / "alpacaeval.jsonl"
    row = json.loads(source.read_text())
    assert row["response"] == "It is blue."
    judged = exports / "result-alpacaeval.jsonl"
    judged.write_text(json.dumps(dict(row, score=["5", "4", "5"])) + "\n")
    check_judged(source, judged)
    judged.write_text(json.dumps(dict(row, response="Red.", score=["5", "4", "5"])) + "\n")
    with pytest.raises(ValueError, match="different responses"):
        check_judged(source, judged)
