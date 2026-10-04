import json
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from d2.hub import load_release, snapshot
from d2.prompts import CONVERSATION_SYSTEM_PROMPT, format_system_prompt
from d2.server import create_app
from d2.weights import restore_parameters


def test_prompt_matches_training():
    assert (
        CONVERSATION_SYSTEM_PROMPT
        == "You are a helpful spoken conversational assistant. Respond naturally when the user finishes speaking."
    )
    assert (
        format_system_prompt(CONVERSATION_SYSTEM_PROMPT)
        == f"<|im_start|>system\n{CONVERSATION_SYSTEM_PROMPT}<|im_end|>\n"
    )


def test_placeholder_fails_before_network():
    with pytest.raises(ValueError, match="not been uploaded"):
        snapshot("HF_ORG/Qwen3-Omni-D2")


def test_offline_environment_forces_cached_snapshot(monkeypatch, tmp_path):
    import huggingface_hub
    from huggingface_hub import constants

    def download(**kwargs):
        assert kwargs["local_files_only"] is True
        return str(tmp_path)

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert snapshot("example/model") == tmp_path


def test_release_rejects_wrong_family_and_latency(tmp_path):
    save_file({"weight": torch.zeros(1)}, str(tmp_path / "d2.safetensors"))
    manifest = dict(format="d2.release.v1", family="qwen", latency_ms=80)
    (tmp_path / "d2.json").write_text(json.dumps(manifest))
    assert load_release(str(tmp_path), family="qwen")[1] == manifest
    with pytest.raises(ValueError, match="Expected a llama"):
        load_release(str(tmp_path), family="llama")
    manifest["latency_ms"] = 100
    (tmp_path / "d2.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="granularity"):
        load_release(str(tmp_path), family="qwen")


def test_restore_requires_exact_inventory_and_finite_values():
    model = torch.nn.Linear(2, 1, bias=False)
    with pytest.raises(ValueError, match="inventory"):
        restore_parameters(model, {})
    with pytest.raises(ValueError, match="Invalid"):
        restore_parameters(model, {"weight": torch.full((1, 2), float("nan"))})
    restore_parameters(model, {"weight": torch.ones(1, 2)})
    assert torch.equal(model.weight, torch.ones(1, 2))


class Session:
    closed = False

    async def push_pcm16(self, payload):
        assert payload == bytes(3200)
        return SimpleNamespace(
            pcm_frames=(bytes(4800),),
            frame_events=({"type": "frame", "token_id": 9, "text_delta": "hello"},),
        )

    async def close(self):
        self.closed = True


class Runtime:
    def __init__(self):
        self.session = Session()

    async def load(self):
        pass

    def metadata(self):
        return dict(latency_ms=100)

    async def create_session(self):
        return self.session

    async def close(self):
        pass


def test_browser_transport_uses_native_pcm_and_closes_session():
    runtime = Runtime()
    with TestClient(create_app(runtime)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/audio.js").status_code == 200
        with client.websocket_connect("/stream") as socket:
            assert socket.receive_json()["type"] == "ready"
            socket.send_bytes(bytes(3200))
            assert socket.receive_json() == dict(type="event", token_id=9, text_delta="hello")
            assert socket.receive_bytes() == bytes(4800)
            ack = socket.receive_json()
            assert ack["type"] == "ack" and ack["samples"] == 1600
            assert ack["processing_seconds"] >= 0
            socket.send_json(dict(type="stop"))
    assert runtime.session.closed
