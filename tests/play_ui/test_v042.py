"""Fixed 10B loading uses the existing average inference and durable HU20 service."""

import pytest

from src.blueprint.average import AveragePolicy
from src.policies import v041, v042
from src.play_api.service import PlayService
from tests.play_ui.test_v041 import model
from tests.play_ui.test_service import create, hand_start, human_action, bot_action


def pinned_fixture(tmp_path, monkeypatch):
    path, view = model(tmp_path, monkeypatch)
    for name in ("MODEL_BYTES", "MODEL_SHA256", "CHECKPOINT_SHA256", "SEED", "ITERATION"):
        monkeypatch.setattr(v042, name, getattr(v041, name))
    return path, view


def test_same_probabilities_and_durable_play_replay(tmp_path, monkeypatch):
    path, view = pinned_fixture(tmp_path, monkeypatch)
    policy = v042.load_policy(path)
    assert policy.name.startswith("v0.4.2 · ")
    assert policy.distribution(view) == AveragePolicy(path, v042.MODEL_SHA256).distribution(view)
    service = PlayService(tmp_path / "private.sqlite", policy)
    try:
        state = hand_start(service, create(service, "free"))
        state = human_action(service, state, "raise", 201)
        for step in range(100):
            if state["phase"] == "finished":
                break
            if state["hand"]["actor"] == 1:
                state = bot_action(service, state, f"v042-bot-step-{step:016d}")
            else:
                kind = "check" if "check" in state["hand"]["legal"]["kinds"] else "call"
                state = human_action(service, state, kind, key=f"v042-human-step-{step:016d}")
        assert state["phase"] == "finished"
        assert service.verify_replay(state["sessionId"]) == 1
        assert service._load(state["sessionId"])["history"][0]["actions"][0]["raiseTo"] == 201
    finally:
        service.close()
    reopened = PlayService(tmp_path / "private.sqlite", v042.load_policy(path))
    try:
        assert reopened.verify_replay(state["sessionId"]) == 1
        assert reopened.state(state["sessionId"])["model"]["sha256"] == v042.MODEL_SHA256
    finally:
        reopened.close()


def test_wrong_lineage_and_bytes_rejected(tmp_path, monkeypatch):
    path, _ = pinned_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(v042, "SEED", 999)
    with pytest.raises(ValueError, match="lineage"):
        v042.load_policy(path)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="byte count"):
        v042.verify(path)
