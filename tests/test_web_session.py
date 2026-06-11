"""Tests for the web poker session layer."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.web.app import REPO_ROOT, create_app
from src.utils.checkpoints import find_checkpoint_dirs
from src.web.session import PokerWebSession, SessionStore


@pytest.fixture
def client():
    return TestClient(create_app())


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_dirs(client):
    response = client.get("/api/model-dirs")
    assert response.status_code == 200
    assert "model_dirs" in response.json()


def test_session_lifecycle(client):
    create = client.post(
        "/api/sessions",
        json={"models_dir": "models/standard/phase1", "hero_seat": 0},
    )
    assert create.status_code == 200
    session_id = create.json()["session_id"]

    hand = client.post(f"/api/sessions/{session_id}/new-hand")
    assert hand.status_code == 200
    payload = hand.json()
    assert payload["hand_number"] == 1
    assert payload["has_active_hand"] is True
    assert len(payload["players"]) == 6
    assert len(payload["players"][0]["cards"]) == 2

    if payload["is_hero_turn"]:
        action = client.post(
            f"/api/sessions/{session_id}/action",
            json={"action": "fold"},
        )
        assert action.status_code == 200
        assert action.json()["final_state"] is True


def test_find_checkpoint_dirs():
    dirs = find_checkpoint_dirs(REPO_ROOT / "models", relative_to=REPO_ROOT)
    assert isinstance(dirs, list)


def test_serialize_without_hand():
    session = PokerWebSession(session_id="test", models_dir=None, hero_seat=0, device="cpu")
    payload = session.serialize()
    assert payload["has_active_hand"] is False
    assert payload["load_warnings"] == []


def test_serialize_includes_load_warnings(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "broken.pt").write_text("not a checkpoint")

    session = PokerWebSession(
        session_id="test",
        models_dir=str(models_dir),
        hero_seat=0,
        device="cpu",
    )
    payload = session.serialize()

    assert len(payload["load_warnings"]) >= 1
    assert payload["load_warnings"][0]["checkpoint"] == "broken.pt"
