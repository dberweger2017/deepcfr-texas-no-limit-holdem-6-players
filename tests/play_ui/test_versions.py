"""Model choice, persistence and retries across the two table versions."""

import pytest

from src.play_api.service import PlayError, PlayService
from src.play_api.versions import VersionedTables
from tests.play_ui.test_service import FixturePolicy, hand_start, human_action


def tables(tmp_path):
    services = {}
    for version, digest in (("v0.4.1", "a"), ("v0.4.0", "b")):
        policy = FixturePolicy()
        policy.spec = type("Identity", (), {"sha256": digest * 64})()
        services[version] = PlayService(tmp_path / version / "private.sqlite", policy)
    return VersionedTables(services)


def test_default_selection_restart_and_cross_model_retry(tmp_path):
    service = tables(tmp_path)
    body = {"playMode": "restricted", "visibility": "developer"}
    try:
        assert service.model_catalog()["default"] == "v0.4.1"
        first = service.create("create-default-key-0001", body)
        old = service.create("create-v040-key-000001", {**body, "modelVersion": "v0.4.0"})
        assert first["model"]["sha256"] == "a" * 64
        assert old["model"]["sha256"] == "b" * 64
        assert service.create("create-default-key-0001", body) == first
        with pytest.raises(PlayError, match="another model"):
            service.create("create-default-key-0001", {**body, "modelVersion": "v0.4.0"})
        for version in ("bad", [], None):
            with pytest.raises(PlayError, match="available model"):
                service.create("invalid-version-key-001", {**body, "modelVersion": version})
        for state in (first, old):
            state = service.new_hand(state["sessionId"], f"new-hand-{state['sessionId']}", {"revision": 0})
            state = human_action(service, state, "fold", key=f"fold-{state['sessionId']}")
            assert service.verify_replay(state["sessionId"]) == 1
        with pytest.raises(PlayError) as error:
            service.state("unknown-session")
        assert error.value.status == 404
    finally:
        service.close()
    reopened = tables(tmp_path)
    try:
        assert reopened.state(first["sessionId"])["model"]["sha256"] == "a" * 64
        assert reopened.state(old["sessionId"])["model"]["sha256"] == "b" * 64
        assert reopened.verify_replay(first["sessionId"]) == 1
        assert reopened.verify_replay(old["sessionId"]) == 1
        assert len(reopened.history(old["sessionId"])["hands"]) == 1
    finally:
        reopened.close()


def test_failed_default_never_silently_loads_incumbent(tmp_path):
    from src.play_api.versions import load_tables
    with pytest.raises(ValueError, match="regular file"):
        load_tables(tmp_path, tmp_path / "data", "test")
    assert not (tmp_path / "data").exists()
