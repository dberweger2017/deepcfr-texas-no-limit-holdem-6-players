"""Shield identity, unchanged arena probabilities and persisted browser play."""

import gzip
import json

import pytest

from src.diagnostics.cfr_average import DiagnosticAverage, extract
from src.diagnostics.saved_hu20 import file_hash
from src.play_api import shield
from src.play_api.service import PlayError, PlayService
from tests.diagnostics.test_cfr_average import fixture
from tests.play_ui.test_service import human_action, bot_action, hand_start


def shield_fixture(tmp_path, monkeypatch):
    _, view, checkpoint, _, spec = fixture(tmp_path)
    rows = gzip.decompress(checkpoint.read_bytes()).decode().splitlines()
    header = json.loads(rows[0])
    header["training_options"] = "regret-floor-0"
    rows[0] = json.dumps(header)
    checkpoint.write_bytes(gzip.compress(("\n".join(rows) + "\n").encode()))
    spec["checkpoint_sha256"] = file_hash(checkpoint)
    output = tmp_path / "shield.gz"
    result = extract(checkpoint, spec, output)
    for name, value in {"MODEL_SHA256": result["sha256"], "MODEL_BYTES": output.stat().st_size,
                        "CHECKPOINT_SHA256": spec["checkpoint_sha256"],
                        "MODEL_SEED": 7, "MODEL_ITERATION": 10}.items():
        monkeypatch.setattr(shield, name, value)
    return output, view


def test_shield_keeps_exact_arena_distribution_and_model_identity(tmp_path, monkeypatch):
    path, view = shield_fixture(tmp_path, monkeypatch)
    policy = shield.ShieldPolicy(path)
    arena = DiagnosticAverage(path, file_hash(path))
    assert policy.distribution(view) == arena.distribution(view)
    assert policy.description == arena.description
    assert policy.spec.sha256 == file_hash(path)
    service = PlayService(tmp_path / "play.sqlite", policy)
    try:
        model = service.model_info()
        assert model["name"].startswith("0.4.0-shield")
        assert model["sha256"] == file_hash(path)
        assert model["adapter"] == "shield-traverser-average-v1"
        assert model["benchmarkOnly"] is True
        with pytest.raises(PlayError, match="restricted benchmark"):
            service.create("casual-create-key-1", {"playMode": "restricted", "visibility": "developer"})
        with pytest.raises(PlayError, match="restricted benchmark"):
            service.create("free-create-key-001", {"sessionType": "benchmark", "playMode": "free", "targetHands": 2})
    finally:
        service.close()


@pytest.mark.parametrize("field,value", [("training_options", "production"),
                                        ("average_rule", "opponent-sampled"),
                                        ("iteration", 11)])
def test_shield_rejects_other_training_and_extraction(tmp_path, monkeypatch, field, value):
    path, _ = shield_fixture(tmp_path, monkeypatch)
    rows = gzip.decompress(path.read_bytes()).decode().splitlines()
    metadata = json.loads(rows[0])
    metadata["checkpoint_header"][field] = value
    rows[0] = json.dumps(metadata)
    path.write_bytes(gzip.compress(("\n".join(rows) + "\n").encode()))
    monkeypatch.setattr(shield, "MODEL_BYTES", path.stat().st_size)
    monkeypatch.setattr(shield, "MODEL_SHA256", file_hash(path))
    with pytest.raises(ValueError, match="pinned Shield"):
        shield.ShieldPolicy(path)


def test_shield_rejects_wrong_bytes_even_with_valid_header(tmp_path, monkeypatch):
    path, _ = shield_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(shield, "MODEL_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="hash"):
        shield.ShieldPolicy(path)
    monkeypatch.setattr(shield, "MODEL_BYTES", 1)
    with pytest.raises(ValueError, match="size"):
        shield.ShieldPolicy(path)


def test_shield_session_reopens_and_replays_without_extra_hands(tmp_path, monkeypatch):
    path, _ = shield_fixture(tmp_path, monkeypatch)
    policy = shield.ShieldPolicy(path)
    database = tmp_path / "play.sqlite"
    service = PlayService(database, policy, source_version="fixture-source")
    state = service.create("benchmark-create-001", {"sessionType": "benchmark", "playMode": "restricted", "targetHands": 2})
    state = hand_start(service, state)
    state = human_action(service, state, "fold")
    assert "sessionChips" not in state
    session_id = state["sessionId"]
    service.close()
    service = PlayService(database, policy, source_version="fixture-source")
    try:
        assert service.state(session_id) == state
        state = service.new_hand(session_id, "second-hand-key-001", {"revision": state["revision"]})
        for step in range(100):
            if state["phase"] == "complete":
                break
            if state["hand"]["actor"] == 1:
                state = bot_action(service, state, key=f"bot-step-{step:016d}")
            else:
                legal = state["hand"]["legal"]["kinds"]
                state = human_action(service, state, "check" if "check" in legal else "fold", key=f"human-step-{step:014d}")
        assert state["phase"] == "complete"
        assert service.verify_replay(session_id) == 2
        assert service.benchmark_result(session_id)["model"]["sha256"] == file_hash(path)
        with pytest.raises(PlayError):
            service.new_hand(session_id, "forbidden-hand-key-1", {"revision": state["revision"]})
    finally:
        service.close()
