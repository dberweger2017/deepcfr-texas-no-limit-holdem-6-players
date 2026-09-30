"""The calibration changes only the opponent distribution, not the game path."""

import json
import sqlite3
import threading
from dataclasses import FrozenInstanceError
from http.server import ThreadingHTTPServer
from random import Random
from urllib.error import HTTPError

import pytest

from src.blueprint.abstraction import choices
from src.game.hand import Hand, Table
from src.play_api.server import handler_for
from src.play_api.service import MODEL_SHA256, PlayError, PlayService
from src.play_api.uniform_random import DEFINITION_SHA256, UniformRestrictedPolicy
from tests.play_ui.test_service import benchmark_create, bot_action, hand_start, http, human_action
from tests.test_blueprint_hu20 import coupled


def test_uniform_menu_native_legality_and_seed_reproducibility():
    policy = UniformRestrictedPolicy()

    def run():
        rng = Random(31)
        actions = []
        for seed in range(20):
            hand = Hand.start(Table(("a", "b"), (2000, 2000), button=seed % 2),
                              hand_id=f"fixture-{seed}", seed=seed)
            for _ in range(100):
                if hand.finished:
                    break
                view = hand.observe(hand.actor)
                menu, probabilities, telemetry = policy.distribution(view)
                assert menu == choices(view, raise_cap=None, free_fold=False)
                assert probabilities == (1 / len(menu),) * len(menu) and telemetry is None
                action = rng.choices(menu, weights=probabilities, k=1)[0].action
                view.legal_actions.validate(action)
                actions.append(action)
                hand = hand.apply(action)
            assert hand.finished and sum(p.stack for p in hand.observe(0).players) == 4000
        return actions

    assert run() == run()
    with pytest.raises((FrozenInstanceError, TypeError)):
        policy.name = "changed"


def test_distribution_ignores_hidden_opponent_cards():
    policy = UniformRestrictedPolicy()
    a = coupled(0, ("Kc", "Kd")).observe(0)
    b = coupled(0, ("Qc", "Qd")).observe(0)
    assert a == b and policy.distribution(a) == policy.distribution(b)


def test_identity_and_server_side_benchmark_restriction(tmp_path):
    service = PlayService(tmp_path / "control.sqlite", UniformRestrictedPolicy(), source_version="fixture")
    try:
        for body in ({"playMode": "restricted", "visibility": "benchmark"},
                     {"sessionType": "benchmark", "playMode": "free", "targetHands": 100}):
            with pytest.raises(PlayError, match="restricted benchmark"):
                service.create("invalid-control-key-001", body)
        assert service.db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
        state = benchmark_create(service, target=100, mode="restricted")
        assert state["model"]["name"] == "Uniform restricted random"
        assert state["model"]["sha256"] == DEFINITION_SHA256 != MODEL_SHA256
        assert state["model"]["adapter"] == "uniform-restricted-v1"
        assert state["model"]["strategy"] == "uniform-restricted"
        assert state["model"]["benchmarkOnly"] is True
    finally:
        service.close()


def test_restart_next_bot_action_and_lost_response_are_equivalent(tmp_path):
    service = PlayService(tmp_path / "original.sqlite", UniformRestrictedPolicy())
    restarted = None
    try:
        state = hand_start(service, benchmark_create(service, target=2, mode="restricted"))
        state = human_action(service, state, "call")
        snapshot = sqlite3.connect(tmp_path / "restarted.sqlite")
        service.db.backup(snapshot)
        snapshot.close()
        restarted = PlayService(tmp_path / "restarted.sqlite", UniformRestrictedPolicy())
        original = bot_action(service, state)
        recovered = bot_action(restarted, state)
        assert original == recovered
        assert bot_action(restarted, state) == recovered
        assert service._load(state["sessionId"])["botRng"] == restarted._load(state["sessionId"])["botRng"]
        assert service._load(state["sessionId"])["current"]["lookup"] == []
    finally:
        if restarted:
            restarted.close()
        service.close()


def test_actual_http_visibility_settlement_export_and_replay(tmp_path, monkeypatch):
    service = PlayService(tmp_path / "http.sqlite", UniformRestrictedPolicy())
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(service, "fixture-access-token", 0))
    port = server.server_address[1]
    server.RequestHandlerClass = handler_for(service, "fixture-access-token", port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        state = http(base, "/api/sessions", body={"sessionType": "benchmark", "playMode": "restricted", "targetHands": 1}, key="random-http-create-001")
        sid = state["sessionId"]
        state = http(base, f"/api/sessions/{sid}/hands", body={"revision": state["revision"]}, key="random-http-deal-0001")
        monkeypatch.setattr("src.play_api.service._hand", lambda row: coupled(0, ("Kc", "Kd")))
        a = http(base, f"/api/sessions/{sid}")
        monkeypatch.setattr("src.play_api.service._hand", lambda row: coupled(0, ("Qc", "Qd")))
        b = http(base, f"/api/sessions/{sid}")
        assert a == b
        monkeypatch.undo()
        state = http(base, f"/api/sessions/{sid}/actions", body={"handId": state["hand"]["id"], "revision": state["revision"], "kind": "fold", "raiseTo": None}, key="random-http-fold-0001")
        assert state["benchmark"]["status"] == "COMPLETE"
        history = http(base, f"/api/sessions/{sid}/history")
        report = http(base, f"/api/sessions/{sid}/benchmark/export")
        assert report["model"]["sha256"] == DEFINITION_SHA256 and report["adapter"] == "uniform-restricted-v1"
        assert report["completedHands"] == 1 and service.verify_replay(sid) == 1
        assert history["hands"][0]["shownBotCards"] == []
        assert not any(key in json.dumps([state, history, report]) for key in ("dealSeed", "botRng", "dealRng", "probabilities", "lookup"))
        with pytest.raises(HTTPError) as denied:
            http(base, f"/api/sessions/{sid}/hands/{state['hand']['id']}/diagnostics")
        assert denied.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        service.close()
