"""Focused native play, persistence, and browser-boundary checks."""

import json
import sqlite3
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA, choices
from src.blueprint.solver import HU20_UNCAPPED_GAME
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind
from src.play_api.server import handler_for
from src.play_api.service import PlayError, PlayService, _hand
from tests.test_blueprint_hu20 import coupled


class FixturePolicy:
    def __init__(self):
        self.spec = type("Spec", (), {"sha256": "a" * 64})()
        self.game = HU20_UNCAPPED_GAME
        self.abstraction = HU20_UNCAPPED_SCHEMA
        self.description = {"strategy": "current"}
        self.calls = 0

    def distribution(self, view):
        self.calls += 1
        menu = choices(view, raise_cap=None, free_fold=False)
        return menu, (1 / len(menu),) * len(menu), self.calls % 2 == 0


@pytest.fixture
def service(tmp_path):
    instance = PlayService(tmp_path / "private" / "play.sqlite", FixturePolicy(), source_version="test")
    yield instance
    instance.close()


def create(service, mode="free", visibility="developer"):
    return service.create("create-key-0000001", {"playMode": mode, "visibility": visibility})


def hand_start(service, state):
    return service.new_hand(state["sessionId"], "new-hand-key-00001", {"revision": state["revision"]})


def human_action(service, state, kind, amount=None, key="human-action-key-00001"):
    return service.act(state["sessionId"], key,
                       {"handId": state["hand"]["id"], "revision": state["revision"],
                        "kind": kind, "raiseTo": amount})


def bot_action(service, state, key="bot-action-key-000001"):
    return service.advance(state["sessionId"], key,
                           {"handId": state["hand"]["id"], "revision": state["revision"]})


def test_exact_free_wager_replay_and_reset(service):
    state = hand_start(service, create(service))
    assert state["hand"]["button"] == 0
    assert state["hand"]["legal"]["minRaiseTo"] == 200
    assert 250 not in [item["raiseTo"] for item in state["hand"]["menu"]]
    state = human_action(service, state, "raise", 250)
    private = service._load(state["sessionId"])["current"]
    assert private["actions"][0]["raiseTo"] == 250
    assert _hand(private).events[-1].seat == 1
    state = bot_action(service, state)
    for step in range(100):
        if state["phase"] == "finished":
            break
        if state["hand"]["actor"] == 1:
            state = bot_action(service, state, f"bot-step-{step:016d}")
        else:
            legal = state["hand"]["legal"]
            kind = "check" if "check" in legal["kinds"] else "call"
            state = human_action(service, state, kind, key=f"human-step-{step:014d}")
    assert state["phase"] == "finished"
    assert service.verify_replay(state["sessionId"]) == 1
    assert state["sessionChips"] == state["hand"]["result"]["humanChips"]
    assert len(service.history(state["sessionId"])["hands"]) == 1
    next_state = service.new_hand(state["sessionId"], "next-hand-key-00001", {"revision": state["revision"]})
    assert next_state["hand"]["button"] == 1
    assert [p["stack"] for p in next_state["hand"]["players"]] == [1900, 1950]


def test_restricted_exact_actions_and_invalid_requests_do_not_mutate(service):
    state = hand_start(service, create(service, "restricted"))
    assert state["hand"]["menu"] and state["hand"]["presets"] == []
    with pytest.raises(PlayError, match="restricted menu"):
        human_action(service, state, "raise", 250)
    with pytest.raises(PlayError, match="exact integer"):
        human_action(service, state, "raise", 250.0)
    with pytest.raises(PlayError, match="legal"):
        human_action(service, state, "raise", 199)
    assert service.state(state["sessionId"]) == state
    allowed = next(item for item in state["hand"]["menu"] if item["kind"] == "raise")
    advanced = human_action(service, state, "raise", allowed["raiseTo"])
    with pytest.raises(PlayError, match="Stale"):
        human_action(service, state, "call", key="stale-key-00000001")
    assert service.state(state["sessionId"]) == advanced


def test_idempotent_commit_lost_reply_and_restart_rng_equivalence(service, tmp_path):
    state = hand_start(service, create(service))
    request = {"handId": state["hand"]["id"], "revision": state["revision"],
               "kind": "call", "raiseTo": None}
    first = service.act(state["sessionId"], "lost-reply-key-0001", request)
    assert service.act(state["sessionId"], "lost-reply-key-0001", request) == first
    with pytest.raises(PlayError, match="conflicts"):
        service.act(state["sessionId"], "lost-reply-key-0001", dict(request, kind="fold"))
    assert service.state(state["sessionId"])["revision"] == first["revision"]
    # Snapshot the committed database into another process's fresh connection.
    clone = tmp_path / "clone.sqlite"
    target = sqlite3.connect(clone)
    service.db.backup(target)
    target.close()
    restarted = PlayService(clone, FixturePolicy(), source_version="test")
    try:
        # Bot acts after the human call; both services must sample the same action.
        original = bot_action(service, first)
        recovered = bot_action(restarted, first)
        assert original == recovered
        assert service._load(state["sessionId"])["botRng"] == restarted._load(state["sessionId"])["botRng"]
        assert bot_action(restarted, first) == recovered
    finally:
        restarted.close()


def test_benchmark_diagnostics_denied_and_public_history_is_sanitized(service):
    state = hand_start(service, create(service, visibility="benchmark"))
    state = human_action(service, state, "fold")
    with pytest.raises(PlayError) as denied:
        service.diagnostics(state["sessionId"], state["hand"]["id"])
    assert denied.value.status == 403
    public = json.dumps({"state": state, "history": service.history(state["sessionId"])})
    private = service._load(state["sessionId"])["current"]
    assert str(private["dealSeed"]) not in public
    assert not any(name in public for name in ("dealSeed", "botRng", "dealRng", "probabilities", "information_key"))
    assert state["hand"]["players"][1]["shownCards"] == []


def test_server_presets_use_native_observation_and_short_all_in(service):
    hand = Hand.start(Table(("human", "trained"), (2000, 2000)), hand_id="preset", seed=7)
    first = service._presets(hand.observe(0), False)
    assert [(p["label"], p["raiseTo"], p["available"]) for p in first[:2]] == [
        ("⅓ pot", 167, False), ("½ pot", 200, True)]
    hand = hand.apply(Action(ActionKind.CALL))
    hand = hand.apply(Action(ActionKind.RAISE, 300))
    view = hand.observe(0)
    assert view.players[0].street_bet == 100 and view.legal_actions.call_amount == 200
    assert next(p for p in service._presets(view, False) if p["label"] == "½ pot")["raiseTo"] == 600
    short = Hand.start(Table(("human", "trained"), (150, 2000)), hand_id="short", seed=7)
    bounds = short.observe(0).legal_actions
    assert bounds.min_raise_to == bounds.max_raise_to == 150
    assert next(p for p in service._presets(short.observe(0), False) if p["label"] == "All-in")["raiseTo"] == 150
    no_raise = Hand.start(Table(("human", "trained"), (100, 2000)), hand_id="no-raise", seed=7)
    assert no_raise.observe(0).legal_actions.call_amount == 50
    assert service._presets(no_raise.observe(0), False) == []


@pytest.fixture
def http_server(service):
    server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHandler)
    port = server.server_address[1]
    server.RequestHandlerClass = handler_for(service, "fixture-access-token", port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


class BaseHandler:
    pass


def http(base, path, *, body=None, token="fixture-access-token", key=None, origin=None):
    headers = {"X-Play-Token": token}
    if body is not None:
        headers["Content-Type"] = "application/json"
        headers["Idempotency-Key"] = key
    if origin is not None:
        headers["Origin"] = origin
    request = Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                      headers=headers, method="POST" if body is not None else "GET")
    with urlopen(request) as response:
        return json.loads(response.read())


def test_http_payload_history_access_and_asset_boundary(http_server):
    base = http_server
    state = http(base, "/api/sessions", body={"playMode": "free", "visibility": "developer"}, key="http-create-key-001", origin=base)
    state = http(base, f"/api/sessions/{state['sessionId']}/hands", body={"revision": 0}, key="http-hand-key-0001", origin=base)
    private_keys = ("dealSeed", "botRng", "dealRng", "probabilities", "entries")
    assert not any(key in json.dumps(state) for key in private_keys)
    assert state["hand"]["players"][1]["shownCards"] == []
    state = http(base, f"/api/sessions/{state['sessionId']}/actions", body={"handId": state["hand"]["id"], "revision": state["revision"], "kind": "fold", "raiseTo": None}, key="http-action-key-01", origin=base)
    history = http(base, f"/api/sessions/{state['sessionId']}/history")
    assert len(history["hands"]) == 1
    assert not any(key in json.dumps(history) for key in private_keys)
    assert history["hands"][0]["shownBotCards"] == []
    for path in ("/../results/private.sqlite", "/src/play_api/service.py", "/api/sessions/../../results"):
        with pytest.raises(HTTPError) as denied:
            http(base, path)
        assert denied.value.code in (403, 404)
    with pytest.raises(HTTPError) as denied:
        http(base, f"/api/sessions/{state['sessionId']}", token="wrong")
    assert denied.value.code == 403


def test_http_hidden_world_invariance_and_benchmark_gate(service, http_server, monkeypatch):
    base = http_server
    state = http(base, "/api/sessions", body={"playMode": "free", "visibility": "benchmark"}, key="hidden-create-key-1", origin=base)
    state = http(base, f"/api/sessions/{state['sessionId']}/hands", body={"revision": 0}, key="hidden-hand-key-01", origin=base)
    monkeypatch.setattr("src.play_api.service._hand", lambda row: coupled(0, ("Kc", "Kd")))
    a = http(base, f"/api/sessions/{state['sessionId']}")
    monkeypatch.setattr("src.play_api.service._hand", lambda row: coupled(0, ("Qc", "Qd")))
    b = http(base, f"/api/sessions/{state['sessionId']}")
    assert a == b
    monkeypatch.undo()
    state = http(base, f"/api/sessions/{state['sessionId']}/actions", body={
        "handId": state["hand"]["id"], "revision": state["revision"], "kind": "fold", "raiseTo": None},
        key="hidden-fold-key-01", origin=base)
    with pytest.raises(HTTPError) as denied:
        http(base, f"/api/sessions/{state['sessionId']}/hands/{state['hand']['id']}/diagnostics")
    assert denied.value.code == 403


def test_http_rejects_bad_origin_and_content_type(http_server):
    base = http_server
    with pytest.raises(HTTPError) as denied:
        http(base, "/api/sessions", body={"playMode": "free", "visibility": "developer"},
             key="bad-origin-key-001", origin="http://evil.example")
    assert denied.value.code == 403
    request = Request(base + "/api/sessions", data=b"{}", method="POST", headers={
        "X-Play-Token": "fixture-access-token", "Idempotency-Key": "bad-type-key-0001",
        "Content-Type": "text/plain", "Origin": base})
    with pytest.raises(HTTPError) as denied:
        urlopen(request)
    assert denied.value.code == 415


def benchmark_create(service, target=2, mode="free"):
    return service.create("benchmark-create-key-001", {
        "sessionType": "benchmark", "playMode": mode, "targetHands": target})


def finish_benchmark_hand(service, state, prefix):
    for step in range(100):
        if state["phase"] in ("finished", "complete"):
            return state
        hand = state["hand"]
        if hand["actor"] == 1:
            state = service.advance(state["sessionId"], f"{prefix}-bot-{step:016d}", {
                "handId": hand["id"], "revision": state["revision"]})
        else:
            kinds = hand["legal"]["kinds"]
            action = "fold" if "fold" in kinds else "check" if "check" in kinds else "call"
            state = service.act(state["sessionId"], f"{prefix}-human-{step:014d}", {
                "handId": hand["id"], "revision": state["revision"],
                "kind": action, "raiseTo": None})
    raise AssertionError("Fixture hand did not settle")


def test_benchmark_freezes_target_identity_and_active_results(service):
    state = benchmark_create(service, 200)
    assert state["benchmark"]["targetHands"] == 200
    assert state["benchmark"]["status"] == "ACTIVE"
    assert state["visibility"] == "benchmark"
    assert "sessionChips" not in state and "benchmarkResult" not in state
    assert service.create("benchmark-create-key-001", {
        "sessionType": "benchmark", "playMode": "free", "targetHands": 200}) == state
    with pytest.raises(PlayError, match="conflicts"):
        service.create("benchmark-create-key-001", {
            "sessionType": "benchmark", "playMode": "restricted", "targetHands": 200})
    with pytest.raises(PlayError, match="conflicts"):
        service.create("benchmark-create-key-001", {
            "sessionType": "benchmark", "playMode": "free", "targetHands": 500})
    with pytest.raises(PlayError) as unavailable:
        service.benchmark_result(state["sessionId"])
    assert unavailable.value.status == 409
    with pytest.raises(PlayError, match="Invalid request fields"):
        service.new_hand(state["sessionId"], "benchmark-change-target-001", {
            "revision": state["revision"], "targetHands": 50})
    assert service.state(state["sessionId"]) == state


def test_benchmark_exact_target_position_restart_and_report(service, tmp_path):
    state = benchmark_create(service)
    session = state["sessionId"]
    first_request = {"revision": state["revision"]}
    state = service.new_hand(session, "benchmark-deal-key-001", first_request)
    assert service.new_hand(session, "benchmark-deal-key-001", first_request) == state
    assert state["hand"]["button"] == 0
    state = finish_benchmark_hand(service, state, "benchmark-first")
    assert state["benchmark"]["completedHands"] == 1
    assert "sessionChips" not in state
    first_history = service.history(session)
    assert "humanChips" not in first_history["hands"][0]
    clone = tmp_path / "benchmark-restart.sqlite"
    target = sqlite3.connect(clone)
    service.db.backup(target)
    target.close()
    restarted = PlayService(clone, FixturePolicy(), source_version="test")
    try:
        assert restarted.state(session) == state
        state = restarted.new_hand(session, "benchmark-deal-key-002", {"revision": state["revision"]})
        assert state["hand"]["button"] == 1
        assert state["benchmark"]["completedHands"] == 1
        state = finish_benchmark_hand(restarted, state, "benchmark-second")
        assert state["phase"] == "complete"
        assert state["benchmark"]["completedHands"] == 2
        report = restarted.benchmark_result(session)
        assert report == state["benchmarkResult"]
        assert report["status"] == "COMPLETE" and report["targetHands"] == 2
        assert report["buttonSB"]["hands"] == report["bigBlind"]["hands"] == 1
        assert report["netChips"] == sum(row["humanChips"] for row in restarted._load(session)["history"])
        assert report["bbPer100"] == report["netChips"] / 2
        assert report["wins"] + report["losses"] + report["ties"] == 2
        assert restarted.verify_replay(session) == 2
        with pytest.raises(PlayError, match="ended"):
            restarted.new_hand(session, "benchmark-deal-key-003", {"revision": state["revision"]})
        assert restarted.state(session) == state
    finally:
        restarted.close()


def test_benchmark_early_end_keeps_records_and_denies_diagnostics(service):
    state = benchmark_create(service)
    session = state["sessionId"]
    state = service.new_hand(session, "benchmark-deal-key-001", {"revision": state["revision"]})
    state = finish_benchmark_hand(service, state, "benchmark-first")
    state = service.new_hand(session, "benchmark-deal-key-002", {"revision": state["revision"]})
    request = {"revision": state["revision"], "handId": state["hand"]["id"], "confirm": True}
    with pytest.raises(PlayError, match="Confirm"):
        service.end_benchmark(session, "benchmark-end-key-000", dict(request, confirm=False))
    ended = service.end_benchmark(session, "benchmark-end-key-001", request)
    assert service.end_benchmark(session, "benchmark-end-key-001", request) == ended
    assert ended["phase"] == "aborted" and ended["hand"]["legal"] is None
    report = service.benchmark_result(session)
    assert report["status"] == "ABORTED" and report["completedHands"] == 1
    assert service._load(session)["benchmark"]["abortedHandId"] == state["hand"]["id"]
    assert len(service._load(session)["history"]) == 1
    assert service.verify_replay(session) == 1
    with pytest.raises(PlayError) as denied:
        service.diagnostics(session, report["handRecords"][0]["handId"])
    assert denied.value.status == 403
    with pytest.raises(PlayError, match="ended"):
        service.act(session, "benchmark-post-end-001", {
            "handId": state["hand"]["id"], "revision": ended["revision"],
            "kind": "fold", "raiseTo": None})


def test_benchmark_can_end_before_first_hand_without_a_rate(service):
    state = benchmark_create(service, target=50)
    ended = service.end_benchmark(state["sessionId"], "benchmark-empty-end-001", {
        "revision": 0, "handId": None, "confirm": True})
    report = ended["benchmarkResult"]
    assert ended["phase"] == "aborted" and ended["hand"] is None
    assert report["completedHands"] == 0 and report["status"] == "ABORTED"
    assert report["netChips"] == 0 and report["bbPer100"] is None
    assert report["averagePotChips"] is None and report["handRecords"] == []


def test_http_benchmark_export_stays_public_and_cannot_delete(http_server, service):
    base = http_server
    state = http(base, "/api/sessions", body={
        "sessionType": "benchmark", "playMode": "free", "targetHands": 1},
        key="http-benchmark-create-001", origin=base)
    session = state["sessionId"]
    assert http(base, "/api/model")["sha256"] == "a" * 64
    with pytest.raises(HTTPError) as unavailable:
        http(base, f"/api/sessions/{session}/benchmark/export")
    assert unavailable.value.code == 409
    state = http(base, f"/api/sessions/{session}/hands", body={"revision": 0},
                 key="http-benchmark-deal-001", origin=base)
    state = http(base, f"/api/sessions/{session}/actions", body={
        "handId": state["hand"]["id"], "revision": state["revision"],
        "kind": "fold", "raiseTo": None}, key="http-benchmark-action-01", origin=base)
    assert state["phase"] == "complete"
    report = http(base, f"/api/sessions/{session}/benchmark/export")
    assert report == http(base, f"/api/sessions/{session}/benchmark/result")
    assert report["netChips"] == service._load(session)["history"][0]["humanChips"]
    public = json.dumps(report)
    assert not any(field in public for field in (
        "dealSeed", "dealRng", "botRng", "hole_cards", "shownBotCards", "probabilities", "lookup"))
    with pytest.raises(HTTPError) as denied:
        http(base, f"/api/sessions/{session}/hands", body={"revision": state["revision"]},
             key="http-benchmark-deal-002", origin=base)
    assert denied.value.code == 409
    request = Request(base + f"/api/sessions/{session}/history", data=b"{}", method="DELETE",
                      headers={"X-Play-Token": "fixture-access-token"})
    with pytest.raises(HTTPError) as denied:
        urlopen(request)
    assert denied.value.code in (404, 405, 501)
    assert len(service._load(session)["history"]) == 1


def test_http_active_benchmark_withholds_totals_and_diagnostics(http_server):
    base = http_server
    state = http(base, "/api/sessions", body={
        "sessionType": "benchmark", "playMode": "free", "targetHands": 2},
        key="http-active-benchmark-001", origin=base)
    session = state["sessionId"]
    assert "sessionChips" not in state and "sessionBB" not in state
    state = http(base, f"/api/sessions/{session}/hands", body={"revision": 0},
                 key="http-active-deal-0001", origin=base)
    state = http(base, f"/api/sessions/{session}/actions", body={
        "handId": state["hand"]["id"], "revision": state["revision"],
        "kind": "fold", "raiseTo": None}, key="http-active-fold-0001", origin=base)
    assert state["phase"] == "finished" and "sessionChips" not in state
    fetched = http(base, f"/api/sessions/{session}")
    history = http(base, f"/api/sessions/{session}/history")
    assert fetched == state and "humanChips" not in history["hands"][0]
    assert "fallbackSummary" not in json.dumps({"state": fetched, "history": history})
    with pytest.raises(HTTPError) as denied:
        http(base, f"/api/sessions/{session}/hands/{state['hand']['id']}/diagnostics")
    assert denied.value.code == 403


def test_benchmark_result_checks_native_replay_before_reporting(service):
    state = benchmark_create(service, target=1)
    session = state["sessionId"]
    state = service.new_hand(session, "benchmark-deal-key-001", {"revision": state["revision"]})
    state = human_action(service, state, "fold", key="benchmark-fold-key-01")
    assert state["phase"] == "complete"
    original = service.benchmark_result(session)
    assert original["netChips"] == state["benchmarkResult"]["netChips"]
    private = service._load(session)
    private["history"][0]["humanChips"] += 1
    service.db.execute("UPDATE sessions SET state=? WHERE id=?", (json.dumps(private), session))
    service.db.commit()
    with pytest.raises(RuntimeError, match="replay mismatch"):
        service.benchmark_result(session)
