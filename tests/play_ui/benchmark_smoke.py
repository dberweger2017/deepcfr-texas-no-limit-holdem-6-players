"""Bounded real-B100M HTTP smoke for completed human benchmark records on M4."""

import argparse
import json
import sqlite3
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from src.blueprint.abstraction import choices
from src.play_api.service import MODEL_SHA256, _hand


def run(base, token, database, mode, target):
    def call(path, body=None):
        headers = {"X-Play-Token": token}
        if body is not None:
            headers.update({"Content-Type": "application/json", "Origin": base,
                            "Idempotency-Key": str(uuid4())})
        request = Request(base + path,
                          data=json.dumps(body).encode() if body is not None else None,
                          headers=headers, method="POST" if body is not None else "GET")
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read())

    state = call("/api/sessions", {
        "sessionType": "benchmark", "playMode": mode, "targetHands": target})
    session = state["sessionId"]
    assert state["model"]["sha256"] == MODEL_SHA256
    assert state["benchmark"]["targetHands"] == target
    assert "sessionChips" not in state
    off_menu = None
    for hand_number in range(target):
        state = call(f"/api/sessions/{session}/hands", {"revision": state["revision"]})
        assert state["hand"]["button"] == hand_number % 2
        for _ in range(100):
            if state["phase"] in ("finished", "complete"):
                break
            hand = state["hand"]
            if hand["actor"] == 1:
                state = call(f"/api/sessions/{session}/advance", {
                    "handId": hand["id"], "revision": state["revision"]})
                continue
            legal = hand["legal"]
            if mode == "free" and off_menu is None and "raise" in legal["kinds"]:
                private = json.loads(sqlite3.connect(database).execute(
                    "SELECT state FROM sessions WHERE id=?", (session,)).fetchone()[0])
                abstract = {item.action.raise_to for item in choices(
                    _hand(private["current"]).observe(0), raise_cap=None, free_fold=False)
                    if item.action.raise_to is not None}
                amount = next(candidate for candidate in range(
                    legal["minRaiseTo"], legal["maxRaiseTo"] + 1) if candidate not in abstract)
                action = {"kind": "raise", "raiseTo": amount}
                off_menu = amount
            elif mode == "restricted":
                item = next((item for item in hand["menu"] if item["kind"] in ("check", "call")), hand["menu"][0])
                action = {"kind": item["kind"], "raiseTo": item["raiseTo"]}
            else:
                action = {"kind": "check" if "check" in legal["kinds"] else "call", "raiseTo": None}
            state = call(f"/api/sessions/{session}/actions", {
                "handId": hand["id"], "revision": state["revision"], **action})
        else:
            raise AssertionError("Benchmark hand did not settle within 100 turns")
    assert state["phase"] == "complete"
    report = call(f"/api/sessions/{session}/benchmark/export")
    assert report == state["benchmarkResult"]
    assert report["completedHands"] == report["targetHands"] == target
    assert report["status"] == "COMPLETE"
    assert report["netChips"] == sum(row["humanChips"] for row in json.loads(
        sqlite3.connect(database).execute("SELECT state FROM sessions WHERE id=?", (session,)).fetchone()[0])["history"])
    public = json.dumps({"state": state, "report": report, "history": call(f"/api/sessions/{session}/history")})
    assert not any(name in public for name in ("dealSeed", "dealRng", "botRng", "probabilities", "hole_cards"))
    if mode == "free":
        assert off_menu is not None and "fallbackSummary" in report
        private = json.loads(sqlite3.connect(database).execute(
            "SELECT state FROM sessions WHERE id=?", (session,)).fetchone()[0])
        assert private["history"][0]["actions"][0]["raiseTo"] == off_menu
    else:
        assert "fallbackSummary" not in report
    try:
        call(f"/api/sessions/{session}/hands", {"revision": state["revision"]})
    except HTTPError as error:
        assert error.code == 409
    else:
        raise AssertionError("N+1 hand was accepted")
    return {"session": session, "mode": mode, "target": target,
            "netChips": report["netChips"], "offMenuRaiseTo": off_menu}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8767")
    parser.add_argument("--data-dir", type=Path, default=Path("results/play-web"))
    args = parser.parse_args()
    token = (args.data_dir / "access.token").read_text().strip()
    for mode, target in (("restricted", 2), ("free", 1)):
        print(json.dumps(run(args.base, token, args.data_dir / "private.sqlite", mode, target)))
