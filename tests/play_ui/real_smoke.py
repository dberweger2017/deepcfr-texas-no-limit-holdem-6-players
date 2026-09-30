"""Bounded HTTP smoke against an already running, hash-verified B100M service.

Run explicitly on M4. This is integration evidence, not a strength test.
"""

import argparse
import json
import sqlite3
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4

from src.blueprint.abstraction import choices
from src.play_api.service import _hand


def run(base, token, database, mode):
    def call(path, body=None, *, key=None):
        headers = {"X-Play-Token": token}
        if body is not None:
            headers["Content-Type"] = "application/json"
            headers["Idempotency-Key"] = key or str(uuid4())
            headers["Origin"] = base
        request = Request(base + path,
                          data=json.dumps(body).encode() if body is not None else None,
                          headers=headers, method="POST" if body is not None else "GET")
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read())

    state = call("/api/sessions", {"playMode": mode, "visibility": "developer"})
    session = state["sessionId"]
    state = call(f"/api/sessions/{session}/hands", {"revision": state["revision"]})
    first_hand = state["hand"]["id"]
    off_menu = False
    amount = None
    for step in range(100):
        if state["phase"] == "finished":
            break
        hand = state["hand"]
        if hand["actor"] == 1:
            state = call(f"/api/sessions/{session}/advance",
                         {"handId": hand["id"], "revision": state["revision"]})
            continue
        legal = hand["legal"]
        if mode == "free" and not off_menu and "raise" in legal["kinds"]:
            private_now = json.loads(sqlite3.connect(database).execute(
                "SELECT state FROM sessions WHERE id=?", (session,)).fetchone()[0])
            native_hand = _hand(private_now["current"])
            abstract = {item.action.raise_to for item in choices(
                native_hand.observe(0), raise_cap=None, free_fold=False)
                if item.action.raise_to is not None}
            amount = next((candidate for candidate in range(legal["minRaiseTo"], legal["maxRaiseTo"] + 1)
                           if candidate not in abstract), None)
            if amount is not None:
                action = {"kind": "raise", "raiseTo": amount}
                off_menu = True
            else:
                action = {"kind": "call", "raiseTo": None}
        elif mode == "restricted":
            item = next((item for item in hand["menu"] if item["kind"] in ("check", "call")), hand["menu"][0])
            action = {"kind": item["kind"], "raiseTo": item["raiseTo"]}
        else:
            action = {"kind": "check" if "check" in legal["kinds"] else "call", "raiseTo": None}
        state = call(f"/api/sessions/{session}/actions",
                     {"handId": hand["id"], "revision": state["revision"], **action})
    assert state["phase"] == "finished", "Hand did not finish within 100 turns"
    assert state["model"]["sha256"] == "4534e7db2f69bedd54098b7eaa3c9bd82450838405ae162270a3b7684db9bedf"
    history = call(f"/api/sessions/{session}/history")
    assert len(history["hands"]) == 1 and history["hands"][0]["handId"] == first_hand
    public = json.dumps({"state": state, "history": history})
    assert "dealSeed" not in public and "botRng" not in public and "probabilities" not in public
    private = json.loads(sqlite3.connect(database).execute(
        "SELECT state FROM sessions WHERE id=?", (session,)).fetchone()[0])
    if mode == "free":
        assert off_menu
        assert private["history"][0]["actions"][0]["raiseTo"] == amount
    else:
        assert not off_menu
    return {"session": session, "mode": mode, "hands": 1,
            "humanChips": state["hand"]["result"]["humanChips"],
            "botLookups": len(private["history"][0]["lookup"]),
            "offMenuRaiseTo": amount if mode == "free" else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8767")
    parser.add_argument("--data-dir", type=Path, default=Path("results/play-web"))
    args = parser.parse_args()
    token = (args.data_dir / "access.token").read_text().strip()
    for mode in ("restricted", "free"):
        print(json.dumps(run(args.base, token, args.data_dir / "private.sqlite", mode)))
