"""Replay private local-play records without loading a policy export."""

import argparse
import json
import sqlite3
from pathlib import Path

from src.arena.runner import public_events
from src.arena.schedule import digest
from src.blueprint.abstraction import choices
from src.policies.v040 import EXPECTED_SHA256 as MODEL_SHA256
from src.play_api.service import _hand


def verify(database, expected_hash=MODEL_SHA256):
    connection = sqlite3.connect(database)
    counts = {"sessions": 0, "hands": 0, "freeOffMenu": 0}
    for (raw,) in connection.execute("SELECT state FROM sessions"):
        state = json.loads(raw)
        counts["sessions"] += 1
        assert state["modelSha256"] == expected_hash
        for record in state["history"]:
            hand = _hand(record)
            assert hand.finished
            assert digest(public_events(hand.events)) == record["publicEventsSha256"]
            assert hand.observe(0).players[0].stack - 2000 == record["humanChips"]
            assert sum(player.stack for player in hand.observe(0).players) == 4000
            counts["hands"] += 1
            if state["playMode"] == "free":
                initial = _hand(dict(record, actions=[]))
                abstract = {item.action for item in choices(initial.observe(0), raise_cap=None, free_fold=False)}
                for action in record["actions"]:
                    if action["seat"] == 0 and action["kind"] == "raise":
                        from src.game.types import Action, ActionKind
                        if Action(ActionKind.RAISE, action["raiseTo"]) not in abstract:
                            counts["freeOffMenu"] += 1
                        break
    connection.close()
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--expected-hash", default=MODEL_SHA256)
    args = parser.parse_args()
    print(json.dumps(verify(args.database, args.expected_hash)))
