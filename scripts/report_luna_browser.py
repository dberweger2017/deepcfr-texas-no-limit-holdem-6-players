"""Reconcile a completed browser experiment; publish no private journal data."""

import argparse
import csv
import json
import sqlite3
import statistics
import urllib.request
from pathlib import Path

from scripts.audit_luna_browser import audit, read
from src.arena.runner import public_events
from src.arena.schedule import digest
from src.game.types import Action, ActionKind
from src.play_api.service import _hand


def reconcile(state, metadata):
    if state["benchmark"]["status"] != "COMPLETE":
        raise ValueError("Only completed sessions may be analyzed")
    attempts = [m for m in metadata if m["type"] == "luna_attempt" and m["decisionOrdinal"] > 0]
    observations = {(m["handOrdinal"], m["decisionOrdinal"]): m for m in metadata
                    if m["type"] == "luna_observed"}
    expected = []
    hands = []
    for ordinal, record in enumerate(state["history"], 1):
        final = _hand(record)
        assert final.finished and sum(p.stack for p in final.observe(0).players) == 4000
        assert final.observe(0).players[0].stack - 2000 == record["humanChips"]
        assert digest(public_events(final.events)) == record["publicEventsSha256"]
        assert record["button"] == (ordinal - 1) % 2
        hand = _hand(dict(record, actions=[]))
        decision = 0
        for action in record["actions"]:
            if action["seat"] == 0:
                decision += 1
                expected.append((ordinal, decision, record["handId"], hand.observe(0).street.value, action))
            hand = hand.apply(Action(ActionKind(action["kind"]), action["raiseTo"]))
        hands.append({"handOrdinal": ordinal, "handId": record["handId"],
                      "button": record["button"], "humanChips": record["humanChips"],
                      "publicEventsSha256": record["publicEventsSha256"]})
    if len(attempts) != len(expected):
        raise ValueError(f"Attempt/action count mismatch: {len(attempts)} / {len(expected)}")
    rows = []
    for attempt, (ordinal, decision, hand_id, street, action) in zip(attempts, expected):
        assert (attempt["handOrdinal"], attempt["decisionOrdinal"]) == (ordinal, decision)
        observed = observations[(ordinal, decision)]
        label = attempt["attemptedButtonLabel"]
        assert label in attempt["legalButtonLabels"]
        if action["kind"] == "raise":
            # Restricted buttons display the exact target after the separator.
            target = label.split("·")[-1].strip().removesuffix(" BB")
            assert float(target) * 100 == action["raiseTo"]
        else:
            assert label.lower().startswith(action["kind"])
        rows.append({"benchmarkId": state["benchmark"]["id"], "handId": hand_id,
                     "handOrdinal": ordinal, "decisionOrdinal": decision, "street": street,
                     "legalButtonLabels": json.dumps(attempt["legalButtonLabels"], ensure_ascii=False),
                     "attemptedButtonLabel": label, "acceptedKind": action["kind"],
                     "acceptedRaiseTo": action["raiseTo"], "attemptedAtMs": attempt["attemptedAtMs"],
                     "observedAtMs": observed["observedAtMs"],
                     "uiConfirmationMs": observed["observedAtMs"] - attempt["attemptedAtMs"],
                     "visibleError": observed.get("visibleError"),
                     "toolRetry": observed.get("toolRetry", ""), "parentIntervention": False})
    assert len(hands) == state["benchmark"]["targetHands"]
    assert sum(h["humanChips"] for h in hands) == state["totalChips"]
    return rows, hands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("rollout", type=Path)
    parser.add_argument("token_file", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with sqlite3.connect(args.database) as connection:
        states = [json.loads(raw) for (raw,) in connection.execute("SELECT state FROM sessions")]
    if len(states) != 1:
        raise ValueError("Expected one isolated experiment session")
    state = states[0]
    result = audit(read(args.rollout))
    if result["violations"] or result["configurations"] != [{"model": "gpt-6-luna", "effort": "high"}]:
        raise ValueError("Invalid player configuration or tool boundary")
    decisions, hands = reconcile(state, result["decisionMetadata"])
    request = urllib.request.Request(
        f'http://127.0.0.1:8765/api/sessions/{state["sessionId"]}/benchmark/export',
        headers={"X-Play-Token": args.token_file.read_text().strip()})
    with urllib.request.urlopen(request) as response:
        export = json.load(response)
    assert export["netChips"] == state["totalChips"] and export["completedHands"] == len(hands)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("decisions", decisions), ("hands", hands)):
        with (args.output / f"{name}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "export.json").write_text(json.dumps(export, indent=2) + "\n")
    latencies = sorted(row["uiConfirmationMs"] for row in decisions)
    lookups = [item for record in state["history"] for item in record["lookup"]]
    summary = {"configurations": result["configurations"], "toolCounts": result["toolCounts"],
               "violations": result["violations"], "verifiedHands": len(hands),
               "reconciledHumanDecisions": len(decisions), "parentPokerInterventions": 0,
               "trainedBotLookups": sum(item["trained"] for item in lookups),
               "fallbackBotLookups": sum(not item["trained"] for item in lookups),
               "uiConfirmationMs": {"mean": statistics.mean(latencies),
                                    "median": statistics.median(latencies),
                                    "p95": latencies[min(len(latencies)-1, int(len(latencies)*.95))]},
               "latencyScope": "Action attempt to observed UI acknowledgment; excludes poker reasoning",
               "usageTotals": result["usageRecords"][-1].get("thread_token_usage") if result["usageRecords"] else None}
    (args.output / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
