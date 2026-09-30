"""Reconcile a completed browser experiment; publish no private journal data."""

import argparse
import csv
import json
import sqlite3
import statistics
import math
from decimal import Decimal
import urllib.request
from pathlib import Path

from scripts.audit_luna_browser import audit, read
from src.blueprint.abstraction import choices
from src.arena.runner import public_events
from src.arena.schedule import digest
from src.game.types import Action, ActionKind
from src.play_api.service import _hand


def reconcile(state, metadata):
    if state["benchmark"]["status"] != "COMPLETE":
        raise ValueError("Only completed sessions may be analyzed")
    attempts = [m for m in metadata if m["type"] == "luna_attempt"
                and not m["attemptedButtonLabel"].lower().startswith("deal hand")]
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
                view = hand.observe(0)
                expected.append((ordinal, decision, record["handId"], view, action))
            hand = hand.apply(Action(ActionKind(action["kind"]), action["raiseTo"]))
        hands.append({"handOrdinal": ordinal, "handId": record["handId"],
                      "button": record["button"], "humanChips": record["humanChips"],
                      "publicEventsSha256": record["publicEventsSha256"]})
    if len(attempts) != len(expected):
        raise ValueError(f"Attempt/action count mismatch: {len(attempts)} / {len(expected)}")
    rows = []
    for attempt, (ordinal, decision, hand_id, view, action) in zip(attempts, expected):
        assert attempt["handOrdinal"] == ordinal
        observed = observations[(ordinal, attempt["decisionOrdinal"])]
        label = attempt["attemptedButtonLabel"]
        matches = False
        if action["kind"] == "raise":
            # Restricted buttons display the exact target after the separator.
            target = label.split("·")[-1].strip().removesuffix(" BB")
            try:
                matches = Decimal(target) * 100 == action["raiseTo"]
            except ArithmeticError:
                matches = False
        else:
            matches = label.lower().startswith(action["kind"])
        rows.append({"benchmarkId": state["benchmark"]["id"], "handId": hand_id,
                     "handOrdinal": ordinal, "decisionOrdinal": decision, "street": view.street.value,
                     "browserDecisionOrdinal": attempt["decisionOrdinal"],
                     "humanVisibleCards": json.dumps(view.hole_cards),
                     "visibleBoard": json.dumps(view.board),
                     "restrictedMenu": json.dumps([
                         {"name": item.name, "kind": item.action.kind.value,
                          "raiseTo": item.action.raise_to}
                         for item in choices(view, raise_cap=None, free_fold=False)]),
                     "legalButtonLabels": json.dumps(attempt["legalButtonLabels"], ensure_ascii=False),
                     "attemptedButtonLabel": label, "acceptedKind": action["kind"],
                     "acceptedRaiseTo": action["raiseTo"], "attemptedAtMs": attempt["attemptedAtMs"],
                     "attemptMatchesAccepted": matches,
                     "reportedMenuContainsAttempt": label in attempt["legalButtonLabels"],
                     "observedAtMs": observed["observedAtMs"],
                     "lastRenderedAtMs": attempt.get("lastRenderedAtMs"),
                     "observedDecisionMs": attempt["attemptedAtMs"] - attempt["lastRenderedAtMs"]
                         if attempt.get("lastRenderedAtMs") is not None else None,
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
    parser.add_argument("--export-file", type=Path,
                        help="Previously saved actual HTTP export, for offline analysis")
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
    if args.export_file:
        export = json.loads(args.export_file.read_text())
    else:
        request = urllib.request.Request(
            f'http://127.0.0.1:8765/api/sessions/{state["sessionId"]}/benchmark/export',
            headers={"X-Play-Token": args.token_file.read_text().strip()})
        with urllib.request.urlopen(request) as response:
            export = json.load(response)
    assert export["netChips"] == state["totalChips"] and export["completedHands"] == len(hands)
    assert export["model"]["sha256"] == state["modelSha256"]
    assert export["sourceVersion"] == state["sourceVersion"]
    assert export["targetHands"] == state["benchmark"]["targetHands"]
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("decisions", decisions), ("hands", hands)):
        with (args.output / f"{name}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "export.json").write_text(json.dumps(export, indent=2) + "\n")
    latencies = sorted(row["uiConfirmationMs"] for row in decisions)
    decision_latencies = sorted(row["observedDecisionMs"] for row in decisions
                                if row["observedDecisionMs"] is not None)
    lookups = [item for record in state["history"] for item in record["lookup"]]
    summary = {"configurations": result["configurations"], "toolCounts": result["toolCounts"],
               "violations": result["violations"], "verifiedHands": len(hands),
               "setupFailures": result["setupFailures"],
               "reconciledHumanDecisions": len(decisions), "parentPokerInterventions": 0,
               "attemptAcceptedMismatches": sum(not r["attemptMatchesAccepted"] for r in decisions),
               "reportedMenuMismatches": sum(not r["reportedMenuContainsAttempt"] for r in decisions),
               "trainedBotLookups": sum(item["trained"] for item in lookups),
               "fallbackBotLookups": sum(not item["trained"] for item in lookups),
               "uiConfirmationMs": {"mean": statistics.mean(latencies),
                                    "median": statistics.median(latencies),
                                    "p95": latencies[math.ceil(len(latencies)*.95)-1]},
               "latencyScope": "Action attempt to observed UI acknowledgment; excludes poker reasoning",
               "observedDecisionMs": {"mean": statistics.mean(decision_latencies),
                                      "median": statistics.median(decision_latencies),
                                      "p95": decision_latencies[math.ceil(len(decision_latencies)*.95)-1]} if decision_latencies else None,
               "percentileMethod": "nearest rank",
               "decisionLatencyScope": "Last rendered result receipt to action attempt; excludes earlier reads/first-ready wait",
               "usageTotals": result["usageRecords"][-1].get("thread_token_usage") if result["usageRecords"] else None}
    (args.output / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
