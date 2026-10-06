"""Compare completed Luna match records using public events only.

This is a post-hoc description, not a paired strength or adaptation test.
Amounts owed are reconstructed from street contributions and checked against
the earlier run's rendered call labels and both runs' accepted-action CSVs.
"""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.analyse_luna_big_pots import _call_bb

ROOTS = {
    "v040": Path("docs/reports/luna-browser/primary"),
    "shield": Path("docs/reports/luna-shield-chrome"),
}


def decisions_from_public(hands):
    rows = []
    for ordinal, hand in enumerate(hands, 1):
        contributions = [0, 0]
        invested = [0, 0]
        street = "preflop"
        raises = []
        decision = 0
        for event in hand["events"]:
            if event["event"] == "board":
                contributions = [0, 0]
                street = event["street"]
            elif event["event"] == "blind":
                contributions[event["seat"]] += event["amount"]
                invested[event["seat"]] += event["amount"]
            elif event["event"] == "action":
                seat = event["seat"]
                assert event["street"] == street
                owed = min(max(contributions) - contributions[seat], 2000 - invested[seat])
                if seat == 0:
                    decision += 1
                    rows.append({"handOrdinal": ordinal, "decisionOrdinal": decision,
                                 "handId": hand["handId"], "street": street,
                                 "kind": event["kind"], "raiseTo": event["raiseTo"],
                                 "owedBB": owed / 100,
                                 "facingThreeBet": street == "preflop" and hand["button"] == 0
                                     and raises == [0, 1]})
                paid = event["paid"]
                if event["kind"] == "call":
                    assert paid == owed
                elif event["kind"] == "raise":
                    assert paid == event["raiseTo"] - contributions[seat]
                    raises.append(seat)
                elif event["kind"] == "check":
                    assert paid == owed == 0
                else:
                    assert event["kind"] == "fold" and paid == 0
                contributions[seat] += paid
                invested[seat] += paid
                assert 0 <= invested[seat] <= 2000
    return rows


def group(hands):
    chips = [h["humanChips"] for h in hands]
    return {"hands": len(chips), "lunaBB": sum(chips) / 100,
            "bbPer100": sum(chips) / len(chips) if chips else None,
            "wins": sum(c > 0 for c in chips), "losses": sum(c < 0 for c in chips),
            "ties": sum(c == 0 for c in chips)}


def analyse(root, *, old):
    history = json.loads((root / "public-history.json").read_text())["hands"]
    export = json.loads((root / "export.json").read_text())
    rows = decisions_from_public(history)
    csv_path = root / ("decisions.csv" if old else "native-decisions.csv")
    with csv_path.open() as stream:
        accepted = list(csv.DictReader(stream))
    assert len(rows) == len(accepted)
    for public, native in zip(rows, accepted):
        assert public["handId"] == native["handId"]
        assert public["handOrdinal"] == int(native["handOrdinal"])
        assert public["street"] == native["street"]
        assert public["kind"] == native["acceptedKind"]
        assert public["raiseTo"] == (int(native["acceptedRaiseTo"]) if native["acceptedRaiseTo"] else None)
        if old:
            rendered = _call_bb(native["legalButtonLabels"])
            assert rendered == (public["owedBB"] if public["owedBB"] > 0 else None)
    assert len(history) == export["completedHands"] == 400
    assert sum(h["humanChips"] for h in history) == export["netChips"]
    assert group(history)["wins"] == export["wins"]
    assert group(history)["losses"] == export["losses"]
    assert group(history)["ties"] == export["ties"]
    thresholds = {}
    for threshold in (5, 8, 12):
        first = {}
        for row in rows:
            if row["owedBB"] >= threshold:
                first.setdefault(row["handOrdinal"], row["kind"])
        groups = {"folded": [], "continued": [], "other": []}
        for ordinal, hand in enumerate(history, 1):
            key = "other" if ordinal not in first else "folded" if first[ordinal] == "fold" else "continued"
            groups[key].append(hand)
        thresholds[threshold] = {name: group(hands) for name, hands in groups.items()}
        assert sum(g["hands"] for g in thresholds[threshold].values()) == 400
        assert sum(g["lunaBB"] for g in thresholds[threshold].values()) == export["netBB"]
    three_bets = {}
    for row in rows:
        if row["facingThreeBet"]:
            three_bets.setdefault(row["handOrdinal"], row["kind"])
    outcomes = {kind: group([history[i - 1] for i, action in three_bets.items() if action == kind])
                for kind in ("fold", "call", "raise")}
    first_actions = {}
    for button, name in ((0, "buttonSB"), (1, "bigBlind")):
        split = [h for h in history if h["button"] == button]
        assert group(split)["lunaBB"] == export[name]["netBB"]
        first_actions[name] = dict(Counter(r["kind"] for r in rows if r["decisionOrdinal"] == 1
                                          and history[r["handOrdinal"] - 1]["button"] == button))
    last = {row["handOrdinal"]: row for row in rows}
    folded = [history[i - 1] for i, row in last.items() if row["kind"] == "fold"]
    ordered = sorted(history, key=lambda h: -h["humanChips"])
    assert all(h["humanChips"] > 0 for h in ordered[:10])
    return {"overall": group(history),
            "positions": {name: group([h for h in history if h["button"] == button])
                          for button, name in ((0, "buttonSB"), (1, "bigBlind"))},
            "acceptedHumanDecisions": len(rows), "firstActionCounts": first_actions,
            "exact20BBWins": sum(h["humanChips"] == 2000 for h in history),
            "exact20BBLosses": sum(h["humanChips"] == -2000 for h in history),
            "largeCallThresholds": thresholds,
            "afterRemovingTenLargestWins": group(ordered[10:]),
            "humanFoldedHands": group(folded),
            "firstResponseToButtonThreeBet": outcomes,
            "postflopFacingBet": {street: {"decisions": len(faced),
                                          "folds": sum(r["kind"] == "fold" for r in faced)}
                                   for street in ("flop", "turn", "river")
                                   for faced in [[r for r in rows if r["street"] == street and r["owedBB"] > 0]]},
            "inputHashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (root / "public-history.json", root / "export.json", csv_path)}}


def main():
    results = {name: analyse(root, old=name == "v040") for name, root in ROOTS.items()}
    old, new = results["v040"], results["shield"]
    # Reproduce the published historical decomposition as a cross-check.
    assert old["largeCallThresholds"][8]["continued"] == {
        "hands": 15, "lunaBB": 276, "bbPer100": 1840, "wins": 14, "losses": 0, "ties": 1}
    assert old["exact20BBWins"] == 12 and old["exact20BBLosses"] == 0
    assert old["afterRemovingTenLargestWins"]["lunaBB"] == -93
    print(json.dumps({"scope": "post-hoc descriptive comparison of unpaired sessions; no strength, decision-EV or adaptation estimate",
                      "matches": results,
                      "lunaShieldMinusV040": {
                          "netBB": new["overall"]["lunaBB"] - old["overall"]["lunaBB"],
                          "bbPer100": new["overall"]["bbPer100"] - old["overall"]["bbPer100"],
                          "buttonSBNetBB": new["positions"]["buttonSB"]["lunaBB"] - old["positions"]["buttonSB"]["lunaBB"],
                          "bigBlindNetBB": new["positions"]["bigBlind"]["lunaBB"] - old["positions"]["bigBlind"]["lunaBB"]}}, indent=2))


if __name__ == "__main__":
    main()
