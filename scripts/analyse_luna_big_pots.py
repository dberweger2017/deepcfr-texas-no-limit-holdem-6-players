"""Descriptive, post-hoc decomposition using committed public-view records.

A hand has large-call exposure when Luna owed at least the threshold at a
recorded decision. Its first such action determines folded versus continued.
These whole-hand outcomes do not estimate decision EV or statistical strength.
"""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

PRIMARY = Path("docs/reports/luna-browser/primary")


def _call_bb(labels: str) -> float | None:
    for label in json.loads(labels):
        if label.lower().startswith("call"):
            return float(re.findall(r"[\d.]+", label)[0])
    return None


def _rate(values: list[float]) -> float | None:
    return 100 * sum(values) / len(values) if values else None


def analyse(root: Path, threshold: float) -> dict:
    with (root / "hands.csv").open() as stream:
        hands = list(csv.DictReader(stream))
    decisions = defaultdict(list)
    with (root / "decisions.csv").open() as stream:
        for row in csv.DictReader(stream):
            decisions[row["handId"]].append(row)
    result_bb = {h["handId"]: int(h["humanChips"]) / 100 for h in hands}
    ordinal = {h["handId"]: int(h["handOrdinal"]) for h in hands}
    large = {}
    first_decision = {}
    for hand_id, rows in decisions.items():
        for row in rows:
            owed = _call_bb(row["legalButtonLabels"])
            if owed is not None and owed >= threshold:
                large[hand_id] = "fold" if row["acceptedKind"] == "fold" else "continue"
                first_decision[hand_id] = row
                break

    def group(ids):
        values = [result_bb[i] for i in ids]
        return {"hands": len(values), "luna_bb": round(sum(values), 2),
                "bb_per_100": _rate(values), "wins": sum(v > 0 for v in values),
                "losses": sum(v < 0 for v in values), "ties": sum(v == 0 for v in values)}

    folded = [i for i, kind in large.items() if kind == "fold"]
    continued = [i for i, kind in large.items() if kind == "continue"]
    other = [h["handId"] for h in hands if h["handId"] not in large]
    all_bb = [result_bb[h["handId"]] for h in hands]
    removed = sorted((v for v in all_bb if v > 0), reverse=True)[:10]
    remaining_count = len(all_bb) - len(removed)
    remaining_bb = sum(all_bb) - sum(removed)
    folded_values = [result_bb[i] for i, rows in decisions.items()
                     if rows[-1]["acceptedKind"] == "fold"]
    street_folds = {}
    for street in ("flop", "turn", "river"):
        faced = [row for rows in decisions.values() for row in rows
                 if row["street"] == street and _call_bb(row["legalButtonLabels"]) is not None]
        count = sum(row["acceptedKind"] == "fold" for row in faced)
        street_folds[street] = {"facing_call_decisions": len(faced), "folds": count,
                               "fold_percent": 100 * count / len(faced) if faced else None}

    # Only published disclosures are read; never recover mucked cards privately.
    history_path = root / "public-history.json"
    history = {h["handId"]: h for h in json.loads(history_path.read_text())["hands"]} if history_path.exists() else {}
    continued_public = []
    for hand_id in sorted(continued, key=ordinal.get):
        row = first_decision[hand_id]
        shown = [e["cards"] for e in history.get(hand_id, {}).get("events", [])
                 if e["event"] == "shown" and e["seat"] == 1]
        continued_public.append({"handOrdinal": ordinal[hand_id], "luna_bb": result_bb[hand_id],
                                 "first_exposure_street": row["street"],
                                 "call_owed_bb": _call_bb(row["legalButtonLabels"]),
                                 "accepted_action": row["acceptedKind"],
                                 "human_cards": json.loads(row["humanVisibleCards"]),
                                 "board_at_first_exposure": json.loads(row["visibleBoard"]),
                                 "legitimately_shown_bot_cards": shown[0] if shown else []})
    return {
        "scope": "Post-hoc descriptive whole-hand outcomes; no confidence interval, significance test or decision-EV estimate",
        "threshold_call_owed_bb": threshold,
        "overall_hands": len(hands), "overall_bb_per_100": _rate(all_bb),
        "per_hand_sd_bb": statistics.stdev(all_bb) if len(all_bb) > 1 else None,
        "large_call_folded": group(folded), "large_call_continued": group(continued),
        "other_hands": group(other),
        "full_stacks_won": sum(v == 20 for v in all_bb),
        "full_stacks_lost": sum(v == -20 for v in all_bb),
        "near_full_stack_wins_19_to_under20": sum(19 <= v < 20 for v in all_bb),
        "full_stack_win_hands": sorted(ordinal[i] for i, v in result_bb.items() if v == 20),
        "without_top10_wins": {"removed_hands": len(removed), "remaining_hands": remaining_count,
                               "remaining_bb": remaining_bb,
                               "bb_per_100": 100 * remaining_bb / remaining_count if remaining_count else None},
        "luna_fold_count": len(folded_values),
        "average_folded_hand_bb": statistics.fmean(folded_values) if folded_values else None,
        "postflop_fold_decisions": street_folds,
        "public_history_available": history_path.exists(),
        "continued_public_hands": continued_public,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PRIMARY)
    args = parser.parse_args()
    print(json.dumps({"by_threshold": [analyse(args.root, t) for t in (5.0, 8.0, 12.0)]}, indent=2))


if __name__ == "__main__":
    main()
