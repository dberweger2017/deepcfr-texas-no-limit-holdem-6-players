"""Post-hoc decomposition of the Luna primary result by large-bet exposure.

Reads only the committed public-view CSVs. A hand is "large-bet" when Luna
faced a call of at least the threshold at any decision; the first such
decision decides whether Luna folded or continued.
"""

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from math import comb
from pathlib import Path

PRIMARY = Path("docs/reports/luna-browser/primary")


def _call_bb(labels: str) -> float | None:
    for label in json.loads(labels):
        if label.lower().startswith("call"):
            return float(re.findall(r"[\d.]+", label)[0])
    return None


def _interval(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return 100 * mean, 100 * (mean - half), 100 * (mean + half)


def analyse(root: Path, threshold: float) -> dict:
    hands = list(csv.DictReader((root / "hands.csv").open()))
    decisions = defaultdict(list)
    for row in csv.DictReader((root / "decisions.csv").open()):
        decisions[row["handId"]].append(row)
    result_bb = {h["handId"]: int(h["humanChips"]) / 100 for h in hands}

    large = {}
    for hand_id, rows in decisions.items():
        for row in rows:
            owed = _call_bb(row["legalButtonLabels"])
            if owed is not None and owed >= threshold:
                large[hand_id] = "fold" if row["acceptedKind"] == "fold" else "continue"
                break

    def group(ids):
        values = [result_bb[i] for i in ids]
        return {"hands": len(values), "luna_bb": round(sum(values), 2),
                "wins": sum(v > 0 for v in values), "losses": sum(v < 0 for v in values)}

    folded = [i for i, kind in large.items() if kind == "fold"]
    continued = [i for i, kind in large.items() if kind == "continue"]
    other = [h["handId"] for h in hands if h["handId"] not in large]
    all_bb = [result_bb[h["handId"]] for h in hands]
    other_bb = [result_bb[i] for i in other]
    return {
        "threshold_bb": threshold,
        "overall_bb_per_100": _interval(all_bb),
        "large_bet_folded": group(folded),
        "large_bet_continued": group(continued),
        "other_hands": group(other) | {"bb_per_100": _interval(other_bb)},
        "stacks_won": sum(v >= 19 for v in all_bb),
        "stacks_lost": sum(v <= -19 for v in all_bb),
        "stack_win_hands": sorted(int(h["handOrdinal"]) for h in hands
                                  if result_bb[h["handId"]] >= 19),
        "without_top10_wins_bb_per_100": 100 * (sum(all_bb) - sum(sorted(all_bb)[-10:])) / len(all_bb),
        "luna_fold_count": sum(1 for rows in decisions.values() if rows[-1]["acceptedKind"] == "fold"),
    }


def tail(n: int, k: int, p: float) -> float:
    """P(at least k wins in n) for an illustrative per-call win probability p."""
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PRIMARY)
    args = parser.parse_args()
    report = {"by_threshold": [analyse(args.root, t) for t in (5.0, 8.0, 12.0)]}
    main_row = report["by_threshold"][1]["large_bet_continued"]
    report["illustrative_tail_probabilities"] = {
        str(p): tail(main_row["hands"], main_row["wins"], p) for p in (0.6, 0.7, 0.8)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
