"""Post-hoc fold/sizing screens from saved HU20 hands; no models or new deals."""
import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path


def fold_measure(action):
    """call/pot matches the review screen, not a general raise MDF formula."""
    view = action["observation"]
    call, pot = view["call_amount"], view["pot"]
    if type(call) is not int or type(pot) is not int or not 0 < call < pot:
        raise ValueError("Invalid facing-wager amounts")
    probabilities, menu = view["probabilities"], view["menu"]
    if len(probabilities) != len(menu) or any(
            not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Invalid policy probabilities")
    if not math.isclose(sum(probabilities), 1, abs_tol=1e-8):
        raise ValueError("Unnormalized policy probabilities")
    folds = [p for choice, p in zip(menu, probabilities) if choice["kind"] == "fold"]
    if len(folds) != 1:
        raise ValueError("Expected one fold action")
    visits = view["visits"]
    if type(visits) is not int or visits < 0:
        raise ValueError("Invalid visits")
    # A prior own street wager identifies a response to a raise. A zero own
    # wager is still not a clean first bet if the call is stack-capped.
    context = view["public_context"]
    rival = context["players"][1 if view["position"] == "button" else 0]
    outstanding = rival["street_bet"] - view["street_bet"]
    if outstanding <= 0 or call != min(outstanding, view["stack"]):
        raise ValueError("Call differs from public commitments")
    spot = "raise_response" if view["street_bet"] else "first_bet"
    if call < outstanding:
        spot += "_capped_call"
    return {"fold_probability": folds[0], "realized_fold": int(action["kind"] == "fold"),
            "call_pot_screen": call / pot, "spot": spot,
            "visits": "under_100" if visits < 100 else "100-999" if visits < 1000 else "1000+",
            "sizing": "under_quarter" if call / pot < .25 else
                      "quarter_to_under_third" if call / pot < 1 / 3 else
                      "third_to_half" if call / pot <= .5 else "over_half"}


def add_screen(cell, measure):
    cell["decisions"] += 1
    for field in ("fold_probability", "realized_fold", "call_pot_screen"):
        cell[field + "_sum"] += measure[field]


def finalize(cell):
    n = cell["decisions"]
    return {**cell, "mean_fold_probability": cell["fold_probability_sum"] / n,
            "actual_fold_fraction": cell["realized_fold_sum"] / n,
            "mean_call_pot_screen": cell["call_pot_screen_sum"] / n,
            "mean_gap": (cell["fold_probability_sum"] - cell["call_pot_screen_sum"]) / n}


def accumulate(rows):
    groups, stops, opportunities = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    seen = set()
    for row in rows:
        if row["status"] != "complete" or not row.get("native_replay_verified"):
            raise ValueError("Incomplete or unreplayed record")
        coordinate = tuple(row[k] for k in ("panel", "seed", "version", "block", "rotation"))
        if coordinate in seen:
            raise ValueError("Duplicate hand coordinate")
        seen.add(coordinate)
        panel, version = row["panel"], row["version"]
        last = row["actions"][-1]
        reason = ("target_fold" if last["logical_player"] == 0 else "rival_fold") if last["kind"] == "fold" else "showdown"
        stops[(panel, version, last["street"], reason)].update(hands=1, chips=row["target_chips"])
        for action in row["actions"]:
            view = action["observation"]
            if action["logical_player"] == 1 and action["street"] == "flop" and view["call_amount"] == 0 and view["street_bet"] == 0:
                opportunities[(panel, version)].update(
                    decisions=1, bets=int(action["kind"] == "raise"),
                    nominal_pot_bets=int(action["kind"] == "raise" and action["raise_to"] == view["pot"]))
            if action["logical_player"] != 0 or view["call_amount"] == 0:
                continue
            measure = fold_measure(action)
            prefix = (panel, version, action["street"])
            for dimension, label in (("all", "all"), ("spot", measure["spot"]),
                                     ("visits", measure["visits"]), ("sizing", measure["sizing"]),
                                     ("seed", str(row["seed"]))):
                add_screen(groups[(*prefix, dimension, label)], measure)
    result = {}
    for (panel, version, street, dimension, label), cell in sorted(groups.items()):
        arm = result.setdefault(panel, {}).setdefault(version, {})
        arm.setdefault("facing_wagers", {}).setdefault(street, {}).setdefault(dimension, {})[label] = finalize(cell)
    for (panel, version, street, reason), cell in sorted(stops.items()):
        arm = result.setdefault(panel, {}).setdefault(version, {})
        arm.setdefault("last_betting_street_and_ending", {}).setdefault(street, {})[reason] = dict(cell)
    for (panel, version), cell in sorted(opportunities.items()):
        result[panel][version]["rival_flop_first_bet_opportunities"] = dict(cell)
    return result


def analyze(root):
    paths = sorted(root.glob("*/evaluation/hands.jsonl.gz"))
    if not paths:
        raise ValueError("No published hands found")
    def rows():
        for path in paths:
            with gzip.open(path, "rt") as stream:
                for line in stream:
                    yield json.loads(line)
    return {"post_hoc": True, "model_loads": 0, "new_hands": 0,
            "analysis_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "definition": "Decision-weighted mean fold probability versus call_amount / current pot; only uncapped first bets have the elementary zero-equity-bluff MDF interpretation. Selected reached decisions, not a range-frequency exploitability estimate. Ending groups carry whole-hand returns, not action EV.",
            "inputs": {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in paths},
            "panels": accumulate(rows())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.out.open("x") as output:
        json.dump(analyze(args.root), output, indent=2, sort_keys=True)
        output.write("\n")
