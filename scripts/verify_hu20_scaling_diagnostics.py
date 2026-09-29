"""Independently recompute #116 panel and paired effects from native hand rows."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from math import sqrt
from pathlib import Path
from statistics import mean

from scipy.stats import t


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def estimate(values, level):
    n = len(values)
    if n < 2:
        raise ValueError("Insufficient independent blocks")
    center = sum(values) / n
    variance = sum((value - center) ** 2 for value in values) / (n - 1)
    half = float(t.ppf((1 + level) / 2, n - 1)) * sqrt(variance / n)
    return center, [center - half, center + half]


def agree(observed, reference, label):
    center, interval = observed
    if abs(center - reference["bb100"]) > 1e-8 or any(
        abs(a - b) > 1e-8 for a, b in zip(interval, reference["interval"])
    ):
        raise ValueError(f"Independent arithmetic disagreement: {label}")


def verify(parent, completion):
    result = json.loads((completion / "combined-results.json").read_text())
    if result["status"] != "complete" or result["pending_panels"]:
        raise ValueError("Expected every frozen panel")
    paths = sorted((parent / "evaluation-primary").glob("*.jsonl.gz")) + sorted(
        (completion / "evaluation-diagnostic").glob("*.jsonl.gz")
    )
    rows = defaultdict(dict)
    hashes = {}
    deal_seeds = defaultdict(set)
    hands = 0
    for path in paths:
        hashes[str(path)] = file_hash(path)
        with gzip.open(path, "rt") as stream:
            for line in stream:
                hand = json.loads(line)
                if hand["status"] != "complete":
                    raise ValueError(f"Incomplete played hand in {path}")
                key = hand["policy"], hand["attacker"], hand["block"]
                if hand["rotation"] in rows[key]:
                    raise ValueError(f"Duplicate rotation: {key}")
                rows[key][hand["rotation"]] = hand["target_chips"]
                deal_seeds[(hand["root_seed"], hand["block"])].add(hand["deal_seed"])
                hands += 1
    if hands != 608256 or hands != result["native_replayed_hands"]:
        raise ValueError("Frozen total hand count differs")
    if any(set(roles) != {0, 1} for roles in rows.values()):
        raise ValueError("Missing paired position")
    if any(len(seeds) != 1 for seeds in deal_seeds.values()):
        raise ValueError("Same schedule coordinate has different deals")
    block = {key: mean(roles.values()) for key, roles in rows.items()}
    checks = 0
    for panel in result["per_policy"]:
        policy, attacker = panel["policy"], panel["attacker"]
        values = [value for (p, a, _), value in block.items() if (p, a) == (policy, attacker)]
        if len(values) != panel["target"]["blocks"]:
            raise ValueError(f"Panel block count differs: {policy} {attacker}")
        agree(estimate(values, .95), panel["target"], f"{policy} {attacker}")
        checks += 1
    seeds = (2026093001, 2026093002, 2026093003)

    def contrast(attacker, milestone, level):
        ids = sorted({b for p, a, b in block if a == attacker and p == f"B-{seeds[0]}-20000000"})
        values = []
        for b in ids:
            differences = [block[(f"B-{seed}-{milestone}", attacker, b)]
                           - block[(f"B-{seed}-20000000", attacker, b)] for seed in seeds]
            values.append(sum(differences) / len(seeds))
        return len(ids), estimate(values, level)

    for attacker, primary in result["primary"].items():
        n, value = contrast(attacker, 100000000, .975)
        if n != primary["long_minus_20M"]["blocks"]:
            raise ValueError(f"Primary block count differs: {attacker}")
        agree(value, primary["long_minus_20M"], f"primary {attacker}")
        checks += 1
    curve_checks = 0
    for curve in result["exploratory_curves"]:
        if curve["status"] != "available":
            continue
        n, value = contrast(curve["attacker"], curve["milestone"], .95)
        if n != curve["long_minus_20M"]["blocks"]:
            raise ValueError(f"Curve block count differs: {curve['attacker']} {curve['milestone']}")
        agree(value, curve["long_minus_20M"], f"curve {curve['attacker']} {curve['milestone']}")
        curve_checks += 1
    return {"status": "complete", "hands": hands, "panels": len(result["per_policy"]),
            "checked_estimates": checks + curve_checks, "checked_curves": curve_checks,
            "unique_schedule_coordinates": len(deal_seeds), "raw_hand_sha256": hashes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    value = verify(args.parent, args.completion)
    args.out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in value.items() if k != "raw_hand_sha256"}))


if __name__ == "__main__":
    main()
