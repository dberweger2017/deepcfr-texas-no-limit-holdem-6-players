"""Replay and independently aggregate the frozen B100M decision diagnosis."""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from hashlib import sha256
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

from scripts.evaluate_hu20 import write_json
from scripts.play_robustness import replay_row
from src.arena.schedule import digest
from src.diagnostics.conditional_values import summarize

SEEDS = (2026093001, 2026093002, 2026093003)
MILESTONES = (20000000, 40000000, 80000000, 100000000)
PANELS = ("pot_pressure", "one_third", "two_thirds", "native_minraise", "passive")


def estimate(data):
    values = tuple(data)
    if not values:
        return None
    center = mean(values)
    width = 1.96 * stdev(values) / sqrt(len(values)) if len(values) > 1 else None
    return {"n_blocks": len(values), "bb100": center,
            "ci95": [center-width, center+width] if width is not None else None}


def visit_band(count):
    if count == 0: return "0"
    if count <= 2: return "1-2"
    if count <= 9: return "3-9"
    if count <= 99: return "10-99"
    return "100+"


def _rows(path):
    with gzip.open(path, "rt") as handle:
        for line in handle:
            yield json.loads(line)


def _file_sha(path):
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decision_report(root):
    selection = json.loads((root / "selection.json").read_text())
    if digest(selection["selected"]) != selection["selection_digest"]:
        raise ValueError("Selection changed after outcome-blind freeze")
    attempts = json.loads((root / "values" / "attempts.json").read_text())
    worlds = defaultdict(list)
    limited = Counter()
    for row in _rows(root / "values" / "worlds.jsonl.gz"):
        key = (row["selection"]["seed"], row["selection"]["block"],
               row["selection"]["rotation"], row["selection"]["action_index"])
        if row["world_index"] != len(worlds[key]):
            raise ValueError("Missing/repeated conditional world index")
        worlds[key].append(tuple(row["action_returns_bb"]))
        limited[key] += row["limited_lbr_batches"]
    grouped = defaultdict(list)
    records = []
    for attempt in attempts:
        entry = attempt["selection"]
        key = (entry["seed"], entry["block"], entry["rotation"], entry["action_index"])
        data = worlds[key]
        if len(data) != attempt["worlds_completed"]:
            raise ValueError("Retained complete world count differs from attempt")
        if attempt["status"] == "complete":
            if len(data) != 96:
                raise ValueError("Completed decision lacks frozen 96 worlds")
            independent = summarize(data, attempt["probabilities"])
            prior = attempt["summary"]
            if abs(independent["policy_gap_bb"] - prior["policy_gap_bb"]) > 1e-10:
                raise ValueError("Independent conditional gap arithmetic differs")
            node = attempt["node"]
            if attempt["trained"] != (node is not None):
                raise ValueError("Training key/node inconsistency")
            band = visit_band(node["visits"] if node else 0)
            p = attempt["probabilities"]
            from math import log
            entropy = -sum(q * log(q) for q in p if q > 0)
            brief = {"selection": entry, "trained": attempt["trained"],
                     "visit_band": band, "visits": node["visits"] if node else 0,
                     "entropy_nats": entropy, "gap_bb": independent["policy_gap_bb"],
                     "gap_interval_bb": independent["policy_gap_95_interval_bb"],
                     "best_action": attempt["menu"][independent["best_estimated_index"]],
                     "lbr_limited_batches": limited[key]}
            records.append(brief)
            for label in ("all", f"street:{entry['street']}", f"seed:{entry['seed']}",
                          f"seat:{entry['seat']}", f"context:{entry['context']}",
                          f"trained:{attempt['trained']}", f"visits:{band}"):
                grouped[label].append(brief)
    primary = {label: {"n": len(rows),
                       "mean_gap_bb": mean(r["gap_bb"] for r in rows),
                       "positive_lower_bound": sum(r["gap_interval_bb"][0] > 0 for r in rows),
                       "median_visits": sorted(r["visits"] for r in rows)[len(rows)//2],
                       "limited_lbr_batches": sum(r["lbr_limited_batches"] for r in rows)}
               for label, rows in grouped.items()}
    exploratory = sorted(records, key=lambda r: -r["gap_bb"])[:8]
    return {"selection_digest": selection["selection_digest"],
            "selected": len(selection["selected"]), "attempted": len(attempts),
            "completed": len(records), "primary": primary,
            "per_decision": records, "exploratory_top_gaps": exploratory,
            "limitations": "Uniform compatible opponent range is not conditioned on prior LBR actions; fixed-policy future rollouts, not exact BR."}


def _evaluate_archive(path):
    data = {}
    issues = []
    for row in _rows(path):
        hand = replay_row(row)
        if row["status"] != "complete":
            issues.append({"block": row["block"], "rotation": row["rotation"],
                           "status": row["status"], "error": row.get("error")})
            continue
        if not hand.finished:
            raise ValueError("Completed row does not finish after native replay")
        key = (row["block"], row["rotation"])
        if key in data:
            raise ValueError("Repeated block/rotation")
        data[key] = row
    return data, issues


def curve_report(root, previous_deal_seeds):
    archive = root / "curve"
    if not (archive / "result.json").exists():
        return {"status": "pending"}
    result = json.loads((archive / "result.json").read_text())
    attempts = json.loads((archive / "attempts.json").read_text())
    values = {}
    issues = []
    hashes = {}
    deals = set()
    for seed in SEEDS:
        for milestone in MILESTONES:
            path = archive / f"B-{seed}-{milestone}.jsonl.gz"
            if not path.exists(): continue
            hashes[path.name] = _file_sha(path)
            values[(seed, milestone)], found = _evaluate_archive(path)
            issues.extend({"policy": path.name, **issue} for issue in found)
            deals.update(row["deal_seed"] for row in values[(seed, milestone)].values())
    if deals & previous_deal_seeds:
        raise ValueError("Fresh LBR curve overlaps #116 deal schedule")
    requested = json.loads((root / "curve-preflight" / "frozen.json").read_text())["selected_blocks"]
    complete = result["status"] == "complete" and len(values) == 12 and all(
        len(rows) == 2*requested for rows in values.values())
    estimates = {}
    if complete:
        for milestone in MILESTONES:
            seed_result = {}
            for seed in SEEDS:
                rows = values[(seed, milestone)]
                per_block = [mean(rows[(b, rot)]["target_chips"] for rot in (0, 1))
                             for b in range(requested)]
                seed_result[str(seed)] = estimate(per_block)
            estimates[str(milestone)] = {"seeds": seed_result,
                "aggregate": estimate(mean(values[(seed, milestone)][(b, rot)]["target_chips"]
                                           for seed in SEEDS for rot in (0, 1))
                                      for b in range(requested))}
            if milestone != 20000000:
                estimates[str(milestone)]["minus_own20m"] = estimate(
                    mean(values[(seed, milestone)][(b, rot)]["target_chips"]
                         - values[(seed, 20000000)][(b, rot)]["target_chips"]
                         for seed in SEEDS for rot in (0, 1))
                    for b in range(requested))
    return {"status": "complete" if complete else "incomplete", "requested_blocks": requested,
            "result": result, "attempts": attempts, "issues": issues,
            "estimates": estimates, "archive_sha256": hashes}


def translation_report(root, previous_deal_seeds):
    archive = root / "translation"
    if not (archive / "result.json").exists():
        return {"status": "pending"}
    result = json.loads((archive / "result.json").read_text())
    attempts = json.loads((archive / "attempts.json").read_text())
    estimates = {}
    hashes = {}
    issues = []
    for panel in PANELS:
        panels = {}
        for seed in SEEDS:
            variants = {}
            for variant in ("exact", "nearest"):
                path = archive / f"{seed}-{variant}-{panel}.jsonl.gz"
                if not path.exists(): continue
                hashes[path.name] = _file_sha(path)
                rows, found = _evaluate_archive(path)
                issues.extend({"policy": path.name, **issue} for issue in found)
                if {row["deal_seed"] for row in rows.values()} & previous_deal_seeds:
                    raise ValueError("Translation schedule overlaps #116 deals")
                variants[variant] = rows
            if set(variants) != {"exact", "nearest"}:
                continue
            if set(variants["exact"]) != set(variants["nearest"]):
                raise ValueError("Translation variants have unequal schedule")
            blocks = sorted({key[0] for key in variants["exact"]})
            exact = estimate(mean(variants["exact"][(b, r)]["target_chips"] for r in (0, 1)) for b in blocks)
            nearest = estimate(mean(variants["nearest"][(b, r)]["target_chips"] for r in (0, 1)) for b in blocks)
            contrast = estimate(mean(variants["nearest"][(b, r)]["target_chips"] -
                                     variants["exact"][(b, r)]["target_chips"] for r in (0, 1)) for b in blocks)
            telemetry = {}
            for variant, rows in variants.items():
                target = [item for row in rows.values() for item in row["actions"] if item["logical_player"] == 0]
                mapped = [item for item in target if item.get("translation", {}).get("attempted")]
                telemetry[variant] = {"target_decisions": len(target),
                    "trained": sum(item["target_trained"] is True for item in target),
                    "fallback": sum(item["target_trained"] is False for item in target),
                    "preceding_off_menu": sum(item["preceding_off_menu"] for item in target),
                    "translated_attempts": len(mapped),
                    "translated_hits": sum(item["translation"]["translated_hit"] for item in mapped),
                    "seconds_per_target_decision": mean(item["seconds"] for item in target) if target else None,
                    "size_pairs": [event for item in mapped for event in item["translation"]["events"]
                                   if event["status"] == "mapped"][:100]}
            panels[str(seed)] = {"exact": exact, "nearest": nearest,
                                  "nearest_minus_exact": contrast, "telemetry": telemetry}
        estimates[panel] = panels
    return {"status": result["status"], "result": result, "attempts": attempts,
            "issues": issues, "estimates": estimates, "archive_sha256": hashes}


def previous_deals(path):
    seeds = set()
    for file in sorted(path.glob("B-*-100000000--LBR-original-cap2.jsonl.gz")):
        for row in _rows(file):
            seeds.add(row["deal_seed"])
    return seeds


def report(root, previous):
    old = previous_deals(previous)
    result = {"schema": "hu20-b100-diagnosis-audit-v1",
              "decisions": decision_report(root),
              "curve": curve_report(root, old),
              "translation": translation_report(root, old)}
    files = [p for p in root.rglob("*") if p.is_file() and p.name not in ("report.json", "artifact-manifest.json")]
    manifest = {str(p.relative_to(root)): {"sha256": _file_sha(p), "bytes": p.stat().st_size}
                for p in sorted(files)}
    write_json(root / "report.json", result)
    write_json(root / "artifact-manifest.json", manifest)
    return {"decision_count": result["decisions"]["completed"],
            "curve": result["curve"]["status"],
            "translation": result["translation"]["status"],
            "files": len(manifest)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.root, args.previous), sort_keys=True))


if __name__ == "__main__":
    main()
