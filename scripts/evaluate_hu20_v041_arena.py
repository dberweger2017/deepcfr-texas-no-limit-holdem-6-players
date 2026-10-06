"""The owner-approved v0.4.1 arena: four policy arms on shared paired deals.

Arms (three lineages each): R, the B100M current policies (lineage one is v0.4.0's shipped export);
O, the 1B-node opponent-sampled average; C, the same runs' current policy; T, the 1B-node production
(traverser-reach) average. Every arm plays #141's 13 panels on identical deal blocks, rival seeds and
target action streams, so contrasts are paired within each block after averaging the three lineages.

`play` writes one model's hands (models run as independent processes); `report` combines all
hands, checks the frozen schedule is complete, and applies the predeclared release rule to O − R.
`timing` runs a small outcome-blind schedule on its own deal root and reports only costs.
"""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter, time

from scripts.evaluate_hu20_cfr_average import play
from src.arena.catalog import Checkpoint
from src.arena.report import estimate
from src.arena.schedule import digest
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
from src.blueprint.artifact import FrozenBlueprint, HU20_UNCAPPED_FORMAT
from src.diagnostics.cfr_average import DiagnosticAverage
from src.diagnostics.saved_hu20 import file_hash

ARMS = ("R", "O", "C", "T")
PRIMARY = ("O", "R")
SECONDARY = (("C", "R"), ("O", "C"), ("O", "T"))
# Predeclared release rule for O − R (BB/100, paired 95% intervals).
LBR_LOWER_ABOVE = 0.0
PRESSURE_LOWER_ABOVE = -10.0
SEVERE_UPPER_BELOW = -20.0


def load(spec, root):
    path = root / spec["path"]
    if path.stat().st_size != spec["bytes"] or file_hash(path) != spec["sha256"]:
        raise ValueError("Policy bytes differ before loading: " + spec["name"])
    source = (FrozenBlueprint(Checkpoint(spec["name"], str(path), spec["sha256"], HU20_UNCAPPED_FORMAT), path)
              if spec["strategy"] == "current" else DiagnosticAverage(path, spec["sha256"]))
    if (source.description["training_seed"] != spec["seed"] or source.description["iteration"] != spec["iteration"]
            or source.abstraction != HU20_UNCAPPED_SCHEMA or source.raise_cap is not None):
        raise ValueError("Policy identity differs: " + spec["name"])
    if spec["strategy"] == "average" and source.description["source_checkpoint_sha256"] != spec["checkpoint_sha256"]:
        raise ValueError("Average checkpoint provenance differs: " + spec["name"])
    return source


def play_model(plan, name, root, out, *, rss_limit):
    """All panels and blocks for one model, as one guarded process."""
    spec = next(s for s in plan["models"] if s["name"] == name)
    deadline = plan["started_at"] + plan["max_seconds"]
    def guard():
        if time() > deadline:
            raise TimeoutError("Frozen arena deadline")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > rss_limit:
            raise MemoryError("Worker peak RSS guard")
    out.mkdir(parents=True, exist_ok=True)
    started = perf_counter(); hands = 0; failure = None; seconds = defaultdict(float)
    try:
        guard(); source = load(spec, root); load_seconds = perf_counter() - started
        with gzip.open(out / f"{name}.hands.jsonl.gz", "xt") as stream:
            for panel in plan["panels"]:
                for block in range(panel["blocks"]):
                    for rotation in (0, 1):
                        row = play(source, spec, panel, plan["root"], block, rotation, guard)
                        row["arm"] = spec["arm"]
                        if plan["stage"] == "timing":
                            seconds[panel["name"]] += row["seconds"]
                        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"); stream.flush()
                        hands += 1
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"; load_seconds = None
    result = {"model": name, "arm": spec["arm"], "status": "incomplete" if failure else "complete",
              "failure": failure, "hands": hands, "plan_sha256": digest(plan), "seconds": perf_counter() - started,
              "load_seconds": load_seconds, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    if plan["stage"] == "timing":
        # Outcome-blind: seconds per paired block (both positions) by panel; payoffs are never read.
        result["seconds_per_block"] = {p["name"]: seconds[p["name"]] / p["blocks"] for p in plan["panels"]}
    (out / f"{name}.result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return result


def block_means(rows):
    """Per arm and panel: block -> three-lineage mean of the position-averaged target chips."""
    cells = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        cells[(r["arm"], r["panel"])][r["block"]].setdefault(r["seed"], {})[r["rotation"]] = r["target_chips"]
    means = {}
    for key, blocks in cells.items():
        series = {}
        for block, lineages in blocks.items():
            if len(lineages) != 3 or any(len(p) != 2 for p in lineages.values()):
                raise ValueError(f"Incomplete paired block {key} {block}")
            series[block] = sum((p[0] + p[1]) / 2 for p in lineages.values()) / 3
        means[key] = series
    return means


def contrast(means, a, b, panel):
    left, right = means[(a, panel)], means[(b, panel)]
    if set(left) != set(right):
        raise ValueError("Arms were not played on the same blocks")
    blocks = sorted(left)
    return estimate([left[k] - right[k] for k in blocks])


def report(plan, out):
    arms = tuple(dict.fromkeys(spec["arm"] for spec in plan["models"]))
    if set(arms) not in ({"R", "O"}, set(ARMS)):
        raise ValueError("Arena requires R/O or all four declared arms")
    comparisons = tuple((a, b) for a, b in (PRIMARY, *SECONDARY) if a in arms and b in arms)
    rows = []
    for spec in plan["models"]:
        result = json.loads((out / f"{spec['name']}.result.json").read_text())
        if result["status"] != "complete" or result["plan_sha256"] != digest(plan):
            raise ValueError("Incomplete or foreign model run: " + spec["name"])
        with gzip.open(out / f"{spec['name']}.hands.jsonl.gz", "rt") as stream:
            # Hands retain full action traces on disk; inference only needs paired payoffs.
            rows.extend({k: row[k] for k in ("arm", "panel", "seed", "block", "rotation", "target_chips")}
                        for row in map(json.loads, stream))
    expected = len(plan["models"]) * 2 * sum(p["blocks"] for p in plan["panels"])
    if len(rows) != expected:
        raise ValueError(f"Frozen schedule coverage differs: {len(rows)} of {expected}")
    means = block_means(rows)
    panels = [p["name"] for p in plan["panels"]]
    absolute = {arm: {panel: estimate(list(means[(arm, panel)].values())) for panel in panels} for arm in arms}
    contrasts = {f"{a}-{b}": {panel: contrast(means, a, b, panel) for panel in panels}
                 for a, b in comparisons}
    primary = contrasts["O-R"]
    lower = lambda e: e["ci95"][0] if e["ci95"] else None
    upper = lambda e: e["ci95"][1] if e["ci95"] else None
    checks = {
        "lbr_lower_above_0": lower(primary["lbr"]) is not None and lower(primary["lbr"]) > LBR_LOWER_ABOVE,
        "native_pressure_lower_above_minus_10": lower(primary["native-pressure"]) is not None
            and lower(primary["native-pressure"]) > PRESSURE_LOWER_ABOVE,
        "no_severe_regression": all(upper(primary[p]) is None or upper(primary[p]) >= SEVERE_UPPER_BELOW
                                    for p in panels if p not in ("lbr", "native-pressure")),
    }
    summary = {"plan_sha256": digest(plan), "hands": len(rows), "absolute_bb_per_100": absolute,
               "contrasts_bb_per_100": contrasts, "release_checks": checks, "release_rule_passed": all(checks.values()),
               "scope": ("paired fresh blocks, three-lineage means; primary O-R gates; C-R, O-C, O-T exploratory"
                         if set(arms) == set(ARMS) else "paired fresh blocks, three-lineage means; R/O-only primary O-R gates")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("play", "report"))
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--policies", type=Path, help="Folder holding every model's policy file")
    p.add_argument("--model", help="Model name to play")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rss-limit-gib", type=float, default=6.0)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    if a.command == "play":
        result = play_model(plan, a.model, a.policies, a.out, rss_limit=int(a.rss_limit_gib * 1024**3))
        print(json.dumps({k: result[k] for k in ("model", "status", "failure", "hands", "seconds", "peak_rss_bytes")}))
        return 0 if result["status"] == "complete" else 1
    summary = report(plan, a.out)
    print(json.dumps({"release_rule_passed": summary["release_rule_passed"], "checks": summary["release_checks"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
