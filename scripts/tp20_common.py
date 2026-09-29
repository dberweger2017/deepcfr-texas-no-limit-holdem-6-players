"""Fixed three-player game identity, schedules and experiment accounting."""

import json
import os
import shutil
import signal
from collections import Counter
from dataclasses import asdict, replace
from time import time

from scripts.train_hu20 import rss, system, write_json
from src.arena.schedule import Plan, build_schedule, digest
from src.blueprint.abstraction import TP20_SCHEMA
from src.blueprint.solver import TP20_GAME
from src.blueprint.windowed import _hash


class Interrupted(RuntimeError):
    pass


def interruptible():
    def stop(signum, frame):
        raise Interrupted(f"Received signal {signum}; incomplete iteration is not published")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)


def append(path, value):
    with path.open("a") as saved:
        saved.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        saved.flush()
        os.fsync(saved.fileno())


def seal(out):
    write_json(out / "checksums.json", {str(p.relative_to(out)): _hash(p)
        for p in out.rglob("*") if p.is_file() and p.name != "checksums.json"})


def guard(plan, out, deadline):
    if time() >= deadline:
        raise TimeoutError("TP20 absolute phase deadline")
    if rss() >= plan["limits"]["max_rss_gib"] * 1024**3:
        raise MemoryError("TP20 process RSS ceiling")
    if shutil.disk_usage(out).free < plan["limits"]["min_free_gib"] * 1024**3:
        raise RuntimeError("TP20 free-disk ceiling")


def validate(plan, *, frozen=False):
    if (plan["schema"] != "tp20-fresh-campaign-v1" or plan["game"] != TP20_GAME
            or plan["abstraction"] != TP20_SCHEMA or plan["primary_extraction"] != "C"
            or len(set(plan["training_seeds"])) != 3
            or plan["limits"]["max_rss_gib"] > 10.5
            or plan["limits"]["max_campaign_seconds"] > 36000
            or plan["limits"]["training_seconds"] > 25200
            or plan["limits"]["report_reserve_seconds"] < 600
            or len(plan["primary_lineups"]) != 6
            or any(len(x) != 2 for x in plan["lineups"].values())):
        raise ValueError("Invalid fixed TP20 campaign contract")
    if frozen and (plan["training_nodes"] not in (20000000, 50000000, 100000000)
            or plan["checkpoints"][-1] != plan["training_nodes"]
            or plan["checkpoints"] != sorted(set(plan["checkpoints"]))
            or (plan["training_nodes"] > 20000000 and 20000000 not in plan["checkpoints"])):
        raise ValueError("Unfrozen work budget")


def schedule(plan, phase, lineup, count=None):
    definition = plan[phase]
    count = definition["blocks_per_lineup"] if count is None else count
    if phase == "crossplay":
        i = int(lineup.removeprefix("hero-"))
        seeds = plan["training_seeds"]
        opponents = (f"C{seeds[(i+1)%3]}", f"C{seeds[(i+2)%3]}")
        index = 100+i
    elif phase == "independent":
        opponents, index = ("tp20_uniform", "tp20_uniform"), 200
    else:
        opponents = tuple(plan["lineups"][lineup])
        index = list(plan["lineups"]).index(lineup)
    p = Plan.from_dict({"scenarios": [{"name": "tp20", "stacks": [2000]*3}],
        "candidate": "check_call", "baseline": "check_call", "opponents": opponents,
        "blocks": count, "root_seed": definition["root_seed"]+index,
        "split": "validation" if phase in ("development", "independent", "preflight") else "test",
        "max_decisions": 1000})
    blocks = tuple(replace(b, opponents=opponents if b.index%2 == 0 else opponents[::-1])
                   for b in build_schedule(p))
    document = {"game": TP20_GAME, "phase": phase, "lineup": lineup,
                "plan": asdict(p), "blocks": [asdict(b) for b in blocks],
                "opponent_order": "reverse on odd block, identical across all arms"}
    return p.scenarios[0], blocks, document


def density(nodes, rows):
    histograms = {}
    for row in rows:
        node = nodes.get(row["key"])
        histogram = histograms.setdefault(row["street"], Counter())
        histogram[node.visits if node else 0] += row["decisions"]
    result = {}
    for street in ("preflop", "flop", "turn", "river"):
        h = histograms.get(street, Counter())
        total = sum(h.values())
        def quantile(q):
            seen = 0
            for visits, count in sorted(h.items()):
                seen += count
                if seen >= max(1, total*q):
                    return visits
            return None
        result[street] = {"decisions": total, "trained_decisions": total-h[0],
            "coverage": (total-h[0])/total if total else None,
            "revisited_decisions": sum(c for v,c in h.items() if v > 1),
            "mean_visits": sum(v*c for v,c in h.items())/total if total else None,
            "visit_quantiles": {f"p{int(q*100)}": quantile(q) for q in (.25,.5,.75,.9,.99)}}
    return result
