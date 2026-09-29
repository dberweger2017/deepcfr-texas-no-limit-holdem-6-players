"""Evaluate whole independent block shards, paired across every required policy."""

import argparse
import gzip
import json
from pathlib import Path
from time import time

from scripts.evaluate_hu20_reopening import ATTACKS, SECONDARY, Target, UniformPlayer
from scripts.evaluate_robustness import play
from scripts.hu20_scaling_common import acquire, check, identity
from scripts.tp20_common import interruptible, seal
from scripts.train_hu20 import rss, system, write_json
from src.arena.policies import make_policy
from src.arena.schedule import digest, stream_seed
from src.diagnostics.robustness import LBRConfig


def all_tasks(plan, specs):
    for spec in specs:
        reference = spec["arm"] == "A"
        attacks = [a for a in ATTACKS if not reference or a[0] in (
            "Pressure-native", "Minraise-original-cap2", "Passive")]
        for label, rule, contract in attacks:
            # The old pairs share deals; each new confirmation root is disjoint.
            index = {"Pressure-original-cap2": 0, "Pressure-native": 0,
                     "Minraise-original-cap2": 1, "Minraise-native": 1, "Passive": 2}[label]
            yield spec, label, rule, contract, plan["cheap_blocks"], plan["stress_root"]+index, "cheap"
        if not reference and spec["milestone"] in (20000000, plan["training_total_nodes"]):
            yield spec, "LBR-original-cap2", "lbr", "menu", plan["lbr_blocks"], plan["lbr_root"], "lbr"
            for index, label in enumerate(SECONDARY):
                yield spec, label, label, "secondary", plan["secondary_blocks"], plan["secondary_root"]+index, "secondary"


def tasks(plan, specs):
    for row in all_tasks(plan, specs):
        spec, label, *_ = row
        primary = spec["arm"] == "B" and spec["milestone"] in (20000000, plan["training_total_nodes"]) and label in ("LBR-original-cap2", "Pressure-native")
        selected = plan.get("panel_filter", "all")
        if selected == "primary" and not primary:
            continue
        if selected == "diagnostic" and primary:
            continue
        yield row


def shard_owner(block, plan):
    return plan["block_host_cycle"][block % len(plan["block_host_cycle"])]


def run(plan, specs, host, out, deadline, swap_before=None):
    acquire(out); interruptible(); started = time()
    before = swap_before or system(["sysctl", "vm.swapusage"])
    result = {"status": "incomplete", "host": host, "started": started,
              "identity": identity(), "plan_digest": digest(plan), "swap_before": before,
              "failure": None, "hands": 0}
    write_json(out / "models.json", specs); write_json(out / "manifest.json", result)
    attempts = []; source = None; current = None
    try:
        for spec, label, rule, contract, blocks, root, phase in tasks(plan, specs):
            check(plan, out, deadline, before)
            if spec["name"] != current:
                source = None
                import gc
                gc.collect()
                source = Target(spec); current = spec["name"]
                check(plan, out, deadline, before)
            selected = [b for b in range(blocks) if shard_owner(b, plan) == host]
            attempt = {"policy": spec["name"], "attacker": label, "phase": phase,
                       "requested_blocks": len(selected), "completed_blocks": 0,
                       "status": "running", "started": time()}
            attempts.append(attempt); write_json(out / "attempts.json", attempts)
            path = out / f'{spec["name"]}--{label}.jsonl.gz'
            with gzip.open(path, "wt") as saved:
                def emit(row):
                    row.update(attacker=label, host=host, training_seed=spec["seed"],
                               milestone=spec["milestone"], arm=spec["arm"])
                    saved.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
                    saved.flush(); result["hands"] += 1
                for b in selected:
                    check(plan, out, deadline, before)
                    for rot in (0, 1):
                        rivals = None
                        if phase == "secondary":
                            s = stream_seed(root, "test", "action", 2, b, 1)
                            rivals = {1: UniformPlayer(s) if label == "hu20_uniform" else make_policy(label, s)}
                        play(source, spec, (rule,), contract, b, rot, root, phase,
                             LBRConfig(4, 5), emit, opponent_policies=rivals)
                    attempt["completed_blocks"] += 1
                    if attempt["completed_blocks"] % 32 == 0:
                        write_json(out / "progress.json", {"hands": result["hands"], "attempt": attempt,
                                   "elapsed_seconds": time()-started, "peak_rss_bytes": rss()})
            attempt.update(status="complete", finished=time()); write_json(out / "attempts.json", attempts)
        result["status"] = "complete"
    except Exception as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
        if attempts and attempts[-1]["status"] == "running":
            attempts[-1].update(status="failed", failure=result["failure"])
    result.update(finished=time(), peak_rss_bytes=rss(), swap_after=system(["sysctl", "vm.swapusage"]))
    write_json(out / "attempts.json", attempts); write_json(out / "result.json", result); seal(out)
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True); p.add_argument("--host", choices=("m1", "m4"), required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--swap-baseline"); a = p.parse_args()
    r = run(json.loads(a.plan.read_text()), json.loads(a.models.read_text()), a.host, a.out,
            a.deadline, a.swap_baseline)
    print(json.dumps(r)); return r["status"] != "complete"


if __name__ == "__main__": raise SystemExit(main())
