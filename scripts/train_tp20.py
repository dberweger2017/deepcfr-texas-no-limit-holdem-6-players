"""From-zero TP20 K1 training with fixed checkpoints and independent density."""

import argparse
import json
import signal
from collections import Counter
from pathlib import Path
from time import monotonic, time

from scripts.tp20_common import (append, density, guard as resource_guard, rss, seal,
                                  system, validate, write_json)
from src.arena.schedule import digest
from src.blueprint.abstraction import TP20_SCHEMA
from src.blueprint.artifact import export_policy, save_training
from src.blueprint.solver import BlueprintTrainer, PilotConfig, TP20_GAME
from src.blueprint.windowed import _hash
from src.game.hand import Table


_stop_requested = False


def guard(plan, out, deadline):
    if _stop_requested:
        raise RuntimeError("TP20 requested stop; completed iteration retained")
    resource_guard(plan, out, deadline)


def install_stop_handler():
    # Raising from a signal handler during in-place publication would expose a
    # partial profile. Collection observes this flag; publication completes.
    def stop(signum, frame):
        global _stop_requested
        _stop_requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)


def train(plan, seed, out, deadline, observations, *, preflight=False):
    validate(plan, frozen=not preflight)
    if out.exists() or seed not in plan["training_seeds"]:
        raise ValueError("Existing output or seed outside fixed plan")
    out.mkdir(parents=True)
    started = monotonic()
    table = Table(tuple(f"player-{i}" for i in range(3)), (2000,)*3)
    trainer = BlueprintTrainer(table, PilotConfig(seed=seed, abstraction=TP20_SCHEMA,
        game=TP20_GAME, raise_cap=2, roots_per_seat=1,
        max_nodes=plan["limits"]["max_nodes_per_iteration"],
        max_entries=plan["limits"]["max_entries"],
        max_seconds=plan["limits"]["max_seconds_per_iteration"]))
    rows = json.loads(observations.read_text())
    write_json(out / "manifest.json", {"schema": "tp20-training-v1", "seed": seed,
        "initialization": "zero regrets; no parent checkpoint", "preflight": preflight,
        "game": TP20_GAME, "abstraction": TP20_SCHEMA, "plan_sha256": digest(plan),
        "independent_observations_sha256": _hash(observations),
        "started_unix_seconds": time(), "deadline_unix_seconds": deadline,
        "swap_before": system(["sysctl", "vm.swapusage"]),
        "memory_pressure_before": system(["memory_pressure", "-Q"])})
    target = plan["preflight_nodes"] if preflight else plan["training_nodes"]
    milestones = [target] if preflight else plan["checkpoints"]
    total = discarded = index = 0
    visits, new, repeated = Counter(), Counter(), Counter()
    result = {"status": "incomplete", "stop_reason": None}
    try:
        while total < target:
            guard(plan, out, deadline)
            try:
                report = trainer.step(cancelled=lambda: _stop_requested)
            except Exception:
                discarded += trainer.last_attempt_nodes
                raise
            total += report.nodes
            visits.update(report.traverser_visits_by_street)
            new.update(report.new_entries_by_street)
            repeated.update(report.revisited_keys_by_street)
            append(out / "iterations.jsonl", {"iteration": trainer.iteration,
                "completed_nodes": total, "nodes": report.nodes, "entries": report.entries,
                "new_entries": report.new_entries, "contributing_infosets": report.contributing_infosets,
                "traverser_visits_by_street": report.traverser_visits_by_street,
                "new_entries_by_street": report.new_entries_by_street,
                "revisited_keys_by_street": report.revisited_keys_by_street,
                "traversal_seconds": report.elapsed_seconds,
                "elapsed_seconds": monotonic()-started, "rss_bytes": rss()})
            while index < len(milestones) and total >= milestones[index]:
                guard(plan, out, deadline)
                checkpoint = out / f"checkpoint-{index}.json.gz"
                begin = monotonic()
                checkpoint_hash = save_training(trainer, checkpoint)
                save_seconds = monotonic()-begin
                guard(plan, out, deadline)
                begin = monotonic()
                policy_hash = export_policy(trainer, out / f"policy-{index}.json.gz")
                append(out / "checkpoints.jsonl", {"index": index,
                    "requested_nodes": milestones[index], "completed_nodes": total,
                    "overshoot_nodes": total-milestones[index], "iteration": trainer.iteration,
                    "entries": len(trainer.nodes), "checkpoint_sha256": checkpoint_hash,
                    "policy_sha256": policy_hash, "save_seconds": save_seconds,
                    "export_seconds": monotonic()-begin,
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "policy_bytes": (out / f"policy-{index}.json.gz").stat().st_size,
                    "independent_density_by_street": density(trainer.nodes, rows),
                    "rss_bytes": rss()})
                index += 1
        if index != len(milestones):
            raise ValueError("Missing fixed checkpoint")
        # Hard link final inference bytes rather than recomputing the profile.
        (out / "current.json.gz").hardlink_to(out / f"policy-{index-1}.json.gz")
        result.update(status="complete", final_checkpoint_sha256=checkpoint_hash,
                      current_export_sha256=policy_hash)
        guard(plan, out, deadline)
    except Exception as exc:
        result.update(status="incomplete", stop_reason=f"{type(exc).__name__}: {exc}")
        # Preserve the last completed iteration even when the next traversal fails.
        try:
            result["partial_checkpoint_sha256"] = save_training(trainer, out / "partial.json.gz")
        except Exception as saving:
            result["partial_save_error"] = f"{type(saving).__name__}: {saving}"
    result.update(completed_nodes=total, overshoot_nodes=max(0,total-target),
        discarded_nodes=discarded, iterations=trainer.iteration, entries=len(trainer.nodes),
        visits_by_street=dict(visits), new_entries_by_street=dict(new),
        repeated_contributions_by_street=dict(repeated), completed_checkpoints=index,
        elapsed_seconds=monotonic()-started, peak_process_rss_bytes=rss(),
        swap_after=system(["sysctl", "vm.swapusage"]),
        memory_pressure_after=system(["memory_pressure", "-Q"]))
    write_json(out / "result.json", result)
    seal(out)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "out", "observations"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args()
    install_stop_handler()
    r = train(json.loads(a.plan.read_text()), a.seed, a.out, a.deadline,
              a.observations, preflight=a.preflight)
    print(json.dumps(r), flush=True)
    return 0 if r["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
