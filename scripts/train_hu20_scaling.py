"""Continue an immutable uncapped parent without resetting iteration weights."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter, time

from scripts.hu20_reopening_common import independent_density
from scripts.hu20_scaling_common import acquire, check, identity, parent_trainer
from scripts.tp20_common import append, interruptible, seal
from scripts.train_hu20 import rss, system, write_json
from src.arena.schedule import digest
from src.blueprint.artifact import export_policy, save_training


def run(plan, parent, out, deadline):
    acquire(out); interruptible()
    started = time(); before = system(["sysctl", "vm.swapusage"])
    trainer = parent_trainer(parent, plan["limits"]["max_entries"])
    total = parent["completed_nodes"]; initial = total
    result = {"status": "incomplete", "started": started, "deadline": deadline,
              "identity": identity(), "parent": parent, "config": asdict(trainer.config),
              "initial_nodes": initial, "initial_iteration": trainer.iteration,
              "initial_entries": len(trainer.nodes), "plan_digest": digest(plan),
              "milestones": [], "failure": None, "swap_before": before}
    write_json(out / "manifest.json", result)
    marker = 0; saved_nodes = total; saved_time = time(); slot = 0
    last_parent_hash = parent["checkpoint_sha256"]
    try:
        while total < plan["training_total_nodes"]:
            check(plan, out, deadline, before)
            t = perf_counter(); step = trainer.step(cancelled=lambda: time() >= deadline)
            total += step.nodes
            append(out / "iterations.jsonl", {**asdict(step), "updated_keys": None,
                   "lifetime_completed_nodes": total, "additional_completed_nodes": total-initial,
                   "complete_outer_seconds": perf_counter()-t})
            milestones = plan["milestones"]
            if marker < len(milestones) and total >= milestones[marker]:
                requested = milestones[marker]; t = perf_counter()
                cp = out / f"checkpoint-{requested}.json.gz"
                policy = out / f"current-{requested}.json.gz"
                checkpoint_hash = save_training(trainer, cp)
                checkpoint_seconds = perf_counter()-t; t = perf_counter()
                ph = export_policy(trainer, policy); export_seconds = perf_counter()-t
                row = {"requested_total_nodes": requested, "completed_nodes": total,
                       "additional_nodes": total-initial, "overshoot_nodes": total-requested,
                       "iteration": trainer.iteration, "entries": len(trainer.nodes),
                       "checkpoint_sha256": checkpoint_hash, "policy_sha256": ph,
                       "parent_checkpoint_sha256": last_parent_hash,
                       "checkpoint_seconds": checkpoint_seconds, "export_seconds": export_seconds,
                       "peak_rss_bytes": rss()}
                t = perf_counter()
                row["independent"] = independent_density(trainer, Path(plan["independent_path"]))
                row["independent_seconds"] = perf_counter()-t
                append(out / "milestones.jsonl", row); result["milestones"].append(row)
                last_parent_hash = checkpoint_hash; marker += 1
                saved_nodes = total; saved_time = time()
            elif total-saved_nodes >= plan["recovery_nodes"] or time()-saved_time >= plan["recovery_seconds"]:
                # Alternate atomic files: the previous valid recovery survives replacement.
                slot = 1-slot; t = perf_counter(); path = out / f"recovery-{slot}.json.gz"
                h = save_training(trainer, path)
                row = {"path": path.name, "sha256": h, "completed_nodes": total,
                       "iteration": trainer.iteration, "seconds": perf_counter()-t}
                append(out / "recovery-saves.jsonl", row); write_json(out / "last-recovery.json", row)
                saved_nodes = total; saved_time = time()
            if total % 100000 < step.nodes:
                write_json(out / "progress.json", {"completed_nodes": total, "additional_nodes": total-initial,
                           "iteration": trainer.iteration, "entries": len(trainer.nodes),
                           "seconds": time()-started, "peak_rss_bytes": rss()})
        if marker != len(plan["milestones"]): raise ValueError("Missing fixed milestone")
        result["status"] = "complete"
    except Exception as exc:
        result.update(failure=f"{type(exc).__name__}: {exc}",
                      discarded_nodes=trainer.last_attempt_nodes,
                      discarded_work=trainer.last_attempt_work, failed_iteration=trainer.iteration+1)
        result["partial_checkpoint_sha256"] = save_training(trainer, out / "partial-last-completed.json.gz")
    result.update(completed_nodes=total, additional_nodes=total-initial,
                  completed_iterations=trainer.iteration,
                  additional_iterations=trainer.iteration-parent["iteration"],
                  entries=len(trainer.nodes), finished=time(), peak_rss_bytes=rss(),
                  swap_after=system(["sysctl", "vm.swapusage"]))
    write_json(out / "result.json", result); seal(out); return result


def main():
    p = argparse.ArgumentParser(); p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--deadline", type=float, required=True); a = p.parse_args()
    r = run(json.loads(a.plan.read_text()), json.loads(a.parent.read_text()), a.out, a.deadline)
    print(json.dumps({k: v for k, v in r.items() if k != "milestones"})); return r["status"] != "complete"


if __name__ == "__main__": raise SystemExit(main())
