"""Continue an immutable uncapped parent without resetting iteration weights."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter, time

from scripts.hu20_reopening_common import independent_density
from scripts.hu20_scaling_common import acquire, check, identity, parent_trainer
from scripts.hu20_scaling_runtime import validate_inputs
from scripts.tp20_common import append, interruptible, seal
from scripts.train_hu20 import rss, system, write_json
from src.arena.schedule import digest
from src.blueprint.artifact import export_policy, save_training
from src.blueprint.windowed import _hash


def publish_milestone(trainer, plan, parent, out, requested, total, anchor, *, recovered=None):
    """Publish each diagnostic stage separately, without a learning step."""
    row = {"requested_total_nodes": requested, "completed_nodes": total,
           "additional_nodes": total-parent["completed_nodes"], "overshoot_nodes": total-requested,
           "iteration": trainer.iteration, "entries": len(trainer.nodes),
           "parent_checkpoint_sha256": anchor[0], "starting_nodes": anchor[1],
           "starting_iteration": anchor[2], "ending_nodes": total,
           "ending_iteration": trainer.iteration, "recovered": recovered is not None,
           "milestone_training_steps": 0 if recovered else None}
    cp = Path(recovered["checkpoint_path"]) if recovered else out/f"checkpoint-{requested}.json.gz"
    policy = Path(recovered["policy_path"]) if recovered else out/f"current-{requested}.json.gz"
    def stage(name, **fields):
        append(out/"milestone-stages.jsonl", {"requested_total_nodes": requested,
               "completed_nodes": total, "iteration": trainer.iteration, "stage": name,
               "recovered": recovered is not None, **fields})
    t = perf_counter()
    h = _hash(cp) if recovered else save_training(trainer, cp)
    if recovered and h != recovered["checkpoint_sha256"]:
        raise ValueError("Recovered milestone checkpoint hash")
    row.update(checkpoint_sha256=h, checkpoint_path=str(cp.resolve()), checkpoint_seconds=perf_counter()-t)
    stage("checkpoint_complete", path=str(cp.resolve()), sha256=h)
    t = perf_counter()
    ph = _hash(policy) if recovered else export_policy(trainer, policy)
    if recovered and ph != recovered["policy_sha256"]:
        raise ValueError("Recovered milestone policy hash")
    row.update(policy_sha256=ph, policy_path=str(policy.resolve()), export_seconds=perf_counter()-t)
    stage("export_complete", path=str(policy.resolve()), sha256=ph)
    if recovered:
        from scripts.evaluate_hu20_reopening import Target
        from scripts.hu20_scaling_common import specification
        source = Target(specification(parent["seed"], requested, trainer.iteration, cp, policy, h, ph))
        del source
        stage("recovered_checkpoint_export_verified")
    t = perf_counter(); stage("density_started")
    row["independent"] = independent_density(trainer, Path(plan["independent_path"]))
    row.update(independent_seconds=perf_counter()-t, peak_rss_bytes=rss())
    stage("density_complete")
    append(out/"milestones.jsonl", row)
    stage("milestone_record_complete")
    return row


def run(plan, parent, out, deadline):
    acquire(out); interruptible()
    validate_inputs(plan)
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
    last_parent_nodes = total; last_parent_iteration = trainer.iteration
    phase = "startup"; unpublished = False
    try:
        for requested in plan["milestones"]:
            if requested > total:
                break
            recovered = next((m for m in parent.get("recovered_milestones", [])
                              if m["requested_total_nodes"] == requested), None)
            if recovered is None:
                raise ValueError("Already crossed milestone requires its exact retained artifact")
            phase = "recovered_milestone"
            row = publish_milestone(trainer, plan, parent, out, requested, total,
                                    (last_parent_hash, last_parent_nodes, last_parent_iteration), recovered=recovered)
            result["milestones"].append(row); marker += 1
            last_parent_hash = row["checkpoint_sha256"]
        while total < plan["training_total_nodes"]:
            check(plan, out, deadline, before)
            phase = "collection"; unpublished = True
            t = perf_counter(); step = trainer.step(cancelled=lambda: time() >= deadline)
            unpublished = False; phase = "iteration_record"
            total += step.nodes
            append(out / "iterations.jsonl", {**asdict(step), "updated_keys": None,
                   "lifetime_completed_nodes": total, "additional_completed_nodes": total-initial,
                   "complete_outer_seconds": perf_counter()-t})
            milestones = plan["milestones"]
            if marker < len(milestones) and total >= milestones[marker]:
                requested = milestones[marker]; phase = "milestone"
                row = publish_milestone(trainer, plan, parent, out, requested, total,
                                        (last_parent_hash, last_parent_nodes, last_parent_iteration))
                check(plan, out, deadline, before)
                result["milestones"].append(row)
                last_parent_hash = row["checkpoint_sha256"]; marker += 1
                last_parent_nodes = total; last_parent_iteration = trainer.iteration
                saved_nodes = total; saved_time = time()
            elif total-saved_nodes >= plan["recovery_nodes"] or time()-saved_time >= plan["recovery_seconds"]:
                phase = "recovery_save"
                # Alternate atomic files: the previous valid recovery survives replacement.
                slot = 1-slot; t = perf_counter(); path = out / f"recovery-{slot}.json.gz"
                h = save_training(trainer, path)
                row = {"path": path.name, "sha256": h, "completed_nodes": total,
                       "iteration": trainer.iteration, "seconds": perf_counter()-t,
                       "parent_checkpoint_sha256": last_parent_hash,
                       "original_resume_sha256": parent["checkpoint_sha256"],
                       "starting_nodes": last_parent_nodes, "starting_iteration": last_parent_iteration}
                append(out / "recovery-saves.jsonl", row); write_json(out / "last-recovery.json", row)
                saved_nodes = total; saved_time = time()
            if total % 100000 < step.nodes:
                write_json(out / "progress.json", {"completed_nodes": total, "additional_nodes": total-initial,
                           "iteration": trainer.iteration, "entries": len(trainer.nodes),
                           "seconds": time()-started, "peak_rss_bytes": rss()})
        if marker != len(plan["milestones"]): raise ValueError("Missing fixed milestone")
        check(plan, out, deadline, before)
        result["status"] = "complete"
    except Exception as exc:
        result.update(failure=f"{type(exc).__name__}: {exc}",
                      failure_phase=phase, unpublished_traversal=unpublished,
                      discarded_nodes=trainer.last_attempt_nodes if unpublished else 0,
                      discarded_work=trainer.last_attempt_work if unpublished else {},
                      failed_iteration=trainer.iteration+1 if unpublished else None)
        result["partial_checkpoint_sha256"] = save_training(trainer, out / "partial-last-completed.json.gz")
    result.update(completed_nodes=total, additional_nodes=total-initial,
                  overshoot_nodes=max(0,total-plan["training_total_nodes"]),
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
