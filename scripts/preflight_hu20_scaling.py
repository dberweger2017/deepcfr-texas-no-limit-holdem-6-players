"""Outcome-free resumed work, host parity and complete save/export/load timings."""

import argparse
from dataclasses import asdict
import gc
import gzip
import json
from pathlib import Path
from time import perf_counter, time

from scripts.evaluate_hu20_reopening import Target
from scripts.evaluate_robustness import play
from scripts.hu20_scaling_common import acquire, check, identity, parent_trainer, specification
from scripts.tp20_common import append, interruptible, seal
from scripts.train_hu20 import rss, system, write_json
from src.blueprint.artifact import export_policy, load_training, save_training
from src.diagnostics.robustness import LBRConfig


def run(plan, parent, out, deadline):
    acquire(out); interruptible()
    start = time(); before = system(["sysctl", "vm.swapusage"])
    r = {"status": "incomplete", "started": start, "deadline": deadline,
         "identity": identity(), "parent": parent, "swap_before": before}
    write_json(out / "manifest.json", r)
    try:
        t = perf_counter(); trainer = parent_trainer(parent)
        r["parent_load_seconds"] = perf_counter() - t
        total = 0; t = perf_counter()
        with (out / "iterations.jsonl").open("w") as rows:
            while total < plan["preflight_nodes"]:
                check(plan, out, deadline, before)
                step = trainer.step(cancelled=lambda: time() >= deadline)
                total += step.nodes
                rows.write(json.dumps({**asdict(step), "updated_keys": None}) + "\n")
        r.update(completed_additional_nodes=total, complete_outer_seconds=perf_counter()-t,
                 iteration=trainer.iteration, entries=len(trainer.nodes))
        checkpoint = out / "continued.json.gz"; policy = out / "current.json.gz"
        t = perf_counter(); cp = save_training(trainer, checkpoint)
        r["checkpoint_seconds"] = perf_counter()-t
        t = perf_counter(); ph = export_policy(trainer, policy)
        r["export_seconds"] = perf_counter()-t
        # Identical future iterations from the saved and in-memory states.
        original = [asdict(trainer.step()) for _ in range(4)]
        h1 = save_training(trainer, out / "next-original.json.gz")
        del trainer; gc.collect()
        trainer = load_training(checkpoint)
        resumed = [asdict(trainer.step()) for _ in range(4)]
        h2 = save_training(trainer, out / "next-resumed.json.gz")
        if h1 != h2:
            raise ValueError("Within-host resume parity failed")
        for a, b in zip(original, resumed):
            for key in ("iteration", "nodes", "entries", "traverser_visits_by_street"):
                if a[key] != b[key]: raise ValueError("Within-host work parity failed")
        r.update(checkpoint_sha256=cp, policy_sha256=ph, next_checkpoint_sha256=h1,
                 recovery_validation_nodes=sum(s["nodes"] for s in original + resumed))
        del trainer; gc.collect()
        spec = specification(parent["seed"], parent["completed_nodes"]+total,
                             r["iteration"], checkpoint, policy, cp, ph)
        t = perf_counter(); source = Target(spec)
        r["checkpoint_export_verification_load_seconds"] = perf_counter()-t
        r["timing_panels"] = []
        with gzip.open(out / "timing-hands.jsonl.gz", "wt") as rows:
            for rule, contract, blocks in (("pressure", "native", 8), ("lbr", "menu", 8)):
                check(plan, out, deadline, before); t = perf_counter()
                def emit(row):
                    assert row.get("target_chips") is None
                    rows.write(json.dumps(row, sort_keys=True) + "\n")
                for b in range(blocks):
                    for rot in (0, 1):
                        play(source, spec, (rule,), contract, b, rot,
                             plan["preflight_root"], "scaling-preflight",
                             LBRConfig(4, 5), emit, resource_only=True)
                r["timing_panels"].append({"rule": rule, "hands": blocks*2,
                                            "seconds": perf_counter()-t})
        check(plan, out, deadline, before); r["status"] = "complete"
    except Exception as exc:
        r["failure"] = f"{type(exc).__name__}: {exc}"
    r.update(finished=time(), peak_rss_bytes=rss(), swap_after=system(["sysctl", "vm.swapusage"]))
    write_json(out / "result.json", r); seal(out)
    return r


def main():
    p = argparse.ArgumentParser(); p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--deadline", type=float, required=True); a = p.parse_args()
    r = run(json.loads(a.plan.read_text()), json.loads(a.parent.read_text()), a.out, a.deadline)
    print(json.dumps(r)); return r["status"] != "complete"


if __name__ == "__main__": raise SystemExit(main())
