"""Run the committed finite six-player learning diagnostic without rental compute."""

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from random import Random
from time import perf_counter

import numpy as np
import torch

from src.arena.artifacts import environment, git, source_fingerprint, write_json
from src.arena.heuristics import STYLES
from src.arena.policies import make_policy
from src.arena.runner import run_schedule
from src.arena.schedule import Plan, Scenario, build_schedule, digest, schedule_document
from src.game.types import ActionKind
from src.holdem.outcome_sampling import collect_outcome
from src.holdem.river_reference import (
    ReferenceProfile,
    check_deadline,
    combine_targets,
    contexts,
    enumerate_reference,
    fit_reference,
    prediction_metrics,
)
from src.holdem.targets import CandidateTargets
from src.solver.neural.network import deterministic_cpu

ARMS = {
    "single-zero": (False, "zero"),
    "single-oracle": (False, "frozen"),
    "first-zero": (True, "zero"),
    "first-oracle": (True, "frozen"),
}


def control(plan, out, deadline):
    arena = Plan(
        (Scenario("check-fold-control"),),
        candidate="check_fold",
        baseline="fold",
        opponents=tuple(STYLES),
        blocks=plan["control_blocks"],
        root_seed=plan["root_seed"],
    )
    rows = []

    class Guard:
        def __init__(self, policy, no_payment):
            self.policy, self.no_payment = policy, no_payment

        def choose_action(self, view):
            check_deadline(deadline)
            action = self.policy.choose_action(view)
            if self.no_payment and action.kind not in (
                ActionKind.CHECK,
                ActionKind.FOLD,
            ):
                raise ValueError("Control invested voluntarily")
            return action

    def factory(name, seed):
        return Guard(make_policy(name, seed), name in ("fold", "check_fold"))

    with (out / "control-outcomes.jsonl").open("w") as stream:

        def emit(row, trace):
            stream.write(json.dumps({"outcome": row, "trace": trace}) + "\n")
            stream.flush()
            rows.append(row)
            if row["status"] == "completed" and sum(row["net_chips"]) != 0:
                raise ValueError("Control settlement is not zero sum")

        if not run_schedule(arena, build_schedule(arena), emit, factory=factory):
            raise RuntimeError("Control arena failed; raw trace retained")
    write_json(out / "control-schedule.json", schedule_document(arena))
    result = {}
    for arm in ("candidate", "baseline"):
        rates = []
        for block in range(arena.blocks):
            selected = [r for r in rows if r["arm"] == arm and r["block"] == block]
            if len(selected) != 6:
                raise ValueError("Incomplete balanced block")
            chips = sum(r["candidate_chips"] for r in selected)
            rate = chips / arena.scenarios[0].big_blind / 6 * 100
            if rate < -25 - 1e-10:
                raise ValueError("Control lost more than posted blinds")
            rates.append(rate)
        result[arm] = {
            "bb_per_100": float(np.mean(rates)),
            "worst_block_bb_per_100": min(rates),
            "hands": len(rates) * 6,
        }
    return result


def moments(rows, reference):
    regrets = np.asarray([r["regrets_bb"] for r in rows])
    variance = float(np.var(regrets, axis=0, ddof=1).sum())
    mean_nodes = float(np.mean([r["nodes"] for r in rows]))
    return {
        "replicates": len(rows),
        "mean_regrets_bb": regrets.mean(axis=0).tolist(),
        "bias_bb": (regrets.mean(axis=0) - reference.target.regrets_bb).tolist(),
        "trace_variance": variance,
        "mean_nodes": mean_nodes,
        "variance_times_nodes": variance * mean_nodes,
        "reference_mse": float(np.mean((regrets - reference.target.regrets_bb) ** 2)),
        "seconds": sum(r["seconds"] for r in rows),
    }


def run(plan, out):
    started = perf_counter()
    deadline = started + plan["max_seconds"]
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "format": plan["format"],
        "plan": plan,
        "revision": git("rev-parse", "HEAD"),
        "source_sha256": source_fingerprint(),
        "plan_sha256": digest(plan),
        "environment": environment(),
        "status": "running",
        "references": [],
        "sampling": [],
        "fits": [],
        "tables": [],
    }
    write_json(out / "report.json", report)
    references, sampled = {}, {}
    all_contexts = contexts(plan)
    try:
        with deterministic_cpu(), (out / "samples.jsonl").open("w") as raw:
            report["control"] = control(plan, out, deadline)
            for context in all_contexts:
                for kind in ("uniform", "increasing"):
                    check_deadline(deadline)
                    profile = ReferenceProfile(kind)
                    reference = enumerate_reference(
                        context,
                        profile,
                        max_nodes=plan["max_tree_nodes"],
                        deadline=deadline,
                    )
                    references[context.name, kind] = reference.target
                    report["references"].append(
                        {
                            "context": context.name,
                            "split": context.split,
                            "profile": kind,
                            "assignments": context.assignments,
                            "target": asdict(reference.target),
                            "nodes": reference.nodes,
                            "seconds": reference.seconds,
                            "hero_information_sets": len(reference.baselines),
                        }
                    )
                    oracle = ReferenceProfile(kind, reference.baselines)
                    for seed in plan["seeds"]:
                        samples = {arm: [] for arm in ARMS}
                        for replicate in range(plan["replicates"]):
                            sample_seed = int(
                                digest((seed, context.name, kind, replicate))[:15], 16
                            )
                            world = Random(sample_seed).randrange(len(context.worlds))
                            hand = context.worlds[world]
                            paths = {}
                            for arm, (expand, baseline) in ARMS.items():
                                before = perf_counter()
                                result = collect_outcome(
                                    hand,
                                    oracle,
                                    hand.actor,
                                    iteration=1,
                                    action_seed=sample_seed,
                                    exploration=plan["exploration"],
                                    baseline=baseline,
                                    branch_first=expand,
                                    max_nodes=plan["max_tree_nodes"],
                                    deadline=deadline,
                                )
                                root = [
                                    d
                                    for d in result.decisions
                                    if d.candidates.decision.source == result.root
                                ]
                                if len(root) != 1 or root[0].own_sample_reach != 1:
                                    raise ValueError(
                                        "Root target is missing or incorrectly weighted"
                                    )
                                row = {
                                    "context": context.name,
                                    "profile": kind,
                                    "seed": seed,
                                    "replicate": replicate,
                                    "world": world,
                                    "action_seed": sample_seed,
                                    "arm": arm,
                                    "values_bb": root[0].values_bb,
                                    "regrets_bb": root[0].regrets_bb,
                                    "nodes": result.nodes,
                                    "seconds": perf_counter() - before,
                                    "path_sha256": digest(
                                        [asdict(e.event) for e in result.executions]
                                    ),
                                }
                                paths[arm] = row["path_sha256"]
                                samples[arm].append(row)
                                raw.write(json.dumps(row) + "\n")
                            if (
                                paths["single-zero"] != paths["single-oracle"]
                                or paths["first-zero"] != paths["first-oracle"]
                            ):
                                raise ValueError(
                                    "Changing only the baseline changed collection paths"
                                )
                            if (
                                context.name.endswith("facing")
                                and samples["first-zero"][-1]["values_bb"]
                                != samples["first-oracle"][-1]["values_bb"]
                            ):
                                raise ValueError(
                                    "Baseline changed a fully expanded final decision"
                                )
                        raw.flush()
                        noisy = samples["first-zero"]
                        target = reference.target
                        sampled[context.name, kind, seed] = CandidateTargets(
                            target.candidates,
                            target.policy,
                            tuple(np.mean([r["values_bb"] for r in noisy], axis=0)),
                            tuple(np.mean([r["regrets_bb"] for r in noisy], axis=0)),
                        )
                        for arm, rows in samples.items():
                            report["sampling"].append(
                                {
                                    "context": context.name,
                                    "profile": kind,
                                    "seed": seed,
                                    "arm": arm,
                                    **moments(rows, reference),
                                }
                            )
                    write_json(out / "report.json", report)
                    print(
                        f"Reference and sampling complete: {context.name} / {kind}",
                        flush=True,
                    )
            exact = {
                c.name: combine_targets(
                    [references[c.name, k] for k in ("uniform", "increasing")]
                )
                for c in all_contexts
            }
            train = [exact[c.name] for c in all_contexts if c.split == "train"]
            validation = [
                exact[c.name] for c in all_contexts if c.split == "validation"
            ]
            for seed in plan["seeds"]:
                noisy = {
                    c.name: combine_targets(
                        [sampled[c.name, k, seed] for k in ("uniform", "increasing")]
                    )
                    for c in all_contexts
                }
                for split in ("train", "validation"):
                    names = [c.name for c in all_contexts if c.split == split]
                    for label, table in (("exact", exact), ("sampled", noisy)):
                        report["tables"].append(
                            {
                                "seed": seed,
                                "split": split,
                                "arm": label,
                                **prediction_metrics(
                                    [table[n].regrets_bb for n in names],
                                    [exact[n] for n in names],
                                ),
                            }
                        )
                for label, table in (("exact", exact), ("sampled", noisy)):
                    fit_targets = [
                        table[c.name] for c in all_contexts if c.split == "train"
                    ]
                    model, metrics = fit_reference(
                        train,
                        validation,
                        fit_targets,
                        seed=seed,
                        plan=plan,
                        deadline=deadline,
                    )
                    path = out / f"{label}-{seed}.pt"
                    torch.save(model.state_dict(), path)
                    report["fits"].append(
                        {
                            "seed": seed,
                            "arm": label,
                            "sha256": sha256(path.read_bytes()).hexdigest(),
                            **metrics,
                        }
                    )
                    write_json(out / "report.json", report)
                    print(f"Fit complete: {label} / {seed}", flush=True)
            report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["seconds"] = perf_counter() - started
        report["completed_context_profiles"] = len(report["references"])
        report["expected_context_profiles"] = len(all_contexts) * 2
        report["expected_fits"] = len(plan["seeds"]) * 2
        report["artifacts"] = {
            p.name: sha256(p.read_bytes()).hexdigest()
            for p in out.iterdir()
            if p.name != "report.json" and p.is_file()
        }
        write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path("configs/holdem/river-reference.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.plan.read_text()), args.out)


if __name__ == "__main__":
    main()
