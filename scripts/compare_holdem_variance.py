"""Compare frozen baselines and first-decision expansion at saved Hold'em roots."""

import argparse
import json
import platform
import resource
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from scripts.compare_holdem_sampling import sample_seed, summary
from scripts.profile_holdem_collection import load_root
from src.arena.artifacts import environment, git, source_fingerprint, write_json
from src.holdem.collection import CollectionLimitExceeded
from src.holdem.outcome_sampling import collect_outcome
from src.holdem.sampled_loss import sampled_betting_loss
from src.solver.neural.network import deterministic_cpu

FORMAT = "holdem-variance-v1"
ARMS = {
    "single-zero": ("zero", False),
    "single-frozen": ("frozen", False),
    "first-frozen": ("frozen", True),
}


def path_digest(result):
    events = [asdict(e.event) for e in result.executions]
    return sha256(
        json.dumps(events, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class GradientMoments:
    def __init__(self, dimensions):
        self.count = 0
        self.mean = torch.zeros(dimensions, dtype=torch.float64)
        self.m2 = 0.0
        self.largest = self.mean.clone()
        self.largest_norm = -1.0

    def add(self, gradient):
        if gradient.shape != self.mean.shape or not torch.isfinite(gradient).all():
            raise FloatingPointError("Invalid gradient vector")
        self.count += 1
        delta = gradient - self.mean
        self.mean += delta / self.count
        self.m2 += float(torch.dot(delta, gradient - self.mean))
        norm = float(torch.linalg.vector_norm(gradient))
        if norm > self.largest_norm:
            self.largest_norm, self.largest = norm, gradient.clone()
        return norm

    def report(self):
        if self.count < 2:
            raise ValueError("Variance requires at least two complete replicates")
        removal = (
            self.count
            / (self.count - 1)
            * float((self.largest - self.mean).square().sum())
        )
        return {
            "trace_sample_covariance": max(0.0, self.m2 / (self.count - 1)),
            "mean_norm": float(torch.linalg.vector_norm(self.mean)),
            "largest_norm": self.largest_norm,
            "largest_sample_variance_fraction": removal / self.m2 if self.m2 > 0 else 0,
            "trace_without_largest_sample": max(
                0.0, (self.m2 - removal) / (self.count - 2)
            )
            if self.count > 2
            else None,
        }


def measure_variance(
    run, job, traverser, arm, replicates, out, max_nodes=50_000, max_seconds=60
):
    if arm not in ARMS or type(replicates) is not int or replicates < 2:
        raise ValueError("Provide a known arm and at least two replicates")
    if type(max_nodes) is not int or max_nodes < 1 or not 0 < max_seconds <= 60:
        raise ValueError("Use a positive node limit and at most 60 seconds")
    out.mkdir(parents=True, exist_ok=False)
    trainer, hand, marker, manifest = load_root(run, job, 1, traverser, 0)
    profile = trainer.current_profile()
    physical = hand.table.seat_numbers[traverser]
    # This private copy measures training gradients; the collection profile stays frozen.
    model = deepcopy(profile._models[physical])
    if model is None:
        raise ValueError("Gradient comparison requires a fitted role model")
    model.requires_grad_(True).eval()
    parameters = tuple(model.parameters())
    moments = GradientMoments(sum(p.numel() for p in parameters))
    baseline, branch_first = ARMS[arm]
    report = {
        "kind": FORMAT,
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "source_sha256": source_fingerprint(),
        "environment": environment(),
        "checkpoint_sha256": marker["sha256"],
        "checkpoint_manifest": manifest,
        "profile_sha256": profile.fingerprint,
        "table": asdict(hand.table),
        "job": job,
        "traverser": traverser,
        "physical_role": physical,
        "iteration": 2,
        "sample": 0,
        "arm": arm,
        "exploration": 0.5,
        "baseline": baseline,
        "branch_first": branch_first,
        "requested_replicates": replicates,
        "max_nodes": max_nodes,
        "max_seconds": max_seconds,
        "rows": [],
        "status": "running",
        "gradient_parameters": [
            {"name": name, "shape": list(p.shape)}
            for name, p in model.named_parameters()
        ],
    }
    remaining = max_nodes
    collection_seconds = gradient_seconds = 0.0
    started = perf_counter()
    deadline = started + max_seconds
    try:
        with deterministic_cpu():
            for replicate in range(replicates):
                report["attempted_replicates"] = replicate + 1
                if remaining <= 0 or perf_counter() >= deadline:
                    raise CollectionLimitExceeded(
                        "Cell budget exhausted; no valid cell estimate"
                    )
                action_seed = (
                    int.from_bytes(
                        sha256(f"{FORMAT}/{job}/{replicate}".encode()).digest()[:8],
                        "big",
                    )
                    if branch_first
                    else sample_seed(job, "outcome-half", replicate)
                )
                before = perf_counter()
                try:
                    result = collect_outcome(
                        hand,
                        profile,
                        traverser,
                        iteration=2,
                        action_seed=action_seed,
                        exploration=0.5,
                        baseline=baseline,
                        branch_first=branch_first,
                        max_nodes=remaining,
                        deadline=deadline,
                    )
                finally:
                    collection_seconds += perf_counter() - before
                remaining -= result.nodes
                before = perf_counter()
                if result.decisions:
                    scores = model([d.candidates for d in result.decisions])
                    loss = sampled_betting_loss(scores, result.decisions, roots=1)
                    gradients = torch.autograd.grad(loss, parameters)
                    flat = torch.cat(
                        [g.detach().reshape(-1).double() for g in gradients]
                    )
                else:
                    flat = torch.zeros_like(moments.mean)
                norm = moments.add(flat)
                gradient_seconds += perf_counter() - before
                if perf_counter() >= deadline:
                    raise CollectionLimitExceeded(
                        "Gradient measurement exceeded cell deadline; no valid cell estimate"
                    )
                root = next(
                    (
                        d
                        for d in result.decisions
                        if d.candidates.decision.source.history == hand.events
                    ),
                    None,
                )
                report["rows"].append(
                    {
                        "replicate": replicate,
                        "action_seed": action_seed,
                        "path_sha256": path_digest(result),
                        "value_bb": result.value_bb,
                        "nodes": result.nodes,
                        "terminals": result.terminals,
                        "decisions": len(result.decisions),
                        "gradient_norm": norm,
                        "max_inverse_own_reach": max(
                            (1 / d.own_sample_reach for d in result.decisions),
                            default=1,
                        ),
                        "max_abs_update_bb": max(
                            (
                                abs(v)
                                for d in result.decisions
                                for v in d.regret_updates_bb
                            ),
                            default=0,
                        ),
                        "max_abs_conditional_target_bb": max(
                            (
                                abs(v)
                                for d in result.decisions
                                for v in (*d.values_bb, *d.regrets_bb)
                            ),
                            default=0,
                        ),
                        "max_abs_baseline_bb": max(
                            (abs(v) for d in result.decisions for v in d.baselines_bb),
                            default=0,
                        ),
                        "root_sampled_action": root.sampled_action if root else None,
                    }
                )
        if perf_counter() >= deadline:
            raise CollectionLimitExceeded(
                "Cell deadline exceeded; no valid cell estimate"
            )
        profile.assert_unchanged()
        report["status"] = "completed"
    except (CollectionLimitExceeded, FloatingPointError) as error:
        report.update(status="invalid", error=f"{type(error).__name__}: {error}")
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(
            cell_seconds=perf_counter() - started,
            collection_seconds=collection_seconds,
            gradient_seconds=gradient_seconds,
            completed_replicates=len(report["rows"]),
            completed_collection_nodes=max_nodes - remaining,
        )
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["process_peak_rss_bytes"] = (
            rss if platform.system() == "Darwin" else rss * 1024
        )
        write_json(out / "report.json", report)
    if report["status"] == "completed":
        rows = report["rows"]
        report["root_value_bb"] = summary([r["value_bb"] for r in rows])
        report["gradient"] = moments.report()
        report["gradient"]["variance_seconds_per_replicate"] = (
            report["gradient"]["trace_sample_covariance"]
            * (collection_seconds + gradient_seconds)
            / replicates
        )
        report["terminals"] = sum(r["terminals"] for r in rows)
        report["decisions"] = sum(r["decisions"] for r in rows)
        for key in (
            "max_inverse_own_reach",
            "max_abs_update_bb",
            "max_abs_conditional_target_bb",
            "max_abs_baseline_bb",
        ):
            report[key] = max(r[key] for r in rows)
        np.save(out / "mean-gradient.npy", moments.mean.numpy(), allow_pickle=False)
        report["mean_gradient_sha256"] = sha256(
            (out / "mean-gradient.npy").read_bytes()
        ).hexdigest()
        write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--traverser", type=int, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--replicates", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = measure_variance(**vars(args))
    print(
        json.dumps(
            {k: report[k] for k in ("status", "completed_replicates", "cell_seconds")}
        )
    )


if __name__ == "__main__":
    main()
