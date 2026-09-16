"""Measure one frozen-root sampling cell; run cells in separate processes."""

import argparse
import json
import platform
import resource
from dataclasses import asdict
from hashlib import sha256
from math import sqrt
from pathlib import Path
from statistics import mean, variance
from time import perf_counter

from scripts.profile_holdem_collection import load_root
from src.arena.artifacts import environment, git, source_fingerprint, write_json
from src.holdem.collection import CollectionLimitExceeded, collect_traversal
from src.holdem.outcome_sampling import collect_outcome

FORMAT = "holdem-sampling-comparison-v1"
ARMS = {"external": None, "outcome-half": 0.5, "outcome-uniform": 1.0}


def sample_seed(job, arm, replicate):
    key = f"{FORMAT}/{job}/{arm}/{replicate}".encode()
    return int.from_bytes(sha256(key).digest()[:8], "big")


def summary(values):
    sample_variance = variance(values) if len(values) > 1 else None
    return {
        "mean": mean(values),
        "sample_variance": sample_variance,
        "standard_error": sqrt(sample_variance / len(values))
        if sample_variance is not None
        else None,
    }


def measure_cell(
    run, job, iteration, traverser, arm, replicates, max_nodes, max_seconds, out
):
    if arm not in ARMS or any(
        type(v) is not int or v < 1 for v in (replicates, max_nodes, iteration)
    ):
        raise ValueError(
            "Provide an arm and positive iteration, replicate and node counts"
        )
    if not 0 < max_seconds <= 300:
        raise ValueError("Use a positive time limit of at most 300 seconds")
    out.mkdir(parents=True, exist_ok=False)
    trainer, hand, marker, manifest = load_root(run, job, iteration, traverser, 0)
    profile = trainer.current_profile()
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
        "iteration": iteration + 1,
        "traverser": traverser,
        "sample": 0,
        "arm": arm,
        "exploration": ARMS[arm],
        "requested_replicates": replicates,
        "max_nodes": max_nodes,
        "max_seconds": max_seconds,
        "rows": [],
        "status": "running",
    }
    remaining = max_nodes
    started = perf_counter()
    deadline = started + max_seconds
    try:
        for replicate in range(replicates):
            report["attempted_replicates"] = replicate + 1
            if remaining <= 0 or perf_counter() >= deadline:
                raise CollectionLimitExceeded(
                    "Cell budget exhausted; no valid cell estimate"
                )
            args = {
                "iteration": iteration + 1,
                "action_seed": sample_seed(job, arm, replicate),
                "max_nodes": remaining,
                "deadline": deadline,
            }
            if arm == "external":
                result = collect_traversal(hand, profile, traverser, **args)
                records = result.targets
                max_inverse_reach = 1.0
                max_update = max(
                    (abs(v) for t in records for v in t.regrets_bb), default=0
                )
            else:
                result = collect_outcome(
                    hand, profile, traverser, exploration=ARMS[arm], **args
                )
                records = result.decisions
                max_inverse_reach = max(
                    (1 / t.own_sample_reach for t in records), default=1
                )
                max_update = max(
                    (abs(v) for t in records for v in t.regret_updates_bb), default=0
                )
            remaining -= result.nodes
            root = next(
                (
                    t
                    for t in records
                    if t.candidates.decision.source.history == hand.events
                ),
                None,
            )
            report["rows"].append(
                {
                    "replicate": replicate,
                    "action_seed": args["action_seed"],
                    "nodes": result.nodes,
                    "terminals": result.terminals if arm == "external" else 1,
                    "decisions": len(records),
                    "value_bb": result.value_bb,
                    "max_inverse_own_reach": max_inverse_reach,
                    "max_abs_update_bb": max_update,
                    "root_values_bb": root.values_bb if root is not None else None,
                    "root_sampled_action": root.sampled_action
                    if root is not None and arm != "external"
                    else None,
                }
            )
        report["status"] = "completed"
    except (CollectionLimitExceeded, FloatingPointError) as error:
        report.update(status="invalid", error=f"{type(error).__name__}: {error}")
    finally:
        report["collection_seconds"] = perf_counter() - started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["process_peak_rss_bytes"] = (
            rss if platform.system() == "Darwin" else rss * 1024
        )
        report["completed_replicates"] = len(report["rows"])
        report["completed_nodes"] = max_nodes - remaining
        # A failed path may have visited further nodes, reported in its error message.
        report["completed_nodes_exclude_failed_attempt"] = (
            report["status"] != "completed"
        )
        write_json(out / "report.json", report)
    if report["status"] == "completed":
        rows = report["rows"]
        report["value_bb"] = summary([r["value_bb"] for r in rows])
        report["terminals"] = sum(r["terminals"] for r in rows)
        report["decisions"] = sum(r["decisions"] for r in rows)
        report["max_inverse_own_reach"] = max(r["max_inverse_own_reach"] for r in rows)
        report["max_abs_update_bb"] = max(r["max_abs_update_bb"] for r in rows)
        if rows[0]["root_values_bb"] is not None:
            count = len(rows[0]["root_values_bb"])
            report["root_action_values_bb"] = [
                summary([r["root_values_bb"][a] for r in rows]) for a in range(count)
            ]
            report["root_actions"] = [asdict(a) for a in root.candidates.actions]
            report["root_sampled_action_counts"] = (
                [sum(r["root_sampled_action"] == a for r in rows) for a in range(count)]
                if arm != "external"
                else None
            )
        write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--traverser", type=int, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--replicates", type=int, required=True)
    parser.add_argument("--max-nodes", type=int, default=50_000)
    parser.add_argument("--max-seconds", type=float, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = measure_cell(**vars(args))
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("status", "completed_replicates", "collection_seconds")
            }
        )
    )


if __name__ == "__main__":
    main()
