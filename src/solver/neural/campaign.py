"""Predeclared, per-seed convergence checks against the retained tabular reference."""

import json
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path

import numpy as np

from src.solver.evaluate import evaluate
from src.solver.experiment import ROOT, canonical, write_json
from src.solver.neural.experiment import Plan, provenance, run
from src.solver.neural.network import deterministic_cpu
from src.solver.sequence_form import solve
from src.solver.tree import GameTree

REFERENCE = ROOT / "docs/reports/tabular-validation.json"


@dataclass(frozen=True)
class Campaign:
    training: Plan
    seeds: tuple[int, ...]
    maximum_exploitability: float
    maximum_value_error: float
    reference_sha256: str
    version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported neural campaign")
        if (
            len(self.seeds) < 3
            or len(set(self.seeds)) != len(self.seeds)
            or any(type(s) is not int or s < 0 for s in self.seeds)
        ):
            raise ValueError("Declare at least three distinct nonnegative seeds")
        if self.training.training.seed != 0:
            raise ValueError(
                "Campaign template seed must be zero; use the declared seeds"
            )
        for limit in (self.maximum_exploitability, self.maximum_value_error):
            if type(limit) not in (int, float) or not np.isfinite(limit) or limit < 0:
                raise ValueError("Campaign tolerances must be finite and nonnegative")
        if (
            not isinstance(self.reference_sha256, str)
            or len(self.reference_sha256) != 64
        ):
            raise ValueError("Pin the retained tabular report hash")

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "training": Plan.from_dict(value["training"])})

    def for_seed(self, seed):
        if seed not in self.seeds:
            raise ValueError("Seed was not declared in this campaign")
        return replace(
            self.training, training=replace(self.training.training, seed=seed)
        )


def reference(campaign):
    if sha256(REFERENCE.read_bytes()).hexdigest() != campaign.reference_sha256:
        raise ValueError("Tabular reference report hash mismatch")
    result = json.loads(REFERENCE.read_text())
    if result["status"] != "passed":
        raise ValueError("Tabular reference did not pass")
    for name, digest in result["manifest"]["source_files"].items():
        if sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Tabular reference source changed: {name}")
    return result


def assess(campaign, seed, training, oracle):
    complete = (
        training["status"] == "completed"
        and training["completed_iterations"] == campaign.training.iterations
        and bool(training["evaluations"])
        and training["evaluations"][-1]["iteration"] == campaign.training.iterations
    )
    final = training["evaluations"][-1] if training["evaluations"] else None
    value_error = (
        None
        if final is None
        else abs(final["neural_average"]["value_player0"] - oracle["value"])
    )
    passed = (
        complete
        and final["neural_average"]["exploitability"] <= campaign.maximum_exploitability
        and value_error <= campaign.maximum_value_error
    )
    return {
        "seed": seed,
        "status": "passed"
        if passed
        else "paused"
        if training["status"] == "paused"
        else "failed",
        "training_status": training["status"],
        "completed_iterations": training["completed_iterations"],
        "value_error": value_error,
        "final": final,
        "final_advantage_fits": training["advantage_fits"][-2:],
        "memory_counts": training.get("memory_counts", {}),
    }


def run_seed(
    campaign: Campaign,
    seed: int,
    output: Path,
    *,
    resume: Path | None = None,
    stop_after: int | None = None,
) -> dict:
    plan = campaign.for_seed(seed)
    baseline = reference(campaign)
    manifest = provenance(asdict(campaign))
    manifest["seed"] = seed
    if resume is not None:
        previous = json.loads((resume / "manifest.json").read_text())
        for field in (
            "plan_sha256",
            "source_sha256",
            "environment",
            "protocol",
            "seed",
        ):
            if canonical(previous[field]) != canonical(manifest[field]):
                raise ValueError(f"Campaign resume mismatch: {field}")
        manifest["resumed_from_manifest_sha256"] = sha256(
            (resume / "manifest.json").read_bytes()
        ).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", manifest)
    report = {"seed": seed, "status": "running"}
    try:
        with deterministic_cpu():
            tree = GameTree(plan.game)
            solution = solve(tree)
            evaluation = evaluate(tree, solution.policy)
            stored = baseline["oracles"][plan.game]
            if (
                max(
                    abs(solution.upper_value - solution.lower_value),
                    solution.maximum_residual,
                    evaluation.exploitability,
                    abs(evaluation.value_player0 - solution.lower_value),
                    abs(solution.lower_value - stored["lower_value"]),
                )
                > 1e-8
            ):
                raise ArithmeticError("Independent equilibrium checks disagree")
            oracle = {"value": solution.lower_value, "evaluation": evaluation.to_dict()}
        training = run(
            plan,
            output / "training",
            resume=None if resume is None else resume / "training",
            stop_after=stop_after,
        )
        report = {
            **assess(campaign, seed, training, oracle),
            "oracle": oracle,
            "training_report_sha256": sha256(
                (output / "training/report.json").read_bytes()
            ).hexdigest(),
            "tabular": [row for row in baseline["runs"] if row["game"] == plan.game],
        }
    except BaseException as exc:
        report["status"], report["error"] = "error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(output / "report.json", report)
    return report


def summarize(campaign: Campaign, runs: list[Path], output: Path) -> dict:
    expected = provenance(asdict(campaign))
    rows, manifests = {}, []
    for path in runs:
        manifest = json.loads((path / "manifest.json").read_text())
        for field in (
            "version",
            "plan_sha256",
            "source_sha256",
            "environment",
            "protocol",
        ):
            if canonical(manifest[field]) != canonical(expected[field]):
                raise ValueError(f"Campaign summary mismatch: {field}")
        report = json.loads((path / "report.json").read_text())
        seed = manifest["seed"]
        if seed not in campaign.seeds or seed in rows or report["seed"] != seed:
            raise ValueError("Unexpected or duplicate campaign seed")
        if "training_report_sha256" in report:
            raw = (path / "training/report.json").read_bytes()
            if sha256(raw).hexdigest() != report["training_report_sha256"]:
                raise ValueError("Campaign training report hash mismatch")
            assessed = assess(campaign, seed, json.loads(raw), report["oracle"])
            if any(canonical(report[k]) != canonical(v) for k, v in assessed.items()):
                raise ValueError(
                    "Campaign assessment differs from the retained training result"
                )
        elif report["status"] != "error":
            raise ValueError("Campaign result is missing its training report")
        rows[seed] = report
        manifests.append(manifest)
    if set(rows) != set(campaign.seeds):
        raise ValueError("Every declared seed must be included, including failures")
    values = [
        row["final"]["neural_average"]["exploitability"]
        for row in rows.values()
        if row.get("final") is not None
    ]
    result = {
        "status": "passed"
        if all(row["status"] == "passed" for row in rows.values())
        else "failed",
        "runs": [rows[seed] for seed in campaign.seeds],
        "exploitability_summary": {
            "count": len(values),
            "mean": float(np.mean(values)) if values else None,
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", {**expected, "run_manifests": manifests})
    write_json(output / "report.json", result)
    return result
