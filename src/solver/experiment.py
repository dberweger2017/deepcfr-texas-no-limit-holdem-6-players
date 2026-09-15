"""Bounded reference runs with declared thresholds and retained strategy artifacts."""

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from src.solver.cfr import CFR
from src.solver.evaluate import evaluate
from src.solver.sequence_form import solve
from src.solver.tree import GameTree

ROOT = Path(__file__).resolve().parents[2]


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class RunSpec:
    game: str
    method: str
    iterations: int
    seeds: tuple[int, ...]
    maximum_exploitability: float
    maximum_value_error: float

    def __post_init__(self):
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if self.game not in {"kuhn", "leduc"} or self.method not in {
            "full",
            "external",
        }:
            raise ValueError("Unknown reference game or solver")
        if type(self.iterations) is not int or self.iterations < 1:
            raise ValueError("Iterations must be a positive integer")
        if (
            not self.seeds
            or any(type(s) is not int or s < 0 for s in self.seeds)
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ValueError("Seeds must be distinct nonnegative integers")
        for limit in (self.maximum_exploitability, self.maximum_value_error):
            if type(limit) not in (int, float) or not np.isfinite(limit) or limit < 0:
                raise ValueError("Acceptance limits must be finite and nonnegative")


@dataclass(frozen=True)
class Plan:
    runs: tuple[RunSpec, ...]
    version: int = 1
    maximum_seconds_per_run: float = 840
    maximum_total_seconds: float = 840
    evaluation_interval: int = 1000
    oracle_tolerance: float = 1e-8

    def __post_init__(self):
        object.__setattr__(self, "runs", tuple(self.runs))
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported reference plan version")
        if type(self.evaluation_interval) is not int or self.evaluation_interval < 1:
            raise ValueError("Evaluation interval must be a positive integer")
        for limit in (self.maximum_seconds_per_run, self.maximum_total_seconds):
            if (
                type(limit) not in (int, float)
                or not np.isfinite(limit)
                or not 0 < limit <= 840
            ):
                raise ValueError("Local reference jobs must be bounded by 840 seconds")
        if (
            type(self.oracle_tolerance) not in (int, float)
            or not np.isfinite(self.oracle_tolerance)
            or not 0 < self.oracle_tolerance <= 1e-6
        ):
            raise ValueError("Oracle tolerance must be positive and at most 1e-6")
        names = [(r.game, r.method, s) for r in self.runs for s in r.seeds]
        if not names or len(set(names)) != len(names):
            raise ValueError(
                "Declare at least one run, without duplicate game/method/seed entries"
            )

    @classmethod
    def from_dict(cls, value: dict) -> "Plan":
        return cls(**{**value, "runs": tuple(RunSpec(**r) for r in value["runs"])})


def provenance(plan: Plan) -> dict:
    paths = sorted((ROOT / "src/solver").glob("*.py")) + [
        ROOT / "scripts/check_solver.py"
    ]
    source = {
        str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths
    }

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True
        ).strip()

    return {
        "version": 1,
        "plan": asdict(plan),
        "plan_sha256": sha256(canonical(asdict(plan)).encode()).hexdigest(),
        "source_sha256": sha256(canonical(source).encode()).hexdigest(),
        "source_files": source,
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
        },
        "protocol": {
            "updates": "simultaneous",
            "averaging": "uniform-iterations-own-reach-exact",
            "units": "ante-per-hand",
            "exploitability": "nash-conv/2",
            "games": {"kuhn": "kuhn-ante1-v1", "leduc": "leduc-2bet-ante1-v1"},
        },
    }


def run(plan: Plan, output: Path) -> dict:
    manifest = provenance(plan)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", manifest)
    started = perf_counter()
    report = {"status": "running", "oracles": {}, "runs": []}
    trees = {}
    try:
        for spec in plan.runs:
            for seed in spec.seeds:
                name = f"{spec.game}-{spec.method}-{seed}"
                if perf_counter() - started >= plan.maximum_total_seconds:
                    report["runs"].append(
                        {"name": name, "status": "not_started_budget_exhausted"}
                    )
                    continue
                run_started = perf_counter()
                if spec.game not in trees:
                    tree = trees[spec.game] = GameTree(spec.game)
                    oracle = solve(tree)
                    oracle_evaluation = evaluate(tree, oracle.policy)
                    gap = abs(oracle.upper_value - oracle.lower_value)
                    if (
                        max(
                            gap,
                            oracle.maximum_residual,
                            oracle_evaluation.exploitability,
                            abs(oracle_evaluation.value_player0 - oracle.lower_value),
                        )
                        > plan.oracle_tolerance
                    ):
                        raise ArithmeticError("Independent equilibrium checks disagree")
                    if (
                        spec.game == "kuhn"
                        and abs(oracle.lower_value + 1 / 18) > plan.oracle_tolerance
                    ):
                        raise ArithmeticError("Kuhn value disagrees with -1/18")
                    report["oracles"][spec.game] = {
                        "lower_value": oracle.lower_value,
                        "upper_value": oracle.upper_value,
                        "maximum_residual": oracle.maximum_residual,
                        "evaluation": oracle_evaluation.to_dict(),
                        "nodes": len(tree.states),
                        "information_sets": len(tree.information_sets),
                    }
                tree = trees[spec.game]
                solver = CFR(tree, method=spec.method, seed=seed)
                evaluations = [
                    {
                        "iteration": 0,
                        **evaluate(tree, solver.average_policy()).to_dict(),
                    }
                ]
                deadline = min(
                    run_started + plan.maximum_seconds_per_run,
                    started + plan.maximum_total_seconds,
                )
                for _ in range(spec.iterations):
                    if perf_counter() >= deadline:
                        break
                    solver.step()
                    if solver.iterations % plan.evaluation_interval == 0:
                        evaluations.append(
                            {
                                "iteration": solver.iterations,
                                **evaluate(tree, solver.average_policy()).to_dict(),
                            }
                        )
                policy = solver.average_policy()
                if evaluations[-1]["iteration"] != solver.iterations:
                    evaluations.append(
                        {
                            "iteration": solver.iterations,
                            **evaluate(tree, policy).to_dict(),
                        }
                    )
                final = evaluations[-1]
                value_error = abs(
                    final["value_player0"] - report["oracles"][spec.game]["lower_value"]
                )
                complete = solver.iterations == spec.iterations
                accepted = (
                    complete
                    and final["exploitability"] <= spec.maximum_exploitability
                    and value_error <= spec.maximum_value_error
                )
                strategy = [
                    {
                        "information_set": asdict(key),
                        "probabilities": policy[i, : len(key.actions)].tolist(),
                    }
                    for i, key in enumerate(tree.information_sets)
                ]
                write_json(output / f"{name}-strategy.json", strategy)
                result = {
                    "name": name,
                    "game": spec.game,
                    "method": spec.method,
                    "seed": seed,
                    "status": "passed"
                    if accepted
                    else "failed"
                    if complete
                    else "timed_out",
                    "requested_iterations": spec.iterations,
                    "completed_iterations": solver.iterations,
                    "value_error": value_error,
                    "evaluations": evaluations,
                    "strategy_sha256": sha256(canonical(strategy).encode()).hexdigest(),
                    "wall_seconds": perf_counter() - run_started,
                }
                report["runs"].append(result)
                write_json(output / "report.json", report)
        report["status"] = (
            "passed"
            if all(r["status"] == "passed" for r in report["runs"])
            else "failed"
        )
    except BaseException as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = perf_counter() - started
        write_json(output / "report.json", report)
    return report


def reproduce(original: Path, output: Path) -> dict:
    manifest = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    plan = Plan.from_dict(manifest["plan"])
    current = provenance(plan)
    for field in ("version", "plan_sha256", "source_sha256", "environment", "protocol"):
        if manifest[field] != current[field]:
            raise ValueError(f"Reference reproduction mismatch: {field}")
    expected = json.loads((original / "report.json").read_text(encoding="utf-8"))
    if expected["status"] != "passed":
        raise ValueError(
            "Only completed passing reference bundles can claim exact reproduction"
        )
    result = run(plan, output)

    def deterministic(report):
        return {
            k: [
                {field: value for field, value in r.items() if field != "wall_seconds"}
                for r in v
            ]
            if k == "runs"
            else v
            for k, v in report.items()
            if k != "wall_seconds"
        }

    if canonical(deterministic(result)) != canonical(deterministic(expected)):
        raise ValueError("Reference results differ; both bundles have been retained")
    for item in expected["runs"]:
        filename = f"{item['name']}-strategy.json"
        if (original / filename).read_bytes() != (output / filename).read_bytes():
            raise ValueError("Reference strategy artifacts differ")
    return result
