"""Bounded diagnostic pilots for the neural baseline, without model promotion."""

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

from src.solver.cfr import regret_delta
from src.solver.evaluate import evaluate
from src.solver.experiment import ROOT, canonical, write_json
from src.solver.neural.artifact import save_policy
from src.solver.neural.encoding import ACTION_SLOT
from src.solver.neural.memory import Reservoir
from src.solver.neural.network import deterministic_cpu, fit
from src.solver.neural.solver import Config, DeepCFR
from src.solver.sequence_form import solve
from src.solver.tree import GameTree


@dataclass(frozen=True)
class Plan:
    game: str
    iterations: int
    training: Config
    evaluation_interval: int = 10
    maximum_seconds: float = 840
    version: int = 1

    def __post_init__(self):
        if (
            self.game not in {"kuhn", "leduc"}
            or type(self.version) is not int
            or self.version != 1
        ):
            raise ValueError("Unsupported neural plan")
        for value in (self.iterations, self.evaluation_interval):
            if type(value) is not int or value < 1:
                raise ValueError(
                    "Iterations and evaluation interval must be positive integers"
                )
        if (
            type(self.maximum_seconds) not in (int, float)
            or not np.isfinite(self.maximum_seconds)
            or not 0 < self.maximum_seconds <= 840
        ):
            raise ValueError("Local pilot must be bounded by 840 seconds")

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "training": Config(**value["training"])})


def provenance(plan: dict) -> dict:
    files = sorted((ROOT / "src/solver").rglob("*.py")) + [
        ROOT / "scripts/check_deep_cfr.py"
    ]
    source = {
        str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in files
    }

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True
        ).strip()

    return {
        "version": 1,
        "plan": plan,
        "plan_sha256": sha256(canonical(plan).encode()).hexdigest(),
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "source_files": source,
        "source_sha256": sha256(canonical(source).encode()).hexdigest(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            **{name: version(name) for name in ("numpy", "scipy", "torch")},
        },
        "protocol": {
            "device": "cpu",
            "threads": 1,
            "deterministic_algorithms": True,
            "update_order": [0, 1],
            "loss_weights": "2*t/T",
            "advantage_initialization": "zero-output-then-retrain-from-scratch",
            "regret_fallback": "highest-legal-prediction-first-slot-tie",
            "strategy_samples": "opponent-nodes-only",
            "encoding": "public-small-game-48-v1",
            "units": "ante-per-hand",
        },
    }


def run(plan: Plan, output: Path) -> dict:
    manifest = provenance(asdict(plan))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", manifest)
    started = perf_counter()
    deadline = started + plan.maximum_seconds
    report = {
        "status": "running",
        "completed_iterations": 0,
        "evaluations": [],
        "advantage_fits": [],
    }
    solver = None
    try:
        with deterministic_cpu():
            tree = GameTree(plan.game)
            solver = DeepCFR(tree, plan.training)
            for _ in range(plan.iterations):
                solver.step(deadline)
                report["completed_iterations"] = solver.iterations
                report["advantage_fits"] = solver.fits
                if (
                    solver.iterations % plan.evaluation_interval == 0
                    or solver.iterations == plan.iterations
                ):
                    metrics = solver.fit_strategy(deadline)
                    policy = solver.average_policy()
                    report["evaluations"].append(
                        {
                            "iteration": solver.iterations,
                            "neural_average": evaluate(tree, policy).to_dict(),
                            "empirical_memory_average": evaluate(
                                tree, solver.empirical_average()
                            ).to_dict(),
                            "exact_played_average": evaluate(
                                tree, solver.played_average()
                            ).to_dict(),
                            "strategy_fit": metrics,
                            "strategy_sha256": sha256(
                                canonical(policy.tolist()).encode()
                            ).hexdigest(),
                        }
                    )
                    write_json(output / "report.json", report)
            report["policy_file_sha256"] = save_policy(solver, output / "policy.pt")
            report["status"] = "completed"
    except TimeoutError as exc:
        report["status"], report["error"] = "timed_out", str(exc)
    except BaseException as exc:
        report["status"], report["error"] = "error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if solver is not None:
            report["advantage_fits"] = solver.fits
            report["memory_counts"] = {
                name: {"seen": memory.seen, "stored": memory.size}
                for name, memory in zip(
                    ("advantage_0", "advantage_1", "strategy"),
                    solver.advantage_memories + [solver.strategy_memory],
                )
            }
        report["wall_seconds"] = perf_counter() - started
        write_json(output / "report.json", report)
    return report


def reproduce(original: Path, output: Path) -> dict:
    previous = json.loads((original / "manifest.json").read_text())
    current = provenance(previous["plan"])
    for field in ("version", "plan_sha256", "source_sha256", "environment", "protocol"):
        if previous[field] != current[field]:
            raise ValueError(f"Neural reproduction mismatch: {field}")
    expected = json.loads((original / "report.json").read_text())
    if expected["status"] != "completed":
        raise ValueError("Only completed pilots can claim exact reproduction")
    if (
        sha256((original / "policy.pt").read_bytes()).hexdigest()
        != expected["policy_file_sha256"]
    ):
        raise ValueError("Original neural policy artifact hash mismatch")
    result = run(Plan.from_dict(previous["plan"]), output)
    if canonical({k: v for k, v in result.items() if k != "wall_seconds"}) != canonical(
        {k: v for k, v in expected.items() if k != "wall_seconds"}
    ):
        raise ValueError("Neural pilot results differ; both bundles have been retained")
    return result


def check_fitting(output: Path) -> dict:
    settings = {
        "kind": "controlled-fitting-v1",
        "hidden": 64,
        "seed": 101,
        "steps": 3000,
        "batch_size": 256,
        "learning_rate": 0.001,
        "maximum_seconds": 840,
        "limits": {"advantage": 0.02, "strategy": 0.01},
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", provenance(settings))
    started = perf_counter()
    report = {"status": "running", "fits": []}
    try:
        with deterministic_cpu():
            for game in ("kuhn", "leduc"):
                tree = GameTree(game)
                solver = DeepCFR(tree, Config(hidden=64, seed=101))
                oracle = solve(tree)
                uniform = tree.mask / tree.mask.sum(axis=1, keepdims=True)
                reach = tree.reaches(tree.edge_probabilities(uniform))
                cf_reach = np.zeros(len(tree.information_sets))
                for node, actor in enumerate(tree.actor):
                    if actor >= 0:
                        cf_reach[tree.info[node]] += (
                            reach[node, 2] * reach[node, 1 - actor]
                        )
                advantage_targets = regret_delta(tree, uniform) / cf_reach[:, None]
                for kind, local in (
                    ("advantage", advantage_targets),
                    ("strategy", oracle.policy),
                ):
                    memory = Reservoir(len(tree.information_sets), 101)
                    for info, key in enumerate(tree.information_sets):
                        target = np.zeros(3)
                        for column, action in enumerate(key.actions):
                            target[ACTION_SLOT[action]] = local[info, column]
                        memory.add(info, 1, target)
                    _, metrics = fit(
                        memory,
                        solver.features,
                        solver.mask,
                        hidden=64,
                        steps=3000,
                        batch_size=256,
                        learning_rate=0.001,
                        iteration=1,
                        seed=101,
                        strategy=kind == "strategy",
                        deadline=started + 840,
                    )
                    report["fits"].append(
                        {
                            "game": game,
                            "kind": kind,
                            **metrics,
                            "limit": settings["limits"][kind],
                            "passed": metrics["excess_mse"] <= settings["limits"][kind],
                        }
                    )
                    write_json(output / "report.json", report)
            report["status"] = (
                "passed" if all(row["passed"] for row in report["fits"]) else "failed"
            )
    except BaseException as exc:
        report["status"], report["error"] = "error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = perf_counter() - started
        write_json(output / "report.json", report)
    return report
