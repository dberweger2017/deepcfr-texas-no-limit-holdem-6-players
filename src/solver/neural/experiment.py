"""Bounded diagnostic pilots for the neural baseline, without model promotion."""

import json
import os
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
from src.solver.neural import snapshot_training
from src.solver.neural.artifact import save_policy
from src.solver.neural.average import (
    StrategyArchive,
    load_archive,
    record_iteration,
    save_archive,
    tabulate,
)
from src.solver.neural.checkpoint import load_training, save_training
from src.solver.neural.encoding import ACTION_SLOT
from src.solver.neural.memory import Reservoir
from src.solver.neural.network import deterministic_cpu, fit, stream_seed
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
    execution: str = "local"
    average: str = "network"
    checkpoint_interval: int | None = None

    def __post_init__(self):
        if (
            self.average not in {"network", "snapshots"}
            or self.game not in {"kuhn", "leduc"}
            or type(self.version) is not int
            or self.version != 1
        ):
            raise ValueError("Unsupported neural plan")
        for value in (self.iterations, self.evaluation_interval):
            if type(value) is not int or value < 1:
                raise ValueError(
                    "Iterations and evaluation interval must be positive integers"
                )
        if self.checkpoint_interval is not None and (
            type(self.checkpoint_interval) is not int or self.checkpoint_interval < 1
        ):
            raise ValueError("Checkpoint interval must be a positive integer")
        limits = {"local": 840, "cpu-campaign": 7200}
        if self.execution not in limits:
            raise ValueError("Unknown neural execution profile")
        if (
            type(self.maximum_seconds) not in (int, float)
            or not np.isfinite(self.maximum_seconds)
            or not 0 < self.maximum_seconds <= limits[self.execution]
        ):
            raise ValueError(f"{self.execution} run exceeds its runtime limit")

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "training": Config(**value["training"])})


def provenance(plan: dict) -> dict:
    files = sorted((ROOT / "src/solver").rglob("*.py")) + [
        ROOT / "scripts/check_deep_cfr.py",
        ROOT / "scripts/check_neural_convergence.py",
        ROOT / "scripts/run_strategy_study.py",
        ROOT / "scripts/study_worker.py",
        ROOT / "scripts/cpu_pilot.py",
        ROOT / "scripts/run_neural_readiness.py",
        ROOT / "scripts/run_snapshot_readiness.py",
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


def run(
    plan: Plan,
    output: Path,
    *,
    resume: Path | None = None,
    stop_after: int | None = None,
) -> dict:
    if stop_after is not None and (
        type(stop_after) is not int or not 0 < stop_after < plan.iterations
    ):
        raise ValueError("Stop iteration must precede the plan's final iteration")
    manifest = provenance(asdict(plan))
    started = perf_counter()
    elapsed = 0.0
    report = {
        "status": "running",
        "completed_iterations": 0,
        "evaluations": [],
        "advantage_fits": [],
    }
    solver = None
    archive = None
    if resume is not None:
        descriptor = json.loads((resume / "checkpoint.json").read_text())
        name = descriptor["file"]
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Invalid checkpoint filename")
        if plan.average == "snapshots":
            solver, archive, progress = snapshot_training.load_training(
                resume / name, descriptor["sha256"], manifest=manifest
            )
        else:
            solver, progress = load_training(
                resume / name, descriptor["sha256"], manifest=manifest
            )
        elapsed = progress["elapsed_seconds"]
        if type(elapsed) not in (int, float) or not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid checkpoint elapsed budget")
        report = progress["report"]
        if (
            solver.tree.game != plan.game
            or solver.config != plan.training
            or solver.iterations > plan.iterations
            or report["completed_iterations"] != solver.iterations
            or report["advantage_fits"] != solver.fits
            or (stop_after is not None and stop_after <= solver.iterations)
        ):
            raise ValueError("Checkpoint progress does not match the requested plan")
        expected = list(
            range(
                plan.evaluation_interval,
                solver.iterations + 1,
                plan.evaluation_interval,
            )
        )
        if solver.iterations == plan.iterations and solver.iterations not in expected:
            expected.append(solver.iterations)
        if [row["iteration"] for row in report["evaluations"]] != expected:
            raise ValueError("Checkpoint evaluation history is incomplete")
        report["status"] = "running"
        manifest["resumed_from"] = {
            "checkpoint_sha256": descriptor["sha256"],
            "iteration": solver.iterations,
        }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", manifest)
    deadline = started + max(0, plan.maximum_seconds - elapsed)

    def checkpoint():
        name = f"iteration-{solver.iterations:06d}.pt"
        save = save_training if archive is None else snapshot_training.save_training
        args = (solver,) if archive is None else (solver, archive)
        digest = save(
            *args,
            output / name,
            manifest=manifest,
            progress={
                "elapsed_seconds": elapsed + perf_counter() - started,
                "report": report,
            },
        )
        temporary = output / ".checkpoint.json"
        write_json(
            temporary, {"file": name, "sha256": digest, "iteration": solver.iterations}
        )
        os.replace(temporary, output / "checkpoint.json")

    try:
        with deterministic_cpu():
            if solver is None:
                solver = DeepCFR(GameTree(plan.game), plan.training)
                if plan.average == "snapshots":
                    archive = StrategyArchive(plan.game, plan.training.hidden)
            tree = solver.tree
            while solver.iterations < plan.iterations:
                if archive is None:
                    solver.step(deadline)
                else:
                    record_iteration(solver, archive, deadline)
                report["completed_iterations"] = solver.iterations
                report["advantage_fits"] = solver.fits
                scheduled = (
                    solver.iterations % plan.evaluation_interval == 0
                    or solver.iterations == plan.iterations
                )
                if scheduled:
                    if archive is None:
                        metrics = solver.fit_strategy(deadline)
                        policy = solver.average_policy()
                    else:
                        metrics = None
                        exported = output / f"average-{solver.iterations:06d}.pt"
                        checksum = save_archive(archive, exported)
                        policy = tabulate(load_archive(exported, checksum), tree)
                        if not np.allclose(
                            policy, solver.played_average(), atol=1e-12, rtol=0
                        ):
                            raise ArithmeticError(
                                "Exported average differs from recorded play"
                            )
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
                save_scheduled = (
                    plan.checkpoint_interval is not None
                    and solver.iterations % plan.checkpoint_interval == 0
                )
                if scheduled or save_scheduled or solver.iterations == stop_after:
                    checkpoint()
                    write_json(output / "report.json", report)
                if solver.iterations == stop_after:
                    report["status"] = "paused"
                    break
            else:
                report["policy_file_sha256"] = (
                    save_policy(solver, output / "policy.pt")
                    if archive is None
                    else save_archive(archive, output / "policy.pt")
                )
                report["average_kind"] = plan.average
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
        report["wall_seconds"] = elapsed + perf_counter() - started
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


def check_refit(plan: Plan, output: Path) -> dict:
    """Reconstruct a pilot's replay, then isolate optimizer budget on frozen targets."""
    settings = {
        "kind": "frozen-advantage-refit-v1",
        "pilot": asdict(plan),
        "steps": 4000,
        "maximum_error_fraction": 0.25,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", provenance(settings))
    started = perf_counter()
    deadline = started + plan.maximum_seconds
    report = {"status": "running", "comparisons": []}
    try:
        with deterministic_cpu():
            solver = DeepCFR(GameTree(plan.game), plan.training)
            for _ in range(plan.iterations):
                solver.step(deadline)
            report["baseline_fits"] = solver.fits[-2:]
            for player, memory in enumerate(solver.advantage_memories):
                data = b"".join(
                    array[: memory.size].tobytes()
                    for array in (memory.infos, memory.iterations, memory.targets)
                )
                before_hash = sha256(data).hexdigest()
                _, metrics = fit(
                    memory,
                    solver.features,
                    solver.mask,
                    hidden=plan.training.hidden,
                    steps=4000,
                    batch_size=plan.training.batch_size,
                    learning_rate=plan.training.learning_rate,
                    iteration=solver.iterations,
                    seed=stream_seed(
                        plan.training.seed, "advantage-fit", solver.iterations, player
                    ),
                    strategy=False,
                    deadline=deadline,
                )
                after = b"".join(
                    array[: memory.size].tobytes()
                    for array in (memory.infos, memory.iterations, memory.targets)
                )
                if sha256(after).hexdigest() != before_hash:
                    raise ArithmeticError("Refitting mutated the frozen replay")
                baseline = solver.fits[-2 + player]["excess_mse"]
                report["comparisons"].append(
                    {
                        "player": player,
                        "memory_sha256": before_hash,
                        "baseline_steps": plan.training.advantage_steps,
                        "baseline_excess_mse": baseline,
                        "refit": metrics,
                        "passed": metrics["excess_mse"] <= 0.25 * baseline,
                    }
                )
                write_json(output / "report.json", report)
            report["status"] = (
                "passed"
                if all(row["passed"] for row in report["comparisons"])
                else "failed"
            )
    except BaseException as exc:
        report["status"], report["error"] = "error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = perf_counter() - started
        write_json(output / "report.json", report)
    return report
