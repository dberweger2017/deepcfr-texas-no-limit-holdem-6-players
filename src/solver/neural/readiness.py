"""A final-checkpoint control for the prospective small-game readiness campaign."""

import json
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from src.solver.evaluate import evaluate
from src.solver.experiment import canonical, write_json
from src.solver.neural.artifact import save_policy
from src.solver.neural.campaign import Campaign, run_seed
from src.solver.neural.checkpoint import load_training
from src.solver.neural.experiment import provenance
from src.solver.neural.network import deterministic_cpu, stream_seed


def fixed_comparison(campaign, seed, path, result, maximum_seconds):
    plan = campaign.for_seed(seed)
    training = path / "campaign/training"
    descriptor = json.loads((training / "checkpoint.json").read_text())
    if descriptor["file"] != f"iteration-{plan.iterations:06d}.pt":
        raise ValueError("Paired comparison requires the final checkpoint")
    with deterministic_cpu():
        solver, _ = load_training(
            training / descriptor["file"],
            descriptor["sha256"],
            manifest=provenance(asdict(plan)),
        )
        if solver.iterations != plan.iterations or solver.config != plan.training:
            raise ValueError("Paired checkpoint differs from the campaign plan")
        candidate = solver.average_policy()
        candidate_hash = sha256(canonical(candidate.tolist()).encode()).hexdigest()
        if candidate_hash != result["final"]["strategy_sha256"]:
            raise ValueError("Candidate policy differs from the final training report")
        # Average fitting never feeds collection. Only this loaded copy is refitted.
        solver.config = replace(
            solver.config,
            strategy_learning_rate_schedule="constant",
            strategy_final_learning_rate=None,
        )
        started = perf_counter()
        metrics = solver.fit_strategy(started + maximum_seconds)
        policy = solver.average_policy()
        fixed = evaluate(solver.tree, policy).to_dict()
        policy_hash = sha256(canonical(policy.tolist()).encode()).hexdigest()
        candidate_evaluation = evaluate(solver.tree, candidate).to_dict()
        return {
            "seed": seed,
            "iteration": solver.iterations,
            "checkpoint_sha256": descriptor["sha256"],
            "fit_seed": stream_seed(seed, "strategy-fit", solver.iterations, 0),
            "fixed_config": asdict(solver.config),
            "candidate_policy_sha256": candidate_hash,
            "fixed_policy_sha256": policy_hash,
            "fixed_policy_file_sha256": save_policy(solver, path / "fixed-policy.pt"),
            "candidate": candidate_evaluation,
            "fixed": fixed,
            "fixed_value_error": abs(
                fixed["value_player0"] - result["oracle"]["value"]
            ),
            "exploitability_delta": candidate_evaluation["exploitability"]
            - fixed["exploitability"],
            "strategy_fit": metrics,
            "wall_seconds": perf_counter() - started,
        }


def run_job(job: dict, output: Path) -> dict:
    campaign = Campaign.from_dict(job["campaign"])
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "error", "seed": job["seed"]}
    try:
        result = run_seed(campaign, job["seed"], output / "campaign")
        report["result"] = result
        if result["training_status"] != "completed":
            raise RuntimeError("Readiness training did not finish its declared budget")
        if job["comparison_seconds"] is not None:
            report["comparison"] = fixed_comparison(
                campaign, job["seed"], output, result, job["comparison_seconds"]
            )
        report["status"] = "completed"
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(output / "report.json", report)
    return report
