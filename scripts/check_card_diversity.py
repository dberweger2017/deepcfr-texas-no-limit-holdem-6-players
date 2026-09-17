"""Prepare and run the bounded nested-board card-diversity diagnostic."""

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import torch

from scripts.check_representation import evaluate, fit, qualifies, reference
from src.arena.artifacts import environment, git, source_fingerprint, write_json
from src.arena.schedule import digest
from src.holdem.card_diversity import expanded_plan, training_targets
from src.holdem.representation_models import make_model
from src.holdem.representation_reference import specifications
from src.solver.neural.network import deterministic_cpu


def _calibrate(plan, specs, out):
    started = perf_counter()
    deadline = started + plan.get("calibration_reference_seconds", 300)
    rows, targets = [], []
    for spec in [s for s in specs if s["split"] == "train"][:2]:
        target, row = reference(spec, plan, deadline)
        targets.append(target)
        rows.append(row)
    worlds = sum(row["worlds"] for row in rows)
    if not worlds:
        raise ValueError("Calibration did not produce any hidden worlds")
    per_world = sum(
        profile["seconds"] for row in rows for profile in row["profiles"]
    ) / worlds
    timings = []
    calibration_plan = {
        **plan,
        "fit_steps": plan["calibration_steps"],
        "measure_steps": [],
    }
    for variant in plan["variants"]:
        _, measurement = fit(
            variant,
            plan["calibration_seed"],
            {"train": targets},
            calibration_plan,
            deadline,
        )
        timings.append(
            {k: measurement[k] for k in ("variant", "parameters", "seconds")}
        )
    allowance = plan["projection_allowance"]
    projected_reference = per_world * sum(len(s["deals"]) for s in specs) * allowance
    projected_fits = (
        sum(row["seconds"] for row in timings)
        / plan["calibration_steps"]
        * plan["fit_steps"]
        * len(plan["seeds"])
        * len(plan["training_board_counts"])
        * allowance
    )
    result = {
        "reference_seconds_with_allowance": projected_reference,
        "fitting_seconds_with_allowance": projected_fits,
        "models": timings,
        "seconds": perf_counter() - started,
        "passed": projected_reference < plan["max_reference_seconds"]
        and projected_fits < plan["max_fit_seconds"],
    }
    write_json(out / "calibration.json", result)
    if not result["passed"]:
        raise RuntimeError("Local calibration failed the committed resource gate")
    return result


def _validation_comparisons(fits, plan):
    result = []
    for comparison in plan["test_comparisons"]:
        baseline = comparison["baseline"]
        candidate = comparison["candidate"]
        checks = []
        for seed in plan["seeds"]:
            candidate_fit = next(
                row
                for row in fits
                if row["arm"] == candidate["arm"]
                and row["variant"] == candidate["variant"]
                and row["seed"] == seed
            )
            baseline_fit = next(
                row
                for row in fits
                if row["arm"] == baseline["arm"]
                and row["variant"] == baseline["variant"]
                and row["seed"] == seed
            )
            candidate_curve = candidate_fit["curves"][-1]
            baseline_curve = baseline_fit["curves"][-1]
            checks.append(
                candidate_curve["train"]["relative_rmse"] is not None
                and candidate_curve["train"]["relative_rmse"]
                <= plan["training_relative_rmse_limit"]
                and qualifies(
                    candidate_curve["validation"],
                    baseline_curve["validation"],
                    plan,
                )
            )
        result.append({**comparison, "seed_passes": checks})
    return result


def _test_comparisons(results, plan):
    indexed = {(row["arm"], row["variant"], row["seed"]): row for row in results}
    comparisons = []
    for comparison in plan["test_comparisons"]:
        checks = []
        baseline = comparison["baseline"]
        candidate = comparison["candidate"]
        for seed in plan["seeds"]:
            checks.append(
                qualifies(
                    indexed[(candidate["arm"], candidate["variant"], seed)],
                    indexed[(baseline["arm"], baseline["variant"], seed)],
                    plan,
                )
            )
        comparisons.append({**comparison, "seed_passes": checks, "passed": all(checks)})
    return comparisons


def _confirm_comparisons(validation, test):
    validation_by_name = {row["name"]: row for row in validation}
    confirmed = []
    for row in test:
        name = row["name"]
        validation_row = validation_by_name[name]
        validation_passed = all(validation_row["seed_passes"])
        confirmed.append(
            {
                **row,
                "validation_passed": validation_passed,
                "confirmed": validation_passed and row["passed"],
            }
        )
    return confirmed


def run(plan, out):
    out.mkdir(parents=False, exist_ok=False)
    report = {
        "format": plan["format"],
        "plan": plan,
        "plan_sha256": digest(plan),
        "revision": git("rev-parse", "HEAD"),
        "source_sha256": source_fingerprint(),
        "environment": environment(),
        "status": "running",
        "references": [],
        "fits": [],
        "test": [],
        "promoted": False,
    }
    started = perf_counter()
    try:
        materialized = expanded_plan(plan)
        specs = specifications(materialized)
        write_json(out / "contexts.json", specs)
        report["materialized_plan"] = materialized
        report["materialized_plan_sha256"] = digest(materialized)
        report["contexts_sha256"] = digest(specs)
        report["contexts_by_split"] = dict(Counter(s["split"] for s in specs))
        with deterministic_cpu():
            report["calibration"] = _calibrate(materialized, specs, out)
            print("Calibration passed", flush=True)
            deadline = perf_counter() + materialized["max_reference_seconds"]
            records = []
            for index, spec in enumerate(specs):
                target, row = reference(spec, materialized, deadline)
                records.append(
                    {
                        "target": target,
                        "split": spec["split"],
                        "board_group": spec["board_group"],
                    }
                )
                report["references"].append(row)
                if index % 24 == 23:
                    write_json(out / "report.json", report)
                    print(f"References {index + 1}/{len(specs)}", flush=True)
            torch.save(records, out / "targets.pt")
            report["training_contexts"] = {}
            fitting_deadline = perf_counter() + materialized["max_fit_seconds"]
            for board_count in materialized["training_board_counts"]:
                arm = f"train{board_count}"
                fitting, _ = training_targets(records, board_count)
                report["training_contexts"][arm] = {
                    "boards": board_count,
                    "contexts": len(fitting["train"]),
                    "updates": materialized["fit_steps"],
                    "sampled_examples": materialized["fit_steps"]
                    * materialized["batch_size"],
                    "effective_epochs": materialized["fit_steps"]
                    * materialized["batch_size"]
                    / len(fitting["train"]),
                }
                for seed in materialized["seeds"]:
                    for variant in materialized["variants"]:
                        model, measurement = fit(
                            variant, seed, fitting, materialized, fitting_deadline
                        )
                        measurement["arm"] = arm
                        measurement["training_boards"] = board_count
                        path = out / f"{arm}-{variant}-{seed}.pt"
                        torch.save(model.state_dict(), path)
                        measurement["sha256"] = sha256(path.read_bytes()).hexdigest()
                        report["fits"].append(measurement)
                        write_json(out / "report.json", report)
                        print(f"Fit complete: {arm}/{variant}/{seed}", flush=True)
            report["validation"] = _validation_comparisons(
                report["fits"], materialized
            )
            write_json(out / "selection.json", report["validation"])
            report["test"] = []
            test_targets = training_targets(records, materialized["training_board_counts"][0])[1]
            test_specs = {
                (entry[key]["arm"], entry[key]["variant"])
                for entry in materialized["test_comparisons"]
                for key in ("baseline", "candidate")
            }
            for arm, variant in sorted(test_specs):
                for seed in materialized["seeds"]:
                    model = make_model(variant, seed)
                    model.load_state_dict(
                        torch.load(
                            out / f"{arm}-{variant}-{seed}.pt",
                            weights_only=True,
                        )
                    )
                    report["test"].append(
                        {
                            "arm": arm,
                            "variant": variant,
                            "seed": seed,
                            **evaluate(model, test_targets),
                        }
                    )
            report["test_comparisons"] = _confirm_comparisons(
                report["validation"],
                _test_comparisons(report["test"], materialized),
            )
            # This field describes sealed-test criteria only; confirmation also
            # requires the separately reported validation criterion above.
            report["test_passed"] = all(
                comparison["passed"] for comparison in report["test_comparisons"]
            )
            report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["seconds"] = perf_counter() - started
        report["artifacts"] = {
            p.name: sha256(p.read_bytes()).hexdigest()
            for p in out.iterdir()
            if p.is_file() and p.name != "report.json"
        }
        write_json(out / "report.json", report)
    return report


def calibrate_only(plan, out):
    """Run only the declared admission calibration; never starts campaign fits."""

    out.mkdir(parents=False, exist_ok=False)
    materialized = expanded_plan(plan)
    specs = specifications(materialized)
    write_json(out / "contexts.json", specs)
    with deterministic_cpu():
        result = _calibrate(materialized, specs, out)
    write_json(
        out / "calibration-report.json",
        {
            "format": materialized["format"],
            "plan_sha256": digest(plan),
            "materialized_plan_sha256": digest(materialized),
            "contexts_sha256": digest(specs),
            "calibration": result,
        },
    )
    return result


def verify(out):
    """Reload every fit in a fresh process and reproduce stored bookkeeping."""

    report = json.loads((out / "report.json").read_text())
    if report["status"] != "completed":
        raise ValueError("Only completed experiments can be verified")
    if source_fingerprint() != report["source_sha256"]:
        raise ValueError("Verification requires the recorded source")
    for name, expected in report["artifacts"].items():
        if sha256((out / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact changed: {name}")
    materialized = report["materialized_plan"]
    specs = specifications(materialized)
    if digest(specs) != report["contexts_sha256"]:
        raise ValueError("Context generation no longer matches the report")
    records = torch.load(out / "targets.pt", weights_only=False)
    expected_counts = {
        "train": materialized["board_counts"]["train"] * 12,
        "validation": materialized["board_counts"]["validation"] * 12,
        "test": materialized["board_counts"]["test"] * 12,
    }
    counts = Counter(record["split"] for record in records)
    if counts != expected_counts:
        raise ValueError("Target bookkeeping has unexpected split counts")
    if any(
        set(curve) - {"step", "train", "validation"}
        for fit_record in report["fits"]
        for curve in fit_record["curves"]
    ):
        raise ValueError("Test metrics entered a fitting curve")
    with deterministic_cpu():
        for fit_record in report["fits"]:
            fitting, _ = training_targets(
                records, fit_record["training_boards"]
            )
            model = make_model(fit_record["variant"], fit_record["seed"])
            model.load_state_dict(
                torch.load(
                    out
                    / f'{fit_record["arm"]}-{fit_record["variant"]}-{fit_record["seed"]}.pt',
                    weights_only=True,
                )
            )
            final = fit_record["curves"][-1]
            if digest(evaluate(model, fitting["train"])) != digest(final["train"]):
                raise ValueError("Reloaded training predictions differ")
            if digest(evaluate(model, fitting["validation"])) != digest(
                final["validation"]
            ):
                raise ValueError("Reloaded validation predictions differ")
        test_targets = training_targets(
            records, min(materialized["training_board_counts"])
        )[1]
        for test_record in report["test"]:
            model = make_model(test_record["variant"], test_record["seed"])
            model.load_state_dict(
                torch.load(
                    out
                    / f'{test_record["arm"]}-{test_record["variant"]}-{test_record["seed"]}.pt',
                    weights_only=True,
                )
            )
            if digest(evaluate(model, test_targets)) != digest(
                {k: v for k, v in test_record.items() if k not in ("arm", "variant", "seed")}
            ):
                raise ValueError("Reloaded test predictions differ")
    if _validation_comparisons(report["fits"], materialized) != report["validation"]:
        raise ValueError("Validation comparison does not reproduce")
    if _confirm_comparisons(
        report["validation"], _test_comparisons(report["test"], materialized)
    ) != report["test_comparisons"]:
        raise ValueError("Sealed test comparison does not reproduce")
    return {"status": "verified", "fits": len(report["fits"]), "test": len(report["test"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/holdem/card-diversity.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.verify:
        print(json.dumps(verify(args.out)))
    elif args.calibrate:
        calibrate_only(plan, args.out)
    else:
        run(plan, args.out)


if __name__ == "__main__":
    main()
