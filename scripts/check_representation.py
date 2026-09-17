"""Compare card representations on a frozen, exact six-player reference set."""

import argparse
import json
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from random import Random
from time import perf_counter

import torch

from src.arena.artifacts import environment, git, source_fingerprint, write_json
from src.arena.schedule import digest
from src.holdem.betting import betting_loss
from src.holdem.representation_models import make_model
from src.holdem.representation_reference import build_context, specifications
from src.holdem.river_reference import (
    ReferenceProfile,
    check_deadline,
    combine_targets,
    enumerate_reference,
    prediction_metrics,
)
from src.solver.neural.network import deterministic_cpu


def evaluate(model, targets):
    regrets = []
    with torch.no_grad():
        for start in range(0, len(targets), 32):
            regrets.extend(
                s.regrets.tolist()
                for s in model([t.candidates for t in targets[start : start + 32]])
            )
    return prediction_metrics(regrets, targets)


def fit(variant, seed, targets, plan, deadline):
    if "train" not in targets or not set(targets) <= {"train", "validation"}:
        raise ValueError("Fitting cannot inspect test targets")
    model = make_model(variant, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=plan["learning_rate"])
    rng = Random(seed + 100_000)
    curves, clipped = [], 0
    started = perf_counter()
    for step in range(plan["fit_steps"] + 1):
        check_deadline(deadline)
        if step in plan["measure_steps"]:
            curves.append(
                {
                    "step": step,
                    **{split: evaluate(model, rows) for split, rows in targets.items()},
                }
            )
        if step == plan["fit_steps"]:
            break
        batch = [
            targets["train"][rng.randrange(len(targets["train"]))]
            for _ in range(plan["batch_size"])
        ]
        optimizer.zero_grad()
        loss = betting_loss(model([t.candidates for t in batch]), batch)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), plan["gradient_clip"], error_if_nonfinite=True
        )
        clipped += int(norm > plan["gradient_clip"])
        optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Nonfinite diagnostic model")
    return model, {
        "variant": variant,
        "seed": seed,
        "curves": curves,
        "parameters": sum(p.numel() for p in model.parameters()),
        "clipped_steps": clipped,
        "seconds": perf_counter() - started,
    }


def qualifies(candidate, control, plan):
    c, b = candidate["mean_decision_cost_bb"], control["mean_decision_cost_bb"]
    if candidate["relative_rmse"] is None or control["relative_rmse"] is None:
        return False
    return (
        b - c >= plan["minimum_absolute_cost_gain_bb"]
        and b - c >= plan["minimum_relative_cost_gain"] * b
        and candidate["relative_rmse"]
        <= control["relative_rmse"] + plan["allowed_relative_rmse_increase"]
    )


def select(fits, plan):
    indexed = {(f["variant"], f["seed"]): f for f in fits}
    eligible = []
    details = {}
    for variant in plan["variants"]:
        if variant == "original":
            continue
        checks, costs = [], []
        for seed in plan["seeds"]:
            candidate = indexed[variant, seed]["curves"][-1]
            original = indexed["original", seed]["curves"][-1]
            train_error = candidate["train"]["relative_rmse"]
            checks.append(
                train_error is not None
                and train_error <= 0.20
                and qualifies(candidate["validation"], original["validation"], plan)
            )
            costs.append(candidate["validation"]["mean_decision_cost_bb"])
        details[variant] = {
            "seed_passes": checks,
            "mean_validation_cost_bb": sum(costs) / len(costs),
        }
        if all(checks):
            eligible.append(
                (
                    sum(costs) / len(costs),
                    indexed[variant, plan["seeds"][0]]["parameters"],
                    variant,
                )
            )
    return {"selected": min(eligible)[2] if eligible else None, "candidates": details}


def reference(spec, plan, deadline):
    context = build_context(spec)
    targets, rows = [], []
    for kind in ("uniform", "increasing"):
        r = enumerate_reference(
            context,
            ReferenceProfile(kind),
            max_nodes=plan["max_tree_nodes"],
            deadline=deadline,
        )
        targets.append(r.target)
        rows.append(
            {
                "profile": kind,
                "nodes": r.nodes,
                "seconds": r.seconds,
                "values_bb": r.target.values_bb,
                "regrets_bb": r.target.regrets_bb,
            }
        )
    combined = combine_targets(targets)
    return combined, {
        "name": spec["name"],
        "split": spec["split"],
        "board_index": spec["board_index"],
        "worlds": len(spec["deals"]),
        "profiles": rows,
        "target": asdict(combined),
    }


def calibrate(plan, specs, out):
    started = perf_counter()
    deadline = started + 180
    rows, targets = [], []
    # Only the first training holding, in its two betting situations.
    for spec in [s for s in specs if s["split"] == "train"][:2]:
        target, row = reference(spec, plan, deadline)
        targets.append(target)
        rows.append(row)
    per_world = sum(r["seconds"] for x in rows for r in x["profiles"]) / sum(
        x["worlds"] for x in rows
    )
    timings = []
    calibration_plan = {**plan, "fit_steps": 8, "measure_steps": []}
    for variant in plan["variants"]:
        _, measurement = fit(
            variant, 809, {"train": targets}, calibration_plan, deadline
        )
        timings.append(
            {k: measurement[k] for k in ("variant", "parameters", "seconds")}
        )
    projected_reference = per_world * sum(len(s["deals"]) for s in specs) * 1.5
    projected_fits = (
        sum(x["seconds"] for x in timings)
        / 8
        * plan["fit_steps"]
        * len(plan["seeds"])
        * 1.5
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


def run(plan, out):
    out.mkdir(parents=True, exist_ok=False)
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
        specs = specifications(plan)
        write_json(out / "contexts.json", specs)
        report["contexts_sha256"] = digest(specs)
        report["contexts_by_split"] = dict(Counter(s["split"] for s in specs))
        with deterministic_cpu():
            report["calibration"] = calibrate(plan, specs, out)
            print("Calibration passed", flush=True)
            deadline = perf_counter() + plan["max_reference_seconds"]
            targets = {s: [] for s in ("train", "validation", "test")}
            for index, spec in enumerate(specs):
                target, record = reference(spec, plan, deadline)
                targets[spec["split"]].append(target)
                report["references"].append(record)
                if index % 12 == 11:
                    write_json(out / "report.json", report)
                    print(f"References {index + 1}/{len(specs)}", flush=True)
            torch.save(targets, out / "targets.pt")
            report["exact_policy"] = {
                split: prediction_metrics([t.regrets_bb for t in rows], rows)
                for split, rows in targets.items()
                if split != "test"
            }
            deadline = perf_counter() + plan["max_fit_seconds"]
            fitting = {k: v for k, v in targets.items() if k != "test"}
            for seed in plan["seeds"]:
                for variant in plan["variants"]:
                    model, measurement = fit(variant, seed, fitting, plan, deadline)
                    path = out / f"{variant}-{seed}.pt"
                    torch.save(model.state_dict(), path)
                    measurement["sha256"] = sha256(path.read_bytes()).hexdigest()
                    report["fits"].append(measurement)
                    write_json(out / "report.json", report)
                    print(f"Fit complete: {variant}/{seed}", flush=True)
            selection = select(report["fits"], plan)
            write_json(out / "selection.json", selection)
            report["selection"] = selection
            report["exact_policy"]["test"] = prediction_metrics(
                [t.regrets_bb for t in targets["test"]], targets["test"]
            )
            for variant in ("original", selection["selected"]):
                if variant is None:
                    continue
                for seed in plan["seeds"]:
                    model = make_model(variant, seed)
                    model.load_state_dict(
                        torch.load(out / f"{variant}-{seed}.pt", weights_only=True)
                    )
                    report["test"].append(
                        {
                            "variant": variant,
                            "seed": seed,
                            **evaluate(model, targets["test"]),
                        }
                    )
            report["test_passed"] = False
            if selection["selected"]:
                tests = {(x["variant"], x["seed"]): x for x in report["test"]}
                report["test_passed"] = all(
                    qualifies(
                        tests[selection["selected"], s], tests["original", s], plan
                    )
                    for s in plan["seeds"]
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


def verify(out):
    report = json.loads((out / "report.json").read_text())
    if source_fingerprint() != report["source_sha256"]:
        raise ValueError("Verification requires the recorded source")
    if report["status"] != "completed":
        raise ValueError("Only completed experiments can be verified")
    for name, expected in report["artifacts"].items():
        if sha256((out / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact changed: {name}")
    if select(report["fits"], report["plan"]) != report["selection"]:
        raise ValueError("Selection does not reproduce")
    # This is our own locally generated, hash-checked cache of typed target objects.
    targets = torch.load(out / "targets.pt", weights_only=False)
    with deterministic_cpu():
        for f in report["fits"]:
            model = make_model(f["variant"], f["seed"])
            model.load_state_dict(
                torch.load(out / f'{f["variant"]}-{f["seed"]}.pt', weights_only=True)
            )
            for split in ("train", "validation"):
                if digest(evaluate(model, targets[split])) != digest(
                    f["curves"][-1][split]
                ):
                    raise ValueError("Reloaded predictions differ")
        for entry in report["test"]:
            model = make_model(entry["variant"], entry["seed"])
            model.load_state_dict(
                torch.load(
                    out / f'{entry["variant"]}-{entry["seed"]}.pt', weights_only=True
                )
            )
            if digest(evaluate(model, targets["test"])) != digest(
                {k: v for k, v in entry.items() if k not in ("variant", "seed")}
            ):
                raise ValueError("Reloaded test predictions differ")
    return {
        "status": "verified",
        "models": len(report["fits"]),
        "artifacts": len(report["artifacts"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path("configs/holdem/representation.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.out)))
    else:
        run(json.loads(args.plan.read_text()), args.out)


if __name__ == "__main__":
    main()
