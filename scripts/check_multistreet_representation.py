"""Run the frozen, bounded flop/turn/river representation diagnostic."""

import argparse
import copy
import json
from hashlib import sha256
from pathlib import Path
from random import Random
from time import perf_counter

import torch
import numpy as np

from src.arena.artifacts import environment, git, source_fingerprint, write_json
from src.arena.schedule import digest
from src.holdem.betting import betting_loss
from src.holdem.multistreet_models import VARIANTS, make_model
from src.holdem.multistreet_reference import enumerate_reference, split_specs
from src.holdem.multistreet_selection import choose_durations, qualify
from src.holdem.river_reference import ReferenceProfile, combine_targets, prediction_metrics
from src.solver.neural.network import deterministic_cpu


def evaluate(model, targets):
    predictions = []
    with torch.no_grad():
        for target in targets:
            predictions.append(model([target.candidates])[0].regrets.tolist())
    return prediction_metrics(predictions, targets)


def fit_arm(variant, seed, targets, plan, deadline):
    """Fit once and retain every declared duration checkpoint."""

    if set(targets) != {"train", "tuning"}:
        raise ValueError("Fitting accepts only train and tuning targets")
    durations = tuple(sorted(set(plan["durations"])))
    model = make_model(variant, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=plan["learning_rate"])
    rng = Random(seed + 100_000)
    measurements, checkpoints = [], {}
    max_steps = durations[-1]
    for step in range(max_steps + 1):
        if perf_counter() >= deadline:
            raise TimeoutError("Multi-street fitting exceeded its deadline")
        if step in durations:
            checkpoints[step] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            measurements.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "duration": step,
                    "metrics": {split: evaluate(model, rows) for split, rows in targets.items() if rows},
                }
            )
        if step == max_steps:
            break
        batch = [targets["train"][rng.randrange(len(targets["train"]))] for _ in range(plan["batch_size"])]
        optimizer.zero_grad()
        loss = betting_loss(model([target.candidates for target in batch]), batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), plan["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()
    return measurements, checkpoints


def _weighted_cost(metric, rows):
    """Equal-weight each street and each canonical board group."""

    by_street = {}
    for row, decision in zip(rows, metric["decisions"], strict=True):
        by_street.setdefault(row["street"], {}).setdefault(row["group"], []).append(
            decision["decision_cost_bb"]
        )
    street_means = []
    for groups in by_street.values():
        street_means.append(sum(sum(values) / len(values) for values in groups.values()) / len(groups))
    return sum(street_means) / len(street_means) if street_means else None


def _breakdown(metric, rows, field):
    grouped = {}
    for row, decision in zip(rows, metric["decisions"], strict=True):
        grouped.setdefault(row[field], []).append(decision["decision_cost_bb"])
    return {
        key: sum(values) / len(values)
        for key, values in grouped.items()
    }


def evaluate_rows(model, rows):
    targets = [row["target"] for row in rows]
    metric = evaluate(model, targets)
    metric["weighted_decision_cost_bb"] = _weighted_cost(metric, rows)
    metric["per_street_decision_cost_bb"] = _breakdown(metric, rows, "street")
    metric["per_situation_decision_cost_bb"] = _breakdown(metric, rows, "situation")
    return metric


def _paired_gain_se(candidate, baseline, rows):
    """SE of candidate minus baseline policy value on paired world Qs."""

    by_street = {}
    for row in rows:
        target = row["target"]
        with torch.no_grad():
            candidate_policy = model_policy(candidate, target)
            baseline_policy = model_policy(baseline, target)
        worlds = torch.tensor(row["world_action_values_bb"], dtype=torch.float64)
        differences = (worlds @ (candidate_policy - baseline_policy)).numpy()
        by_street.setdefault(row["street"], {}).setdefault(row["group"], []).append(differences)
    street_variances = []
    for groups in by_street.values():
        group_variances = []
        for context_values in groups.values():
            context_variances = [
                float(np.var(values, ddof=1) / len(values))
                if len(values) > 1
                else 0.0
                for values in context_values
            ]
            group_variances.append(
                sum(context_variances) / (len(context_variances) ** 2)
            )
        street_variances.append(sum(group_variances) / (len(group_variances) ** 2))
    street_count = len(street_variances)
    return (sum(street_variances) / (street_count**2)) ** 0.5


def model_policy(model, target):
    with torch.no_grad():
        return model([target.candidates])[0].probabilities().cpu().double()


def _reference_rows(plan, deadline):
    rows = []
    for context, group in split_specs(plan):
        if perf_counter() >= deadline:
            raise TimeoutError("Multi-street reference exceeded its deadline")
        profiles = [
            enumerate_reference(
                context,
                ReferenceProfile(kind),
                max_nodes=plan["max_tree_nodes"],
                deadline=deadline,
            )
            for kind in ("uniform", "increasing")
        ]
        combined = combine_targets([profile.target for profile in profiles])
        world_values = tuple(
            tuple((a + 2 * b) / 3 for a, b in zip(first, second, strict=True))
            for first, second in zip(
                profiles[0].world_action_values_bb,
                profiles[1].world_action_values_bb,
                strict=True,
            )
        )
        rows.append(
            {
                "name": context.name,
                "split": context.split,
                "street": context.street,
                "situation": "facing" if context.facing else "open",
                "group": repr(group),
                "context": context,
                "target": combined,
                "world_action_values_bb": world_values,
                "worlds": context.worlds,
                "uncertainty": [profile.uncertainty_status for profile in profiles],
                "action_se": [profile.action_standard_error_bb for profile in profiles],
            }
        )
    return rows


def run(plan, out):
    out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    report = {
        "format": plan["format"],
        "plan": plan,
        "plan_sha256": digest(plan),
        "revision": git("rev-parse", "HEAD"),
        "source_sha256": source_fingerprint(),
        "environment": environment(),
        "status": "running",
        "fits": [],
        "test": [],
        "promoted": False,
    }
    reference_deadline = perf_counter() + plan["max_reference_seconds"]
    write_json(out / "report.json", report)
    try:
        rows = _reference_rows(plan, reference_deadline)
    except Exception as error:
        report["status"] = "failed_reference"
        report["error"] = repr(error)
        report["seconds"] = perf_counter() - started
        write_json(out / "report.json", report)
        raise
    serial = [
        {
            key: value
            for key, value in row.items()
            if key not in ("context", "target", "worlds")
        }
        for row in rows
    ]
    write_json(out / "contexts.json", serial)
    torch.save(
        {split: [row["target"] for row in rows if row["split"] == split] for split in ("train", "tuning", "validation", "test")},
        out / "targets.pt",
    )
    report["contexts_sha256"] = digest(serial)
    report["worlds"] = sum(len(row["worlds"]) for row in rows)
    by_split = {split: [row for row in rows if row["split"] == split] for split in ("train", "tuning", "validation", "test")}
    if any(not row["uncertainty"] or "insufficient_worlds" in row["uncertainty"] for row in rows):
        report["status"] = "failed_uncertainty"
        write_json(out / "report.json", report)
        return report
    try:
        with deterministic_cpu():
            # Duration fitting can inspect train and the separate tuning board
            # only. Validation and sealed test never enter this phase.
            targets = {
                key: [row["target"] for row in by_split[key]]
                for key in ("train", "tuning")
            }
            deadline = perf_counter() + plan["max_fit_seconds"]
            for seed in plan["seeds"]:
                for variant in plan["variants"]:
                    measurements, checkpoints = fit_arm(variant, seed, targets, plan, deadline)
                    for measurement in measurements:
                        for split in ("train", "tuning"):
                            metric = measurement["metrics"][split]
                            metric["weighted_decision_cost_bb"] = _weighted_cost(metric, by_split[split])
                            metric["per_street_decision_cost_bb"] = _breakdown(metric, by_split[split], "street")
                            metric["per_situation_decision_cost_bb"] = _breakdown(metric, by_split[split], "situation")
                        step = measurement["duration"]
                        path = out / f"{variant}-{seed}-{step}.pt"
                        torch.save(checkpoints[step], path)
                        measurement["sha256"] = sha256(path.read_bytes()).hexdigest()
                        report["fits"].append(measurement)

            tuning_rows = [
                {
                    "variant": measurement["variant"],
                    "seed": measurement["seed"],
                    "duration": measurement["duration"],
                    "weighted_decision_cost_bb": measurement["metrics"]["tuning"]["weighted_decision_cost_bb"],
                }
                for measurement in report["fits"]
            ]
            duration_selection = choose_durations(tuning_rows, plan)
            report["duration_selection"] = duration_selection
            write_json(out / "duration-selection.json", duration_selection)

            # Reload only the selected duration for validation. The baseline
            # model for each seed is retained only long enough to compute the
            # paired policy-value uncertainty of each candidate.
            selected_models = {}
            validation_rows = []
            for variant in plan["variants"]:
                duration = duration_selection["durations"][variant]
                for seed in plan["seeds"]:
                    model = make_model(variant, seed)
                    model.load_state_dict(torch.load(out / f"{variant}-{seed}-{duration}.pt", weights_only=True))
                    selected_models[variant, seed] = model
                    metric = evaluate_rows(model, by_split["validation"])
                    validation_rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "duration": duration,
                            "weighted_decision_cost_bb": metric["weighted_decision_cost_bb"],
                            "relative_rmse": metric["relative_rmse"],
                            "metric": metric,
                        }
                    )
            baseline = plan["variants"][0]
            for row in validation_rows:
                if row["variant"] == baseline:
                    row["paired_gain_standard_error_bb"] = 0.0
                else:
                    row["paired_gain_standard_error_bb"] = _paired_gain_se(
                        selected_models[row["variant"], row["seed"]],
                        selected_models[baseline, row["seed"]],
                        by_split["validation"],
                    )
            qualification = qualify(
                [
                    {key: value for key, value in row.items() if key not in ("metric",)}
                    for row in validation_rows
                ],
                duration_selection["durations"],
                plan,
            )
            report["validation"] = validation_rows
            report["qualification"] = qualification
            write_json(out / "qualification.json", qualification)

            # Open the sealed test exactly once for baseline and all qualifiers.
            selected = [baseline, *qualification["eligible"]]
            test_models = {}
            for variant in selected:
                duration = duration_selection["durations"][variant]
                for seed in plan["seeds"]:
                    model = make_model(variant, seed)
                    model.load_state_dict(torch.load(out / f"{variant}-{seed}-{duration}.pt", weights_only=True))
                    test_models[variant, seed] = model
                    metric = evaluate_rows(model, by_split["test"])
                    metric["paired_gain_standard_error_bb"] = (
                        0.0
                        if variant == baseline
                        else _paired_gain_se(
                            model,
                            test_models[baseline, seed],
                            by_split["test"],
                        )
                    )
                    report["test"].append({"variant": variant, "seed": seed, "duration": duration, **metric})
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        report["seconds"] = perf_counter() - started
        write_json(out / "report.json", report)
        raise
    report["status"] = "completed"
    report["seconds"] = perf_counter() - started
    report["artifacts"] = {
        name: sha256((out / name).read_bytes()).hexdigest()
        for name in ["contexts.json", "targets.pt", "duration-selection.json", "qualification.json"]
    }
    report["artifacts"].update(
        {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in out.glob("*.pt")
            if path.name != "targets.pt"
        }
    )
    write_json(out / "report.json", report)
    return report


def verify(out):
    """Recheck source/artifact hashes and fresh model reload predictions."""

    out = Path(out)
    report = json.loads((out / "report.json").read_text())
    if report["status"] != "completed":
        raise ValueError("Only completed runs can be verified")
    if source_fingerprint() != report["source_sha256"]:
        raise ValueError("The recorded source fingerprint changed")
    for name, expected in report["artifacts"].items():
        if sha256((out / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact changed: {name}")
    targets = torch.load(out / "targets.pt", weights_only=False)
    serial = json.loads((out / "contexts.json").read_text())
    records = {
        split: [
            {
                "target": target,
                "street": meta["street"],
                "situation": meta["situation"],
                "group": meta["group"],
                "world_action_values_bb": meta["world_action_values_bb"],
            }
            for meta, target in zip(
                (row for row in serial if row["split"] == split),
                targets[split],
                strict=True,
            )
        ]
        for split in ("train", "tuning", "validation", "test")
    }
    for fit in report["fits"]:
        if set(fit["metrics"]) != {"train", "tuning"}:
            raise ValueError("Fit record contains a validation or test metric")
    with deterministic_cpu():
        for fit in report["fits"]:
            model = make_model(fit["variant"], fit["seed"])
            model.load_state_dict(
                torch.load(
                    out / f'{fit["variant"]}-{fit["seed"]}-{fit["duration"]}.pt',
                    weights_only=True,
                )
            )
            for split, expected in fit["metrics"].items():
                actual = evaluate_rows(model, records[split])
                if digest(actual) != digest(expected):
                    raise ValueError("Reloaded train/tuning predictions differ")

        durations = choose_durations(
            [
                {
                    "variant": fit["variant"],
                    "seed": fit["seed"],
                    "duration": fit["duration"],
                    "weighted_decision_cost_bb": fit["metrics"]["tuning"]["weighted_decision_cost_bb"],
                }
                for fit in report["fits"]
            ],
            report["plan"],
        )
        if digest(durations) != digest(report["duration_selection"]):
            raise ValueError("Duration selection does not reproduce")
        models = {}
        validation_inputs = []
        for variant in report["plan"]["variants"]:
            for seed in report["plan"]["seeds"]:
                duration = durations["durations"][variant]
                model = make_model(variant, seed)
                model.load_state_dict(torch.load(out / f"{variant}-{seed}-{duration}.pt", weights_only=True))
                models[variant, seed] = model
                metric = evaluate_rows(model, records["validation"])
                validation_inputs.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "duration": duration,
                        "weighted_decision_cost_bb": metric["weighted_decision_cost_bb"],
                        "relative_rmse": metric["relative_rmse"],
                        "metric": metric,
                    }
                )
        baseline = report["plan"]["variants"][0]
        for row in validation_inputs:
            row["paired_gain_standard_error_bb"] = 0.0 if row["variant"] == baseline else _paired_gain_se(
                models[row["variant"], row["seed"]], models[baseline, row["seed"]], records["validation"]
            )
        for actual, expected in zip(validation_inputs, report["validation"], strict=True):
            if digest(actual) != digest(expected):
                raise ValueError("Reloaded validation predictions differ")
        qualification = qualify(
            [{key: value for key, value in row.items() if key != "metric"} for row in validation_inputs],
            durations["durations"],
            report["plan"],
        )
        if digest(qualification) != digest(report["qualification"]):
            raise ValueError("Qualification does not reproduce")
        selected = [baseline, *qualification["eligible"]]
        for expected in report["test"]:
            if expected["variant"] not in selected:
                raise ValueError("Test contains an unqualified architecture")
            model = models[expected["variant"], expected["seed"]]
            actual = evaluate_rows(model, records["test"])
            actual["paired_gain_standard_error_bb"] = (
                0.0
                if expected["variant"] == baseline
                else _paired_gain_se(
                    model, models[baseline, expected["seed"]], records["test"]
                )
            )
            if digest(actual) != digest({key: value for key, value in expected.items() if key not in ("variant", "seed", "duration")}):
                raise ValueError("Reloaded sealed-test predictions differ")
    return {"status": "verified", "fits": len(report["fits"]), "artifacts": len(report["artifacts"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/holdem/multistreet-representation.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.out) if args.verify else run(json.loads(args.plan.read_text()), args.out), default=str))


if __name__ == "__main__":
    main()
