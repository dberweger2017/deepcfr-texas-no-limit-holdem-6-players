"""Complete decision diagnostics and final-only screening for frozen replay fits."""

from dataclasses import asdict
from hashlib import sha256

import numpy as np

from src.solver.evaluate import evaluate
from src.solver.experiment import canonical
from src.solver.neural.network import fitting_metrics, predict


def diagnose(solver, memory, model, iteration, oracle_value):
    probabilities = predict(model, solver.features, solver.mask, strategy=True)
    policy = solver.local_policy(probabilities)
    evaluation = evaluate(solver.tree, policy).to_dict()
    means, weights = memory.means(len(probabilities))
    counts = np.bincount(memory.infos[: memory.size], minlength=len(weights))
    errors = (probabilities - means) ** 2 * solver.mask.numpy()
    contributions = errors.sum(axis=1) * weights * 2 / (memory.size * iteration)
    rows = [
        {
            "information_set": asdict(info),
            "samples": int(counts[i]),
            "weight": float(weights[i]),
            "target": means[i].tolist() if counts[i] else None,
            "prediction": probabilities[i].tolist(),
            "action_squared_errors": errors[i].tolist() if counts[i] else None,
            "weighted_loss_contribution": float(contributions[i]),
        }
        for i, info in enumerate(solver.tree.information_sets)
    ]
    bands = {}
    for name, selected in (
        ("0", counts == 0),
        ("1-9", (counts > 0) & (counts < 10)),
        ("10-99", (counts >= 10) & (counts < 100)),
        ("100+", counts >= 100),
    ):
        entry = {
            "information_sets": int(selected.sum()),
            "samples": int(counts[selected].sum()),
            "weighted_loss_contribution": float(contributions[selected].sum()),
        }
        if name != "0":
            hybrid = probabilities.copy()
            target = means[selected]
            hybrid[selected] = target / target.sum(axis=1, keepdims=True)
            entry["hybrid_evaluation"] = evaluate(
                solver.tree, solver.local_policy(hybrid)
            ).to_dict()
        bands[name] = entry
    metrics = fitting_metrics(
        model, solver.features, solver.mask, memory, strategy=True
    )
    scale = 2 * float(weights.sum()) / (memory.size * iteration)
    return {
        "evaluation": evaluation,
        "value_error": abs(evaluation["value_player0"] - oracle_value),
        "fit": metrics,
        "full_replay_loss": metrics["empirical_mse"] * scale,
        "policy_sha256": sha256(canonical(policy.tolist()).encode()).hexdigest(),
        "bands": bands,
    }, rows


def screen(plan, reports):
    canonical(reports)
    names = [r["name"] for r in plan["recipes"]]
    seeds = [r["seed"] for r in plan["source"]["inputs"]]
    expected = {(s, n, r) for s in seeds for n in names for r in plan["fit_replicates"]}
    keys = [(r["seed"], r["recipe"], r["replicate"]) for r in reports]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Screen requires every planned fit exactly once")
    by_key = dict(zip(keys, reports))
    for r in reports:
        if (
            r["status"] != "completed"
            or not r["scored"]
            or [e["step"] for e in r["evaluations"]]
            != plan["fixed"]["evaluation_steps"]
        ):
            raise ValueError("Screen requires complete scored evaluations")
    for spec in plan["source"]["inputs"]:
        control = by_key[spec["seed"], "minibatch-fixed", 0]
        if (
            control.get("original_control_reproduced") is not True
            or control["evaluations"][-1]["policy_sha256"]
            != spec["original_control_policy_sha256"]
        ):
            raise ValueError("Original control was not reproduced")
    for seed in seeds:
        for replicate in plan["fit_replicates"]:
            peers = [by_key[seed, name, replicate]["evaluations"][-1] for name in names]
            if len({e["optimizer"]["initial_weights_sha256"] for e in peers}) != 1:
                raise ValueError("Paired initial weights differ")
            sampled = [
                by_key[seed, recipe["name"], replicate]["evaluations"][-1]
                for recipe in plan["recipes"]
                if recipe["objective"] == "sampled-original"
            ]
            if len({e["optimizer"]["minibatch_indices_sha256"] for e in sampled}) != 1:
                raise ValueError("Paired minibatch streams differ")
    limits = plan["screen"]
    summaries = []
    for recipe in plan["recipes"]:
        rows = [
            by_key[s, recipe["name"], r] for s in seeds for r in plan["fit_replicates"]
        ]
        final = [r["evaluations"][-1] for r in rows]
        values = [e["evaluation"]["exploitability"] for e in final]
        regressions = [
            e["evaluation"]["exploitability"]
            - by_key[r["seed"], "minibatch-fixed", r["replicate"]]["evaluations"][-1][
                "evaluation"
            ]["exploitability"]
            for r, e in zip(rows, final)
        ]
        passed = sum(
            e["evaluation"]["exploitability"] <= limits["maximum_exploitability"]
            and e["value_error"] <= limits["maximum_value_error"]
            for e in final
        )
        summaries.append(
            {
                "recipe": recipe["name"],
                "count": len(rows),
                "passing_fits": passed,
                "worst_exploitability": max(values),
                "mean_exploitability": sum(values) / len(values),
                "maximum_paired_regression": max(regressions),
                "eligible": recipe["eligible_for_confirmation"]
                and passed == len(rows)
                and max(regressions)
                <= limits["maximum_paired_regression_against_minibatch_fixed"],
            }
        )
    candidates = [r for r in summaries if r["eligible"]]
    chosen = (
        min(
            candidates,
            key=lambda r: (
                r["worst_exploitability"],
                r["mean_exploitability"],
                r["recipe"],
            ),
        )
        if candidates
        else None
    )
    return {
        "status": "shortlisted" if chosen else "no_candidate",
        "selected": chosen["recipe"] if chosen else None,
        "recipes": summaries,
        "model_promoted": False,
        "confirmation_started": False,
    }
