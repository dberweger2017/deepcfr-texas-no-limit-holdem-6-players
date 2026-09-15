import json
from copy import deepcopy
from time import perf_counter

import numpy as np
import pytest
import torch

from scripts.fitting.data import PLAN
from scripts.fitting.optimizer import grouped_objective, model_hash, optimize, rate_at
from scripts.fitting.report import diagnose, screen
from src.solver.neural.memory import Reservoir
from src.solver.neural.network import deterministic_cpu, fit, new_network, weighted_loss
from src.solver.neural.solver import Config, DeepCFR
from src.solver.tree import GameTree


def problem():
    solver = DeepCFR(GameTree("leduc"), Config(hidden=8, capacity=64, seed=123))
    memory = Reservoir(64, 19)
    for i in range(24):
        info = i % 6
        legal = solver.mask[info].numpy()
        target = (
            np.array([1.0, 2.0, 3.0]) if i % 2 else np.array([3.0, 1.0, 2.0])
        ) * legal
        memory.add(info, i % 4 + 1, target / target.sum())
    return solver, memory


@pytest.mark.parametrize("dtype,atol", [(torch.float64, 1e-10), (torch.float32, 1e-5)])
def test_grouping_preserves_loss_and_parameter_gradients(dtype, atol):
    solver, memory = problem()
    with deterministic_cpu():
        model = new_network(8, 23).to(dtype)
        logits = model(solver.features.to(dtype))
        p = torch.softmax(logits.masked_fill(~solver.mask, -torch.inf), dim=-1)
        ids = torch.from_numpy(memory.infos[: memory.size])
        times = torch.from_numpy(memory.iterations[: memory.size]).to(dtype)
        targets = torch.from_numpy(memory.targets[: memory.size]).to(dtype)
        means, weights = memory.means(len(solver.features))
        means, weights = (
            torch.tensor(means, dtype=dtype),
            torch.tensor(weights, dtype=dtype),
        )
        raw = weighted_loss(p[ids], targets, solver.mask[ids], times, 4)
        grouped = grouped_objective(p, means, weights, solver.mask, memory.size, 4)
        noise = weighted_loss(means[ids], targets, solver.mask[ids], times, 4)
        assert torch.allclose(raw, grouped + noise, atol=atol, rtol=0)
        a = torch.autograd.grad(raw, tuple(model.parameters()), retain_graph=True)
        b = torch.autograd.grad(grouped, tuple(model.parameters()))
        assert all(torch.allclose(x, y, atol=atol, rtol=0) for x, y in zip(a, b))
        assert (weights[6:] == 0).all()


def test_control_matches_original_and_all_recipes_are_paired():
    solver, memory = problem()
    fixed = {
        **json.loads(PLAN.read_text())["fixed"],
        "strategy_hidden": 8,
        "steps": 8,
        "evaluation_steps": [2, 8],
        "batch_size": 8,
    }
    before = [
        x.copy()
        for x in (
            memory.infos[: memory.size],
            memory.targets[: memory.size],
            memory.iterations[: memory.size],
        )
    ]
    advantages = [model_hash(m) for m in solver.advantages]
    reports = {}
    with deterministic_cpu():
        original, _ = fit(
            memory,
            solver.features,
            solver.mask,
            hidden=8,
            steps=8,
            batch_size=8,
            learning_rate=0.001,
            iteration=4,
            seed=31,
            strategy=True,
        )
        for recipe in json.loads(PLAN.read_text())["recipes"]:
            observations = []

            def observe(step, model, metrics, observations=observations):
                diagnose(solver, memory, model, 4, 0)
                observations.append((step, metrics))

            model = optimize(
                solver, memory, recipe, fixed, 31, 4, perf_counter() + 30, observe
            )
            if recipe["name"] == "minibatch-fixed":
                assert model_hash(model) == model_hash(original)
            if recipe["name"] == "minibatch-decay":
                integrated, metrics = fit(
                    memory,
                    solver.features,
                    solver.mask,
                    hidden=8,
                    steps=8,
                    batch_size=8,
                    learning_rate=0.001,
                    iteration=4,
                    seed=31,
                    strategy=True,
                    learning_rate_schedule="cosine",
                    final_learning_rate=0.00001,
                )
                assert model_hash(model) == model_hash(integrated)
                assert metrics["learning_rate"] == {
                    "schedule": "cosine",
                    "first": 0.001,
                    "last": 0.00001,
                }
            reports[recipe["name"]] = observations
    assert [m for m in advantages] == [model_hash(m) for m in solver.advantages]
    assert all(
        np.array_equal(a, b)
        for a, b in zip(
            before,
            (
                memory.infos[: memory.size],
                memory.targets[: memory.size],
                memory.iterations[: memory.size],
            ),
        )
    )
    assert (
        len({rows[-1][1]["initial_weights_sha256"] for rows in reports.values()}) == 1
    )
    assert (
        reports["minibatch-fixed"][-1][1]["minibatch_indices_sha256"]
        == reports["minibatch-decay"][-1][1]["minibatch_indices_sha256"]
    )
    decay = json.loads(PLAN.read_text())["recipes"][1]
    assert rate_at(decay, 0, 48000) == 0.001
    assert rate_at(decay, 47999, 48000) == 0.00001
    assert rate_at(decay, 24000, 48000) < 0.001


def test_diagnostics_keep_absent_targets_unknown_and_hybrids_legal():
    solver, memory = problem()
    with deterministic_cpu():
        result, rows = diagnose(solver, memory, new_network(8, 21), 4, 0)
    assert len(rows) == 288
    assert all(
        r["target"] is None
        and r["action_squared_errors"] is None
        and r["weighted_loss_contribution"] == 0
        for r in rows
        if not r["samples"]
    )
    assert "hybrid_evaluation" not in result["bands"]["0"]
    assert sum(b["information_sets"] for b in result["bands"].values()) == 288
    assert all(
        np.isfinite(b["hybrid_evaluation"]["exploitability"])
        for n, b in result["bands"].items()
        if n != "0"
    )


def fake_reports(plan):
    return [
        {
            "status": "completed",
            "scored": True,
            "original_control_reproduced": True,
            "seed": s["seed"],
            "recipe": recipe["name"],
            "replicate": rep,
            "evaluations": [
                {
                    "step": step,
                    "policy_sha256": s["original_control_policy_sha256"],
                    "evaluation": {"exploitability": 0.1},
                    "value_error": 0.01,
                    "optimizer": {
                        "initial_weights_sha256": "same",
                        "minibatch_indices_sha256": "same",
                    },
                }
                for step in plan["fixed"]["evaluation_steps"]
            ],
        }
        for s in plan["source"]["inputs"]
        for recipe in plan["recipes"]
        for rep in plan["fit_replicates"]
    ]


def test_screen_retains_every_replicate_and_excludes_diagnostic_recipes():
    plan = json.loads(PLAN.read_text())
    rows = fake_reports(plan)
    assert screen(plan, rows)["selected"] == "minibatch-decay"
    for row in rows:
        if (
            row["recipe"].startswith("minibatch")
            and row["seed"] == 211
            and row["replicate"] == 2
        ):
            row["evaluations"][-1]["evaluation"]["exploitability"] = 0.16
    assert screen(plan, rows)["selected"] is None
    with pytest.raises(ValueError, match="every planned"):
        screen(plan, rows[:-1])
    with pytest.raises(ValueError, match="every planned"):
        screen(plan, rows + rows[:1])
    bad = deepcopy(rows)
    bad[0]["evaluations"].pop()
    with pytest.raises(ValueError, match="complete"):
        screen(plan, bad)
    bad = deepcopy(rows)
    bad[0]["evaluations"][-1]["optimizer"]["initial_weights_sha256"] = "different"
    with pytest.raises(ValueError, match="initial weights"):
        screen(plan, bad)
    bad = deepcopy(rows)
    bad[0]["evaluations"][-1]["evaluation"]["exploitability"] = float("nan")
    with pytest.raises(ValueError):
        screen(plan, bad)
    bad = deepcopy(rows)
    bad[0]["original_control_reproduced"] = False
    with pytest.raises(ValueError, match="Original control"):
        screen(plan, bad)
    bad = deepcopy(rows)
    bad[0]["seed"] = 401
    with pytest.raises(ValueError, match="every planned"):
        screen(plan, bad)


def test_deadline_interrupts_fitting():
    solver, memory = problem()
    plan = json.loads(PLAN.read_text())
    with deterministic_cpu(), pytest.raises(TimeoutError):
        optimize(
            solver,
            memory,
            plan["recipes"][0],
            plan["fixed"],
            31,
            4,
            perf_counter() - 1,
            lambda *a: None,
        )
