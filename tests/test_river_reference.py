import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
import torch

from scripts.check_river_learning import control, moments, run
from src.arena.policies import CheckFold
from src.game.types import ActionKind
from src.holdem import outcome_sampling
from src.holdem.actions import bet_candidates
from src.holdem.collection import CollectionLimitExceeded
from src.holdem.outcome_sampling import collect_outcome
from src.holdem.river_reference import (
    ReferenceProfile,
    combine_targets,
    contexts,
    enumerate_reference,
    fit_reference,
    prediction_metrics,
    probabilities,
)
from src.holdem.targets import CandidateTargets
from src.solver.neural.network import deterministic_cpu
from tests.test_holdem_outcome_sampling import enumerate_samples


@pytest.fixture
def plan():
    value = json.loads(Path("configs/holdem/river-reference.json").read_text())
    value["boards"] = [{**value["boards"][0], "hands": value["boards"][0]["hands"][:1]}]
    return value


@pytest.fixture
def facing(plan):
    return contexts(plan)[1]


def test_worlds_share_hero_information_and_all_six_are_live(plan):
    for context in contexts(plan):
        one, two = context.worlds
        hero = one.actor
        assert one.observe(hero) == two.observe(hero)
        assert context.assignments[0][hero] == context.assignments[1][hero]
        assert context.assignments[0] != context.assignments[1]
        assert len(one.observe(hero).players) == 6
        assert not any(p.folded for p in one.observe(hero).players)
        for node, hands in zip(context.worlds, context.assignments):
            assert tuple(node.observe(i).hole_cards for i in range(6)) == hands
            cards = [c for hand in hands for c in hand] + list(node.observe(hero).board)
            assert len(set(cards)) == 17


def test_exact_values_integrate_worlds_and_have_correct_perspective(facing):
    profile = ReferenceProfile("uniform")
    reference = enumerate_reference(
        facing, profile, max_nodes=20000, deadline=float("inf")
    )
    singles = [
        enumerate_reference(
            replace(facing, worlds=(w,)),
            profile,
            max_nodes=20000,
            deadline=float("inf"),
        ).target
        for w in facing.worlds
    ]
    expected = np.mean([t.values_bb for t in singles], axis=0)
    assert reference.target.values_bb == pytest.approx(expected)
    assert sum(
        p * r for p, r in zip(reference.target.policy, reference.target.regrets_bb)
    ) == pytest.approx(0)
    fold = next(
        i
        for i, a in enumerate(reference.target.candidates.actions)
        if a.kind == ActionKind.FOLD
    )
    assert reference.target.values_bb[fold] == -1
    oracle = ReferenceProfile("uniform", reference.baselines)
    for world in facing.worlds:
        assert (
            oracle.action_values(bet_candidates(world.observe(world.actor)))
            == reference.target.values_bb
        )
    with pytest.raises(TypeError):
        oracle.baselines[reference.target.candidates.decision.source] = (999, 999)


@pytest.mark.parametrize("baseline", ["zero", "frozen"])
def test_production_sample_expectation_matches_hidden_world_reference(
    monkeypatch, facing, baseline
):
    reference = enumerate_reference(
        facing, ReferenceProfile("increasing"), max_nodes=20000, deadline=float("inf")
    )
    oracle = ReferenceProfile("increasing", reference.baselines)
    expected = np.zeros(len(reference.target.values_bb))
    mass = 0
    for world in facing.worlds:
        samples = enumerate_samples(
            monkeypatch,
            outcome_sampling,
            collect_outcome,
            world,
            oracle,
            world.actor,
            exploration=0.5,
            baseline=baseline,
        )
        for probability, sample in samples:
            root = next(
                d
                for d in sample.decisions
                if d.candidates.decision.source == sample.root
            )
            expected += probability / 2 * np.asarray(root.values_bb)
            mass += probability / 2
    assert mass == pytest.approx(1)
    assert expected == pytest.approx(reference.target.values_bb, abs=1e-10)


def test_baseline_cannot_change_paths_or_expanded_final_decision(facing):
    reference = enumerate_reference(
        facing, ReferenceProfile("uniform"), max_nodes=20000, deadline=float("inf")
    )
    profile = ReferenceProfile("uniform", reference.baselines)
    for world in facing.worlds:
        for seed in range(3):
            kwargs = {
                "iteration": 1,
                "action_seed": seed,
                "exploration": 0.5,
                "branch_first": True,
            }
            a = collect_outcome(world, profile, world.actor, baseline="zero", **kwargs)
            b = collect_outcome(
                world, profile, world.actor, baseline="frozen", **kwargs
            )
            assert a.executions == b.executions
            assert a.decisions[0].values_bb == b.decisions[0].values_bb


def test_full_tree_limits_fail_without_returning_partial_targets(facing):
    for kwargs in (
        {"max_nodes": 1, "deadline": float("inf")},
        {"max_nodes": 20000, "deadline": 0},
    ):
        with pytest.raises(CollectionLimitExceeded):
            enumerate_reference(facing, ReferenceProfile("uniform"), **kwargs)


def test_weighted_schedule_does_not_recenter_cumulative_regrets(facing):
    c = bet_candidates(facing.worlds[0].observe(facing.worlds[0].actor))
    a = CandidateTargets(c, (0.5, 0.5), (-1, 3), (-2, 2))
    b = CandidateTargets(c, (0.25, 0.75), (-1, 1), (-1.5, 0.5))
    combined = combine_targets([a, b])
    assert combined.regrets_bb == pytest.approx((-5 / 3, 1))
    assert combined.values_bb == pytest.approx((-1, 5 / 3))
    assert prediction_metrics([combined.regrets_bb], [combined])["regret_rmse_bb"] == 0


def test_centered_variance_is_not_gradient_or_target_magnitude(facing):
    reference = enumerate_reference(
        facing, ReferenceProfile("uniform"), max_nodes=20000, deadline=float("inf")
    )
    rows = [{"regrets_bb": [100, 200], "nodes": 10, "seconds": 0.1}] * 3
    result = moments(rows, reference)
    assert result["trace_variance"] == 0
    assert result["reference_mse"] > 0
    assert probabilities([1, 2]) == pytest.approx(probabilities([10, 20]))
    assert probabilities([-0.1, -0.2]) == pytest.approx([1, 0])


def test_check_fold_checks_when_free_and_bounds_rotated_losses(plan, tmp_path):
    context = contexts(plan)[0]
    view = context.worlds[0].observe(context.worlds[0].actor)
    assert CheckFold().choose_action(view).kind == ActionKind.CHECK
    report = control({**plan, "control_blocks": 2}, tmp_path, perf_counter() + 60)
    assert all(r["worst_block_bb_per_100"] >= -25 for r in report.values())
    assert all(r["hands"] == 12 for r in report.values())


def test_exact_and_noisy_fits_share_initialization_and_reproduce(plan, facing):
    reference = enumerate_reference(
        facing, ReferenceProfile("uniform"), max_nodes=20000, deadline=float("inf")
    )
    target = reference.target
    state = torch.get_rng_state().clone()
    with deterministic_cpu():
        args = {"seed": 9, "plan": {**plan, "fit_steps": 2}, "deadline": float("inf")}
        a, first = fit_reference([target], [target], [target], **args)
        b, second = fit_reference([target], [target], [target], **args)
    assert torch.equal(state, torch.get_rng_state())
    assert first["initial"] == second["initial"]
    assert first["final"] == second["final"]
    assert all(
        torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict()
    )


def test_runner_retains_raw_samples_models_and_all_declared_cells(plan, tmp_path):
    plan = {**plan, "seeds": [19], "replicates": 2, "fit_steps": 2, "control_blocks": 1}
    plan["boards"].append({**plan["boards"][0], "split": "validation"})
    out = tmp_path / "complete"
    report = run(plan, out)
    assert report["status"] == "completed"
    assert len(report["references"]) == 8
    assert len(report["sampling"]) == 32
    assert len(report["fits"]) == 2
    assert len((out / "samples.jsonl").read_text().splitlines()) == 64
    assert (out / "exact-19.pt").is_file()
    assert json.loads((out / "report.json").read_text())["status"] == "completed"


def test_runner_records_failure_instead_of_extending_budget(plan, tmp_path):
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="Control arena failed"):
        run({**plan, "max_seconds": 0}, out)
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["fits"] == []
    assert report["completed_context_profiles"] == 0
    assert report["expected_context_profiles"] == 4
