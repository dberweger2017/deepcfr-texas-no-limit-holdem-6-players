import json
from dataclasses import asdict, replace
from math import fsum
from pathlib import Path

import pytest

from src.arena.schedule import digest
from src.game.hand import Hand
from src.game.types import ActionKind, Street
from src.holdem import outcome_sampling
from src.holdem.collection import CollectionLimitExceeded
from src.holdem.outcome_sampling import collect_outcome
from src.holdem.policy import FrozenProfile
from src.holdem.sampled_collection import SampledCollection, split_sampled_collection
from tests.test_hand_observations import table
from tests.test_holdem_encoding import advance
from tests.test_holdem_outcome_sampling import (
    enumerate_samples,
    exact_tree,
    moments,
    river,
)


def test_original_sampling_fingerprints_are_unchanged():
    expected = json.loads(Path("tests/fixtures/branching-defaults.json").read_text())
    hand = river()
    for case in expected:
        result = collect_outcome(
            hand,
            FrozenProfile([None] * 2),
            hand.actor,
            iteration=1,
            action_seed=case["seed"],
            exploration=0.5,
            branch_first=case["branch_first"],
        )
        record = asdict(result)
        assert record.pop("branch_second") is False
        assert digest(record) == case["sha256"]


@pytest.mark.parametrize("zero_reach", [False, True])
@pytest.mark.parametrize("baseline", ["zero", "frozen"])
def test_second_expansion_preserves_all_expected_updates(
    monkeypatch, zero_reach, baseline
):
    hand = advance(
        Hand.start(table(2, (11, 11)), hand_id="deep-tree", seed=7), Street.RIVER
    )
    hero = hand.actor

    def distribution(self, candidates):
        n = len(candidates.actions)
        if candidates.decision.source.seat != hero:
            raises = [
                i
                for i, a in enumerate(candidates.actions)
                if a.kind == ActionKind.RAISE
            ]
            index = raises[0] if raises else n - 1
            return tuple(float(i == index) for i in range(n))
        if zero_reach:
            return (1.0,) + (0.0,) * (n - 1)
        return tuple((i + 1) / (n * (n + 1) / 2) for i in range(n))

    monkeypatch.setattr(FrozenProfile, "distribution", distribution)
    monkeypatch.setattr(
        FrozenProfile,
        "action_values",
        lambda self, c: tuple(-1 + 0.125 * i for i in range(len(c.actions))),
    )
    profile = FrozenProfile([None] * 2)
    value, expected = exact_tree(hand, profile, hero)
    samples = enumerate_samples(
        monkeypatch,
        outcome_sampling,
        collect_outcome,
        hand,
        profile,
        hero,
        exploration=0.5,
        baseline=baseline,
        branch_first=True,
        branch_second=True,
    )
    import torch

    from src.holdem.betting import betting_loss
    from tests.test_holdem_sampled_loss import full_tree, gradient, predictions

    _, records = full_tree(hand, profile, hero)
    parameter = torch.tensor([0.7, -0.3, 1.1], dtype=torch.float64, requires_grad=True)
    loss = sum(
        reach * betting_loss([predictions(t.candidates, parameter)], [t])
        for t, reach in records.values()
    )
    expected_gradient = torch.autograd.grad(loss, parameter)[0]
    actual_gradient = sum(p * gradient(s.decisions) for p, s in samples)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-10, rtol=0)
    mean, _ = moments(samples, True)
    assert fsum(p * s.value_bb for p, s in samples) == pytest.approx(value, abs=1e-10)
    for history, values in expected.items():
        for action, target in enumerate(values):
            assert mean.get((history, action), 0) == pytest.approx(target, abs=1e-10)
    assert any(d.sampled_action is not None for _, s in samples for d in s.decisions)
    assert any(d.own_sample_reach < 1 for _, s in samples for d in s.decisions)
    for _, sample in samples:
        initial = len(hand.observe(hero).history)
        for decision in sample.decisions:
            history = decision.candidates.decision.source.history
            count = sum(
                getattr(e, "seat", None) == hero and type(e).__name__ == "ActionTaken"
                for e in history[initial:]
            )
            assert (decision.sampled_action is None) == (count < 2)
            if count < 2:
                assert decision.own_sample_reach == 1
                assert set(decision.inclusion_probabilities) == {1.0}


def test_second_branch_requires_matching_phase_configuration():
    hand = river()
    profile = FrozenProfile([None] * 2)
    kwargs = {
        "iteration": 1,
        "action_seed": 1,
        "exploration": 0.5,
        "baseline": "frozen",
    }
    with pytest.raises(ValueError, match="requires"):
        collect_outcome(hand, profile, hand.actor, branch_second=True, **kwargs)
    result = collect_outcome(
        hand, profile, hand.actor, branch_first=True, branch_second=True, **kwargs
    )
    batch = SampledCollection(hand.table, 1, 1, 1, profile.fingerprint, (result,), 0.5)
    with pytest.raises(ValueError, match="another sampling design"):
        split_sampled_collection(batch)
    with pytest.raises(ValueError, match="requires"):
        collect_outcome(
            hand, profile, hand.actor, branch_first=True, branch_second=1, **kwargs
        )
    assert replace(result, branch_second=False).branch_first


def test_profile_selection_uses_completed_fit_indices():
    from types import SimpleNamespace

    from scripts.check_collector_branching import select_profiles

    profiles = [FrozenProfile([None] * 2) for _ in range(4)]
    # Distinct checked identities expose indexing mistakes without fitting networks.
    for index, profile in enumerate(profiles):
        object.__setattr__(profile, "fingerprint", str(index))
    reports = tuple(
        SimpleNamespace(collection_profile=str(i), fitted_profile=str(i + 1))
        for i in range(3)
    )
    trainer = SimpleNamespace(
        iteration=3,
        _state=SimpleNamespace(archive=tuple(profiles[:3])),
        reports=reports,
        current_profile=lambda: profiles[3],
    )
    from unittest.mock import patch

    with patch.object(FrozenProfile, "assert_unchanged"):
        assert select_profiles(trainer, [0, 1, 3]) == {
            0: profiles[0],
            1: profiles[1],
            3: profiles[3],
        }
        reports[0].fitted_profile = "wrong"
        with pytest.raises(ValueError, match="completed fit"):
            select_profiles(trainer, [1])
        with pytest.raises(ValueError, match="outside"):
            select_profiles(trainer, [4])


def test_study_runs_and_verifies_a_small_complete_schedule(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from scripts import check_collector_branching as study
    from src.holdem.critic import CriticConfig, PersistentCritic
    from src.holdem.river_reference import ReferenceProfile

    plan = json.loads(Path("configs/holdem/collector-branching.json").read_text())
    river_plan = json.loads(Path(plan["river_plan"]).read_text())
    river_plan["boards"] = [
        {**river_plan["boards"][0], "hands": river_plan["boards"][0]["hands"][:1]}
    ]
    (tmp_path / "river.json").write_text(json.dumps(river_plan))
    (tmp_path / "manifest.json").write_text("{}")
    plan.update(
        sampling_seeds=[29],
        river_replicates=2,
        full_replicates=2,
        full_roots_per_street=1,
        coverage_iterations=[0, 256],
        coverage_blocks=1,
        river_plan=str(tmp_path / "river.json"),
        historical_manifest=str(tmp_path / "manifest.json"),
    )
    profile = FrozenProfile([None] * 6)
    reports = tuple(
        SimpleNamespace(
            collection_profile=profile.fingerprint, fitted_profile=profile.fingerprint
        )
        for _ in range(256)
    )
    monkeypatch.setattr(
        study,
        "load_training",
        lambda *a, **k: SimpleNamespace(
            iteration=256,
            config=SimpleNamespace(seed=307),
            current_profile=lambda: profile,
            _state=SimpleNamespace(archive=(profile,) * 256),
            reports=reports,
        ),
    )

    def small_root(seed, *, street=None, button=0):
        root = Hand.start(
            table(6, (4,) * 6, button), hand_id="coverage-test", seed=seed
        )
        return root if street is None else advance(root, street)

    monkeypatch.setattr(study, "full_root", small_root)

    def critic(path, fingerprint):
        result = PersistentCritic(CriticConfig(31, width=4))
        policy = (
            ReferenceProfile("increasing")
            if "river-reference" in str(path)
            else profile
        )
        result.begin_phase(policy, "test")
        return result

    monkeypatch.setattr(study.PersistentCritic, "load", critic)
    report = study.run(plan, tmp_path / "run")
    assert report["status"] == "completed"
    assert len(report["sampling"]) == report["expected_cells"] == 44
    assert len(report["coverage"]) == 48
    verified = study.verify(tmp_path / "run")
    assert verified["verified"] and verified["coverage_cases"] == 72
    rows = [
        json.loads(line)
        for line in (tmp_path / "run/coverage.jsonl").read_text().splitlines()
    ]
    assert {r["position"] for r in rows} == set(study.POSITIONS)
    assert all(
        sum(r[mode]["net_chips"]) == 0 for r in rows for mode in ("selfplay", "styles")
    )
    with pytest.raises(CollectionLimitExceeded, match="time budget"):
        study.run({**plan, "max_seconds": 0}, tmp_path / "failed")
    assert (
        json.loads((tmp_path / "failed/report.json").read_text())["status"] == "failed"
    )


def test_cost_screen_requires_both_cost_metrics_and_all_streets():
    from scripts.check_collector_branching import summarize_sampling

    plan = {"sampling_seeds": [1], "required_ratio": 0.75, "maximum_street_ratio": 1.25}
    cells = [
        {
            "seed": 1,
            "suite": "full-" + street,
            "arm": "historical",
            "depth": depth,
            "variance_times_nodes": float(3 - depth),
            "variance_times_seconds": float(3 - depth),
        }
        for street in ("flop", "turn", "river")
        for depth in (1, 2)
    ]
    assert summarize_sampling(plan, cells)["pass"]
    cells[1]["variance_times_seconds"] = 4
    assert not summarize_sampling(plan, cells)["pass"]
    for c in cells:
        c["variance_times_nodes"] = c["variance_times_seconds"] = 0.0
    assert not summarize_sampling(plan, cells)["pass"]
