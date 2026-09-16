import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from random import Random

import pytest
import torch

from src.game.types import ActionKind
from src.holdem.actions import bet_candidates
from src.holdem.collection import CollectionLimitExceeded
from src.holdem.critic import (
    BaselineProfile,
    CriticConfig,
    PersistentCritic,
    ValueRecord,
    model_digest,
    monte_carlo_records,
)
from src.holdem.outcome_sampling import collect_outcome
from src.holdem.river_reference import ReferenceProfile, contexts
from src.solver.neural.network import deterministic_cpu


@pytest.fixture(autouse=True)
def cpu():
    with deterministic_cpu():
        yield


@pytest.fixture
def roots():
    plan = json.loads(Path("configs/holdem/river-reference.json").read_text())
    plan["boards"] = [plan["boards"][0]]
    return contexts(plan)


def fitted(root):
    critic = PersistentCritic(
        CriticConfig(11, width=8, batch_size=4, replay_capacity=16)
    )
    profile = ReferenceProfile("uniform")
    critic.begin_phase(profile, "test-river")
    for _ in range(4):
        critic.collect(root, profile, deadline=float("inf"), max_nodes=100)
    critic.fit(3, deadline=float("inf"))
    return critic, profile


def test_returns_belong_to_each_actor_and_use_initial_stack(roots):
    root = roots[0].worlds[0]

    class AlwaysCall(ReferenceProfile):
        def distribution(self, candidates):
            index = next(
                i
                for i, a in enumerate(candidates.actions)
                if a.kind in (ActionKind.CHECK, ActionKind.CALL)
            )
            return tuple(float(i == index) for i in range(len(candidates.actions)))

    class RootRaise(Random):
        def randrange(self, n):
            return n - 1

    records = monte_carlo_records(
        root, AlwaysCall("uniform"), RootRaise(2), deadline=float("inf"), max_nodes=100
    )
    hand = root
    for record in records:
        assert record.candidates.decision.source == hand.observe(hand.actor)
        hand = hand.apply(record.candidates.actions[record.action])
    assert hand.finished
    for record in records:
        seat = record.candidates.decision.source.seat
        assert record.value_bb == (hand.events[-1].stacks[seat] - 4) / 2
    assert len({r.value_bb for r in records}) > 1
    assert sum(r.value_bb for r in records) == 0


def test_replay_clears_on_profile_or_distribution_change_but_adam_persists(roots):
    critic, profile = fitted(roots[0].worlds[0])
    model = model_digest(critic.model)
    steps = next(iter(critic.optimizer.state.values()))["step"].item()
    replay = list(critic.replay)
    critic.begin_phase(profile, "test-river")
    assert critic.replay == replay
    critic.begin_phase(ReferenceProfile("increasing"), "test-river")
    assert critic.replay == [] and critic.cursor == 0
    assert model_digest(critic.model) == model
    assert next(iter(critic.optimizer.state.values()))["step"].item() == steps
    with pytest.raises(ValueError, match="continuation"):
        critic.collect(
            roots[0].worlds[0], profile, deadline=float("inf"), max_nodes=100
        )
    critic.begin_phase(profile, "another-distribution")
    assert critic.phases == 4


def test_frozen_predictions_ignore_hidden_world_and_later_training(roots):
    critic, policy = fitted(roots[0].worlds[0])
    baseline = BaselineProfile(policy, kind="learned", critic=critic)
    candidates = [bet_candidates(w.observe(w.actor)) for w in roots[0].worlds]
    before = baseline.action_values(candidates[0])
    assert before == baseline.action_values(candidates[1])
    critic.fit(2, deadline=float("inf"))
    assert baseline.action_values(candidates[0]) == before
    baseline.assert_unchanged()
    with torch.no_grad():
        next(baseline.model.parameters()).add_(1)
    with pytest.raises(RuntimeError, match="changed"):
        baseline.assert_unchanged()


@pytest.mark.parametrize("context_index", [0, 1])
def test_baseline_preserves_paths_and_final_expanded_values(roots, context_index):
    root = roots[context_index].worlds[0]
    critic, policy = fitted(root)
    results = []
    for kind in ("zero", "accounting", "historical", "learned"):
        baseline = BaselineProfile(
            policy,
            kind=kind,
            critic=critic if kind == "learned" else None,
            historical=policy if kind == "historical" else None,
        )
        if kind == "historical":
            # The reference profile has no Q head; a uniform production profile does.
            from src.holdem.policy import FrozenProfile

            baseline = BaselineProfile(
                policy, kind=kind, historical=FrozenProfile([None] * 6)
            )
        result = collect_outcome(
            root,
            baseline,
            root.actor,
            iteration=1,
            action_seed=17,
            exploration=0.5,
            baseline="frozen",
            branch_first=True,
        )
        results.append(result)
        c = bet_candidates(root.observe(root.actor))
        assert baseline.distribution(c) == policy.distribution(c)
        if kind == "accounting":
            assert baseline.action_values(c) == (-1.0,) * len(c.actions)
    assert all(
        r.executions == results[0].executions and r.nodes == results[0].nodes
        for r in results
    )
    if context_index == 1:
        assert all(r.value_bb == results[0].value_bb for r in results)


def test_recovery_preserves_fit_replay_random_and_optimizer_in_fresh_process(
    roots, tmp_path
):
    critic, profile = fitted(roots[0].worlds[0])
    path = tmp_path / "phase.pt"
    digest = critic.save(path)
    restored = PersistentCritic.load(path, digest)
    assert restored.replay == critic.replay and restored.cursor == critic.cursor
    assert restored.identity == critic.identity
    for instance in (critic, restored):
        instance.collect(
            roots[1].worlds[1], profile, deadline=float("inf"), max_nodes=100
        )
        instance.fit(4, deadline=float("inf"))
    assert model_digest(critic.model) == model_digest(restored.model)
    assert critic.random.getstate() == restored.random.getstate()
    code = """
import sys
from pathlib import Path
from src.holdem.critic import PersistentCritic, model_digest
from src.solver.neural.network import deterministic_cpu
with deterministic_cpu():
    critic = PersistentCritic.load(Path(sys.argv[1]), sys.argv[2])
    critic.fit(4, deadline=float("inf"))
    print(model_digest(critic.model))
"""
    original = PersistentCritic.load(path, digest)
    original.fit(4, deadline=float("inf"))
    output = subprocess.check_output(
        [sys.executable, "-c", code, str(path), digest],
        text=True,
    )
    assert output.strip() == model_digest(original.model)
    with pytest.raises(FileExistsError):
        original.save(path)
    with pytest.raises(ValueError, match="hash"):
        PersistentCritic.load(path, "bad")


def test_failed_fit_requires_recovery(roots, tmp_path):
    critic, _ = fitted(roots[0].worlds[0])
    with pytest.raises(CollectionLimitExceeded):
        critic.fit(10, deadline=0)
    with pytest.raises(RuntimeError, match="failed"):
        critic.save(tmp_path / "bad.pt")


def test_failed_rollout_publishes_no_partial_records(roots):
    critic = PersistentCritic(CriticConfig(13))
    profile = ReferenceProfile("uniform")
    critic.begin_phase(profile, "test")
    with pytest.raises(CollectionLimitExceeded):
        critic.collect(roots[0].worlds[0], profile, deadline=0, max_nodes=1)
    assert not critic.replay and critic.rollouts == 0


def test_circular_replay_retains_latest_records(roots):
    critic = PersistentCritic(CriticConfig(11, replay_capacity=2))
    profile = ReferenceProfile("uniform")
    critic.begin_phase(profile, "test")
    expected_random = Random(11)
    expected = []
    for root in (roots[0].worlds[0], roots[1].worlds[1]):
        expected.extend(
            monte_carlo_records(
                root, profile, expected_random, deadline=float("inf"), max_nodes=100
            )
        )
        critic.collect(root, profile, deadline=float("inf"), max_nodes=100)
    ordered = critic.replay[critic.cursor :] + critic.replay[: critic.cursor]
    assert ordered == expected[-2:]


def test_invalid_records_and_configuration_fail(roots):
    with pytest.raises(ValueError):
        CriticConfig(1, learning_rate=float("nan"))
    with pytest.raises(ValueError):
        CriticConfig(1, replay_capacity=0)
    c = bet_candidates(roots[0].worlds[0].observe(1))
    with pytest.raises(ValueError):
        ValueRecord(c, len(c.actions), 0)
    with pytest.raises(ValueError):
        ValueRecord(c, 0, float("nan"))
    with pytest.raises(ValueError):
        replace(CriticConfig(1), batch_size=False)


def test_study_runs_all_arms_and_retains_failure(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from scripts import check_persistent_critic as study
    from src.holdem.policy import FrozenProfile

    plan = json.loads(Path("configs/holdem/persistent-critic.json").read_text())
    river = json.loads(Path(plan["river_plan"]).read_text())
    river["boards"] = [
        {**b, "hands": b["hands"][:1]} for b in (river["boards"][0], river["boards"][2])
    ]
    (tmp_path / "river.json").write_text(json.dumps(river))
    (tmp_path / "manifest.json").write_text("{}")
    plan.update(
        seeds=[19],
        width=4,
        fit_steps_per_phase=1,
        river_rollouts_per_phase=2,
        full_rollouts_per_phase=2,
        full_evaluation_roots_per_street=1,
        replicates=2,
        coverage_hands=2,
        replay_capacity=32,
        batch_size=2,
        historical_manifest=str(tmp_path / "manifest.json"),
        river_plan=str(tmp_path / "river.json"),
    )
    profile = FrozenProfile([None] * 6)
    monkeypatch.setattr(
        study,
        "load_training",
        lambda *a, **k: SimpleNamespace(
            iteration=256,
            config=SimpleNamespace(seed=307),
            current_profile=lambda: profile,
        ),
    )
    report = study.run(plan, tmp_path / "success")
    assert report["status"] == "completed"
    assert len(report["fits"]) == report["expected_fits"] == 4
    assert len(report["sampling"]) == report["expected_sampling_cells"] == 32
    assert all(p["recovery_verified"] for f in report["fits"] for p in f["phases"])
    assert len(report["artifacts"]) == 11
    assert report["coverage"]["hands_per_distribution"] == 2
    with pytest.raises(CollectionLimitExceeded):
        study.run({**plan, "max_seconds": 0}, tmp_path / "failed")
    failure = json.loads((tmp_path / "failed/report.json").read_text())
    assert failure["status"] == "failed" and failure["error"]


def test_cost_screen_includes_training_cost_and_street_veto():
    from scripts.check_persistent_critic import cost_screen

    plan = {
        "seeds": [1],
        "amortization_traversals": 100,
        "maximum_street_ratio": 1.25,
        "required_cost_variance_ratio": 0.75,
    }
    fits, rows = [], []
    for street in ("flop", "turn", "river"):
        suite = "full-" + street
        fits.append({"suite": suite, "seed": 1, "training_seconds": 10})
        for arm in ("zero", "accounting", "historical", "learned"):
            rows.append(
                {
                    "suite": suite,
                    "seed": 1,
                    "arm": arm,
                    "trace_variance": 1,
                    "variance_times_seconds": 0.5 if arm == "learned" else 1,
                }
            )
    report = cost_screen(plan, fits, rows)
    assert report["pass"]
    assert report["seeds"][0]["aggregate"][0]["ratio"] == pytest.approx(0.6)
    assert report["seeds"][0]["streets"][0]["break_even_traversals"] == 20
    fits[0]["training_seconds"] = 100
    assert not cost_screen(plan, fits, rows)["pass"]


def test_cannot_freeze_a_failed_or_wrong_profile_critic(roots):
    critic, profile = fitted(roots[0].worlds[0])
    with pytest.raises(ValueError, match="policy profile"):
        BaselineProfile(ReferenceProfile("increasing"), kind="learned", critic=critic)
    critic.failed = True
    with pytest.raises(RuntimeError, match="failed"):
        BaselineProfile(profile, kind="learned", critic=critic)
