import gzip
import json
from dataclasses import replace
from math import log, sqrt
from statistics import stdev

import pytest
import torch
from scipy.stats import t

from scripts.compare_holdem_policies import (
    evaluate_pair,
    load_refits,
    simultaneous_interval,
)
from src.arena.schedule import Plan, Scenario, build_schedule
from src.game.hand import Hand
from src.holdem.average import AveragePolicy
from src.holdem.betting import ActionScores, BettingNetwork
from src.holdem.policy import FrozenProfile
from src.holdem.policy_diagnostics import (
    policy_change,
    replay_policy_changes,
    summarize_changes,
)
from src.solver.neural.network import deterministic_cpu
from tests.test_hand_observations import DECK, table
from tests.test_holdem_frozen_fitting import memory  # noqa: F401
from tests.test_holdem_replay import example


@pytest.fixture(autouse=True)
def cpu():
    with deterministic_cpu(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(613)
        yield


def scores(regrets):
    candidates = example(1).target.candidates
    assert len(candidates.actions) == len(regrets)
    return ActionScores(
        candidates, torch.tensor(regrets, dtype=torch.float), torch.zeros(len(regrets))
    )


def test_policy_distance_ignores_positive_scaling_but_detects_sign_changes():
    change = policy_change(scores([40, 10, -20]), scores([20, 5, -10]))
    assert change["tv"] == 0
    assert change["entropy_nats"] == pytest.approx(-0.8 * log(0.8) - 0.2 * log(0.2))
    assert sum(change["action_mass"].values()) == pytest.approx(1)
    assert not change["fallback"]
    swapped = policy_change(scores([0.1, -0.1, 0]), scores([-0.1, 0.1, 0]))
    assert swapped["tv"] == 1
    assert swapped["entropy_nats"] == 0
    for values in ([0, 0, 0], [-1, -2, -3]):
        result = policy_change(scores([1, 0, 0]), scores(values))
        assert result["fallback"] and result["tv"] == 0
    summary = summarize_changes([change, swapped])
    assert summary["records"] == 2 and summary["mean_tv"] == 0.5
    assert summary["tv_above_010_fraction"] == 0.5


def test_policy_diagnostics_reject_mismatched_decisions_and_nonfinite_predictions():
    first = scores([1, 2, 3])
    candidate = first.candidates
    altered = replace(
        candidate,
        decision=replace(
            candidate.decision,
            source=replace(candidate.decision.source, hand_id="different"),
        ),
    )
    with pytest.raises(ValueError, match="same public"):
        policy_change(first, replace(first, candidates=altered))
    with pytest.raises(ValueError, match="finite"):
        policy_change(first, scores([1, float("nan"), 3]))


def test_replay_control_has_zero_distance_and_street_counts_reconcile(memory):  # noqa: F811
    model = BettingNetwork(8).eval()
    result = replay_policy_changes(
        memory, {"control": model, "same": model}, control="control"
    )
    assert result["control"] == result["same"]
    assert result["control"]["all"]["mean_tv"] == 0
    assert sum(v["records"] for v in result["control"]["streets"].values()) == len(
        memory
    )
    assert sum(result["control"]["all"]["mean_action_mass"].values()) == pytest.approx(
        1
    )
    with pytest.raises(TimeoutError):
        replay_policy_changes(memory, {"control": model}, control="control", deadline=0)


@pytest.mark.parametrize("players", [4, 5, 6])
def test_current_player_matches_one_component_average_and_preserves_profile(players):
    profile = FrozenProfile([BettingNetwork(8) for _ in range(players)])
    current = [profile.player(i) for i in range(players)]
    average = [AveragePolicy((profile,)).player(i) for i in range(players)]
    for deal in range(5):
        hand = Hand.start(table(players), hand_id=f"game-{deal}", seed=deal)
        while not hand.finished:
            view = hand.observe(hand.actor)
            action = current[hand.actor].choose_action(view)
            assert action == average[hand.actor].choose_action(view)
            view.legal_actions.validate(action)
            hand = hand.apply(action)
    profile.assert_unchanged()


def test_current_player_uses_only_owner_observation():
    hand = Hand.from_deck(table(6), hand_id="private", deck=DECK)
    view = hand.observe(hand.actor)
    positions = [i for i, card in enumerate(DECK) if card not in view.hole_cards]
    alternate = list(DECK)
    for i, j in zip(positions, reversed(positions)):
        alternate[i] = DECK[j]
    other = Hand.from_deck(table(6), hand_id="private", deck=tuple(alternate))
    profile = FrozenProfile([BettingNetwork(8) for _ in range(6)])
    for seed in range(20):
        assert profile.player(seed).choose_action(view) == profile.player(
            seed
        ).choose_action(other.observe(other.actor))


def test_arena_pair_reproduces_outcomes_and_preserves_historical_average(tmp_path):
    original = FrozenProfile([BettingNetwork(8) for _ in range(4)])
    average = AveragePolicy((FrozenProfile([None] * 4), original))
    fingerprints = average.fingerprints
    plan = Plan(
        (Scenario("test", (20,) * 4, small_blind=1, big_blind=2, chip_unit="1"),),
        candidate="current",
        baseline="average",
        opponents=("check_call",),
        blocks=2,
    )
    policies = {"current": original, "average": average}
    first = evaluate_pair(
        plan,
        policies,
        tmp_path / "a",
        deadline=float("inf"),
        family_size=12,
        material_bb100=100,
    )
    second = evaluate_pair(
        plan,
        policies,
        tmp_path / "b",
        deadline=float("inf"),
        family_size=12,
        material_bb100=100,
    )
    assert first["status"] == second["status"] == "valid"
    assert first["outcome_file"]["sha256"] == second["outcome_file"]["sha256"]
    assert first["outcomes_sha256"] == second["outcomes_sha256"]
    assert first["completed_hands"] == 16
    assert average.fingerprints == fingerprints
    assert build_schedule(plan) == build_schedule(
        replace(plan, candidate="other", baseline="control")
    )


def test_arena_failure_retains_partial_trace(tmp_path):
    profile = FrozenProfile([None] * 4)
    plan = Plan(
        (Scenario("test", (20,) * 4, small_blind=1, big_blind=2),),
        candidate="current",
        baseline="current",
        blocks=2,
    )
    with pytest.raises(RuntimeError, match="Invalid comparison"):
        evaluate_pair(
            plan,
            {"current": profile},
            tmp_path / "failed",
            deadline=0,
            family_size=12,
            material_bb100=100,
        )
    report = json.loads((tmp_path / "failed/report.json").read_text())
    assert report["status"] == "invalid" and report["failed_hands"] == 1
    with gzip.open(tmp_path / "failed/outcomes.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 1 and "TimeoutError" in rows[0]["error"]
    assert rows[0]["events"]


def test_family_interval_counts_blocks_and_widens_for_multiple_comparisons():
    plan = Plan((Scenario("test", (20,) * 4),), blocks=40)
    values = [(-1) ** i * (i + 1) for i in range(40)]
    rows = [
        {
            "block": i,
            "arm": arm,
            "candidate_chips": value if arm == "candidate" else 0,
            "big_blind": 100,
        }
        for i, value in enumerate(values)
        for _ in range(4)
        for arm in ("candidate", "baseline")
    ]
    one = simultaneous_interval(plan, rows, comparisons=1, material_bb100=100)
    family = simultaneous_interval(plan, rows, comparisons=12, material_bb100=100)
    width = family["ci_family95"][1] - family["ci_family95"][0]
    assert width == pytest.approx(
        2 * t.ppf(1 - 0.05 / 24, 39) * stdev(values) / sqrt(40)
    )
    assert family["ci_family95"][0] < one["ci_family95"][0]
    assert family["within_material_margin"]
    assert not family["candidate_better"] and not family["baseline_better"]


def test_refit_loader_rejects_missing_roles_and_wrong_hashes(tmp_path):
    with pytest.raises(ValueError, match="six roles"):
        load_refits({"fits": []}, tmp_path, 8)
    path = tmp_path / "weights.pt"
    path.write_bytes(b"wrong weights")
    job = {
        "fits": [
            {
                "gradient_clip": 1.0,
                "steps": 64,
                "role": r,
                "weights": {"path": path.name, "sha256": "0" * 64},
            }
            for r in range(6)
        ]
    }
    with pytest.raises(ValueError, match="checksum"):
        load_refits(job, tmp_path, 8)
