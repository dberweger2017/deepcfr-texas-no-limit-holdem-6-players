from dataclasses import replace

import pytest
import torch

from src.game.hand import Hand
from src.game.observation import replay
from src.game.types import Action, ActionKind
from src.holdem.actions import bet_candidates
from src.holdem.average import AveragePolicy, own_path
from src.holdem.policy import FrozenProfile
from src.holdem.training import HoldemTrainer
from src.solver.neural.network import deterministic_cpu
from tests.test_hand_observations import table
from tests.test_holdem_training import config


class Profile:
    capacity = 4

    def __init__(self, first, later):
        self.first, self.later = first, later

    def distribution(self, candidates):
        view = candidates.decision.source
        values = [0.0] * len(candidates.actions)
        target = ActionKind.CALL if view.street.value == "preflop" else ActionKind.CHECK
        index = next(i for i, a in enumerate(candidates.actions) if a.kind == target)
        p = self.first if view.street.value == "preflop" else self.later
        values[index] = p
        values[(index + 1) % len(values)] = 1 - p
        return tuple(values)


def flop():
    hand = Hand.start(table(4, (200,) * 4), hand_id="average-test", seed=12)
    while hand.observe(hand.actor).street.value == "preflop":
        view = hand.observe(hand.actor)
        hand = hand.apply(
            Action(
                ActionKind.CHECK
                if ActionKind.CHECK in view.legal_actions.kinds
                else ActionKind.CALL
            )
        )
    return hand


def test_average_conditions_on_own_reach_not_opponents():
    hand = flop()
    view = hand.observe(hand.actor)
    candidates, path = own_path(view)
    assert len(path) == 1
    assert path[0][0].decision.source.board == ()
    policy = AveragePolicy((Profile(0.8, 0.1), Profile(0.2, 0.9)))
    index = candidates.actions.index(Action(ActionKind.CHECK))
    assert policy.distribution(view)[index] == pytest.approx(
        (0.8 * 0.1 + 2 * 0.2 * 0.9) / (0.8 + 2 * 0.2)
    )
    assert policy.distribution(view)[index] != pytest.approx((0.1 + 2 * 0.9) / 3)


def test_zero_reach_has_explicit_uniform_completion():
    view = flop().observe(1)
    policy = AveragePolicy((Profile(0, 0.1), Profile(0, 0.9)))
    result = policy.distribution(view)
    assert result == (1 / len(result),) * len(result)


def test_own_off_menu_action_is_rejected_but_opponent_off_menu_is_supported():
    hand = Hand.start(table(4, (200,) * 4), hand_id="off-menu", seed=12)
    actor = hand.actor
    candidates = bet_candidates(hand.observe(actor))
    amount = next(
        x for x in range(5, 30) if Action(ActionKind.RAISE, x) not in candidates.actions
    )
    hand = hand.apply(Action(ActionKind.RAISE, amount))
    policy = AveragePolicy((FrozenProfile([None] * 4),))
    policy.distribution(hand.observe(hand.actor))
    while hand.actor != actor:
        legal = hand.observe(hand.actor).legal_actions.kinds
        hand = hand.apply(
            Action(ActionKind.CHECK if ActionKind.CHECK in legal else ActionKind.CALL)
        )
        if hand.finished:
            pytest.fail("Expected another decision")
    with pytest.raises(ValueError, match="outside"):
        policy.distribution(hand.observe(actor))


def test_sample_is_fixed_per_hand_and_reproducible():
    policy = AveragePolicy(tuple(FrozenProfile([None] * 4) for _ in range(3)))
    one, two = policy.player(2), policy.player(2)
    hand = flop()
    view = hand.observe(hand.actor)
    assert one.choose_action(view) == two.choose_action(view)
    iteration = one.iteration
    with pytest.raises(ValueError, match="forward"):
        one.choose_action(view)
    hand = hand.apply(Action(ActionKind.CHECK))
    while hand.actor != view.seat:
        hand = hand.apply(Action(ActionKind.CHECK))
    one.choose_action(hand.observe(hand.actor))
    assert one.iteration == iteration


def test_archive_contains_collection_profiles_and_is_transactional(monkeypatch):
    with deterministic_cpu(), torch.random.fork_rng(devices=[]):
        trainer = HoldemTrainer(table(4, (10,) * 4), config())
        with pytest.raises(ValueError, match="average"):
            trainer.average_policy()
        first = trainer.step()
        frozen = trainer.average_policy()
        second = trainer.step()
        assert frozen.fingerprints == (first.collection_profile,)
        assert trainer.average_policy().fingerprints == (
            first.collection_profile,
            second.collection_profile,
        )
        before = trainer.average_policy().fingerprints
        monkeypatch.setattr(
            "src.holdem.training.fit_role",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("failed")),
        )
        with pytest.raises(RuntimeError):
            trainer.step()
        assert trainer.average_policy().fingerprints == before


def test_distribution_depends_only_on_visible_world():
    hand = flop()
    view = hand.observe(hand.actor)
    policy = AveragePolicy((FrozenProfile([None] * 4),))
    equivalent = replay(view.history, view.seat, view.hole_cards)
    assert policy.distribution(view) == policy.distribution(equivalent)
    with pytest.raises(ValueError):
        policy.distribution(replace(view, capacity=5))
