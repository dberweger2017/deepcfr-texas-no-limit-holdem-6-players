import random
from dataclasses import replace
from fractions import Fraction
from math import fsum

import numpy as np
import pytest
import torch

from src.game.hand import Hand
from src.game.types import Action, ActionKind, Street, TableSeat
from src.holdem import collection
from src.holdem.actions import bet_candidates
from src.holdem.betting import BettingNetwork
from src.holdem.collection import (
    CollectionLimitExceeded,
    collect_phase,
    collect_traversal,
    collection_seed,
)
from src.holdem.policy import FrozenProfile
from src.solver.neural.network import deterministic_cpu
from tests.test_hand_observations import table
from tests.test_holdem_encoding import advance
from tests.test_holdem_policy import folding_model


@pytest.fixture(autouse=True)
def cpu():
    with deterministic_cpu(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(827)
        yield


def exact_value(hand, traverser):
    if hand.finished:
        return Fraction(
            hand.events[-1].stacks[traverser] - hand.table.stacks[traverser]
        )
    actions = bet_candidates(hand.observe(hand.actor)).actions
    return sum(exact_value(hand.apply(a), traverser) for a in actions) / len(actions)


def test_sampling_expectation_matches_an_exhaustive_river_tree(monkeypatch):
    hand = advance(Hand.start(table(2, (3, 3)), hand_id="exact", seed=7), Street.RIVER)
    traverser = hand.actor
    candidates = bet_candidates(hand.observe(traverser))
    profile = FrozenProfile([None] * 2)
    exact = tuple(
        exact_value(hand.apply(a), traverser) / hand.table.big_blind
        for a in candidates.actions
    )

    class NeedChoice(Exception):
        def __init__(self, probabilities):
            self.probabilities = probabilities

    class BranchRandom:
        def __init__(self, seed):
            self.offset = 0

        def choices(self, population, weights, k):
            assert k == 1
            if self.offset == len(script):
                raise NeedChoice(tuple(weights))
            index = script[self.offset]
            self.offset += 1
            return [population[index]]

    monkeypatch.setattr(collection, "Random", BranchRandom)
    pending, completed = [((), 1.0)], []
    while pending:
        script, weight = pending.pop()
        try:
            result = collect_traversal(
                hand, profile, traverser, iteration=1, action_seed=0
            )
        except NeedChoice as choice:
            pending.extend(
                (script + (i,), weight * p)
                for i, p in enumerate(choice.probabilities)
                if p > 0
            )
        else:
            root_target = next(
                t
                for t in result.targets
                if t.candidates.decision.source.history == hand.events
            )
            completed.append((weight, result.value_bb, root_target))
    assert len(completed) == 6
    assert fsum(w for w, _, _ in completed) == pytest.approx(1)
    expected_baseline = sum(exact) / len(exact)
    assert fsum(w * v for w, v, _ in completed) == pytest.approx(
        float(expected_baseline), abs=1e-12
    )
    for i, expected in enumerate(exact):
        assert fsum(w * t.values_bb[i] for w, _, t in completed) == pytest.approx(
            float(expected), abs=1e-12
        )
        assert fsum(w * t.regrets_bb[i] for w, _, t in completed) == pytest.approx(
            float(expected - expected_baseline), abs=1e-12
        )


@pytest.mark.parametrize("n", [4, 5, 6])
def test_every_role_collects_against_one_profile_and_records_exact_actions(n):
    config = table(n, (10,) * n)
    profile = FrozenProfile([None] * n)
    result = collect_phase(
        config, profile, iteration=2, seed=7, traversals_per_player=2
    )
    assert len(result.traversals) == 2 * n
    assert {r.root.seat for r in result.traversals} == set(range(n))
    for index, r in enumerate(result.traversals):
        deal = Hand.start(
            config,
            hand_id=r.root.hand_id,
            seed=collection_seed(7, 2, r.root.seat, index % 2, "deal"),
        )
        assert r.root == deal.observe(r.root.seat)
        assert r.profile == result.profile == profile.fingerprint
        assert r.iteration == result.iteration == 2
        assert r.terminals > 0 and r.nodes == len(r.executions) + 1
        for target in r.targets:
            view = target.candidates.decision.source
            assert view.seat == view.actor == r.root.seat
            assert view.player_id == r.root.player_id
            assert view.hole_cards == r.root.hole_cards
            assert not view.previous_hands
            assert fsum(
                p * v for p, v in zip(target.policy, target.regrets_bb)
            ) == pytest.approx(0, abs=1e-12)
            assert all(
                abs(v) <= sum(config.stacks) / config.big_blind
                for v in target.values_bb
            )
        for executed in r.executions:
            view = executed.candidates.decision.source
            assert view.seat == view.actor == executed.event.seat
            assert view.player_id == config.player_ids[view.seat]
            assert view.hole_cards == deal.observe(view.seat).hole_cards
            assert executed.event.action == executed.candidates.actions[executed.index]
            view.legal_actions.validate(executed.event.action)


def test_zero_probability_traverser_branches_are_explored_but_opponents_are_sampled():
    hand = Hand.start(table(4, (10,) * 4), hand_id="zero-reach", seed=17)
    profile = FrozenProfile([folding_model()] * 4)
    result = collect_traversal(hand, profile, hand.actor, iteration=3, action_seed=31)
    candidates = bet_candidates(hand.observe(hand.actor))
    root = next(
        t for t in result.targets if t.candidates.decision.source.history == hand.events
    )
    assert root.policy == (1.0,) + (0.0,) * (len(candidates.actions) - 1)
    root_edges = [
        e
        for e in result.executions
        if e.candidates.decision.source.history == hand.events
    ]
    assert tuple(e.event.action for e in root_edges) == candidates.actions
    assert any(v != root.values_bb[0] for v in root.values_bb[1:])
    assert all(
        e.event.action.kind == ActionKind.FOLD
        for e in result.executions
        if e.event.seat != hand.actor
    )


def test_conditional_terminal_values_use_each_players_net_stack_not_a_two_player_sign():
    config = table(4, (3, 7, 9, 13))
    hand = Hand.start(config, hand_id="settlement", seed=83)
    while not hand.finished:
        view = hand.observe(hand.actor)
        if ActionKind.RAISE in view.legal_actions.kinds:
            action = Action(ActionKind.RAISE, view.legal_actions.max_raise_to)
        else:
            action = Action(
                ActionKind.CALL
                if ActionKind.CALL in view.legal_actions.kinds
                else ActionKind.CHECK
            )
        hand = hand.apply(action)
    profile = FrozenProfile([None] * 4)
    values = []
    for player in range(4):
        result = collect_traversal(hand, profile, player, iteration=1, action_seed=10)
        expected = (
            hand.events[-1].stacks[player] - config.stacks[player]
        ) / config.big_blind
        assert result.value_bb == expected
        assert result.nodes == result.terminals == 1
        assert not result.targets and not result.executions
        values.append(result.value_bb)
    assert sum(values) == 0 and len(set(values)) > 2


def test_collection_is_reproducible_and_isolates_global_random_streams():
    config = table(4, (10,) * 4)
    profile = FrozenProfile([BettingNetwork(8) for _ in range(4)])
    state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    first = collect_phase(config, profile, iteration=1, seed=17)
    assert torch.equal(state, torch.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    assert numpy_state[2:] == np.random.get_state()[2:]
    assert python_state == random.getstate()
    torch.rand(11)
    np.random.random(11)
    random.random()
    second = collect_phase(config, profile, iteration=1, seed=17)
    assert first == second
    assert collection_seed(17, 1, 0, 0, "deal") != collection_seed(
        17, 1, 0, 0, "opponents"
    )
    coordinates = {
        collection_seed(17, 1, p, k, stream)
        for p in range(6)
        for k in range(3)
        for stream in ("deal", "opponents")
    }
    assert len(coordinates) == 36
    assert all(
        str(collection_seed(17, 1, r.root.seat, 0, "deal")) not in r.root.hand_id
        for r in first.traversals
    )


def test_traversals_can_be_reconstructed_individually_in_reverse_order():
    config = table(4, (10,) * 4)
    profile = FrozenProfile([None] * 4)
    phase = collect_phase(config, profile, iteration=3, seed=19)
    for expected in reversed(phase.traversals):
        role = config.seat_numbers[expected.root.seat]
        hand = Hand.start(
            config,
            hand_id=expected.root.hand_id,
            seed=collection_seed(19, 3, role, 0, "deal"),
        )
        actual = collect_traversal(
            hand,
            profile,
            expected.root.seat,
            iteration=3,
            action_seed=collection_seed(19, 3, role, 0, "opponents"),
        )
        assert actual == expected


def test_changing_lineups_keep_physical_policy_ownership():
    profile = FrozenProfile([None] * 6)
    for seats in ((0, 1, 2, 3, 4, 5), (0, 1, 3, 4, 5), (0, 2, 3, 5)):
        config = replace(
            table(len(seats), (10,) * len(seats)),
            player_ids=tuple(f"player-{s}" for s in seats),
            seat_numbers=seats,
            capacity=6,
            table_seats=tuple(
                TableSeat(s, f"player-{s}", 10, "playing") for s in seats
            ),
        )
        phase = collect_phase(config, profile, iteration=1, seed=31)
        assert {r.root.seat_numbers[r.root.seat] for r in phase.traversals} == set(
            seats
        )
        assert {r.root.player_id for r in phase.traversals} == set(config.player_ids)


def test_budgets_abort_the_entire_batch_and_leave_profile_reusable():
    config = table(4, (10,) * 4)
    profile = FrozenProfile([None] * 4)
    baseline = collect_phase(config, profile, iteration=1, seed=7)
    with pytest.raises(CollectionLimitExceeded, match="no batch returned"):
        collect_phase(
            config, profile, iteration=1, seed=7, max_nodes=baseline.traversals[0].nodes
        )
    hand = Hand.start(config, hand_id="limits", seed=31)
    for kwargs in ({"max_nodes": 1}, {"deadline": 0}):
        with pytest.raises(CollectionLimitExceeded):
            collect_traversal(
                hand, profile, hand.actor, iteration=1, action_seed=3, **kwargs
            )
    assert collect_phase(config, profile, iteration=1, seed=7) == baseline


def test_internal_profile_changes_fail_collection_before_samples_are_returned(
    monkeypatch,
):
    profile = FrozenProfile([folding_model()] * 4)
    original = FrozenProfile.distribution

    def changed(self, candidates):
        result = original(self, candidates)
        with torch.no_grad():
            self._models[0].regret.bias.add_(0.01)
        return result

    monkeypatch.setattr(FrozenProfile, "distribution", changed)
    hand = Hand.start(table(4, (3,) * 4), hand_id="mutation", seed=4)
    with pytest.raises(RuntimeError, match="changed"):
        collect_traversal(hand, profile, hand.actor, iteration=1, action_seed=4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iteration": 0},
        {"seed": -1},
        {"max_nodes": 0},
        {"max_seconds": 0},
        {"max_seconds": float("nan")},
        {"traversals_per_player": True},
    ],
)
def test_phase_rejects_invalid_configuration(kwargs):
    args = {"iteration": 1, "seed": 0, **kwargs}
    with pytest.raises(ValueError):
        collect_phase(table(4), FrozenProfile([None] * 4), **args)


def test_traversal_rejects_invalid_ownership_and_raw_policy_inputs():
    hand = Hand.start(table(4), hand_id="invalid", seed=2)
    profile = FrozenProfile([None] * 4)
    for player in (-1, 4, True):
        with pytest.raises(ValueError):
            collect_traversal(hand, profile, player, iteration=1, action_seed=0)
    with pytest.raises(ValueError):
        collect_traversal(
            hand, FrozenProfile([None] * 6), 0, iteration=1, action_seed=0
        )
    with pytest.raises(TypeError):
        collect_traversal(
            hand.observe(hand.actor), profile, 0, iteration=1, action_seed=0
        )


def test_zero_own_reach_does_not_prune_later_traverser_decisions(monkeypatch):
    hand = advance(
        Hand.start(table(2, (3, 3)), hand_id="unreachable", seed=7), Street.RIVER
    )
    original = tuple(hand.observe(p) for p in range(2))
    models = [None, None]
    models[hand.actor] = folding_model()
    profile = FrozenProfile(models)

    class RaiseOrCall:
        def __init__(self, seed):
            pass

        def choices(self, population, weights, k):
            return [population[-1]]

    monkeypatch.setattr(collection, "Random", RaiseOrCall)
    result = collect_traversal(hand, profile, hand.actor, iteration=1, action_seed=0)
    assert len(result.targets) == 2
    later = next(
        t
        for t in result.targets
        if len(t.candidates.decision.source.history) > len(hand.events)
    )
    assert later.candidates.decision.source.legal_actions.call_amount == 1
    root = next(
        t for t in result.targets if t.candidates.decision.source.history == hand.events
    )
    assert root.policy[0] == 1 and sum(root.policy[1:]) == 0
    assert later.values_bb[0] != later.values_bb[1]
    assert tuple(hand.observe(p) for p in range(2)) == original
