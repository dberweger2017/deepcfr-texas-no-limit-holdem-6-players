"""Exercise the installed engine through the bot's actual action adapters."""

import random

import pokers
import pytest

from src.utils.actions import build_raise_action, raise_bounds
from src.utils.logging import apply_action_with_logging


def test_minimum_raise_comes_from_the_engine():
    state = pokers.State.from_seed(6, 0, 1, 2, 200, 0)
    state = state.apply_action(build_raise_action(state, 8, strict=True))
    bounds = raise_bounds(state)
    assert state.min_bet == 10
    assert bounds.min_raise == 8
    action = build_raise_action(state, None, strict=True)
    assert state.apply_action(action).min_bet == 18


def test_short_stack_can_raise_all_in_below_the_full_increment():
    state = pokers.State.from_seed(3, 0, 1, 2, 200, 0, stakes=[3, 200, 200])
    action = build_raise_action(state, 100, strict=True)
    assert action.action == pokers.ActionEnum.Raise
    assert action.amount == 1
    state = state.apply_action(action)
    assert state.status == pokers.StateStatus.Ok
    assert state.players_state[0].stake == 0
    assert state.min_raise == 2


@pytest.mark.parametrize("players", [4, 5, 6])
def test_unequal_stack_hands_through_strict_adapters(players):
    for seed in range(100):
        rng = random.Random(1000 * players + seed)
        stacks = [rng.randint(1, 20000) / 100 for _ in range(players)]
        state = pokers.State.from_seed(
            players,
            seed % players,
            0.5,
            1,
            100,
            seed,
            stakes=stacks,
        )
        initial_chips = sum(round(stack * 100) for stack in stacks)
        for turn in range(500):
            if state.final_state:
                break
            action_type = rng.choice(state.legal_actions)
            if action_type == pokers.ActionEnum.Raise:
                bounds = raise_bounds(state)
                amount = rng.choice(
                    [
                        bounds.min_raise,
                        bounds.max_raise,
                        state.pot * rng.uniform(0.1, 3),
                    ]
                )
                action = build_raise_action(state, amount, strict=True)
                assert action.action == pokers.ActionEnum.Raise
            else:
                action = pokers.Action(action_type)
            state, log_path, status = apply_action_with_logging(
                state, action, strict=True
            )
            assert status == pokers.StateStatus.Ok
            assert log_path is None
            assert (
                sum(
                    round((p.stake + p.bet_chips + p.pot_chips) * 100)
                    for p in state.players_state
                )
                == initial_chips
            )
        assert state.final_state, (players, seed, turn)
        assert state.pot == 0
        assert sum(round(p.reward * 100) for p in state.players_state) == 0
        assert all(
            round(p.reward * 100) == round((p.stake - start) * 100)
            for p, start in zip(state.players_state, stacks)
        )
