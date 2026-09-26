"""Controlled rollout and frozen CFR play share the declared hidden-card law."""

import numpy as np

from scripts.evaluate_river_quality import _ranges
from src.arena.endgame_quality import _world, fixture_hand
from src.arena.river_conditional import (
    ConditionalRiverRollout, FrozenRiverProfile, conditional_opponent_weights,
)
from src.blueprint.abstraction import choices
from src.blueprint.river_cfr import RiverCFR
from src.blueprint.river_game import RiverGame, river_root_history
from src.blueprint.search import SearchConfig


class _UniformBlueprint:
    def distribution(self, view):
        menu = choices(view)
        return menu, (1 / len(menu),) * len(menu), False


def test_conditional_rollout_sees_only_the_hero_hand_and_public_joint_law():
    case = {
        "id": "conditional-root", "board": ["2h", "7d", "9c", "Js", "Qs"],
        "stack": 1000, "button": 0, "deck_shift": 18, "survivors": 2,
    }
    hand = fixture_hand(case)
    root = river_root_history(hand.events)
    game = RiverGame(root, _ranges(hand.observe(hand.actor)))
    hero = hand.actor
    own = game.seats.index(hero)
    other = 1 - own
    for own_id, holding in enumerate(game.holdings[own]):
        mass = game.joint[own_id] if own == 0 else game.joint[:, own_id]
        compatible = np.flatnonzero(mass > 0)
        if len(compatible) >= 2:
            break
    else:
        raise AssertionError("Fixture needs two compatible unseen hands")
    blueprint = _UniformBlueprint()
    observations = []
    for index in compatible[:2]:
        world = _world(root, game.board,
                       {hero: holding, game.seats[other]: game.holdings[other][index]})
        observations.append(world.observe(hero))
    assert observations[0] == observations[1]
    pairs, weights = conditional_opponent_weights(game, blueprint, observations[0])
    assert pairs == game.holdings[other]
    np.testing.assert_allclose(weights, mass / mass.sum(), atol=1e-15)

    config = SearchConfig(max_seconds=3, worlds=2, range_samples=2,
                          styles=("blueprint",), variant="corrected")
    controls = [ConditionalRiverRollout(blueprint, game, 239, config)
                for _ in observations]
    actions = [control.choose_action(view)
               for control, view in zip(controls, observations)]
    assert actions[0] == actions[1]
    observations[0].legal_actions.validate(actions[0])
    assert all(control.completed == 1 and control.fallbacks == 0
               for control in controls)

    solver = RiverCFR(game)
    profile = solver.solve(max_sweeps=1).average
    candidate = FrozenRiverProfile(blueprint, game, profile, 239, config)
    candidate_action = candidate.choose_action(observations[0])
    observations[0].legal_actions.validate(candidate_action)
    assert candidate.delegations == 0
