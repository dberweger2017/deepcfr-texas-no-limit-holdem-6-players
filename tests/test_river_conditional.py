"""Controlled rollout and frozen CFR play share the declared hidden-card law."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_river_quality import _ranges
from scripts.evaluate_river_conditional import _draw, _play, _seed
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
    assert ConditionalRiverRollout(blueprint, game, 239, config,
                                   worlds_override=8192).worlds == 8192
    with pytest.raises(ValueError, match="worlds override"):
        ConditionalRiverRollout(blueprint, game, 239, config,
                                worlds_override=16385)

    solver = RiverCFR(game)
    profile = solver.solve(max_sweeps=1).average
    candidate = FrozenRiverProfile(blueprint, game, profile, 239, config)
    candidate_action = candidate.choose_action(observations[0])
    observations[0].legal_actions.validate(candidate_action)
    assert candidate.delegations == 0


def test_frozen_confirmation_roots_are_fresh_legal_and_balanced():
    base = Path(__file__).resolve().parents[1] / "configs/blueprint"
    cases = json.loads((base / "river-confirmation-cases.json").read_text())["cases"]
    prior = []
    for name in ("river-reference-fixtures.json", "river-development-m4.json",
                 "river-range-amendment-m4.json"):
        source = json.loads((base / name).read_text())
        prior.extend(source.get("cases", source.get("full_range_cases", [])))
    excluded = {frozenset(case["board"]) for case in prior}
    assert len(cases) == 32
    assert len({frozenset(case["board"]) for case in cases}) == 32
    assert not any(frozenset(case["board"]) in excluded for case in cases)
    assert {case["hero_position"] for case in cases} == {"first", "second"}
    assert {case["opponent_style"] for case in cases} == {
        "tight_passive", "loose_passive", "tight_aggressive",
        "loose_aggressive", "pot_pressure", "train_pressure",
    }
    pots = set()
    for case in cases:
        hand = fixture_hand(case)
        view = hand.observe(hand.actor)
        assert len([player for player in view.players if not player.folded]) == 2
        assert list(view.board) == case["board"]
        pots.add(view.pot / view.big_blind)
    assert pots == {2, 4, 6, 12}


def test_paired_river_attempts_replay_the_same_private_deal():
    case = {"id": "paired-river", "board": ["2h", "7d", "9c", "Js", "Qs"],
            "stack": 1000, "button": 0, "deck_shift": 18, "survivors": 2}
    hand = fixture_hand(case)
    game = RiverGame(river_root_history(hand.events), _ranges(hand.observe(hand.actor)))
    deal_seed = _seed(20260927, case["id"], 0, "deal")
    holes = _draw(game, deal_seed)
    assert holes == _draw(game, deal_seed)
    assert len(set(holes[game.seats[0]] + holes[game.seats[1]] + game.board)) == 9
    blueprint = _UniformBlueprint()
    config = SearchConfig(max_seconds=3, worlds=2, range_samples=2,
                          styles=("blueprint",), variant="corrected")
    profile = RiverCFR(game).solve(max_sweeps=2).average
    players = (
        FrozenRiverProfile(blueprint, game, profile, 11, config),
        ConditionalRiverRollout(blueprint, game, 12, config),
    )
    for player in players:
        result = _play(game, holes, hand.actor, "tight_passive", 23, player)
        assert result["finished"] and result["actions"]
        assert isinstance(result["payoff_bb"], float)
