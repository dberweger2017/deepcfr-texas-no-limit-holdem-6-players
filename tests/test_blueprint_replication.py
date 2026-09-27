"""Conditional continuation sampling uses the production aggregation path."""

from dataclasses import asdict
from itertools import product
from math import isclose
from dataclasses import replace

import pytest

from src.blueprint.artifact import save_training
from src.blueprint.abstraction import BUTTON_ZERO_COMPAT_LOOKUP, SCHEMA
from src.blueprint.lookup import (
    DESCENDANT_LINEAGE_SCHEMA, REPLICATION_PARENT, SAMPLERS,
    TableDistribution,
)
from src.blueprint.solver import (
    BlueprintTrainer, CollectionLimitExceeded, PilotConfig, _Delta,
    _mean_continuations, _resample_flop_future,
)
from src.game.hand import Hand, Table, card_name
from src.game.types import Action, ActionKind, Street


def _table():
    return Table(tuple(f"p{i}" for i in range(6)), (200,) * 6,
                 small_blind=1, big_blind=2, chip_unit="1")


def _flop_prefix():
    hand = Hand.start(_table(), hand_id="replicated", seed=17)
    folded = False
    while not hand.finished and hand.observe(hand.actor).street == Street.PREFLOP:
        view = hand.observe(hand.actor)
        if not folded and ActionKind.CHECK not in view.legal_actions.kinds:
            action = Action(ActionKind.FOLD)
            folded = True
        else:
            kind = (ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds
                    else ActionKind.CALL)
            action = Action(kind)
        hand = hand.apply(action)
    assert not hand.finished and hand.observe(hand.actor).street == Street.FLOP
    assert any(player.folded for player in hand.observe(hand.actor).players)
    return hand


def test_k1_checkpoint_is_byte_identical_to_legacy_fixture(tmp_path):
    trainer = BlueprintTrainer(_table(), PilotConfig(
        seed=1, raise_cap=0, max_nodes=30_000, max_seconds=30,
        postflop_replicates=1,
    ))
    report = trainer.step()
    assert (report.nodes, report.terminals, report.entries) == (132, 24, 18)
    assert report.continuation_samples == 0
    assert save_training(trainer, tmp_path / "k1.json.gz") == (
        "c90170cf2d1b0fef8a1a67b7cc9dada73d466c84e739b0477484010c2444c360"
    )


def test_flop_replay_preserves_every_assigned_card_and_public_prefix():
    hand = _flop_prefix()
    original_holes = tuple(tuple(card_name(c) for c in player.hand)
                           for player in hand._state.players_state)
    decks = []
    for seed in (11, 22, 33, 44):
        world, actions = _resample_flop_future(hand, seed)
        assert actions > 0 and world.events == hand.events
        assert tuple(tuple(card_name(c) for c in player.hand)
                     for player in world._state.players_state) == original_holes
        assert tuple(world.observe(seat) for seat in range(6)) == (
            tuple(hand.observe(seat) for seat in range(6))
        )
        cards = [card_name(c) for player in world._state.players_state
                 for c in player.hand]
        cards += [card_name(c) for c in world._state.public_cards]
        cards += [card_name(c) for c in world._state.deck]
        assert len(cards) == len(set(cards)) == 52
        decks.append(tuple(card_name(c) for c in world._state.deck))
    assert len(set(decks)) == 4


def _finite_outcomes():
    """Opponent's hidden H/L card changes its nonuniform response law."""
    outcomes = []
    policy = (0.4, 0.6)
    iteration = 7
    for hidden, p_hidden in (("H", 0.3), ("L", 0.7)):
        for response, p_response in (("A", 0.2 if hidden == "H" else 0.6),
                                     ("B", 0.8 if hidden == "H" else 0.4)):
            probability = p_hidden * p_response
            if response == "A":
                values = (2.0, -1.0) if hidden == "H" else (-2.0, 1.0)
                utility = sum(p * v for p, v in zip(policy, values))
                delta = _Delta(("left", "right"),
                               [iteration * (value - utility) for value in values],
                               [iteration * p for p in policy], 1)
                outcomes.append((probability, utility, {"shared-I": delta}))
            else:
                utility = 0.5 if hidden == "H" else -0.5
                outcomes.append((probability, utility, {}))
    return outcomes


def test_nonuniform_hidden_game_matches_exact_enumeration_for_k1_and_k4():
    outcomes = _finite_outcomes()
    assert isclose(sum(p for p, _, _ in outcomes), 1.0)
    exact_value = sum(p * utility for p, utility, _ in outcomes)
    exact_regrets = [sum(p * delta["shared-I"].regrets[action]
                         for p, _, delta in outcomes if delta)
                     for action in range(2)]
    exact_average = [sum(p * delta["shared-I"].average[action]
                         for p, _, delta in outcomes if delta)
                     for action in range(2)]
    variance_one = sum(p * (utility - exact_value) ** 2
                       for p, utility, _ in outcomes)
    for count in (1, 4):
        expected_value = 0.0
        expected_regrets = [0.0, 0.0]
        expected_average = [0.0, 0.0]
        expected_square = 0.0
        for draws in product(outcomes, repeat=count):
            probability = 1.0
            for draw in draws:
                probability *= draw[0]
            value, deltas = _mean_continuations(
                ((utility, contribution) for _, utility, contribution in draws),
                count,
            )
            expected_value += probability * value
            expected_square += probability * value * value
            if "shared-I" in deltas:
                for action in range(2):
                    expected_regrets[action] += probability * deltas["shared-I"].regrets[action]
                    expected_average[action] += probability * deltas["shared-I"].average[action]
        assert isclose(expected_value, exact_value, abs_tol=1e-12)
        assert all(isclose(a, b, abs_tol=1e-12)
                   for a, b in zip(expected_regrets, exact_regrets))
        assert all(isclose(a, b, abs_tol=1e-12)
                   for a, b in zip(expected_average, exact_average))
        assert isclose(expected_square - expected_value ** 2,
                       variance_one / count, abs_tol=1e-12)


def test_k4_publishes_only_complete_outer_iterations(monkeypatch):
    trainer = BlueprintTrainer(_table(), PilotConfig(
        seed=1, raise_cap=0, max_nodes=30_000, max_seconds=30,
        postflop_replicates=4,
    ))
    first = trainer.step()
    assert first.sampled_postflop_prefixes == 6
    assert first.continuation_samples == 24
    assert first.raw_traverser_visits > first.normalized_update_mass
    assert first.replay_actions > 0 and first.replay_seconds > 0
    before = {key: asdict(node) for key, node in trainer.nodes.items()}
    iteration = trainer.iteration

    # Fail while a replicated continuation is being rebuilt on the next step.
    from src.blueprint import solver
    original = solver._resample_flop_future
    called = 0

    def interrupt(hand, seed):
        nonlocal called
        called += 1
        if called == 2:
            raise CollectionLimitExceeded("injected mid-batch interruption")
        return original(hand, seed)

    monkeypatch.setattr(solver, "_resample_flop_future", interrupt)
    with pytest.raises(CollectionLimitExceeded):
        trainer.step()
    assert called == 2
    assert trainer.iteration == iteration
    assert {key: asdict(node) for key, node in trainer.nodes.items()} == before


def test_incomplete_replication_batch_cannot_publish_mean():
    with pytest.raises(ValueError, match="Incomplete"):
        _mean_continuations(iter([(1.0, {})]), 4)


def test_descendant_compatibility_requires_matching_verified_lineage(tmp_path):
    table = Table(tuple(f"player-{i}" for i in range(6)), (10_000,) * 6)
    trainer = BlueprintTrainer(table, PilotConfig(seed=101, postflop_replicates=4))
    trainer.iteration = 8734
    output_hash = save_training(trainer, tmp_path / "descendant.json.gz")
    lineage = {
        "schema": DESCENDANT_LINEAGE_SCHEMA,
        "parent_checkpoint_sha256": REPLICATION_PARENT,
        "output_checkpoint_sha256": output_hash,
        "source_revision": "a" * 40,
        "source_dirty": False,
        "key_schema": SCHEMA,
        "action_menu_raise_cap": trainer.config.raise_cap,
        "sampler_version": SAMPLERS[4],
        "continuation_seed": trainer.config.seed,
        "parent_iteration": 8733,
        "output_iteration": trainer.iteration,
        "completed_nodes": 100,
        "completed_outer_iterations": 1,
        "training_table": {
            "player_ids": list(table.player_ids),
            "stacks": list(table.stacks),
            "button": table.button,
            "small_blind": table.small_blind,
            "big_blind": table.big_blind,
            "chip_unit": table.chip_unit,
        },
    }
    with pytest.raises(ValueError, match="verified"):
        TableDistribution(trainer, lookup_mode=BUTTON_ZERO_COMPAT_LOOKUP,
                          checkpoint_sha256=output_hash)
    TableDistribution(trainer, lookup_mode=BUTTON_ZERO_COMPAT_LOOKUP,
                      checkpoint_sha256=output_hash, lineage=lineage)
    for mutation in (
        {"parent_checkpoint_sha256": "0" * 64},
        {"output_checkpoint_sha256": REPLICATION_PARENT},
        {"source_dirty": True},
        {"sampler_version": SAMPLERS[1]},
        {"completed_outer_iterations": 2},
    ):
        with pytest.raises(ValueError, match="verified"):
            TableDistribution(trainer, lookup_mode=BUTTON_ZERO_COMPAT_LOOKUP,
                              checkpoint_sha256=output_hash,
                              lineage={**lineage, **mutation})
