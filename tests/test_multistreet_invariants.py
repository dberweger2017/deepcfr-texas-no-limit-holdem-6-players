"""Focused invariants for the shallow multi-street reference and features.

These checks deliberately use tiny contexts and one frozen model forward pass.
They exercise information-boundary and finite-reference identities rather than
the diagnostic runner's campaign screens.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.game.hand import card_name
from src.game.types import ActionKind
from src.holdem.actions import bet_candidates
from src.holdem.multistreet_models import VARIANTS, make_model
from src.holdem.multistreet_reference import (
    DECK,
    _ordered_deck,
    build_context,
    enumerate_reference,
    flop_key,
    range_support,
    split_specs,
)
from src.holdem.river_reference import (
    ReferenceProfile,
    river_context,
)
from src.holdem.river_reference import (
    enumerate_reference as enumerate_river_reference,
)
from src.holdem.visible_features import partial_visible_features


@pytest.fixture(scope="module")
def support():
    plan = json.loads(Path("configs/holdem/representation.json").read_text())
    return range_support(plan["range_templates"])


STREET_CASES = (
    ("flop", ("Ac", "Kd", "7h")),
    ("turn", ("Ac", "Kd", "7h", "4s")),
    ("river", ("Ac", "Kd", "7h", "4s", "2c")),
)


@pytest.mark.parametrize("street, board", STREET_CASES)
@pytest.mark.parametrize("facing", (False, True))
def test_open_and_facing_contexts_have_legal_hero_roots(support, street, board, facing):
    context = build_context(
        name=f"{street}/{facing}",
        split="train",
        street=street,
        board=board,
        holding=("Qs", "Jc"),
        support=support,
        samples=1,
        deals_per_sample=1,
        seed=19,
        facing=facing,
    )
    root = context.worlds[0]
    view = root.observe(context.hero_seat)

    assert not root.finished
    assert root.actor == context.hero_seat
    assert view.actor == context.hero_seat
    assert view.street.value == street
    assert view.legal_actions.kinds
    assert view.legal_actions.call_amount >= 0
    if ActionKind.RAISE in view.legal_actions.kinds:
        assert view.legal_actions.min_raise_to <= view.legal_actions.max_raise_to


@pytest.mark.parametrize("street, board", STREET_CASES[:2])
def test_hidden_world_and_future_cards_do_not_change_any_model_output(
    support, street, board
):
    contexts = tuple(
        build_context(
            name="same-public-prefix",
            split="train",
            street=street,
            board=board,
            holding=("Qs", "Jc"),
            support=support,
            samples=2,
            deals_per_sample=1,
            seed=seed,
        )
        for seed in (13, 29)
    )
    observations = [
        world.observe(context.hero_seat)
        for context in contexts
        for world in context.worlds
    ]
    assert all(observation == observations[0] for observation in observations)
    assert contexts[0].assignments != contexts[1].assignments
    # The complete simulator decks differ too; the extra cards are future
    # board material hidden from the current observation.
    future_decks = {
        tuple(card_name(card) for card in world._state.deck)
        for context in contexts
        for world in context.worlds
    }
    assert len(future_decks) > 1

    candidates = [
        bet_candidates(observation)
        for observation in observations
    ]
    for variant in VARIANTS:
        outputs = make_model(variant, seed=31)(candidates)
        for output in outputs[1:]:
            assert torch.equal(outputs[0].regrets, output.regrets)
            assert torch.equal(outputs[0].values, output.values)


def test_suit_relabel_preserves_wheel_and_straight_completion_features():
    # The wheel is represented by the ace-low window.  The second case checks
    # an ordinary five-rank window; neither assertion uses an equity estimate.
    cases = (
        (("4c", "Qd"), ("As", "2d", "3h"), 0.1),
        (("8c", "Qd"), ("5s", "6d", "7h"), 0.2),
    )
    mapping = dict(zip("cdhs", "shdc", strict=True))

    def relabel(cards):
        return tuple(card[0] + mapping[card[1]] for card in cards)

    for holding, board, completion_count in cases:
        features = partial_visible_features(holding, board)
        relabeled = partial_visible_features(relabel(holding), relabel(board))
        assert features == pytest.approx(relabeled)
        # PARTIAL_FEATURE_SIZE layout puts one-card completion and nearest
        # window immediately after the four suit-count fields.
        assert features[24] == pytest.approx(completion_count)
        assert 0.0 <= features[25] <= 1.0


def test_finite_world_q_mean_and_covariance_match_legacy_river_reference():
    board = ("Ac", "Kd", "7h", "4s", "2c")
    holding = ("Qs", "Jc")
    deal_a = ("6d", "3h", "4h", "Qc", "4d", "5c", "8c", "Ts", "7d", "Kh")
    deal_b = ("5d", "6c", "8s", "5c", "6s", "3d", "Js", "Ad", "8h", "9h")
    profile = ReferenceProfile("uniform")

    modern = enumerate_reference(
        build_context(
            name="finite-support",
            split="train",
            street="river",
            board=board,
            holding=holding,
            support=(deal_a, deal_b),
            samples=2,
            deals_per_sample=1,
            seed=4,  # Random.choice selects deal_a then deal_b.
        ),
        profile,
        max_nodes=100_000,
        deadline=float("inf"),
    )
    legacy = enumerate_river_reference(
        river_context("finite-support", "train", board, holding, (deal_a, deal_b), False),
        profile,
        max_nodes=100_000,
        deadline=float("inf"),
    )

    rows = np.asarray(modern.world_action_values_bb, dtype=float)
    assert modern.target.values_bb == pytest.approx(rows.mean(axis=0))
    assert modern.target.values_bb == pytest.approx(legacy.target.values_bb)
    assert modern.action_standard_error_bb == pytest.approx(
        rows.std(axis=0, ddof=1) / np.sqrt(rows.shape[0])
    )
    assert np.any(rows[0] != rows[1])
    assert modern.uncertainty_status == "estimated"


def test_flop_family_cannot_cross_splits():
    template = [
        "2c", "2d", "3h", "3s", "4c", "4d", "5h", "5s", "6c", "6d"
    ]
    plan = {
        "context_seed": 7,
        "range_templates": [template],
        "world_samples": 1,
        "deals_per_sample": 1,
        "contexts": [
            {
                "split": "train",
                "street": "flop",
                "board": ["Ac", "Kd", "7h"],
                "holding": ["Qc", "Js"],
            },
            {
                "split": "validation",
                "street": "turn",
                "board": ["As", "Kc", "7d", "4h"],
                "holding": ["Qs", "Jc"],
            },
        ],
    }
    assert flop_key(plan["contexts"][0]["board"]) == flop_key(
        plan["contexts"][1]["board"]
    )
    with pytest.raises(ValueError, match="one split"):
        split_specs(plan)


def test_range_prior_keeps_template_multiplicity_and_order():
    template = (
        "2c", "2d", "3h", "3s", "4c", "4d", "5h", "5s", "6c", "6d"
    )
    support = range_support((template, template))
    assert len(support) == 48
    assert support[:24] == support[24:]
    assert support[0] == template


def test_ordered_deck_interleaves_holes_then_board_then_remaining_deck():
    hero = ("Qs", "Jc")
    board = ("Ac", "Kd", "7h", "4s", "2c")
    deal = ("Th", "Ts", "9h", "9s", "8h", "8s", "7c", "7d", "6h", "6s")
    deck, hands = _ordered_deck(hero, board, deal, hero_seat=1)
    order = (1, 2, 3, 4, 5, 0)
    prefix = tuple(
        hands[seat][round_index]
        for round_index in range(2)
        for seat in order
    ) + board
    assert hands[1] == hero
    assert deck[:17] == prefix
    assert deck[17:] == tuple(card for card in DECK if card not in prefix)
