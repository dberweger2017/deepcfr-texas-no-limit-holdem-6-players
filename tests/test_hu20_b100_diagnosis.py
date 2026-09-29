"""Focused information and lookup boundaries for the B100M diagnostic."""

from random import Random

import pytest

from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA, choices, information_key
from src.blueprint.search import DECK, _sample_world
from src.diagnostics.action_translation import NearestOpponentRaiseLookup, translated_labels
from src.diagnostics.conditional_values import summarize, world_action_returns
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind


class CallTarget:
    def distribution(self, view):
        menu = choices(view, raise_cap=None, free_fold=False)
        wanted = ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds else ActionKind.CALL
        return menu, tuple(float(c.action.kind == wanted) for c in menu), True


def test_conditional_comparison_cannot_see_real_hidden_cards_or_future_deck():
    first = Hand.start(Table(("hero", "rival"), (2000, 2000)), hand_id="same-visible", seed=31)
    view = first.observe(first.actor)
    compatible = [c for c in DECK if c not in view.hole_cards]
    a = _sample_world(view, {1-view.seat: ((tuple(compatible[:2]), 1),)}, Random(1))
    b = _sample_world(view, {1-view.seat: ((tuple(compatible[-2:]), 1),)}, Random(2))
    assert a.observe(view.seat) == b.observe(view.seat) == view
    left = world_action_returns(a.observe(view.seat), CallTarget(), 702, 0)
    right = world_action_returns(b.observe(view.seat), CallTarget(), 702, 0)
    assert left == right


def test_world_clustered_gap_uses_policy_mix_after_aggregation():
    result = summarize(((1, -1), (-1, 1), (2, -2)), (.75, .25))
    assert result["action_mean_bb"] == pytest.approx((2/3, -2/3))
    assert result["policy_mean_bb"] == pytest.approx(1/3)
    assert result["policy_gap_bb"] == pytest.approx(1/3)


def test_translation_changes_only_lookup_label_and_preserves_native_chips():
    hand = Hand.start(Table(("rival", "hero"), (2000, 2000)), hand_id="off-menu", seed=15)
    before = hand.observe(0)
    actual = Action(ActionKind.RAISE, 260)
    assert all(c.action != actual for c in choices(before, raise_cap=None, free_fold=False))
    hand = hand.apply(actual)
    view = hand.observe(1)
    overrides, events = translated_labels(view)
    assert len(overrides) == 1 and events[0]["source_raise_to"] == 260
    assert events[0]["target_raise_to"] == 300
    menu = choices(view, raise_cap=None, free_fold=False)
    original_key = information_key(view, menu, schema=HU20_UNCAPPED_SCHEMA)
    mapped_key = information_key(view, menu, schema=HU20_UNCAPPED_SCHEMA,
                                 history_label_overrides=overrides)
    assert original_key != mapped_key

    class Saved:
        abstraction = HU20_UNCAPPED_SCHEMA
        entries = {mapped_key: (tuple(c.name for c in menu),
                                tuple(float(i == 0) for i in range(len(menu))))}

    class Target:
        source = Saved()
        raise_cap = None
        visits = {}
        def distribution(self, seen):
            opts = choices(seen, raise_cap=None, free_fold=False)
            return opts, (1/len(opts),)*len(opts), False

    original_events = hand.events
    original_pot = view.pot
    original_stacks = tuple(p.stack for p in view.players)
    translated = NearestOpponentRaiseLookup(Target())
    opts, probabilities, hit = translated.distribution(view)
    assert hit and sum(probabilities) == pytest.approx(1)
    assert opts == menu and translated.last_translation["translated_hit"]
    assert hand.events == original_events and view.pot == original_pot
    assert tuple(p.stack for p in view.players) == original_stacks
    for option in opts:
        view.legal_actions.validate(option.action)
