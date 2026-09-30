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


def test_selection_context_ignores_deal_seed_and_terminal_payoff():
    from scripts.diagnose_hu20_decisions import _trace
    row = {"rotation": 0, "block": 0,
           "actions": [{"seat": 0, "street": "preflop", "kind": "call",
                        "raise_to": None}]}
    first = list(_trace({**row, "deal_seed": 31, "target_chips": -2000}, synthetic=True))
    second = list(_trace({**row, "deal_seed": 982, "target_chips": 2000}, synthetic=True))
    assert first[0][1] == second[0][1]
    assert first[0][1].legal_actions.call_amount == 50


def test_evaluation_worlds_cannot_change_the_selected_action():
    worlds = ((2, 0),) * 48 + ((0, 20),) * 48
    result = summarize(worlds, (.5, .5))
    assert result["selection_worlds"] == result["evaluation_worlds"] == 48
    assert result["selection_action_mean_bb"] == pytest.approx((2, 0))
    assert result["descriptive_action_mean_bb"][1] > result["descriptive_action_mean_bb"][0]
    assert result["selected_action_index"] == 0
    assert result["evaluation_policy_gap_bb"] == pytest.approx(-10)
    assert result["evaluation_policy_gap_95_interval_bb"] == pytest.approx((-10, -10))


def test_selection_magnitudes_change_descriptive_means_not_heldout_gap():
    evaluation = ((3, 1),) * 24 + ((5, 2),) * 24
    first = summarize(((2, 0),) * 48 + evaluation, (.25, .75))
    second = summarize(((200, 0),) * 48 + evaluation, (.25, .75))
    assert first["selected_action_index"] == second["selected_action_index"] == 0
    assert first["evaluation_policy_gap_bb"] == second["evaluation_policy_gap_bb"]
    assert first["evaluation_policy_gap_95_interval_bb"] == second["evaluation_policy_gap_95_interval_bb"]
    assert first["descriptive_action_mean_bb"] != second["descriptive_action_mean_bb"]
    assert first["descriptive_policy_mean_bb"] != second["descriptive_policy_mean_bb"]


def test_heldout_gap_is_paired_with_saved_policy_mixture():
    result = summarize(((1, -1), (-1, 1), (2, -2), (0, 0)),
                       (.75, .25), selection_worlds=2)
    assert result["selected_action_index"] == 0
    assert result["evaluation_selected_action_mean_bb"] == pytest.approx(1)
    assert result["evaluation_policy_mean_bb"] == pytest.approx(.5)
    assert result["evaluation_policy_gap_bb"] == pytest.approx(.5)
    assert result["descriptive_action_mean_bb"] == pytest.approx((.5, -.5))
    assert result["descriptive_policy_mean_bb"] == pytest.approx(.25)


def test_primary_summary_rejects_incomplete_96_world_split():
    with pytest.raises(ValueError, match="equal"):
        summarize(((1, 0),) * 95, (.5, .5))


def test_prespecified_card_probe_imports():
    from scripts.probe_b100_card_collisions import _alternatives

    assert callable(_alternatives)


def test_late_street_fold_value_matches_independent_ledger():
    prefix = ("3c", "Ac", "3d", "Ad", "2c", "5d", "8h", "Ts", "Jc")
    hand = Hand.from_deck(Table(("hero", "rival"), (2000, 2000)),
                          hand_id="river-reference",
                          deck=prefix + tuple(c for c in DECK if c not in prefix))
    while len(hand.observe(hand.actor).board) < 5:
        seen = hand.observe(hand.actor)
        hand = hand.apply(Action(ActionKind.CHECK if ActionKind.CHECK in seen.legal_actions.kinds
                                 else ActionKind.CALL))
    assert hand.actor == 1
    hand = hand.apply(Action(ActionKind.RAISE, hand.observe(1).legal_actions.min_raise_to))
    view = hand.observe(0)
    menu = choices(view, raise_cap=None, free_fold=False)
    returns, _ = world_action_returns(view, CallTarget(), 9102, 0)
    fold = next(i for i, item in enumerate(menu) if item.action.kind == ActionKind.FOLD)
    assert returns[fold] == pytest.approx(-view.players[0].contributed / 100)


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
    assert translated.source is Target.source
    assert translated.distribution(view) == (opts, probabilities, hit)
    assert hand.events == original_events and view.pot == original_pot
    assert tuple(p.stack for p in view.players) == original_stacks
    for option in opts:
        view.legal_actions.validate(option.action)
