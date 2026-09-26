"""Native coupled rotations and checkpoint-compatible blueprint lookup."""

from dataclasses import replace

import pytest

from src.blueprint.abstraction import (
    BUTTON_ZERO_COMPAT_LOOKUP, LEGACY_LOOKUP, SCHEMA, SUMMARY_SCHEMA,
    choices, information_key,
)
from src.blueprint.lookup import NoFreeFoldDistribution, TableDistribution
from src.blueprint.search import LiveBlueprint
from src.blueprint.solver import BlueprintTrainer, Node, PilotConfig
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind, Street


DECK = tuple(rank + suit for suit in "cdhs" for rank in "23456789TJQKA")
HASH = "94d41c737d0e8f41a548d58c64dd219dfb9593459c80d7a7382569bf6963e33a"


def _coupled_hand(rotation, *, swap_unseen=False):
    board = ("As", "Kd", "7c", "2h", "9d")
    holes = (("Ac", "Ah"), ("Qc", "Qd"), ("Jc", "Jh"),
             ("Tc", "Td"), ("8c", "8h"), ("6c", "6d"))
    if swap_unseen:
        holes = (holes[1], holes[0], *holes[2:])
    dealt = tuple(holes[seat][round_] for round_ in range(2)
                  for seat in (1, 2, 3, 4, 5, 0))
    remaining = tuple(card for card in DECK if card not in set(dealt + board))
    ids = tuple(f"logical-{(seat - rotation) % 6}" for seat in range(6))
    stacks = tuple(300 if name == "logical-3" else 10_000 for name in ids)
    table = Table(ids, stacks, button=rotation)
    return Hand.from_deck(table, hand_id="coupled-rotation", deck=dealt + board + remaining)


def _trace(rotation, *, swap_unseen=False):
    hand = _coupled_hand(rotation, swap_unseen=swap_unseen)
    records = []
    off_menu_seen = False
    for _ in range(80):
        if hand.finished:
            break
        view = hand.observe(hand.actor)
        menu = choices(view)
        records.append(view)
        actor = view.player_id
        prior = sum(event.seat == view.seat and event.street == view.street
                    for event in view.history if hasattr(event, "action"))
        if (view.street == Street.PREFLOP and actor == "logical-3"
                and prior == 0 and ActionKind.RAISE in view.legal_actions.kinds):
            action = Action(ActionKind.RAISE, view.legal_actions.max_raise_to)
        elif (view.street == Street.PREFLOP and actor == "logical-4"
              and prior == 0 and ActionKind.RAISE in view.legal_actions.kinds):
            candidates = (target for target in range(view.legal_actions.min_raise_to,
                                                      view.legal_actions.max_raise_to + 1)
                          if all(item.action != Action(ActionKind.RAISE, target)
                                 for item in menu))
            action = Action(ActionKind.RAISE, next(candidates))
            off_menu_seen = True
        elif view.street == Street.PREFLOP and actor == "logical-5" and prior == 0:
            action = Action(ActionKind.FOLD)
        else:
            action = Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds
                            else ActionKind.CALL)
        view.legal_actions.validate(action)
        hand = hand.apply(action)
    assert hand.finished and off_menu_seen
    assert {view.street for view in records} == {
        Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER,
    }
    assert any(p.folded for view in records for p in view.players)
    assert any(p.all_in for view in records for p in view.players)
    return records


def test_six_coupled_native_rotations_preserve_canonical_keys_and_probabilities():
    traces = [_trace(rotation) for rotation in range(6)]
    assert all(len(trace) == len(traces[0]) for trace in traces)
    legacy_differs = False
    for positions in zip(*traces, strict=True):
        assert len({view.player_id for view in positions}) == 1
        assert len({view.hole_cards for view in positions}) == 1
        for schema in (SCHEMA, SUMMARY_SCHEMA):
            canonical = [information_key(view, choices(view), schema=schema,
                                         lookup_mode=BUTTON_ZERO_COMPAT_LOOKUP)
                         for view in positions]
            assert len(set(canonical)) == 1
            assert canonical[0] == information_key(positions[0], choices(positions[0]),
                                                   schema=schema, lookup_mode=LEGACY_LOOKUP)
        if len({information_key(view, choices(view)) for view in positions}) > 1:
            legacy_differs = True
    assert legacy_differs

    trainer = BlueprintTrainer(Table(tuple(f"p{i}" for i in range(6)), (10_000,) * 6), PilotConfig())
    trainer.iteration = 8733
    view_zero = traces[0][-1]
    menu = choices(view_zero)
    trainer.nodes[information_key(view_zero, menu)] = Node(
        tuple(item.name for item in menu), [float(i == 0) for i in range(len(menu))],
        [0.0] * len(menu), 3,
    )
    legacy = TableDistribution(trainer)
    canonical = TableDistribution(trainer, lookup_mode=BUTTON_ZERO_COMPAT_LOOKUP,
                                  checkpoint_sha256=HASH)
    search_interface = LiveBlueprint(
        trainer, lookup_mode=BUTTON_ZERO_COMPAT_LOOKUP, checkpoint_sha256=HASH,
    )
    assert legacy.distribution(view_zero) == canonical.distribution(view_zero)
    for view in (trace[-1] for trace in traces):
        assert canonical.distribution(view) == canonical.distribution(view_zero)
        assert search_interface.distribution(view) == canonical.distribution(view)


def test_compatibility_rejects_unverified_provenance():
    trainer = BlueprintTrainer(Table(tuple(f"p{i}" for i in range(6)), (10_000,) * 6), PilotConfig())
    with pytest.raises(ValueError, match="verified button-zero"):
        TableDistribution(trainer, lookup_mode=BUTTON_ZERO_COMPAT_LOOKUP,
                          checkpoint_sha256=HASH)
    trainer.iteration = 8733
    with pytest.raises(ValueError, match="verified button-zero"):
        TableDistribution(trainer, lookup_mode=BUTTON_ZERO_COMPAT_LOOKUP,
                          checkpoint_sha256="0" * 64)
    rotated = BlueprintTrainer(replace(trainer.table, button=1), trainer.config)
    rotated.iteration = 8733
    with pytest.raises(ValueError, match="verified button-zero"):
        TableDistribution(rotated, lookup_mode=BUTTON_ZERO_COMPAT_LOOKUP,
                          checkpoint_sha256=HASH)


def test_no_free_fold_preserves_menu_hits_and_other_fold_probabilities():
    first = _coupled_hand(0).observe(_coupled_hand(0).actor)
    assert ActionKind.CHECK not in first.legal_actions.kinds
    while_first = _coupled_hand(0)
    while while_first.observe(while_first.actor).street == Street.PREFLOP:
        view = while_first.observe(while_first.actor)
        while_first = while_first.apply(Action(
            ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds
            else ActionKind.CALL))
    free = while_first.observe(while_first.actor)
    menu = choices(free)
    fold = next(i for i, item in enumerate(menu) if item.action.kind == ActionKind.FOLD)
    check = next(i for i, item in enumerate(menu) if item.action.kind == ActionKind.CHECK)

    class Source:
        def __init__(self, weights, trained):
            self.weights = weights
            self.trained = trained
        def distribution(self, view):
            return menu, self.weights, self.trained

    weights = [0.0] * len(menu)
    weights[fold] = 0.4
    weights[check] = 0.6
    wrapped = NoFreeFoldDistribution(Source(tuple(weights), True))
    selected_menu, probabilities, trained = wrapped.distribution(free)
    assert selected_menu == menu and trained
    assert probabilities[fold] == 0 and probabilities[check] == 1
    assert wrapped.interventions["removed_probability"] == pytest.approx(0.4)
    assert wrapped.interventions["changed"] == 1

    missing = NoFreeFoldDistribution(Source((1 / len(menu),) * len(menu), False))
    _, uniform_safe, trained = missing.distribution(free)
    assert not trained and uniform_safe[fold] == 0
    assert sum(uniform_safe) == pytest.approx(1)
    assert all(value >= 0 for value in uniform_safe)

    all_fold = [0.0] * len(menu)
    all_fold[fold] = 1.0
    forced = NoFreeFoldDistribution(Source(tuple(all_fold), True))
    _, deterministic, _ = forced.distribution(free)
    assert deterministic[check] == 1 and forced.interventions["all_fold_mass"] == 1

    no_check_menu = choices(first)
    class NoCheck:
        def distribution(self, view):
            return no_check_menu, (1 / len(no_check_menu),) * len(no_check_menu), True
    before = NoCheck().distribution(first)
    assert NoFreeFoldDistribution(NoCheck()).distribution(first) == before


def test_lookup_ignores_other_unseen_hands():
    original = _trace(0)
    altered = _trace(0, swap_unseen=True)
    assert len(original) == len(altered)
    for a, b in zip(original, altered, strict=True):
        if a.player_id in {"logical-3", "logical-4"}:
            assert a == b
            assert information_key(a, choices(a),
                                   lookup_mode=BUTTON_ZERO_COMPAT_LOOKUP) == (
                information_key(b, choices(b),
                                lookup_mode=BUTTON_ZERO_COMPAT_LOOKUP)
            )
