from dataclasses import replace

import pytest

from scripts.benchmark_observation_reuse import original_observe
from src.game.hand import Hand, Table
from src.game.observation import replay
from src.game.types import Action, ActionKind


def hand():
    return Hand.start(Table(("a", "b"), (2000, 2000)), hand_id="reuse", seed=17)


def test_same_instance_actor_reuses_but_still_validates(monkeypatch):
    import src.game.hand as module
    calls = []

    def counted(*args, **kwargs):
        calls.append(args)
        return replay(*args, **kwargs)

    h = hand()
    monkeypatch.setattr(module, "replay", counted)
    view = h.observe(h.actor)
    assert h.observe(h.actor) is view
    with pytest.raises(ValueError):
        h.apply(Action(ActionKind.RAISE, 101))
    assert len(calls) == 1
    assert h.observe(h.actor) is view
    h.apply(Action(ActionKind.CALL))
    # The original Hand's acting view was not replayed by apply.
    assert sum(args[0] is h.events and args[1] == h.actor for args in calls) == 1


def test_new_instances_branches_and_replaced_history_do_not_share_cache():
    h = hand()
    view = h.observe(h.actor)
    other = hand()
    assert other.observe(other.actor) == view
    assert other.observe(other.actor) is not view
    for branch in (h.apply(Action(ActionKind.CALL)), h.apply(Action(ActionKind.RAISE, 300))):
        assert branch._acting_view is None
        new = branch.observe(branch.actor)
        assert new is not view
        assert new == replay(branch.events, branch.actor, new.hole_cards)
    altered = replace(h, events=(replace(h.events[0], hand_id="different"), *h.events[1:]))
    assert altered._acting_view is None
    assert altered.observe(altered.actor).hand_id == "different"
    assert h.observe(h.actor) is view


def test_seat_and_prior_private_history_are_not_reused():
    h = hand()
    actor = h.actor
    cached = h.observe(actor)
    other = h.observe(1-actor)
    assert other.seat == 1-actor and other.hole_cards != cached.hole_cards
    settled = h.apply(Action(ActionKind.FOLD))
    prior = settled.observe(actor).record()
    context = h.observe(actor, (prior,))
    assert context.previous_hands == (prior,)
    assert context is not cached and h.observe(actor) is cached
    foreign = settled.observe(1-actor).record()
    with pytest.raises(ValueError, match="different player"):
        h.observe(actor, (foreign,))
    for invalid in (True, -1, 2, None):
        with pytest.raises(ValueError):
            h.observe(invalid)


def test_original_control_and_cached_views_match_all_seats_and_branches():
    original = original_observe()
    h = hand()
    while not h.finished:
        for seat in (0, 1):
            assert h.observe(seat) == original(h, seat)
        view = h.observe(h.actor)
        kind = ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds else ActionKind.CALL
        h = h.apply(Action(kind))
    assert h._acting_view is None
    for seat in (0, 1):
        assert h.observe(seat) == original(h, seat)
