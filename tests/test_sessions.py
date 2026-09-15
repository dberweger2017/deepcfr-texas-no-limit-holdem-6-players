from dataclasses import FrozenInstanceError

import pytest

from src.game.observation import BlindPosted, replay
from src.game.session import SESSION_PROFILE, Session, replay_session
from src.game.types import Action, ActionKind


def table(n=6):
    session = Session(
        "test", small_blind=1, big_blind=2, chip_unit="1", min_buy_in=20, max_buy_in=400
    )
    for seat in range(n):
        session.join(f"p{seat}", seat, 100 + seat * 10)
    return session


def finish(session):
    while session.actor is not None:
        view = session.observe(session.actor)
        kind = (
            ActionKind.CHECK
            if ActionKind.CHECK in view.legal_actions.kinds
            else ActionKind.FOLD
        )
        session.apply(session.actor, Action(kind))
    session.settle()


def physical_blinds(session):
    view = session.observe(session.actor or session.seats[0].player_id)
    return tuple(
        view.seat_numbers[e.seat] for e in view.history if isinstance(e, BlindPosted)
    )


def test_bankrolls_and_public_replay_survive_six_five_four_players():
    session = table()
    session.start_hand(seed=1, opening_button=0)
    finish(session)
    stacks = {p.player_id: p.stack for p in session.seats}
    paid = session.leave("p5")
    assert paid == stacks.pop("p5")
    session.start_hand(seed=2)
    view = session.observe("p0")
    assert tuple(p.starting_stack for p in view.players) == tuple(stacks.values())
    assert len(view.players) == 5
    finish(session)
    session.leave("p4")
    session.start_hand(seed=3)
    assert len(session.observe("p0").players) == 4
    finish(session)
    restored = replay_session(session.events)
    assert restored.seats == session.seats
    assert restored.chips_in - restored.chips_out == sum(p.stack for p in session.seats)
    assert restored.profile == SESSION_PROFILE
    assert restored.big_blind == 2
    assert restored.capacity == 6


@pytest.mark.parametrize("departed", range(6))
def test_button_moves_to_next_incumbent_after_any_departure(departed):
    session = table()
    session.start_hand(seed=1, opening_button=0)
    finish(session)
    session.leave(f"p{departed}")
    session.start_hand(seed=2)
    remaining = [s for s in range(6) if s != departed]
    button = next(s for s in range(1, 6) if s in remaining)
    index = remaining.index(button)
    assert physical_blinds(session) == (
        remaining[(index + 1) % 5],
        remaining[(index + 2) % 5],
    )


def test_return_waits_for_big_blind_and_seat_owner_keeps_private_history():
    session = table(4)
    session.start_hand(seed=1, opening_button=0)
    old_cards = session.observe("p0").hole_cards
    finish(session)
    session.move("p0", 5)
    session.start_hand(seed=2)
    view = session.observe("p0")
    assert view.seat == -1 and view.hole_cards == () and not view.legal_actions.kinds
    assert view.previous_hands[0].hole_cards == old_cards
    assert all(p.player_id != "p0" for p in view.players)
    assert view.table_seats[-1].status == "waiting"
    assert replay(view.history, -1, (), view.previous_hands, observer_id="p0") == view
    finish(session)
    session.start_hand(seed=3)
    view = session.observe("p0")
    assert view.seat_numbers[view.seat] == 5
    assert physical_blinds(session)[1] == 5
    assert view.previous_hands[-1].seat == -1
    assert view.previous_hands[-1].hole_cards == ()
    assert view.previous_hands[0].hole_cards == old_cards


def test_replacement_has_no_previous_occupants_private_records():
    session = table()
    session.start_hand(seed=1, opening_button=0)
    finish(session)
    session.leave("p3")
    session.join("new", 3, 100)
    session.start_hand(seed=2)
    view = session.observe("new")
    assert physical_blinds(session)[1] == 3
    assert view.previous_hands == ()
    assert "p3" not in [p.player_id for p in view.players]
    with pytest.raises(ValueError):
        session.observe("p3")


def test_sit_out_return_and_top_up_do_not_reset_other_stacks():
    session = table()
    session.start_hand(seed=1, opening_button=0)
    finish(session)
    session.sit_out("p4")
    session.top_up("p4", 50)
    session.start_hand(seed=2)
    view = session.observe("p0")
    assert view.table_seats[4].status == "sitting_out"
    assert view.table_seats[4].stack == 190
    with pytest.raises(ValueError):
        session.observe("p4")
    finish(session)
    session.return_to_play("p4")
    session.start_hand(seed=3)
    assert physical_blinds(session)[1] == 4
    assert len(session.observe("p4").previous_hands) == 1


@pytest.mark.parametrize(
    "operation",
    [
        lambda s: s.join("new", 5, 100),
        lambda s: s.leave("p0"),
        lambda s: s.top_up("p0", 10),
        lambda s: s.sit_out("p0"),
        lambda s: s.return_to_play("p0"),
        lambda s: s.move("p0", 5),
        lambda s: s.start_hand(seed=2),
        lambda s: s.settle(),
    ],
)
def test_table_mutations_fail_atomically_during_a_hand(operation):
    session = table(4)
    session.start_hand(seed=1, opening_button=0)
    before = session.events
    view = session.observe("p0")
    with pytest.raises(ValueError):
        operation(session)
    assert session.events == before
    assert session.observe("p0") == view


def test_departure_into_heads_up_and_waiter_restores_three_handed_blinds():
    session = table(3)
    session.start_hand(seed=1, opening_button=0)
    finish(session)
    session.leave("p2")
    session.start_hand(seed=2)
    assert physical_blinds(session) == (1, 0)
    assert session.actor == "p1"
    finish(session)
    session.join("new", 3, 100)
    session.start_hand(seed=3)
    assert physical_blinds(session) == (1, 3)
    assert len(session.observe("p0").players) == 3
    assert session.actor == "p0"


def test_short_incumbent_can_play_but_returning_player_needs_full_blind():
    session = table(2)
    session.top_up("p0", 10)
    session.start_hand(seed=1, opening_button=0)
    while session.actor:
        view = session.observe(session.actor)
        legal = view.legal_actions
        action = (
            Action(ActionKind.RAISE, legal.max_raise_to)
            if ActionKind.RAISE in legal.kinds
            else Action(
                ActionKind.CALL if ActionKind.CALL in legal.kinds else ActionKind.CHECK
            )
        )
        session.apply(session.actor, action)
    session.settle()
    busted = next(p for p in session.seats if p.stack == 0)
    assert busted.status == "busted"
    with pytest.raises(ValueError):
        session.top_up(busted.player_id, 1)
    session.top_up(busted.player_id, 20)
    session.start_hand(seed=2)
    assert physical_blinds(session)[1] == busted.seat


def test_cashout_cannot_shed_chips_on_return_and_bad_inputs_do_not_change_ledger():
    session = table(4)
    chips = session.leave("p0")
    with pytest.raises(ValueError):
        session.join("p0", 0, chips - 1)
    session.join("p0", 0, chips)
    for chips in (True, 0, -1, 1.5, 500):
        before = session.events
        with pytest.raises(ValueError):
            session.top_up("p0", chips)
        assert session.events == before
    with pytest.raises(FrozenInstanceError):
        session.seats[0].stack = 0


def test_policies_only_receive_their_own_session_observations():
    from src.game.play import play_session_hand

    class Observer:
        def __init__(self, identity):
            self.identity = identity
            self.views = []

        def choose_action(self, view):
            assert view.player_id == self.identity
            assert all(h.player_id == self.identity for h in view.previous_hands)
            self.views.append(view)
            return Action(
                ActionKind.CHECK
                if ActionKind.CHECK in view.legal_actions.kinds
                else ActionKind.CALL
            )

    session = table(4)
    policies = {p.player_id: Observer(p.player_id) for p in session.seats}
    for index in range(3):
        session.start_hand(seed=index, opening_button=0 if index == 0 else None)
        play_session_hand(session, policies)
    for policy in policies.values():
        assert len(policy.views[-1].previous_hands) == 2


def test_failed_policy_keeps_table_locked_and_preserves_public_trace():
    from src.game.play import play_session_hand

    class Broken:
        def choose_action(self, view):
            raise RuntimeError("broken policy")

    session = table(4)
    session.start_hand(seed=1, opening_button=0)
    with pytest.raises(RuntimeError, match="broken policy"):
        play_session_hand(session, {p.player_id: Broken() for p in session.seats})
    assert session.hand_active
    assert session.events[-1].kind == "hand_started"
    with pytest.raises(ValueError):
        session.leave("p0")
    finish(session)
    assert session.events[-1].kind == "hand_settled"


@pytest.mark.parametrize("seed", range(20))
def test_generated_sessions_conserve_chips_through_lineup_changes(seed):
    from scripts.check_session import run

    result = run(seed, hands=30)
    assert result["completed_hands"] == 30
    assert result["table_chips"] == result["chips_in"] - result["chips_out"]
    assert 6 in result["participant_counts"]
    assert all(2 <= count <= 6 for count in result["participant_counts"])


def test_waiters_do_not_receive_unrevealed_cards_from_other_players():
    session = table(4)
    session.start_hand(seed=1, opening_button=0)
    finish(session)
    session.join("watcher", 5, 100)
    session.start_hand(seed=2)
    view = session.observe("watcher")
    assert view.seat == -1 and view.hole_cards == ()
    assert all(p.shown_cards == () for p in view.players)
    with pytest.raises(ValueError, match="private cards"):
        replay(view.history, -1, ("Ac", "Ad"), observer_id="watcher")
    with pytest.raises(ValueError):
        replay(view.history, 0, (), observer_id="watcher")


def test_every_waiting_seat_eventually_enters_by_posting_the_big_blind():
    for waiting_seat in range(6):
        session = table()
        session.start_hand(seed=1, opening_button=0)
        finish(session)
        session.sit_out(f"p{waiting_seat}")
        session.return_to_play(f"p{waiting_seat}")
        for hand in range(8):
            session.start_hand(seed=hand)
            entered = f"p{waiting_seat}" in session.participants
            if entered:
                assert physical_blinds(session)[1] == waiting_seat
                break
            finish(session)
        assert entered


def test_all_players_away_requires_an_explicit_new_opening_and_preserves_histories():
    session = table(4)
    session.start_hand(seed=1, opening_button=0)
    cards = session.observe("p0").hole_cards
    finish(session)
    for player in session.seats:
        session.sit_out(player.player_id)
    for player in session.seats:
        session.return_to_play(player.player_id)
    with pytest.raises(ValueError, match="explicit new opening"):
        session.start_hand(seed=2)
    session.start_hand(seed=2, opening_button=2)
    assert physical_blinds(session) == (3, 0)
    assert session.observe("p0").previous_hands[0].hole_cards == cards


def test_returning_identity_recovers_its_own_history_after_cashout():
    session = table()
    session.start_hand(seed=1, opening_button=0)
    cards = session.observe("p3").hole_cards
    finish(session)
    chips = session.leave("p3")
    session.join("p3", 3, chips)
    session.start_hand(seed=2)
    assert session.observe("p3").previous_hands[0].hole_cards == cards
    assert physical_blinds(session)[1] == 3


def test_settlement_cannot_be_applied_twice():
    session = table()
    session.start_hand(seed=1, opening_button=0)
    finish(session)
    before = session.events
    with pytest.raises(ValueError):
        session.settle()
    assert session.events == before
