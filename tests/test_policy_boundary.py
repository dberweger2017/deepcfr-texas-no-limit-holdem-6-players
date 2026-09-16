import pytest

from src.game import Action, ActionKind
from src.game.hand import Hand
from src.game.play import PlayerHistory, RandomPolicy, play_hand
from tests.test_hand_observations import table


@pytest.mark.parametrize("n", [4, 5, 6])
def test_headless_runner_keeps_policy_instances_and_histories_separate(n):
    policies = {f"player-{i}": RandomPolicy(100 + i) for i in range(n)}
    histories = {}
    for deal in range(3):
        hand = play_hand(
            Hand.start(table(n), hand_id=f"hand-{deal}", seed=deal), policies, histories
        )
        next_histories = {
            identity: histories.get(identity, PlayerHistory(identity)).append(
                hand.observe(seat)
            )
            for seat, identity in enumerate(hand.table.player_ids)
        }
        assert all(len(h.hands) == deal + 1 for h in next_histories.values())
        assert all(len(h.hands) == deal for h in histories.values())
        histories = next_histories
    shared = RandomPolicy(1)
    with pytest.raises(ValueError, match="own policy"):
        play_hand(
            Hand.start(table(n), hand_id="shared", seed=1),
            {identity: shared for identity in policies},
        )


def test_own_cards_and_distinct_betting_histories_remain_visible():
    hand = Hand.start(table(3), hand_id="history", seed=13)
    raised_first = (
        hand.apply(Action(ActionKind.RAISE, 4))
        .apply(Action(ActionKind.CALL))
        .apply(Action(ActionKind.CALL))
    )
    raised_second = (
        hand.apply(Action(ActionKind.CALL))
        .apply(Action(ActionKind.RAISE, 4))
        .apply(Action(ActionKind.CALL))
        .apply(Action(ActionKind.CALL))
    )
    first, second = raised_first.observe(1), raised_second.observe(1)
    assert first.board == second.board
    assert first.players == second.players
    assert first.pots == second.pots
    assert first.history != second.history
    assert hand.observe(0).hole_cards != hand.observe(1).hole_cards
    assert hand.observe(0).player_id != hand.observe(1).player_id
