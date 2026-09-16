"""Default cash-game disclosure: show in order, muck hands already beaten."""

from collections import Counter
from functools import lru_cache
from itertools import combinations

from src.game.observation import ActionTaken, CardsMucked, CardsShown
from src.game.types import ActionKind, Street


@lru_cache(maxsize=8192)
def hand_value(cards: tuple[str, ...]) -> tuple[int, ...]:
    def five(hand):
        ranks = sorted(("23456789TJQKA".index(c[0]) + 2 for c in hand), reverse=True)
        groups = sorted(
            ((count, rank) for rank, count in Counter(ranks).items()), reverse=True
        )
        flush = len({c[1] for c in hand}) == 1
        straight = 5 if ranks == [14, 5, 4, 3, 2] else 0
        if len(groups) == 5 and ranks[0] - ranks[-1] == 4:
            straight = ranks[0]
        ordered = tuple(rank for _, rank in groups)
        if flush and straight:
            return (8, straight)
        if groups[0][0] == 4:
            return (7, *ordered)
        if [g[0] for g in groups] == [3, 2]:
            return (6, *ordered)
        if flush:
            return (5, *ranks)
        if straight:
            return (4, straight)
        if groups[0][0] == 3:
            return (3, *ordered)
        if [g[0] for g in groups[:2]] == [2, 2]:
            return (2, *ordered)
        if groups[0][0] == 2:
            return (1, *ordered)
        return (0, *ranks)

    return max(five(hand) for hand in combinations(cards, 5))


def disclosures(events, players, board, pots, button):
    first = (button + 1) % len(players)
    for event in events:
        if (
            isinstance(event, ActionTaken)
            and event.street == Street.RIVER
            and event.action.kind == ActionKind.RAISE
        ):
            first = event.seat
    best_shown = {}
    for offset in range(len(players)):
        seat = (first + offset) % len(players)
        cards = players[seat]
        if cards is None:
            continue
        value = hand_value(cards + board)
        contested = [
            i
            for i, pot in enumerate(pots)
            if seat in pot.eligible_seats and pot.refund_to is None
        ]
        # Compare only with hands already tabled in pots this player can win.
        # Looking at later players' cards would make mucking use hidden knowledge.
        if any(i not in best_shown or value >= best_shown[i] for i in contested):
            yield CardsShown(seat, cards)
            for i in contested:
                best_shown[i] = max(best_shown.get(i, value), value)
        else:
            yield CardsMucked(seat)
