"""Optional exact seven-card ranker for diagnostic LBR chance comparisons.

Original implementation using rank multiplicities and per-suit bit masks.
The game/showdown evaluator remains the independent reference and default.
"""

from functools import lru_cache
from types import FunctionType

from src.diagnostics.cached_lbr import CachedLocalBestResponse
from src.diagnostics.robustness import LocalBestResponse
from src.game.showdown import hand_value


def _straight_high(mask):
    for high in range(14, 5, -1):
        run = 31 << (high - 6)
        if mask & run == run:
            return high
    return 5 if mask & 0x100F == 0x100F else 0


_STRAIGHTS = tuple(_straight_high(mask) for mask in range(8192))
_CARDS = {rank + suit: (index + 2, suit_index)
          for index, rank in enumerate("23456789TJQKA")
          for suit_index, suit in enumerate("cdhs")}


@lru_cache(maxsize=8192)
def exact_seven_card(cards: tuple[str, ...]) -> tuple[int, ...]:
    if len(cards) != 7:
        return hand_value(cards)
    counts = [0] * 15
    suits = [0] * 4
    mask = 0
    for card in cards:
        rank, suit = _CARDS[card]
        bit = 1 << (rank - 2)
        counts[rank] += 1
        suits[suit] |= bit
        mask |= bit
    flush_mask = next((suit for suit in suits if suit.bit_count() >= 5), 0)
    if flush_mask and (high := _STRAIGHTS[flush_mask]):
        return (8, high)
    ranks = [rank for rank in range(14, 1, -1) if counts[rank]]
    quads = [rank for rank in ranks if counts[rank] == 4]
    if quads:
        return (7, quads[0], next(rank for rank in ranks if rank != quads[0]))
    trips = [rank for rank in ranks if counts[rank] >= 3]
    pairs = [rank for rank in ranks if counts[rank] >= 2]
    if trips and (full_pairs := [rank for rank in pairs if rank != trips[0]]):
        return (6, trips[0], full_pairs[0])
    if flush_mask:
        return (5, *[rank for rank in ranks
                     if flush_mask & (1 << (rank - 2))][:5])
    if high := _STRAIGHTS[mask]:
        return (4, high)
    if trips:
        return (3, trips[0], *[rank for rank in ranks if rank != trips[0]][:2])
    if len(pairs) >= 2:
        return (2, pairs[0], pairs[1],
                next(rank for rank in ranks if rank not in pairs[:2]))
    if pairs:
        return (1, pairs[0], *[rank for rank in ranks if rank != pairs[0]][:3])
    return (0, *ranks[:5])


def _bind_choose_action(ranker, *, clock=None):
    # A private globals dictionary reuses the frozen native code verbatim.
    # Only its ranking primitive changes; no engine/native module is patched.
    # Clock injection is for semantic batch-boundary tests, never production.
    native = LocalBestResponse.choose_action
    namespace = dict(native.__globals__, hand_value=ranker)
    if clock is not None:
        namespace["perf_counter"] = clock
    return FunctionType(native.__code__, namespace, native.__name__)


class RankedCachedLocalBestResponse(CachedLocalBestResponse):
    """Shared saved-query cache plus exact ranking, explicitly selected only."""

    choose_action = _bind_choose_action(exact_seven_card)
