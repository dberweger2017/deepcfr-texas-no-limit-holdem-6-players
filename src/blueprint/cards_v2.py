"""Declared HU20 private-card refinement; no opponent/range information."""

from functools import lru_cache

from src.game.showdown import hand_value

VERSION = "hu20-contribution-kicker-draw-descriptor-v2"
RANKS = "23456789TJQKA"
WINDOWS = ((14, 2, 3, 4, 5),) + tuple(tuple(range(n, n + 5)) for n in range(2, 11))


def rank_band(rank):
    """2–7 / 8–T / J–Q / K–A, fixed before any trained outcomes."""
    return sum(rank >= threshold for threshold in (8, 11, 13))


def _relative(rank, board_ranks):
    return sum(r > rank for r in board_ranks), int(rank in board_ranks), rank_band(rank)


def _flush_quality(cards, board):
    candidates = []
    for suit in "cdhs":
        own = [RANKS.index(c[0]) + 2 for c in cards if c[1] == suit]
        public = [RANKS.index(c[0]) + 2 for c in board if c[1] == suit]
        count = len(own) + len(public)
        # Backdoor draws exist only on the flop. River retains made-flush
        # contribution/blocker quality, with no claim of a future draw.
        stage = 3 if count >= 5 else 2 if count == 4 and len(board) < 5 else 1 if count == 3 and len(board) == 3 else 0
        if not stage:
            continue
        higher = sum(r > max(own) and r not in public for r in range(2, 15)) if own else 13
        nut_band = 0 if higher == 0 else 1 if higher == 1 else 2 if higher <= 3 else 3
        candidates.append((stage, len(own), 3 - nut_band))
    return max(candidates, default=(0, 0, 0))


def _straight_quality(cards, board, made_category):
    if len(board) == 5 or made_category >= 4:
        return (0, 0, 0, 0)
    own = {RANKS.index(c[0]) + 2 for c in cards}
    public = {RANKS.index(c[0]) + 2 for c in board}
    ranks = own | public
    draws = []
    backdoor = False
    for ordered in WINDOWS:
        window = set(ordered); missing = window - ranks
        contribution = len((own - public) & window)
        if len(missing) == 1:
            draws.append((next(iter(missing)), contribution, ordered[-1]))
        if len(board) == 3 and len(missing) == 2 and contribution:
            backdoor = True
    # Distinguish board-only draws, one/two private ranks and one/two-or-more
    # distinct out ranks (gutshot versus open/double-gutshot). Not betting EV.
    return (min(len({r for r, _, _ in draws}), 2),
            max((n for _, n, _ in draws), default=0),
            rank_band(max((top for _, _, top in draws), default=2)), int(backdoor))


@lru_cache(maxsize=8192)
def _descriptor(cards, board):
    from src.blueprint.abstraction import _postflop

    value = hand_value(cards + board); category = value[0]
    own = [RANKS.index(c[0]) + 2 for c in cards]
    public = {RANKS.index(c[0]) + 2 for c in board}
    contribution = 2
    if len(board) == 5 and hand_value(board) == value:
        contribution = 0
    elif len(board) >= 4 and any(hand_value(board + (c,)) == value for c in cards):
        contribution = 1
    groups = value[1:3] if category in (2, 6) else value[1:2] if category in (1, 3, 7) else ()
    # Full houses have two made groups; other multiplicity hands retain every
    # kicker band. Straights retain their top relative rank, flush/high card
    # retain all rank bands. No future card or hidden opponent is inspected.
    kickers = value[1 + len(groups):] if groups else value[1:] if category in (0, 5) else ()
    made_relative = tuple(_relative(r, public) for r in groups)
    if category in (4, 8):
        made_relative = (_relative(value[1], public),)
    return (_postflop(cards, board), contribution,
            tuple(own.count(r) for r in groups), made_relative,
            tuple(rank_band(r) for r in kickers),
            _flush_quality(cards, board), _straight_quality(cards, board, category))


def postflop_v2(cards, board):
    """Order/suit invariant descriptor of two own cards and current board."""
    if len(cards) != 2 or len(board) not in (3, 4, 5) or len(set(cards + board)) != len(cards + board):
        raise ValueError("Need distinct private cards and flop/turn/river board")
    return _descriptor(tuple(sorted(cards)), tuple(sorted(board)))
