"""Deterministic own-hand/board features from the player-visible cards."""

from src.game.showdown import hand_value


RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = frozenset(rank + suit for rank in RANKS for suit in SUITS)
HAND_VALUE_WIDTH = 6
FEATURE_SIZE = 2 * HAND_VALUE_WIDTH
PARTIAL_FEATURE_SIZE = HAND_VALUE_WIDTH + 1 + 13 + 4 + 4


def _checked_cards(cards, expected, name):
    cards = tuple(cards)
    if (
        len(cards) != expected
        or len(set(cards)) != expected
        or any(card not in DECK for card in cards)
    ):
        count = "two" if expected == 2 else "five"
        raise ValueError(f"{name} needs {count} distinct standard cards")
    return cards


def _padded_value(cards):
    value = hand_value(cards)
    # Categories are 0..8 and tie-break ranks are 2..14.  Keep each field on a
    # fixed [0, 1] scale without fitting a data-dependent normalization.
    normalized = (value[0] / 8,) + tuple(rank / 14 for rank in value[1:])
    return normalized + (0.0,) * (HAND_VALUE_WIDTH - len(value))


def visible_card_features(holding, board):
    """Return fixed-width hand ranks for the visible hero cards and board.

    The extractor accepts exactly the owner's two cards and the complete public
    five-card board.  It deliberately has no range, opponent, target, or engine
    state argument, so a feature control cannot inspect hidden worlds.
    """

    holding = _checked_cards(holding, 2, "The visible holding")
    board = _checked_cards(board, 5, "The visible board")
    if set(holding).intersection(board):
        raise ValueError("The visible holding and board need seven distinct cards")
    return _padded_value(holding + board) + _padded_value(board)


def partial_visible_features(holding, board):
    """Features available at flop, turn, or river.

    Made-hand ranks are emitted only once five cards are visible.  The other
    fields summarize public rank/suit texture and straight/flush completion
    potential; no future card, range, or equity information is consulted.
    """

    holding = _checked_cards(holding, 2, "The visible holding")
    board = tuple(board)
    if len(board) not in (3, 4, 5):
        raise ValueError("A partial visible board needs three to five cards")
    if (
        len(set(board)) != len(board)
        or any(card not in DECK for card in board)
        or set(holding).intersection(board)
    ):
        raise ValueError("The visible holding and board need distinct cards")
    visible = holding + board
    made = _padded_value(visible) if len(visible) >= 5 else (0.0,) * HAND_VALUE_WIDTH
    ranks = tuple("23456789TJQKA".index(card[0]) for card in board)
    rank_hist = tuple(ranks.count(rank) / 4.0 for rank in range(13))
    suit_hist = tuple(
        count / 5.0
        for count in sorted(
            (sum(card[1] == suit for card in board) for suit in SUITS), reverse=True
        )
    )
    all_ranks = {"23456789TJQKA".index(c[0]) for c in visible}
    # Count public five-rank windows that are one card away.  The wheel is
    # represented explicitly; these are completion features, not winning-outs
    # or an equity estimate.
    windows = ({12, 0, 1, 2, 3},) + tuple(
        frozenset(range(start, start + 5)) for start in range(0, 9)
    )
    missing = tuple(len(window - all_ranks) for window in windows)
    one_card_completion = sum(value == 1 for value in missing) / len(windows)
    nearest_window = min(missing) / 5.0
    flush_potential = max(
        (sum(card[1] == suit for card in visible) / 5.0 for suit in SUITS),
        default=0.0,
    )
    return made + (float(len(visible) >= 5),) + rank_hist + suit_hist + (
        one_card_completion,
        nearest_window,
        flush_potential,
        len(board) / 5.0,
    )
