"""Deterministic own-hand/board features from the player-visible cards."""

from src.game.showdown import hand_value


RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = frozenset(rank + suit for rank in RANKS for suit in SUITS)
HAND_VALUE_WIDTH = 6
FEATURE_SIZE = 2 * HAND_VALUE_WIDTH


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
