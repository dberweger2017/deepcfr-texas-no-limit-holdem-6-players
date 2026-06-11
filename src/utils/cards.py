"""Card and stage formatting shared across clients and logging."""

from __future__ import annotations

RANK_LABELS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUIT_SYMBOLS = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}
SUIT_COLORS = {0: "black", 1: "red", 2: "red", 3: "black"}
STAGE_LABELS = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River", 4: "Showdown"}


def card_to_string(card) -> str:
    """Convert a poker card to a readable string."""
    return f"{RANK_LABELS[int(card.rank)]}{SUIT_SYMBOLS[int(card.suit)]}"


def card_code(card) -> str:
    return card_to_string(card)


def card_display_payload(card) -> dict:
    suit = int(card.suit)
    return {
        "code": card_to_string(card),
        "rank": RANK_LABELS[int(card.rank)],
        "suit": SUIT_SYMBOLS[suit],
        "color": SUIT_COLORS[suit],
    }
