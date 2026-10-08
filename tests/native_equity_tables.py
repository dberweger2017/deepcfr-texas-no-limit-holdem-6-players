"""Synthetic K=50 equity tables covering exactly the situations a test reaches.

Each class's bucket is a function of its suit-isomorphism class key, so isomorphic
situations share a bucket as they do in #163's tables.
"""

from itertools import combinations
from random import Random

from src.blueprint.equity_buckets import class_key, hash_key, write_table

DECK = tuple(rank + suit for rank in "23456789TJQKA" for suit in "cdhs")
STREETS = {3: "flop", 4: "turn", 5: "river"}


def write_tables(directory, situations):
    """`{flop,turn,river}-k50.bin` in `directory` for (hole, board) situations."""
    buckets = {street: {} for street in STREETS.values()}
    for hole, board in situations:
        key = class_key(hole, board)
        buckets[STREETS[len(board)]][key] = hash_key(key) % 50
    for street, table in buckets.items():
        write_table(directory / f"{street}-k50.bin", street, 50, table)
    return directory


def fixture_situations(seeds):
    """Every pair of the first four cards of `native_parity_fixtures.record`'s deck on its
    flop, turn and river, so the holdings it deals are covered whichever way they go."""
    for seed in seeds:
        deck = list(DECK)
        Random(seed).shuffle(deck)
        for hole in combinations(deck[:4], 2):
            for size in (3, 4, 5):
                yield hole, tuple(deck[4:4 + size])
