"""Read equity-distribution bucket tables built by `native/hu20-buckets`.

A situation is a two-card holding plus a flop, turn or river board. Its class
key is suit-isomorphism invariant and matches the Rust builder bit for bit.
"""

from pathlib import Path

import numpy as np

RANKS = "23456789TJQKA"
SUITS = "cdhs"
STREETS = {3: "flop", 4: "turn", 5: "river"}
MASK64 = (1 << 64) - 1


def class_key(hole, board):
    signature = [0, 0, 0, 0]
    for card in hole:
        signature[SUITS.index(card[1])] |= 1 << (13 + RANKS.index(card[0]))
    for card in board:
        signature[SUITS.index(card[1])] |= 1 << RANKS.index(card[0])
    key = 0
    for value in sorted(signature, reverse=True):
        key = (key << 26) | value
    return key


def _splitmix(x):
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def hash_key(key):
    return _splitmix(((key >> 64) & MASK64) ^ _splitmix(key & MASK64))


class BucketTable:
    """One street's table: sorted 64-bit class hashes and their bucket ids."""

    def __init__(self, path):
        raw = np.memmap(Path(path), dtype=np.uint8, mode="r")
        if bytes(raw[:8]) != b"HU20BKT1":
            raise ValueError("Not an HU20 bucket table")
        street, self.k = (int(v) for v in np.frombuffer(raw[8:16], dtype="<u4"))
        count = int(np.frombuffer(raw[16:24], dtype="<u8")[0])
        self.street = {1: "flop", 2: "turn", 3: "river"}[street]
        self.hashes = np.frombuffer(raw, dtype="<u8", count=count, offset=24)
        self.buckets = np.frombuffer(raw, dtype="<u2", count=count, offset=24 + 8 * count)

    def bucket(self, hole, board):
        if STREETS[len(board)] != self.street:
            raise ValueError("Board size belongs to another street's table")
        target = np.uint64(hash_key(class_key(hole, board)))
        index = int(np.searchsorted(self.hashes, target))
        if index >= len(self.hashes) or self.hashes[index] != target:
            raise KeyError("Situation is missing from the bucket table")
        return int(self.buckets[index])
