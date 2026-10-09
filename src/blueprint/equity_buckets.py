"""Read equity-distribution bucket tables built by `native/hu20-buckets`.

A situation is a two-card holding plus a flop, turn or river board. Its class
key is suit-isomorphism invariant and matches the Rust builder bit for bit.
`EquityCards` holds one K=50 table per street for the equity-bucket information
keys; the native trainer's `cards.rs` reads the same files.
"""

from hashlib import sha256
from pathlib import Path

import numpy as np

RANKS = "23456789TJQKA"
SUITS = "cdhs"
STREETS = {3: "flop", 4: "turn", 5: "river"}
# #163's K=50 tables (docs/reports/hu20-equity-buckets.md), validated by #190.
EQUITY_K50_SHA256 = {
    "flop": "084e243ded9fe93ad03cd59601f22f16c3210592f51e1e3646a60983b572c111",
    "turn": "c0bc6a8aa0535553118109d18a32d3b4dc6880e937c263cdc87472b1ae9f168f",
    "river": "70c1cb5ecf4cd292e4c89dc2200f7c8381100be3b0288ceeb0a12a3e83977629",
}
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


def write_table(path, street, k, buckets):
    """A table in the builder's format, from {class key: bucket}; for test fixtures."""
    code = {"flop": 1, "turn": 2, "river": 3}[street]
    rows = sorted((hash_key(key), bucket) for key, bucket in buckets.items())
    if any(a[0] == b[0] for a, b in zip(rows, rows[1:])):
        raise ValueError("64-bit class hash collision")
    header = b"HU20BKT1" + np.array([code, k], dtype="<u4").tobytes() + np.array([len(rows)], dtype="<u8").tobytes()
    hashes = np.array([r[0] for r in rows], dtype="<u8").tobytes()
    ids = np.array([r[1] for r in rows], dtype="<u2").tobytes()
    Path(path).write_bytes(header + hashes + ids)


def file_sha256(path):
    digest = sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class EquityCards:
    """`{flop,turn,river}-k50.bin` from one directory. Unless `pinned=False` (test tables
    only), each file must be #163's, so the schema name identifies its labels."""

    def __init__(self, directory, *, pinned=True):
        self.tables, self.sha256 = {}, {}
        for street in ("flop", "turn", "river"):
            path = Path(directory) / f"{street}-k50.bin"
            digest = file_sha256(path)
            if pinned and digest != EQUITY_K50_SHA256[street]:
                raise ValueError(f"{path} is not #163's {street} K=50 table")
            table = BucketTable(path)
            if table.street != street or table.k != 50:
                raise ValueError(f"{path} is not a {street} K=50 table")
            self.tables[street], self.sha256[street] = table, digest

    def bucket(self, hole, board):
        return self.tables[STREETS[len(board)]].bucket(hole, board)


_registered = None


def register(cards):
    """Tables that equity-bucket information keys use from now on in this process."""
    global _registered
    _registered = cards


def registered():
    if _registered is None:
        raise ValueError("Equity-bucket keys need tables; call equity_buckets.register first")
    return _registered
