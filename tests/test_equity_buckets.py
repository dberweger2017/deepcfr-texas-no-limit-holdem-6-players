import os
import random
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.blueprint.equity_buckets import BucketTable, class_key, hash_key

BINARY = Path("native/hu20-buckets/target/release/hu20-buckets")

# Produced by `hu20-buckets key HOLE BOARD`; Python must match bit for bit.
GOLDEN = [
    (("Ah", "Kh"), ("2h", "7c", "9d"), 0x000000c0000040000800000080000000, 0x569f5133b1765a33),
    (("As", "Ks"), ("2s", "7d", "9c"), 0x000000c0000040000800000080000000, 0x569f5133b1765a33),
    (("Ah", "Kd"), ("2h", "7c", "9d", "Qs"), 0x00000080000050000800001000000020, 0xcedd4a5bfce4855c),
    (("Tc", "9c"), ("2c", "3c", "4d", "5h", "Ks"), 0x0000000c0000c0008000000020000004, 0xad83f36d639fd63e),
    (("2d", "2h"), ("Ad", "Kd", "Qd"), 0x000000000f0000020000000000000000, 0xe54792613f12b1c2),
]


@pytest.mark.parametrize("hole,board,key,digest", GOLDEN)
def test_class_key_matches_rust_builder(hole, board, key, digest):
    assert class_key(hole, board) == key
    assert hash_key(key) == digest


def test_reader_finds_buckets_and_rejects_other_streets(tmp_path):
    rows = sorted((hash_key(class_key(h, b)), bucket) for h, b, bucket in [
        (("Ah", "Kh"), ("2h", "7c", "9d"), 3), (("Ah", "Kd"), ("2h", "7c", "9d"), 7)])
    path = tmp_path / "flop-k8.bin"
    with path.open("wb") as out:
        out.write(b"HU20BKT1" + np.array([1, 8], "<u4").tobytes() + np.array([len(rows)], "<u8").tobytes())
        out.write(np.array([r[0] for r in rows], "<u8").tobytes() + np.array([r[1] for r in rows], "<u2").tobytes())
    table = BucketTable(path)
    assert (table.street, table.k) == ("flop", 8)
    assert table.bucket(("Ks", "As"), ("9c", "2s", "7d")) == 3  # suit-isomorphic, any order
    assert table.bucket(("Ah", "Kd"), ("2h", "7c", "9d")) == 7
    with pytest.raises(ValueError):
        table.bucket(("Ah", "Kd"), ("2h", "7c", "9d", "Qs"))
    with pytest.raises(KeyError):
        table.bucket(("2c", "3d"), ("4h", "5s", "7c"))


@pytest.mark.skipif(not BINARY.exists(), reason="build native/hu20-buckets first")
def test_native_evaluator_orders_hands_like_the_engine():
    from src.game.showdown import hand_value
    deck = [r + s for r in "23456789TJQKA" for s in "cdhs"]
    rng = random.Random(7)
    hands = [rng.sample(deck, 7) for _ in range(5000)]
    out = subprocess.run([str(BINARY), "eval"], input="\n".join(" ".join(h) for h in hands),
                         capture_output=True, text=True, check=True).stdout.split()
    native = [int(v) for v in out]
    engine = [hand_value(tuple(h)) for h in hands]
    for _ in range(50000):
        i, j = rng.randrange(len(hands)), rng.randrange(len(hands))
        assert (engine[i] > engine[j]) - (engine[i] < engine[j]) == (native[i] > native[j]) - (native[i] < native[j])
