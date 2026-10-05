"""Native #162 bench parity; runs when native/hu20-trainer has been built in release mode."""

import json
from pathlib import Path
from random import Random
import subprocess

import pytest

from src.diagnostics.subgame_bench import FrozenRoot, SubgameTrainer

BINARY = Path("native/hu20-trainer/target/release/hu20-trainer")
CORPUS = Path("docs/reports/hu20-board-pooling-artifacts/corpus.json")
DECK = tuple(rank + suit for rank in "23456789TJQKA" for suit in "cdhs")
pytestmark = pytest.mark.skipif(not BINARY.exists(), reason="build native/hu20-trainer first")


def roots(count, seed):
    """Real #149 turn roots with full random-weight ranges, so holdings collide and get redrawn."""
    rng = Random(seed)
    built = []
    for record in json.loads(CORPUS.read_text())["roots"][:count]:
        live = [c for c in DECK if c not in record["board"]]
        hands = [(a, b) for i, a in enumerate(live) for b in live[i + 1:]]
        ranges = [[{"hand": list(h), "weight": rng.choice((0.0, rng.random(), 1.0, 3.5))} for h in hands]
                  for _ in (0, 1)]
        built.append(FrozenRoot.from_request(record, {"board": record["board"], "spot": record["spot"], "ranges": ranges}))
    return built


@pytest.mark.parametrize("count,checkpoints", [(1, (40, 150)), (5, (100, 400))])
def test_native_bench_equals_subgame_trainer(tmp_path, count, checkpoints):
    frozen = roots(count, 7 + count)
    (tmp_path / "roots.json").write_text(json.dumps([r.to_native() for r in frozen]))
    out = subprocess.run([str(BINARY), "bench-train", "--roots", str(tmp_path / "roots.json"), "--seed", "202610050001",
                          "--iterations", str(checkpoints[-1]), "--checkpoints", ",".join(map(str, checkpoints)),
                          "--lineage", "L", "--out", str(tmp_path / "native")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    nodes = [int(line.split()[3]) for line in out.stdout.splitlines()]
    trainer = SubgameTrainer(frozen, seed=202610050001)
    for checkpoint, native_nodes in zip(checkpoints, nodes, strict=True):
        while trainer.iteration < checkpoint:
            trainer.step()
        assert native_nodes == trainer.nodes
        for strategy in ("current", "average-traverser-reach", "average-opponent-sampled"):
            native = json.loads((tmp_path / "native" / f"iteration-{checkpoint}" / f"base.{strategy}.json").read_text())
            assert native == trainer.export("L", strategy), (checkpoint, strategy)
