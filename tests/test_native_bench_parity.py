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


@pytest.mark.parametrize("flags,label", [(["--regret-floor", "0"], "regret-floor-0"), (["--dcfr", "1.5,0,2"], "dcfr-1.5-0-2")])
def test_options_are_labeled_and_keep_both_averages_on_one_trajectory(tmp_path, flags, label):
    """bench-train refuses to export unless both averaging trainers kept bit-identical regrets."""
    (tmp_path / "roots.json").write_text(json.dumps([r.to_native() for r in roots(3, 5)]))
    common = [str(BINARY), "bench-train", "--roots", str(tmp_path / "roots.json"), "--seed", "3", "--iterations", "300",
              "--lineage", "L", "--out", str(tmp_path / "out")]
    subprocess.run([*common, "--variant", "base"], check=True, capture_output=True)
    subprocess.run([*common, "--variant", "option", *flags], check=True, capture_output=True)
    for strategy in ("current", "average-traverser-reach", "average-opponent-sampled"):
        base = json.loads((tmp_path / "out/iteration-300" / f"base.{strategy}.json").read_text())
        option = json.loads((tmp_path / "out/iteration-300" / f"option.{strategy}.json").read_text())
        assert "training_options" not in base and option["training_options"] == label
        assert option["groups"] != base["groups"]


def test_python_refuses_to_continue_a_checkpoint_trained_with_options(tmp_path):
    import gzip
    from src.blueprint.artifact import load_training
    checkpoint = tmp_path / "native.json.gz"
    subprocess.run([str(BINARY), "train", "--nodes", "20000", "--seed", "4", "--out", str(checkpoint)], check=True,
                   capture_output=True)
    lines = gzip.open(checkpoint, "rt").read().splitlines()
    header = json.loads(lines[0])
    header["training_options"] = "regret-floor-0"
    labeled = tmp_path / "labeled.json.gz"
    with gzip.open(labeled, "wt") as stream:
        stream.write("\n".join([json.dumps(header), *lines[1:]]) + "\n")
    load_training(checkpoint)
    with pytest.raises(ValueError, match="native options"):
        load_training(labeled)


@pytest.mark.parametrize("flags", [["--dcfr", "1.5,0,2"], ["--regret-floor", "0"], ["--dcfr", "1.5,0,2", "--regret-floor", "-2"]])
def test_export_cadence_cannot_change_training(tmp_path, flags):
    """Long enough for beta=0 discounts to underflow negative regrets to -0.0 between exports (#169 review)."""
    (tmp_path / "roots.json").write_text(json.dumps([r.to_native() for r in roots(5, 12)]))
    common = [str(BINARY), "bench-train", "--roots", str(tmp_path / "roots.json"), "--seed", "3", "--iterations", "3000",
              "--lineage", "L", *flags]
    subprocess.run([*common, "--checkpoints", "500,1500,2999", "--out", str(tmp_path / "often")], check=True, capture_output=True)
    subprocess.run([*common, "--out", str(tmp_path / "once")], check=True, capture_output=True)
    for strategy in ("current", "average-traverser-reach", "average-opponent-sampled"):
        name = f"iteration-3000/base.{strategy}.json"
        assert (tmp_path / "often" / name).read_bytes() == (tmp_path / "once" / name).read_bytes()


def test_dcfr_rejects_alpha_at_most_one(tmp_path):
    (tmp_path / "roots.json").write_text(json.dumps([r.to_native() for r in roots(1, 1)]))
    out = subprocess.run([str(BINARY), "bench-train", "--roots", str(tmp_path / "roots.json"), "--seed", "3",
                          "--iterations", "10", "--lineage", "L", "--dcfr", "0,0,2", "--out", str(tmp_path / "out")],
                         capture_output=True, text=True)
    assert out.returncode != 0 and "alpha > 1" in out.stderr
