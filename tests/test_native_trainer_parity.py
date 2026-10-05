"""Native HU20 trainer parity; runs when native/hu20-trainer has been built in release mode."""

import subprocess
import sys
from pathlib import Path

import pytest

BINARY = Path("native/hu20-trainer/target/release/hu20-trainer")
pytestmark = pytest.mark.skipif(not BINARY.exists(), reason="build native/hu20-trainer first")


def run(*args):
    return subprocess.run([sys.executable, *args], check=True, capture_output=True, text=True)


def test_rules_menus_and_keys_match_the_python_engine(tmp_path):
    fixtures = tmp_path / "hands.jsonl"
    run("-m", "scripts.native_parity_fixtures", "--hands", "500", "--passive", "0.6", "--out", str(fixtures))
    out = subprocess.run([str(BINARY), "parity", str(fixtures)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "mismatched_hands 0" in out.stdout


def test_traversals_reproduce_python_deltas_bit_for_bit(tmp_path):
    fixture = tmp_path / "traversal.json"
    run("-m", "scripts.native_traversal_fixtures", "--train-iterations", "100", "--cases", "40", "--out", str(fixture))
    out = subprocess.run([str(BINARY), "traversal-parity", str(fixture)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "mismatched 0" in out.stdout


def test_python_loads_and_continues_a_native_checkpoint(tmp_path):
    from src.blueprint.artifact import export_policy, load_training
    checkpoint = tmp_path / "native.json.gz"
    subprocess.run([str(BINARY), "train", "--nodes", "200000", "--seed", "3", "--out", str(checkpoint)], check=True)
    trainer = load_training(checkpoint)
    assert trainer.nodes and trainer.iteration > 0
    for _ in range(20):
        trainer.step()  # raises if any native key carries a different menu
    export_policy(trainer, tmp_path / "current.json.gz")
