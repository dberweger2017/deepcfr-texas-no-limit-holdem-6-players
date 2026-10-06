"""The checkpoint diagnosis joins lockstep T/O runs and a floor run; runs when the native trainer is built."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

BINARY = Path("native/hu20-trainer/target/release/hu20-trainer")
pytestmark = pytest.mark.skipif(not BINARY.exists(), reason="build native/hu20-trainer first")


def test_lockstep_join_and_cells(tmp_path):
    for name, flags in (("T", []), ("O", ["--average-rule", "opponent-sampled"]), ("F", ["--regret-floor", "0"])):
        subprocess.run([str(BINARY), "train", "--nodes", "300000", "--seed", "5", *flags, "--out", str(tmp_path / f"{name}.json.gz")],
                       check=True, capture_output=True)
    subprocess.run([sys.executable, "-m", "scripts.diagnose_hu20_checkpoints", "--traverser", str(tmp_path / "T.json.gz"),
                    "--opponent", str(tmp_path / "O.json.gz"), "--floor", str(tmp_path / "F.json.gz"), "--out", str(tmp_path / "out")],
                   check=True, capture_output=True)
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["F options"] == "regret-floor-0" and summary["iterations"]["T"] == summary["iterations"]["O"]
    cells = {(c["situation"], c["T visits"]) for c in summary["cells"]}
    assert ("facing-jam", "T-missing") in cells and ("unopened", "1") in cells
    # Keys O stores only as the sampled opponent are absent from T, which then plays uniformly.
    missing = next(c for c in summary["cells"] if c["T visits"] == "T-missing" and c["situation"] == "facing-jam")
    assert missing["T zero-mass"] == 1 and missing["O zero-mass"] == 0


def test_rejects_runs_that_are_not_lockstep(tmp_path):
    for name, seed, flags in (("T", 5, []), ("O", 6, ["--average-rule", "opponent-sampled"]), ("F", 5, ["--regret-floor", "0"])):
        subprocess.run([str(BINARY), "train", "--nodes", "100000", "--seed", str(seed), *flags, "--out", str(tmp_path / f"{name}.json.gz")],
                       check=True, capture_output=True)
    out = subprocess.run([sys.executable, "-m", "scripts.diagnose_hu20_checkpoints", "--traverser", str(tmp_path / "T.json.gz"),
                          "--opponent", str(tmp_path / "O.json.gz"), "--floor", str(tmp_path / "F.json.gz"), "--out", str(tmp_path / "out")],
                         capture_output=True, text=True)
    assert out.returncode != 0 and "lockstep" in out.stderr
