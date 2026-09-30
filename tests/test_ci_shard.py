import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.ci_shard import shard_for


def test_every_collected_test_is_selected_once_in_isolated_file_shards(tmp_path):
    # Running real pytest validates collection, deselection and parametrizations.
    for i in range(8):
        (tmp_path / f"test_case_{i}.py").write_text(
            'import pytest\n@pytest.mark.parametrize("x", [1, 2])\ndef test_value(x):\n    assert x > 0\n')
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root)}
    collected = []
    for shard in (0, 1):
        run = subprocess.run([sys.executable, "-m", "pytest", "-p", "scripts.ci_shard", "--ci-shard", str(shard),
                              "--collect-only", "-q"], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
        collected.append({line for line in run.stdout.splitlines() if line.startswith("test_case_") and "::" in line})
    assert not collected[0] & collected[1]
    assert len(collected[0] | collected[1]) == 16
    for shard, items in enumerate(collected):
        assert items
        assert all(shard_for(node.split("::")[0], 2) == shard for node in items)
        for i in range(8):
            assert sum(node.startswith(f"test_case_{i}.py::") for node in items) in (0, 2)
    for shard in (0, 1):
        subprocess.run([sys.executable, "-m", "pytest", "-p", "scripts.ci_shard", "--ci-shard", str(shard), "-q"],
                       cwd=tmp_path, env=env, check=True, capture_output=True)


@pytest.mark.parametrize("shard,count", [(-1, 2), (2, 2), (0, 0)])
def test_invalid_shard_fails_instead_of_silently_skipping_tests(tmp_path, shard, count):
    (tmp_path / "test_case.py").write_text("def test_ok():\n    pass\n")
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run([sys.executable, "-m", "pytest", "-p", "scripts.ci_shard", "--ci-shard", str(shard),
                          "--ci-shards", str(count), "-q"], cwd=tmp_path,
                         env={**os.environ, "PYTHONPATH": str(root)}, capture_output=True, text=True)
    assert run.returncode != 0 and "CI shard must be" in run.stderr
