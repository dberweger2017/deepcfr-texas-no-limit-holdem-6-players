"""N-player CLI admission and optional pinned-K50 cash traversal/recovery gates."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

BINARY = Path("native/hu20-trainer/target/release/examples/nplayer_train")
pytestmark = pytest.mark.skipif(not BINARY.exists(), reason="build the nplayer_train example")


@pytest.mark.parametrize("args", [
    ["--players", "1"], ["--players", "7"], ["--players", "3", "--players", "3"],
    ["--unknown", "1"], ["--players"],
    ["--players", "3", "--stacks", "2000,2000"],
    ["--players", "3", "--stack-bb", "20", "--stacks", "2000,2000,2000"],
    ["--players", "3", "--menu", "v2"],
])
def test_invalid_admission_never_starts_a_run(args):
    result = subprocess.run([str(BINARY), *args], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "nplayer_train:" in result.stderr
    assert not result.stdout


@pytest.fixture
def pinned_tables():
    path = os.environ.get("NPLAYER_K50_TABLES")
    if not path:
        pytest.skip("set NPLAYER_K50_TABLES to the verified #163 input directory")
    return Path(path)


def run(tmp_path, tables, name, *, players=3, nodes=20_000, scope=3, cap=1_000_000,
        freeze=False, resume=None, stack=20, expected=0, seed=81):
    path = tmp_path / f"{name}.gz"
    cmd = [str(BINARY), "--players", str(players), "--stack-bb", str(stack),
           "--card-buckets", str(tables), "--seed", str(seed), "--nodes", str(nodes),
           "--last-average-street", str(scope), "--max-entries", str(cap),
           "--max-seconds", "60", "--out", str(path), "--telemetry", str(tmp_path / f"{name}.jsonl")]
    if freeze:
        cmd.append("--freeze-at-cap")
    if resume:
        cmd.extend(["--resume", str(resume), "--resume-sha256", hashlib.sha256(resume.read_bytes()).hexdigest()])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    assert result.returncode == expected, result.stdout + result.stderr
    if expected == 2:
        assert not path.exists()
        return None
    report = json.loads(result.stdout)
    assert report["stop_reason"] == ("capacity" if expected == 3 else "nodes")
    assert report["identity"]["players"] == players
    assert report["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads((tmp_path / f"{name}.jsonl").read_text().splitlines()[-1]) == report
    return path


def rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


@pytest.mark.parametrize("players", range(2, 7))
def test_same_traversal_acts_for_every_cash_seat(tmp_path, pinned_tables, players):
    path = run(tmp_path, pinned_tables, f"n{players}", players=players, stack=2, nodes=1000)
    header, *entries = rows(path)
    assert all(n > 0 for n in header["counts"]["traverser"][:players])
    assert all(n == 0 for n in header["counts"]["traverser"][players:])
    assert {e[1] for e in entries} == set(range(players))


@pytest.mark.parametrize("freeze,scope", [(False, 3), (True, 0)])
def test_cash_fresh_resume_and_frozen_bytes(tmp_path, pinned_tables, freeze, scope):
    options = dict(freeze=freeze, scope=scope, cap=200 if freeze else 1_000_000)
    prefix = run(tmp_path, pinned_tables, "prefix", nodes=10_000, **options)
    fresh = run(tmp_path, pinned_tables, "fresh", **options)
    resumed = run(tmp_path, pinned_tables, "resumed", resume=prefix, **options)
    assert fresh.read_bytes() == resumed.read_bytes()
    header, *entries = rows(fresh)
    if freeze:
        assert header["frozen_at"] is not None
        assert header["counts"]["discarded_keys"] > 0
        assert {e[0] for e in entries} == {e[0] for e in rows(prefix)[1:]}
    for entry in entries:
        assert (entry[6] is not None) == (entry[2] <= scope)


def test_resume_configuration_cannot_silently_change(tmp_path, pinned_tables):
    prefix = run(tmp_path, pinned_tables, "prefix", nodes=1000)
    for i, override in enumerate([{"players": 4}, {"seed": 82}, {"scope": 0}, {"cap": 200}]):
        run(tmp_path, pinned_tables, f"reject{i}", resume=prefix, expected=2, **override)
    capped = run(tmp_path, pinned_tables, "capacity", cap=200, expected=3)
    assert rows(capped)[0]["frozen_at"] is None
