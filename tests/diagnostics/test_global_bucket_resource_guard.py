"""Resource stops preserve their triggering sample without relaxing limits."""

import json
from types import SimpleNamespace

import pytest

from scripts import run_global_bucket_validation as runner


@pytest.fixture
def resources(monkeypatch):
    values = dict(rss=3*runner.GIB, family=4*runner.GIB,
                  swap=runner.GIB, free=20*runner.GIB, now=10000)
    monkeypatch.setattr(runner, "rss_for_tree", lambda _: values["rss"])
    monkeypatch.setattr(runner, "family_rss", lambda *_: values["family"])
    monkeypatch.setattr(runner, "swap_usage", lambda: values["swap"])
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=values["free"]))
    monkeypatch.setattr(runner, "time", lambda: values["now"])
    monkeypatch.setattr(runner.guard, "last_log", values["now"], raising=False)
    return values


@pytest.mark.parametrize("field,value,reason", [
    ("family", 8*runner.GIB+1, "8-GiB total owned family ceiling"),
    ("rss", 7*runner.GIB+1, "Owned RSS ceiling"),
    ("swap", 2*runner.GIB+1, "Swap growth ceiling"),
    ("free", 15*runner.GIB-1, "15-GiB free-disk floor"),
    ("now", 16400, "48-hour cap reached closeout reserve"),
])
def test_stop_logs_exact_sample_despite_logging_throttle(tmp_path, resources, field, value, reason):
    resources[field] = value
    budget = dict(swap_baseline_bytes=runner.GIB, deadline_epoch=20000)
    with pytest.raises(RuntimeError, match=reason):
        runner.guard(tmp_path, budget, 123, worker=True)
    sample = json.loads((tmp_path/"family-resources.jsonl").read_text())
    assert sample["guard_failure"] == reason
    assert sample["swap_bytes"] == resources["swap"]
    assert sample["rss_bytes"] == resources["rss"]
    assert sample["family_rss_bytes"] == resources["family"]
    assert sample["free_disk_bytes"] == resources["free"]
    assert sample["timestamp"] == resources["now"]
    assert sample["swap_baseline_bytes"] == runner.GIB
    assert sample["deadline_epoch"] == 20000


def test_original_inclusive_resource_boundaries_still_pass(tmp_path, resources):
    resources.update(rss=7*runner.GIB, family=8*runner.GIB,
                     swap=2*runner.GIB, free=15*runner.GIB, now=16399)
    runner.guard(tmp_path, dict(swap_baseline_bytes=runner.GIB, deadline_epoch=20000),
                 123, worker=True)
    assert "guard_failure" not in (tmp_path/"family-resources.jsonl").read_text()


def test_failed_log_write_cannot_allow_work_to_continue(tmp_path, resources, monkeypatch):
    resources["swap"] = 2*runner.GIB+1
    def failed_append(*_):
        raise OSError("Log unavailable")
    monkeypatch.setattr(runner, "append", failed_append)
    with pytest.raises(RuntimeError, match="Swap growth ceiling"):
        runner.guard(tmp_path, dict(swap_baseline_bytes=runner.GIB, deadline_epoch=20000), 123)
