"""Owned-job accounting excludes the idle PID sentinel and includes children."""

import pytest

from scripts import run_exact_ranker_experiment as supervisor


@pytest.mark.parametrize("child,expected_kib", [(0, 300), (20, 1000)])
def test_owned_rss_excludes_system_tree_and_tracks_active_child(monkeypatch, child, expected_kib):
    monkeypatch.setattr(supervisor.os, "getpid", lambda: 10)
    # launchd/system descendants must not become owned when child=0.
    listing = "1 0 900 launchd\n2 1 800 system\n10 1 100 reporter\n11 10 200 helper\n20 1 300 benchmark\n21 20 400 helper\n"
    monkeypatch.setattr(supervisor.subprocess, "check_output", lambda *a, **kw: listing)
    owned, foreign = supervisor.owned_rss(child)
    assert owned == expected_kib * 1024
    assert foreign == []
