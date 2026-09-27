import gzip
import json
from pathlib import Path
from random import Random
from types import SimpleNamespace

import pytest

from src.blueprint.windowed import build_index, collect_one


class NeedChoice(Exception):
    def __init__(self, weights):
        self.weights = weights


class EnumeratingRandom:
    def __init__(self, prefix):
        self.prefix = prefix
        self.position = 0

    def choices(self, population, *, weights, k):
        assert k == 1
        if self.position == len(self.prefix):
            raise NeedChoice(weights)
        choice = self.prefix[self.position]
        self.position += 1
        return [choice]


class HiddenGame:
    def __init__(self, profile):
        self.profile = profile

    def stop(self, state, target):
        return state[1] == 3

    def actor(self, state):
        return 1 if state[1] == 1 else 0

    def decision(self, state):
        hidden, phase, first, public = state
        if phase == 0:
            assert first is None and public is None
            # The opponent's private card is deliberately unavailable here.
            p = (0.25, 0.75)[self.profile]
            return (SimpleNamespace(name="A"), SimpleNamespace(name="B")), "first", (p, 1-p)
        if phase == 1:
            return (SimpleNamespace(name="X"), SimpleNamespace(name="Y")), "opponent", (0.9, 0.1)
        p = (0.6, 0.1)[self.profile]
        return (SimpleNamespace(name="L"), SimpleNamespace(name="R")), "second", (p, 1-p)

    def advance(self, state, action):
        hidden, phase, first, public = state
        if phase == 0:
            return hidden, 3 if action.name == "B" else 1, action.name, public
        if phase == 1:
            return hidden, 2, first, action.name
        return hidden, 3, first, public


def exact_production_counters(adapter, hidden):
    """Enumerate the production collector's random choices, then weight paths."""
    result = {}

    def visit(prefix, mass):
        counters = {}
        try:
            collect_one((hidden, 0, None, None), 0, EnumeratingRandom(prefix), adapter, counters)
        except NeedChoice as need:
            for index, weight in enumerate(need.weights):
                visit(prefix + (index,), mass * weight)
        else:
            for key, (_, counts) in counters.items():
                values = result.setdefault(key, [0.0] * len(counts))
                for index, count in enumerate(counts):
                    values[index] += mass * count

    visit((), 1.0)
    return result


def test_collector_matches_independent_own_reach_expectation():
    combined = {"first": [0.0, 0.0], "second": [0.0, 0.0]}
    for profile in (0, 1):
        p_first = (0.25, 0.75)[profile]
        p_second = (0.6, 0.1)[profile]
        expected = {"first": (p_first, 1-p_first),
                    "second": (2*p_first*p_second, 2*p_first*(1-p_second))}
        for hidden, chance in (("rare", 0.2), ("common", 0.8)):
            actual = exact_production_counters(HiddenGame(profile), hidden)
            for key in expected:
                assert actual[key] == pytest.approx(expected[key])
                for index in (0, 1):
                    combined[key][index] += chance * actual[key][index]
    assert combined["second"] == pytest.approx([0.45, 1.55])
    # Opponent reach would halve this mass; a second own-reach factor would
    # produce 2*(.25²+.75²), rather than the exact total of two.
    assert sum(combined["second"]) == pytest.approx(2.0)


def snapshot(path, rows):
    with gzip.open(path, "wt") as destination:
        for row in rows:
            destination.write(json.dumps(row) + "\n")


def test_streamed_snapshot_mean_includes_missing_uniform(tmp_path):
    paths = [tmp_path / f"{index}.gz" for index in range(8)]
    for index, path in enumerate(paths):
        rows = [["a", ["fold", "call"], [1, 0]]]
        if index >= 6:
            rows.append(["b", ["fold", "call"], [0, 1]])
        snapshot(path, rows)
    result = build_index(paths, {"a": (("fold", "call"), [3, 1])}, tmp_path / "index.sqlite")
    assert result["profile_coverage"] == {2: 1, 8: 1}
    import sqlite3
    with sqlite3.connect(tmp_path / "index.sqlite") as db:
        a = db.execute("SELECT current,snapshot,preflop FROM policies WHERE key='a'").fetchone()
        b = db.execute("SELECT current,snapshot FROM policies WHERE key='b'").fetchone()
    assert json.loads(a[0]) == [1, 0]
    assert json.loads(a[1]) == [1, 0]
    assert json.loads(a[2]) == [3, 1]
    assert json.loads(b[0]) == [0, 1]
    assert json.loads(b[1]) == pytest.approx([0.375, 0.625])


def test_streamed_snapshot_rejects_menu_mismatch(tmp_path):
    paths = [tmp_path / f"{index}.gz" for index in range(8)]
    for index, path in enumerate(paths):
        snapshot(path, [["a", ["fold", "call" if index else "check"], [0.5, 0.5]]])
    with pytest.raises(ValueError, match="action-menu"):
        build_index(paths, {}, tmp_path / "index.sqlite")
