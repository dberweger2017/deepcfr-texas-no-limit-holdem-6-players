from copy import deepcopy
from math import sqrt
from statistics import mean, stdev

import pytest

from src.arena.report import comparison, estimate, summarize
from src.arena.schedule import Plan, Scenario
from tests.test_arena_runner import collect


def test_interval_matches_a_known_t_critical_value():
    values = list(range(30))
    result = estimate(values)
    margin = 2.045229642132703 * stdev(values) / sqrt(30)
    assert result["bb_per_100"] == mean(values)
    assert result["ci95"] == pytest.approx([14.5 - margin, 14.5 + margin])


def test_small_or_degenerate_samples_are_inconclusive():
    assert estimate([1, 2])["ci95"] is None
    assert estimate([1] * 100)["ci95"] is None
    assert comparison([1] * 100, [0] * 100)["conclusion"] == "inconclusive"
    assert (
        comparison([float(i) for i in range(30)], [float(i) for i in range(30)])[
            "conclusion"
        ]
        == "inconclusive"
    )


def test_paired_interval_uses_block_differences():
    baseline = [float(i * 100) for i in range(30)]
    candidate = [x + 1 + (i % 2) for i, x in enumerate(baseline)]
    result = comparison(candidate, baseline)
    assert result["conclusion"] == "candidate_better"
    assert result["paired_difference"]["bb_per_100"] == 1.5
    assert (
        result["paired_difference"]["ci95"][1] - result["paired_difference"]["ci95"][0]
        < 1
    )
    assert comparison(baseline, candidate)["conclusion"] == "baseline_better"


def test_rotations_are_not_independent_samples_and_failures_null_all_estimates():
    plan = Plan((Scenario("four", (2000,) * 4),), blocks=2)
    _, rows, _ = collect(plan)
    report = summarize(plan, rows)
    assert report["status"] == "valid"
    assert report["scenarios"]["four"]["comparison"]["paired_difference"]["blocks"] == 2
    assert (
        report["scenarios"]["four"]["comparison"]["paired_difference"]["ci95"] is None
    )
    for corrupt in (rows[:-1], rows + rows[:1]):
        invalid = summarize(plan, corrupt)
        assert invalid["status"] == "invalid"
        assert invalid["scenarios"]["four"]["comparison"] is None
    altered = deepcopy(rows)
    altered[0]["candidate_chips"] += 1
    assert summarize(plan, altered)["status"] == "invalid"
