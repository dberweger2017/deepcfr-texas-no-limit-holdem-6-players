"""Prospective selection cannot substitute accuracy, missing roots or paid work."""

from copy import deepcopy

import pytest

from src.diagnostics.hu20_search_protocol import freeze_part_a, power, qualify_curve, select_base


def row(config, root, *, residual=.3, seconds=10, blueprint=50, **extra):
    return {"configuration_id":config, "config":{"menu":"native", "opponent_likelihood_floor":.01},
            "root":root, "weight":1, "cold_seconds":seconds, "residual_pct_pot":residual,
            "blueprint_pct_pot":blueprint, "full_native_verified":True,
            "reference_supported":True, "played_strategy_verified":True, **extra}


def test_fastest_strict_configuration_wins_over_more_accurate():
    rows = [row("accurate",r,residual=.1,seconds=20) for r in ("a","b")]
    rows += [row("fast",r,residual=.49,seconds=8) for r in ("a","b")]
    result=qualify_curve(rows,["a","b"])
    assert result["tier"]=="strict" and result["selected"]["configuration_id"]=="fast"
    assert len(result["full_curve"])==2


def test_relaxed_is_used_only_when_no_strict_candidate_qualifies():
    rows=[row("strict",r,seconds=25) for r in ("a","b")]
    rows += [row("relaxed",r,residual=.8,seconds=5) for r in ("a","b")]
    assert qualify_curve(rows,["a","b"])["selected"]["configuration_id"]=="strict"
    result=qualify_curve(rows[2:],["a","b"])
    assert result["tier"]=="relaxed"
    rows[2]["blueprint_pct_pot"]=8
    assert qualify_curve(rows[2:],["a","b"])["status"]=="owner-decision-needed"


def test_missing_duplicate_tampered_or_reduced_roots_cannot_qualify():
    assert qualify_curve([row("one","a")],["a","b"])["selected"] is None
    assert qualify_curve([row("one","a"),row("one","a")],["a","b"])["selected"] is None
    for field in ("full_native_verified","reference_supported","played_strategy_verified"):
        assert qualify_curve([row("one","a",**{field:False})],["a"])["selected"] is None
    reduced=row("one","a");reduced["config"]["menu"]="cap2"
    assert qualify_curve([reduced],["a"])["selected"] is None


def test_fallback_quality_is_included_in_the_gate():
    rows=[row("one","a"),row("one","b",residual=50,fallback=True)]
    assert qualify_curve(rows,["a","b"])["selected"] is None


def test_latency_tails_and_timeout_causes_are_reported_per_host():
    rows=[row("one","a",host="m4",seconds=5),
          row("one","b",host="pod",seconds=30,fallback=True,
              failures=[{"phase":"play","cause":"timeout"}])]
    selected=qualify_curve(rows,["a","b"])["selected"]
    assert selected["cold_p99_seconds"]==pytest.approx(29.75)
    assert selected["cold_max_seconds"]==30
    assert selected["latency_by_host"]["m4"]["timeout_fallback_rate"]==0
    assert selected["latency_by_host"]["pod"]["timeout_fallback_rate"]==1
    assert selected["latency_by_host"]["pod"]["fallback_causes"]=={"timeout":1}


def test_base_defaults_average_and_switches_only_beyond_frozen_upper_margin():
    summary={"three_lineage_changes":[{"panel":p,"average_minus_current":{"ci95":[-50,upper]}}
        for p,upper in [("lbr",-10),("native-pressure",-20),("selective-stackoff",-10)]]}
    assert select_base(summary)["base"]=="average"
    summary["three_lineage_changes"][1]["average_minus_current"]["ci95"][1]=-20.001
    assert select_base(summary)["base"]=="current"
    assert select_base(summary)["regressions"]==["native-pressure"]
    summary["three_lineage_changes"].pop()
    with pytest.raises(ValueError):select_base(summary)


def test_power_and_budget_freeze_are_outcome_independent():
    assert power(2048)>.8 and power(128)<.2
    plan={"stage":"planned-not-admitted","panels":[{"name":"lbr","blocks":2048},
        {"name":"native-pressure","blocks":2048}],"power":{"prior_paired_sd":317.8}}
    saved=deepcopy(plan)
    frozen=freeze_part_a(plan,{"lbr":[1,2],"native-pressure":[.1]},load_seconds=10,pilot_hash="sealed")
    assert plan==saved and frozen["stage"]=="frozen-final"
    assert frozen["panels"][0]["blocks"]<2048
    assert frozen["timing_admission"]["forecast_seconds"]<=21600
    assert frozen["power"]["achieved_design_power"]==power(frozen["panels"][0]["blocks"])
    with pytest.raises(ValueError):freeze_part_a(frozen,{},load_seconds=0,pilot_hash="sealed")
