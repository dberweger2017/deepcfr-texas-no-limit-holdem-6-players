"""The v0.4.1 arena report: block pairing, three-lineage means and the predeclared release rule."""

import gzip
import json

import pytest

from scripts import evaluate_hu20_v041_arena as arena
from src.arena.schedule import digest

PANELS = [{"name": "lbr", "blocks": 32}, {"name": "native-pressure", "blocks": 32}, {"name": "uniform", "blocks": 32}]


def write(tmp_path, plan, chips):
    """chips(arm, seed, panel, block, rotation) gives each hand's target chips."""
    for spec in plan["models"]:
        with gzip.open(tmp_path / f"{spec['name']}.hands.jsonl.gz", "wt") as stream:
            for panel in plan["panels"]:
                for block in range(panel["blocks"]):
                    for rotation in (0, 1):
                        stream.write(json.dumps({"arm": spec["arm"], "seed": spec["seed"], "panel": panel["name"],
                            "block": block, "rotation": rotation,
                            "target_chips": chips(spec["arm"], spec["seed"], panel["name"], block, rotation)}) + "\n")
        (tmp_path / f"{spec['name']}.result.json").write_text(json.dumps({"status": "complete", "plan_sha256": digest(plan)}))


def noise(arm, seed, block, rotation):
    return 10 * ((block * (3 + "ROCT".index(arm)) + seed + rotation) % 5)


def plan():
    models = [{"name": f"{arm}-{seed}", "arm": arm, "seed": seed} for arm in arena.ARMS for seed in (1, 2, 3)]
    return {"stage": "frozen-final", "root": 1, "panels": PANELS, "models": models}


def test_candidate_gain_passes_the_release_rule(tmp_path):
    p = plan()
    write(tmp_path, p, lambda arm, seed, panel, block, rotation: noise(arm, seed, block, rotation) + (40 if arm == "O" else 0))
    summary = arena.report(p, tmp_path)
    assert summary["hands"] == 12 * 2 * 96
    assert summary["contrasts_bb_per_100"]["O-R"]["lbr"]["bb_per_100"] == pytest.approx(40, abs=3)
    assert summary["release_rule_passed"]


def test_a_severe_scenario_regression_fails_the_release_rule(tmp_path):
    p = plan()
    def chips(arm, seed, panel, block, rotation):
        gain = 40 if panel != "uniform" else -60
        return (gain if arm == "O" else 0) + noise(arm, seed, block, rotation)
    write(tmp_path, p, chips)
    summary = arena.report(p, tmp_path)
    assert summary["release_checks"] == {"lbr_lower_above_0": True, "native_pressure_lower_above_minus_10": True,
                                         "no_severe_regression": False}
    assert not summary["release_rule_passed"]


def test_missing_hands_are_refused(tmp_path):
    p = plan()
    write(tmp_path, p, lambda *a: 0)
    with gzip.open(tmp_path / "O-1.hands.jsonl.gz", "wt") as stream:
        stream.write("")
    with pytest.raises(ValueError, match="coverage"):
        arena.report(p, tmp_path)


def test_ro_only_reports_no_undeclared_arms(tmp_path):
    p = plan()
    p['models'] = [m for m in p['models'] if m['arm'] in ('R', 'O')]
    write(tmp_path, p, lambda arm, seed, panel, block, rotation:
          noise(arm, seed, block, rotation) + (40 if arm == 'O' else 0))
    result = arena.report(p, tmp_path)
    assert set(result['absolute_bb_per_100']) == {'R', 'O'}
    assert set(result['contrasts_bb_per_100']) == {'O-R'}
    assert result['release_rule_passed']
