"""Frozen conditional cases exercise only public observations at solve time."""

import json
from hashlib import sha256
from pathlib import Path

from scripts.diagnose_local_cfr import _measure, _plan_hash, _view
from src.blueprint.abstraction import choices


ROOT = Path(__file__).resolve().parents[1]


class UniformBlueprint:
    def distribution(self, view):
        menu = choices(view)
        return menu, (1 / len(menu),) * len(menu), False


def test_frozen_conditional_cases_are_eligible_and_reproducible():
    plan = json.loads((ROOT / "configs/blueprint/local-cfr-conditional-m4.json").read_text())
    frozen = json.loads((ROOT / "configs/blueprint/local-cfr-conditional-cases.json").read_text())
    assert frozen["plan_sha256"] == _plan_hash(plan)
    assert len(frozen["cases"]) == 48
    assert len({case["observation_sha256"] for case in frozen["cases"]}) == 48
    assert {key for case in frozen["cases"] for key in case} == {
        "id", "deal_seed", "button", "position", "prefix", "search_seed",
        "hero_seat", "observation_sha256",
    }
    for case in frozen["cases"]:
        view = _view(case)
        assert view.seat == case["hero_seat"]
        assert sha256(repr(view).encode()).hexdigest() == case["observation_sha256"]


def test_conditional_modes_measure_the_same_first_decision():
    plan = json.loads((ROOT / "configs/blueprint/local-cfr-conditional-m4.json").read_text())
    plan["local_cfr"] = {**plan["local_cfr"], "range_samples": 8,
                         "min_cycles": 1, "max_cycles": 1}
    case = json.loads((ROOT / "configs/blueprint/local-cfr-conditional-cases.json").read_text())[
        "cases"
    ][0]
    view = _view(case)
    targeted = _measure(UniformBlueprint(), view, case, True, plan)
    ordinary = _measure(UniformBlueprint(), view, case, False, plan)
    assert targeted["status"] == "completed"
    assert targeted["target_visits"] >= 1
    assert targeted["cycles"] == ordinary["cycles"] == 1
    assert ordinary["status"] in {"completed", "target_unvisited"}
    assert targeted["target_public_range_holdings_visited"] >= 1
