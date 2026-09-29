"""The scaling instrument shares ranges but leaves traversal draws independent."""

import json
from collections import Counter
from pathlib import Path
from random import Random
from time import monotonic

from scripts.diagnose_local_cfr import _view
from scripts.scale_local_cfr import _attempt, _seed
from src.blueprint.abstraction import choices
from src.blueprint.local_cfr import LocalCFRConfig, _LocalSolver, _flop_root, _root_ranges


ROOT = Path(__file__).resolve().parents[1]


class UniformBlueprint:
    def distribution(self, view):
        menu = choices(view)
        return menu, (1 / len(menu),) * len(menu), False


def test_frozen_ranges_are_shared_without_consuming_traversal_randomness():
    case = json.loads((ROOT / "configs/blueprint/local-cfr-conditional-cases.json").read_text())["cases"][0]
    plan = json.loads((ROOT / "configs/blueprint/local-cfr-scaling-m4.json").read_text())
    view = _view(case)
    blueprint = UniformBlueprint()
    ranges = _root_ranges(blueprint, view, _flop_root(view)[0],
                          Random(_seed(plan["seed_namespace"], case["id"], "public-range")),
                          8, monotonic() + 30, Counter())
    config = LocalCFRConfig(max_seconds=5, range_samples=8,
                            min_cycles=1, max_cycles=1)
    seed = _seed(plan["seed_namespace"], case["id"], "traversal", 0)
    random = Random(seed)
    solver = _LocalSolver(blueprint, view, random, config, monotonic() + 5,
                          Counter(), root_ranges=ranges, snapshot_seconds=(0.0,))
    assert solver.root_ranges is ranges
    assert random.getstate() == Random(seed).getstate()
    solver.solve()
    assert solver.time_snapshots[0]["cycles"] == 1
    assert solver.time_snapshots[0]["target_visits"] >= 1


def test_scaling_attempt_retains_final_completed_cycle_and_shared_seed():
    case = json.loads((ROOT / "configs/blueprint/local-cfr-conditional-cases.json").read_text())["cases"][0]
    plan = json.loads((ROOT / "configs/blueprint/local-cfr-scaling-m4.json").read_text())
    plan["local_cfr"] = {**plan["local_cfr"], "range_samples": 8,
                         "min_cycles": 1, "max_cycles": 1}
    view = _view(case)
    blueprint = UniformBlueprint()
    ranges = _root_ranges(blueprint, view, _flop_root(view)[0], Random(91),
                          8, monotonic() + 30, Counter())
    targeted = _attempt(blueprint, view, ranges, case, 0, True, plan, 5, 1)
    ordinary = _attempt(blueprint, view, ranges, case, 0, False, plan, 5, 1)
    assert targeted["traversal_seed"] == ordinary["traversal_seed"]
    assert targeted["status"] == "completed"
    assert targeted["final_completed_cycle_snapshot"]["cycles"] == 1
    assert ordinary["final_completed_cycle_snapshot"]["cycles"] == 1
