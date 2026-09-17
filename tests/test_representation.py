import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from scripts.check_representation import evaluate, fit, qualifies, run, select, verify
from src.holdem.betting import BettingNetwork
from src.holdem.encoding import _canonical_cards
from src.holdem.representation_models import VARIANTS, make_model, scaled_candidates
from src.holdem.representation_reference import (
    board_key,
    build_context,
    compatible_deals,
    range_support,
    specifications,
)
from src.holdem.river_reference import ReferenceProfile, enumerate_reference
from src.solver.neural.network import deterministic_cpu


@pytest.fixture
def plan():
    return json.loads(Path("configs/holdem/representation.json").read_text())


@pytest.fixture
def target(plan):
    context = build_context(specifications(plan)[1])
    return enumerate_reference(
        context, ReferenceProfile("uniform"), max_nodes=20000, deadline=float("inf")
    ).target


def test_shared_range_conditions_on_visible_cards_and_preserves_suit_symmetry(plan):
    support = range_support(plan["range_templates"])
    assert len(support) == 48
    mapping = dict(zip("cdhs", "hscd", strict=True))
    transform = lambda cards: tuple(c[0] + mapping[c[1]] for c in cards)
    assert Counter(transform(d) for d in support) == Counter(support)
    spec = specifications(plan)[0]
    original = compatible_deals(support, spec["board"], spec["holding"])
    changed = compatible_deals(
        support, transform(spec["board"]), transform(spec["holding"])
    )
    assert Counter(transform(d) for d in original) == Counter(changed)
    assert all(
        not set(d).intersection(spec["board"] + list(spec["holding"])) for d in original
    )
    with pytest.raises(ValueError, match="seven distinct"):
        compatible_deals(support, spec["board"], spec["board"][:2])


def test_splits_keep_whole_boards_and_reject_equivalent_boards(plan):
    specs = specifications(plan)
    assert Counter(s["split"] for s in specs) == {
        "train": 192,
        "validation": 48,
        "test": 48,
    }
    assert len({board_key(s["board"]) for s in specs}) == 24
    assert all(
        len({s["split"] for s in specs if s["board_index"] == i}) == 1
        for i in range(24)
    )
    changed = json.loads(json.dumps(plan))
    changed["boards"][-1]["cards"] = changed["boards"][0]["cards"][::-1]
    with pytest.raises(ValueError, match="equivalent"):
        specifications(changed)


def test_exact_reference_is_invariant_to_suit_relabeling(plan):
    spec = specifications(plan)[1]
    mapping = dict(zip("cdhs", "shdc", strict=True))
    transform = lambda cards: tuple(c[0] + mapping[c[1]] for c in cards)
    changed = {
        **spec,
        "board": transform(spec["board"]),
        "holding": transform(spec["holding"]),
        "deals": tuple(transform(d) for d in spec["deals"]),
    }
    contexts = [build_context(s) for s in (spec, changed)]
    targets = []
    for c in contexts:
        assert all(
            w.observe(w.actor) == c.worlds[0].observe(c.worlds[0].actor)
            for w in c.worlds
        )
        targets.append(
            enumerate_reference(
                c,
                ReferenceProfile("increasing"),
                max_nodes=20000,
                deadline=float("inf"),
            ).target
        )
    assert targets[0].candidates.decision == targets[1].candidates.decision
    assert targets[0].values_bb == pytest.approx(targets[1].values_bb)
    assert targets[0].regrets_bb == pytest.approx(targets[1].regrets_bb)


def test_scaling_preserves_cards_legality_and_source(target):
    before = target.candidates
    after = scaled_candidates(before)
    assert after.actions == before.actions
    assert after.decision.source is before.decision.source
    assert after.decision.cards == before.decision.cards
    assert after.decision.context != before.decision.context
    assert after.features != before.features
    assert all(
        a[-52:] == b[-52:]
        for a, b in zip(after.decision.events, before.decision.events)
    )


def test_original_model_is_bit_identical_and_larger_models_are_parameter_matched(
    target,
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        original = BettingNetwork(32)
    reference = make_model("original", 17)
    scaled = make_model("scaled", 17)
    assert all(
        torch.equal(v, reference.state_dict()[k])
        for k, v in original.state_dict().items()
    )
    assert all(
        torch.equal(v, scaled.state_dict()[k]) for k, v in original.state_dict().items()
    )
    with deterministic_cpu(), torch.no_grad():
        assert torch.equal(
            original([target.candidates])[0].regrets,
            reference([target.candidates])[0].regrets,
        )
        for variant in VARIANTS:
            scores = make_model(variant, 17)([target.candidates])[0]
            assert scores.candidates is target.candidates
            assert torch.isfinite(scores.regrets).all()
            assert scores.probabilities().sum() == pytest.approx(1)
    counts = {
        v: sum(p.numel() for p in make_model(v, 17).parameters()) for v in VARIANTS
    }
    assert counts["original"] == counts["scaled"] == 25602
    assert all(abs(counts[v] / counts["wide"] - 1) < 0.06 for v in ("deep", "cards"))


def test_fits_reproduce_without_changing_global_rng(plan, target):
    plan = {**plan, "fit_steps": 2, "measure_steps": [0, 2], "batch_size": 2}
    state = torch.get_rng_state().clone()
    with deterministic_cpu():
        a, first = fit(
            "deep", 17, {"train": [target], "validation": [target]}, plan, float("inf")
        )
        b, second = fit(
            "deep", 17, {"train": [target], "validation": [target]}, plan, float("inf")
        )
    assert torch.equal(state, torch.get_rng_state())
    assert first["curves"] == second["curves"]
    assert all(
        torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict()
    )


def test_selection_requires_each_seed_and_does_not_use_test(plan):
    def measurement(v, seed, cost):
        return {
            "variant": v,
            "seed": seed,
            "parameters": 10,
            "curves": [
                {
                    "train": {"relative_rmse": 0.1},
                    "validation": {"mean_decision_cost_bb": cost, "relative_rmse": 0.2},
                }
            ],
        }

    fits = [
        measurement(v, seed, 0.3 if v == "original" else 0.2)
        for v in plan["variants"]
        for seed in plan["seeds"]
    ]
    assert select(fits, plan)["selected"] == "cards"
    for f in fits:
        if f["seed"] == plan["seeds"][0] and f["variant"] != "original":
            f["curves"][-1]["validation"]["mean_decision_cost_bb"] = 0.4
    assert select(fits, plan)["selected"] is None
    assert not qualifies(
        {"mean_decision_cost_bb": 0, "relative_rmse": 0.5},
        {"mean_decision_cost_bb": 0.3, "relative_rmse": 0.2},
        plan,
    )


def test_failure_is_retained_without_fits(plan, tmp_path):
    bad = {**plan, "boards": [plan["boards"][0], plan["boards"][0]]}
    with pytest.raises(ValueError, match="equivalent"):
        run(bad, tmp_path / "failed")
    report = json.loads((tmp_path / "failed/report.json").read_text())
    assert report["status"] == "failed"
    assert report["fits"] == []


def test_complete_run_reloads_all_models_and_rejects_changed_artifacts(plan, tmp_path):
    small = {
        **plan,
        "boards": [plan["boards"][0], plan["boards"][16], plan["boards"][20]],
        "hands_per_board": 1,
        "seeds": [19],
        "variants": ["original", "scaled"],
        "fit_steps": 2,
        "batch_size": 2,
        "measure_steps": [0, 2],
    }
    out = tmp_path / "complete"
    result = run(small, out)
    assert result["status"] == "completed"
    assert len(result["fits"]) == 2
    assert len(result["references"]) == 6
    assert result["selection"]["selected"] is None
    assert [x["variant"] for x in result["test"]] == ["original"]
    assert verify(out)["models"] == 2
    with (out / "original-19.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Artifact changed"):
        verify(out)


def test_fitting_rejects_test_targets(plan, target):
    with pytest.raises(ValueError, match="test targets"):
        fit("original", 19, {"train": [target], "test": [target]}, plan, float("inf"))
