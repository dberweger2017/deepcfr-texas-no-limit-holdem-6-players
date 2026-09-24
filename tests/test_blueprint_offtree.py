"""Public-action replay distinguishes off-menu play from sparse table lookup."""

from dataclasses import replace

import pytest

from scripts.check_blueprint_offtree import main as diagnose
from src.arena.schedule import Plan
from src.blueprint.artifact import save_training
from src.blueprint.evaluation import evaluate
from src.blueprint.offtree import classify_history, lookup_source
from src.blueprint.solver import BlueprintTrainer, PilotConfig
from src.game.hand import Hand, Table
from src.game.observation import ActionTaken
from src.game.types import Action, ActionKind, Street


def _hand(players=2):
    table = Table(
        tuple(f"p{seat}" for seat in range(players)),
        (200,) * players,
        small_blind=1,
        big_blind=2,
        chip_unit="1",
    )
    return Hand.start(table, hand_id="off-tree", seed=17)


def test_exact_raise_sizes_and_cap_are_classified_from_public_replay():
    hand = _hand()
    for _ in range(3):
        view = hand.observe(hand.actor)
        hand = hand.apply(Action(ActionKind.RAISE, view.legal_actions.min_raise_to))
    observed = hand.observe(hand.actor)
    trace = classify_history(observed)
    assert [(row.in_menu, row.reason) for row in trace] == [
        (True, None),
        (True, None),
        (False, "raise_cap"),
    ]
    assert all(row.street == Street.PREFLOP for row in trace)
    assert classify_history(replace(observed, hole_cards=("As", "Ah"))) == trace


def test_early_jam_is_off_menu_even_though_legally_executed():
    hand = _hand(6)
    view = hand.observe(hand.actor)
    hand = hand.apply(Action(ActionKind.RAISE, view.legal_actions.max_raise_to))
    observed = hand.observe(hand.actor)
    assert classify_history(observed)[0].reason == "raise_size"
    assert lookup_source(observed, trained=False) == "fallback_after_off_tree"
    assert lookup_source(observed, trained=True) == "trained_after_off_tree"


def test_regular_nonraise_stays_in_menu():
    hand = _hand()
    hand = hand.apply(Action(ActionKind.CALL))
    assert classify_history(hand.observe(hand.actor))[0].in_menu
    assert lookup_source(hand.observe(hand.actor), trained=False) == "fallback_in_tree"


def test_mismatched_decision_action_pair_is_rejected():
    hand = _hand()
    hand = hand.apply(Action(ActionKind.CALL))
    observed = hand.observe(hand.actor)
    events = list(observed.history)
    index = next(i for i, event in enumerate(events) if isinstance(event, ActionTaken))
    events[index] = replace(events[index], seat=1 - events[index].seat)
    with pytest.raises(ValueError, match="different actors"):
        classify_history(replace(observed, history=tuple(events)))


def test_diagnostic_counts_reconcile_with_candidate_decisions(tmp_path):
    from json import dumps, loads

    trainer = BlueprintTrainer(
        _hand().table,
        PilotConfig(seed=13, raise_cap=2, max_nodes=5000, max_seconds=10),
    )
    trainer.step()
    checkpoint = tmp_path / "checkpoint.json.gz"
    digest = save_training(trainer, checkpoint)
    plan = tmp_path / "plan.json"
    schedule = {
        "scenarios": [
            {
                "name": "two-player",
                "stacks": [200, 200],
                "small_blind": 1,
                "big_blind": 2,
                "chip_unit": "1",
            }
        ],
        "candidate": "blueprint_live",
        "baseline": "blueprint_uniform",
        "opponents": ["random"],
        "blocks": 2,
        "root_seed": 99,
        "split": "validation",
    }
    plan.write_text(dumps({"evaluations": {"random": schedule}}))
    output = tmp_path / "diagnostic"
    assert (
        diagnose(
            [
                "--checkpoint",
                str(checkpoint),
                "--expected-sha256",
                digest,
                "--plan",
                str(plan),
                "--out",
                str(output),
            ]
        )
        == 0
    )
    result = loads((output / "result.json").read_text())["evaluations"]["random"]
    assert result["arena"]["report"]["status"] == "valid"
    plain = evaluate(trainer, Plan.from_dict(schedule))
    assert result["arena"]["report"] == plain["report"]
    assert result["arena"]["coverage"] == plain["coverage"]
    for street, coverage in result["arena"]["coverage"].items():
        assert sum(result["lookup_sources"][street].values()) == coverage["decisions"]
