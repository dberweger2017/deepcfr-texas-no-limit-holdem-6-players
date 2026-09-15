from dataclasses import asdict, replace

import pytest

from src.arena.catalog import OpponentPool
from src.arena.heuristics import STYLES, hand_score
from src.arena.policies import make_policy
from src.arena.schedule import Plan, Scenario
from src.game.hand import Hand, Table
from src.game.types import ActionKind
from tests.test_arena_runner import collect


@pytest.mark.parametrize("players", [4, 5, 6])
def test_style_pool_plays_legal_unequal_stack_hands(players):
    plan = Plan(
        (Scenario("styles", tuple(2000 * (i + 1) for i in range(players))),),
        candidate="tight_aggressive",
        baseline="loose_passive",
        opponents=tuple(STYLES),
        blocks=5,
    )
    ok, rows, _ = collect(plan)
    assert ok and all(sum(row["net_chips"]) == 0 for row in rows)
    assert rows == collect(plan)[1]


def test_styles_have_distinct_tendencies_and_only_use_owner_information():
    table = Table(tuple(f"p{i}" for i in range(6)), (10000,) * 6, button=3)
    view = Hand.start(table, hand_id="test", seed=4).observe(0)
    premium = replace(view, hole_cards=("As", "Ah"))
    trash = replace(view, hole_cards=("7s", "2h"))
    assert hand_score(premium) > hand_score(trash)
    rates = {}
    for name in STYLES:
        actions = [
            make_policy(name, seed).choose_action(premium) for seed in range(100)
        ]
        rates[name] = sum(a.kind == ActionKind.RAISE for a in actions)
        for action in actions:
            premium.legal_actions.validate(action)
    assert rates["tight_aggressive"] > rates["tight_passive"] + 50
    assert rates["loose_aggressive"] > rates["loose_passive"] + 50
    for name in STYLES:
        policy = make_policy(name, 0)
        with pytest.raises(TypeError):
            policy.choose_action(object())
        with pytest.raises(ValueError):
            policy.choose_action(replace(view, actor=1))


def test_named_pool_roundtrips_and_cannot_cross_split_purposes():
    pool = OpponentPool(
        "evaluation-v1", "evaluation", ("tight_passive", "pot_pressure")
    )
    plan = Plan((Scenario("six"),), pool=pool)
    assert plan.opponents == pool.members
    assert Plan.from_dict(asdict(plan)) == plan
    with pytest.raises(ValueError, match="pool"):
        replace(plan, split="train")
    with pytest.raises(ValueError, match="pool"):
        replace(plan, opponents=("random",))
    training = OpponentPool("training-v1", "training", ("random", "train_pressure"))
    assert not set(pool.members).intersection(training.members)
    with pytest.raises(ValueError, match="pool"):
        Plan((Scenario("six"),), pool=training, split="test")
