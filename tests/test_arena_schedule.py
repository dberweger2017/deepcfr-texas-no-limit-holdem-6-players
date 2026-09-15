from dataclasses import asdict, replace

import pytest

from src.arena.schedule import (
    Plan,
    Scenario,
    build_schedule,
    digest,
    schedule_document,
    stream_seed,
)


def test_schedule_roundtrip_and_expansion_preserve_existing_blocks():
    plan = Plan((Scenario("six"),), blocks=3, root_seed=42)
    assert Plan.from_dict(asdict(plan)) == plan
    assert schedule_document(plan) == schedule_document(Plan.from_dict(asdict(plan)))
    assert build_schedule(replace(plan, blocks=5))[:3] == build_schedule(plan)
    assert build_schedule(
        replace(plan, scenarios=plan.scenarios + (Scenario("four", (2000,) * 4),))
    )[:3] == build_schedule(plan)


def test_streams_and_splits_are_separate():
    streams = ("deal", "action", "opponent", "training")
    seeds = {
        stream_seed(0, split, stream, i)
        for split in ("train", "validation", "test")
        for stream in streams
        for i in range(100)
    }
    assert len(seeds) == 1200
    for index, split in enumerate(("train", "validation", "test")):
        assert all(stream_seed(0, split, "deal", i) >> 62 == index for i in range(100))


def test_candidate_and_pool_changes_do_not_change_deals_or_action_streams():
    plan = Plan((Scenario("six"),), blocks=2)
    other = replace(plan, candidate="random", opponents=("random", "fold"))
    for left, right in zip(build_schedule(plan), build_schedule(other)):
        assert left.deal_seeds == right.deal_seeds
        assert left.action_seeds == right.action_seeds
    assert digest(schedule_document(plan)) != digest(schedule_document(other))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"blocks": 0},
        {"blocks": True},
        {"split": "unknown"},
        {"root_seed": -1},
        {"root_seed": 2**64},
        {"max_decisions": 0},
    ],
)
def test_invalid_plans_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Plan((Scenario("six"),), **kwargs)


def test_session_schedule_contains_a_sequence_of_deals():
    plan = Plan((Scenario("session", mode="session", hands_per_rotation=10),))
    assert len(build_schedule(plan)[0].deal_seeds) == 10
    with pytest.raises(ValueError):
        Scenario("bad", mode="fixed", hands_per_rotation=2)
