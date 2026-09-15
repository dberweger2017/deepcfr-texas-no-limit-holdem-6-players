from collections import Counter
from dataclasses import replace
from itertools import product
from random import Random

import pytest

from src.game.hand import Hand
from src.holdem.actions import bet_candidates
from src.holdem.collection import collect_phase
from src.holdem.policy import FrozenProfile
from src.holdem.replay import ReplaySample, RoleReservoir, split_collection
from src.holdem.targets import action_targets
from tests.test_hand_observations import table


def example(iteration=1):
    hand = Hand.start(table(4, (3,) * 4), hand_id="memory", seed=7)
    candidates = bet_candidates(hand.observe(hand.actor))
    target = action_targets(
        candidates,
        (1 / len(candidates.actions),) * len(candidates.actions),
        {a: float(i) for i, a in enumerate(candidates.actions)},
    )
    return ReplaySample(hand.actor, iteration, "a" * 64, 17, 0, target)


@pytest.fixture(scope="module")
def collected():
    return collect_phase(
        table(4, (10,) * 4), FrozenProfile([None] * 4), iteration=1, seed=7
    )


def test_reservoir_is_exactly_uniform_over_subsets():
    samples = tuple(example(i + 1) for i in range(5))
    counts = Counter()

    class Draws:
        def __init__(self, values):
            self.values = iter(values)

        def randrange(self, stop):
            result = next(self.values)
            assert 0 <= result < stop
            return result

    # All equally likely replacement streams for a capacity-two, five-item reservoir.
    for draws in product(range(3), range(4), range(5)):
        memory = RoleReservoir(3, 2, 0)
        memory._random = Draws(draws)
        memory.extend(samples)
        assert memory.seen == 5 and len(memory) == 2
        counts[tuple(sorted(s.iteration for s in memory.items))] += 1
    assert len(counts) == 10 and set(counts.values()) == {6}


def test_sampling_does_not_change_admission_and_clones_are_independent():
    samples = tuple(example(i + 1) for i in range(10))
    memory = RoleReservoir(3, 3, 51)
    memory.extend(samples[:5])
    clone = memory.clone()
    before = memory.items, memory.seen, memory._random.getstate()
    batch = memory.sample(100, Random(1))
    assert set(batch) <= set(memory.items)
    assert (memory.items, memory.seen, memory._random.getstate()) == before
    memory.extend(samples[5:])
    assert clone.items == before[0] and clone.seen == 5
    clone.extend(samples[5:])
    assert (
        clone.items == memory.items
        and clone._random.getstate() == memory._random.getstate()
    )


def test_invalid_admission_is_atomic_even_after_valid_samples():
    memory = RoleReservoir(3, 2, 1)
    memory.extend([example()])
    before = memory.items, memory.seen, memory._random.getstate()
    with pytest.raises(ValueError):
        memory.extend([example(2), replace(example(3), role=2)])
    assert (memory.items, memory.seen, memory._random.getstate()) == before
    with pytest.raises(ValueError):
        RoleReservoir(2, 2, 1).sample(1, Random(0))


def test_split_keeps_owner_actions_and_provenance(collected):
    roles = split_collection(collected)
    assert len(roles) == 4
    for role, samples in enumerate(roles):
        expected = collected.traversals[role]
        assert len(samples) == len(expected.targets)
        for index, sample in enumerate(samples):
            assert sample.role == role and sample.iteration == 1
            assert sample.profile == collected.profile
            assert sample.action_seed == expected.action_seed
            assert sample.target_index == index
            assert sample.target is expected.targets[index]
            sample.validate()


@pytest.mark.parametrize(
    "bad",
    [
        "missing-role",
        "duplicate-role",
        "profile",
        "seed",
        "missing-target",
        "repeated-target",
        "owner",
    ],
)
def test_split_rejects_incomplete_or_misattributed_batches(collected, bad):
    first = collected.traversals[0]
    if bad == "missing-role":
        broken = replace(collected, traversals=collected.traversals[:-1])
    elif bad == "duplicate-role":
        broken = replace(collected, traversals=(first,) * 4)
    elif bad == "profile":
        broken = replace(collected, profile="b" * 64)
    elif bad == "seed":
        broken = replace(collected, seed=999)
    else:
        if bad == "missing-target":
            first = replace(first, targets=first.targets[:-1])
        elif bad == "repeated-target":
            first = replace(first, targets=first.targets + first.targets[:1])
        else:
            first = replace(first, targets=collected.traversals[1].targets)
        broken = replace(collected, traversals=(first,) + collected.traversals[1:])
    with pytest.raises(ValueError):
        split_collection(broken)


@pytest.mark.parametrize(
    "bad", ["iteration", "profile", "nan", "policy", "regret", "amount"]
)
def test_replay_checks_exact_candidate_inputs_and_target_consistency(bad):
    sample = example()
    if bad == "iteration":
        sample = replace(sample, iteration=0)
    elif bad == "profile":
        sample = replace(sample, profile="unknown")
    else:
        target = sample.target
        if bad == "nan":
            target = replace(target, values_bb=(float("nan"),) * len(target.values_bb))
        elif bad == "policy":
            target = replace(target, policy=(0.0,) * len(target.policy))
        elif bad == "regret":
            target = replace(target, regrets_bb=tuple(v + 1 for v in target.regrets_bb))
        else:
            candidates = target.candidates
            target = replace(
                target,
                candidates=replace(candidates, features=candidates.features[::-1]),
            )
        sample = replace(sample, target=target)
    with pytest.raises(ValueError):
        sample.validate()
