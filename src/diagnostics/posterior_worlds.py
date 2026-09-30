"""Paired visible-state-only conditional worlds for the frozen v2 audit."""

from random import Random

from src.arena.schedule import stream_seed
from src.blueprint.search import _sample_world
from src.diagnostics.exact_ranker import RankedCachedLocalBestResponse
from src.diagnostics.posterior_audit_v2 import WORLD_ROOT, holding_from_uniform
from src.diagnostics.robustness import LBRConfig


def paired_world(view, holdings, weights, rank, index):
    # Range labels do not alter these draws: the two ranges share a CDF
    # uniform and corresponding shuffled-runout randomness, not hidden cards.
    uniform = Random(stream_seed(WORLD_ROOT, "test", "deal", "holding", rank, index)).random()
    pair = holding_from_uniform(holdings, weights, uniform)
    random = Random(stream_seed(WORLD_ROOT, "test", "deal", "runout", rank, index))
    return _sample_world(view, {1 - view.seat: ((pair, 1.0),)}, random)


def conditional_row(view, source, cache, holdings, weights, rank, index, guard=lambda: None):
    menu, probabilities, trained = source.distribution(view)
    world = paired_world(view, holdings, weights, rank, index)
    results, work = [], []
    for choice in menu:
        guard()
        hand = world.apply(choice.action)
        random = Random(stream_seed(WORLD_ROOT, "test", "action", "target", rank, index))
        attacker = RankedCachedLocalBestResponse(
            source, stream_seed(WORLD_ROOT, "test", "opponent", "continuation", rank, index),
            cache, LBRConfig(4, 5))
        for _ in range(1000):
            guard()
            if hand.finished:
                final = hand.events[-1].stacks
                if sum(final) != sum(hand.table.stacks):
                    raise ValueError("Conditional native chip settlement mismatch")
                results.append((final[view.seat] - view.players[view.seat].starting_stack) / view.big_blind)
                work.append({"limited": sum(not r["completed"] for r in attacker.telemetry),
                             "requested": sum(r["requested_samples"] for r in attacker.telemetry),
                             "completed": sum(r["samples"] for r in attacker.telemetry)})
                break
            observation = hand.observe(hand.actor)
            if observation.seat == view.seat:
                choices, probs, _ = source.distribution(observation)
                action = random.choices(choices, weights=probs, k=1)[0].action
            else:
                action = attacker.choose_action(observation)
            hand = hand.apply(action)
        else:
            raise RuntimeError("Conditional world exceeded 1000 legal actions")
    return {"values_bb": results, "menu": [repr(c.action) for c in menu],
            "probabilities": list(probabilities), "trained": trained,
            "attacker_work": work, "limited": sum(r["limited"] for r in work)}
