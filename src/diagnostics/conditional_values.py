"""Information-safe conditional HU20 action values for saved policies.

The only input representing a played hand is the acting player's Observation.
Fresh opponent cards, future board and policy random streams are generated here.
"""

from hashlib import sha256
from math import sqrt
from random import Random
from statistics import mean, stdev

from src.arena.schedule import stream_seed
from src.blueprint.search import DECK, _sample_world
from src.diagnostics.robustness import LBRConfig, LocalBestResponse
from src.game.types import ActionKind


def _rollout(hand, hero_seat, target, target_seed, attacker_seed):
    target_random = Random(target_seed)
    attacker = LocalBestResponse(target, attacker_seed, LBRConfig(4, 5))
    for _ in range(1000):
        if hand.finished:
            return hand.events[-1].stacks[hero_seat] - 2000, attacker.telemetry
        view = hand.observe(hand.actor)
        if view.seat == hero_seat:
            menu, probabilities, _ = target.distribution(view)
            action = target_random.choices(menu, weights=probabilities, k=1)[0].action
        else:
            action = attacker.choose_action(view)
        hand = hand.apply(action)
    raise RuntimeError("Conditional rollout exceeded 1000 actions")


def _world(view, seed):
    """Generate a world from the view alone, with a uniform compatible pair."""
    random = Random(seed)
    available = [card for card in DECK if card not in view.hole_cards + view.board]
    pair = tuple(random.sample(available, 2))
    return _sample_world(view, {1 - view.seat: ((pair, 1.0),)}, random)


def world_action_returns(view, target, seed, world_index):
    """One complete, paired comparison batch; no realized-hand data enter."""
    menu, _, _ = target.distribution(view)
    root = stream_seed(seed, "test", "opponent", "conditional", world_index)
    world = _world(view, stream_seed(root, "test", "deal", "world"))
    returns = []
    limited = 0
    for choice in menu:
        result, telemetry = _rollout(
            world.apply(choice.action), view.seat, target,
            stream_seed(root, "test", "action", "target"),
            stream_seed(root, "test", "opponent", "attacker"),
        )
        returns.append(result / view.big_blind)
        limited += sum(not row["completed"] for row in telemetry)
    return tuple(returns), limited


def summarize(values, probabilities):
    """Paired world-clustered 95% normal intervals, explicitly descriptive."""
    if not values:
        raise ValueError("No complete action-comparison worlds")
    n = len(values)
    if any(len(row) != len(probabilities) for row in values):
        raise ValueError("Incomplete action comparison")
    columns = tuple(tuple(row[i] for row in values) for i in range(len(probabilities)))
    estimate = [mean(column) for column in columns]
    uncertainty = [1.96 * stdev(column) / sqrt(n) if n > 1 else None for column in columns]
    policy = tuple(sum(p * value for p, value in zip(probabilities, row)) for row in values)
    policy_mean = mean(policy)
    best = max(range(len(estimate)), key=lambda i: estimate[i])
    differences = tuple(row[best] - own for row, own in zip(values, policy))
    gap = mean(differences)
    gap_halfwidth = 1.96 * stdev(differences) / sqrt(n) if n > 1 else None
    return {
        "worlds": n,
        "action_mean_bb": estimate,
        "action_95_halfwidth_bb": uncertainty,
        "policy_mean_bb": policy_mean,
        "best_estimated_index": best,
        "policy_gap_bb": gap,
        "policy_gap_95_interval_bb": [gap-gap_halfwidth, gap+gap_halfwidth] if gap_halfwidth is not None else None,
    }


def visible_fingerprint(view):
    """Useful test aid: no opponent holding or future deck is present."""
    return sha256(repr((view.hole_cards, view.board, view.history,
                        view.legal_actions, view.seat)).encode()).hexdigest()
