"""Target-perspective posterior over an observed bounded LBR's private cards.

The saved target's observation is the only played-hand input. In particular,
LocalBestResponse.update() estimates the *opposite* posterior and is never
used as this module's Bayesian update. It is called only inside fresh LBR
action simulations for hypothetical attacker holdings.
"""

from itertools import combinations
from math import log

from src.arena.schedule import stream_seed
from src.blueprint.search import DECK
from src.diagnostics.robustness import LBRConfig, LocalBestResponse
from src.game.observation import ActionTaken, replay


def compatible_holdings(view):
    visible = set(view.hole_cards + view.board)
    return tuple(combinations((card for card in DECK if card not in visible), 2))


def observed_lbr_actions(view):
    """Public prefixes of actions made by the opponent of this target seat."""
    opponent = 1 - view.seat
    return tuple((i, view.history[:i], event.action)
                 for i, event in enumerate(view.history)
                 if isinstance(event, ActionTaken) and event.seat == opponent)


def likelihood_sample(source, prefix, opponent_seat, pair, observed, seed,
                      config=LBRConfig(4, 5)):
    """Simulate the unchanged LBR using only its hypothetical visible view."""
    attacker_view = replay(prefix, opponent_seat, pair)
    attacker = LocalBestResponse(source, seed, config)
    predicted = attacker.choose_action(attacker_view)
    return int(predicted == observed), sum(not row["completed"] for row in attacker.telemetry)


def _diagnostics(holdings, weights, zeros, samples, limited):
    n = len(holdings)
    positive = [weight for weight in weights if weight > 0]
    entropy = -sum(weight * log(weight) for weight in positive)
    uniform = 1 / n
    return {
        "compatible_holdings": n,
        "positive_mass_holdings": len(positive),
        "effective_sample_size": 1 / sum(weight * weight for weight in weights),
        "entropy_nats": entropy,
        "maximum_holding_weight": max(weights),
        "normalization": sum(weights),
        "uniform_total_variation": .5 * sum(abs(weight - uniform) for weight in weights),
        "zero_likelihood_events": zeros,
        "likelihood_samples_per_holding_action": samples,
        "limited_lbr_samples": limited,
    }


def reverse_lbr_posterior(view, source, *, root, samples, coordinate,
                          likelihood_fn=likelihood_sample):
    """P(LBR hand | target cards, board, public actions) on all compatible hands.

    ``likelihood_fn`` is injectable only for deterministic reference tests.
    Production invokes unchanged LBR for every hypothetical holding/sample.
    """
    if samples < 1 or len(view.players) != 2:
        raise ValueError("Expected positive samples and heads-up observation")
    holdings = compatible_holdings(view)
    if not holdings:
        raise ValueError("No compatible LBR holdings")
    weights = [1 / len(holdings)] * len(holdings)
    zeros = []
    limited = 0
    for event_index, prefix, observed in observed_lbr_actions(view):
        likelihoods = [0.0] * len(holdings)
        for index, pair in enumerate(holdings):
            if weights[index] == 0:
                continue
            matched = 0
            for sample in range(samples):
                seed = stream_seed(root, "test", "opponent", "reverse-lbr",
                                   coordinate, event_index, pair, sample)
                equal, limited_here = likelihood_fn(
                    source, prefix, 1 - view.seat, pair, observed, seed)
                if equal not in (0, 1):
                    raise ValueError("LBR action equality must be binary")
                matched += equal
                limited += limited_here
            likelihoods[index] = matched / samples
        updated = [weight * likelihood for weight, likelihood in zip(weights, likelihoods)]
        normalizer = sum(updated)
        if normalizer == 0:
            zeros.append({"public_event_index": event_index,
                          "action": repr(observed),
                          "handling": "preserve preceding compatible posterior"})
        else:
            weights = [weight / normalizer for weight in updated]
    return holdings, tuple(weights), _diagnostics(holdings, weights, zeros, samples, limited)
