"""Exact profile values and information-set-consistent best responses."""

from dataclasses import asdict, dataclass

import numpy as np

from src.solver.tree import GameTree


@dataclass(frozen=True)
class Evaluation:
    value_player0: float
    best_response_values: tuple[float, float]
    improvements: tuple[float, float]
    nash_conv: float
    exploitability: float

    def to_dict(self) -> dict:
        return asdict(self)


def best_response(
    tree: GameTree, policy: np.ndarray, player: int
) -> tuple[float, np.ndarray]:
    if type(player) is not int or player not in (0, 1):
        raise ValueError("Best responder must be player 0 or 1")
    tree.validate_policy(policy)
    probabilities = tree.edge_probabilities(policy)
    reach = tree.reaches(probabilities)
    counterfactual = reach[:, 2] * reach[:, 1 - player]
    values = tree.terminal_values.copy() * (1 - 2 * player)
    response = policy.copy()
    for layer in reversed(tree.layers):
        own = tree.actor[layer.parents] == player
        parents, children, actions = (
            layer.parents[own],
            layer.children[own],
            layer.actions[own],
        )
        scores = np.zeros_like(policy)
        np.add.at(
            scores,
            (tree.info[parents], actions),
            counterfactual[parents] * values[children],
        )
        scores[~tree.mask] = -np.inf
        selected = scores.argmax(axis=1)
        infos = np.unique(tree.info[parents])
        response[infos] = 0
        response[infos, selected[infos]] = 1
        chosen = actions == selected[tree.info[parents]]
        values[parents[chosen]] = values[children[chosen]]
        other = ~own
        totals = np.bincount(
            layer.parents[other],
            weights=probabilities[layer.children[other]]
            * values[layer.children[other]],
            minlength=len(values),
        )
        other_nodes = layer.nodes[tree.actor[layer.nodes] != player]
        values[other_nodes] = totals[other_nodes]
    return float(values[0]), response


def evaluate(tree: GameTree, policy: np.ndarray) -> Evaluation:
    tree.validate_policy(policy)
    value = float(tree.values(tree.edge_probabilities(policy))[0])
    responses = tuple(best_response(tree, policy, player)[0] for player in (0, 1))
    gains = (responses[0] - value, responses[1] + value)
    if min(gains) < -1e-10:
        raise ArithmeticError("Best response is worse than the supplied strategy")
    gains = tuple(max(0.0, g) for g in gains)
    nash_conv = sum(gains)
    return Evaluation(value, responses, gains, nash_conv, nash_conv / 2)
