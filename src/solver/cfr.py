"""Simultaneous vanilla CFR and external sampling with exact strategy averaging."""

from random import Random

import numpy as np

from src.solver.tree import GameTree


def normalize(weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    positive = np.maximum(weights, 0) * mask
    total = positive.sum(axis=1, keepdims=True)
    return np.divide(
        positive, total, out=mask / mask.sum(axis=1, keepdims=True), where=total > 0
    )


def regret_delta(tree: GameTree, policy: np.ndarray) -> np.ndarray:
    probabilities = tree.edge_probabilities(policy)
    reach = tree.reaches(probabilities)
    values = tree.values(probabilities)
    delta = np.zeros_like(policy)
    for layer in tree.layers:
        decision = tree.actor[layer.parents] >= 0
        parents, children, actions = (
            layer.parents[decision],
            layer.children[decision],
            layer.actions[decision],
        )
        owners = tree.actor[parents]
        counterfactual = reach[parents, 2] * reach[parents, 1 - owners]
        advantages = (values[children] - values[parents]) * (1 - 2 * owners)
        np.add.at(delta, (tree.info[parents], actions), counterfactual * advantages)
    return delta


def sampled_delta(
    tree: GameTree, policy: np.ndarray, player: int, choose
) -> np.ndarray:
    """Sample chance/opponents, enumerate the traverser; keep the profile frozen."""
    delta = np.zeros_like(policy)

    def visit(node):
        actor = tree.actor[node]
        if actor == -2:
            return tree.terminal_values[node] * (1 - 2 * player)
        children = tree.children[node]
        if actor == -1:
            return visit(children[choose(tree.chance[node])])
        info = tree.info[node]
        probabilities = policy[info, : len(children)]
        if actor != player:
            return visit(children[choose(probabilities)])
        values = np.array([visit(child) for child in children])
        value = probabilities @ values
        # Sampling already accounts for chance/opponent reach; weighting again biases it.
        delta[info, : len(children)] += values - value
        return value

    visit(0)
    return delta


class CFR:
    def __init__(self, tree: GameTree, *, method: str = "full", seed: int = 0):
        if method not in {"full", "external"}:
            raise ValueError("Method must be full or external")
        if type(seed) is not int or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        self.tree, self.method = tree, method
        self.regrets = np.zeros(tree.mask.shape)
        self.strategy_sum = np.zeros(tree.mask.shape)
        self.iterations = 0
        self.random = Random(seed)

    def current_policy(self) -> np.ndarray:
        return normalize(self.regrets, self.tree.mask)

    def average_policy(self) -> np.ndarray:
        return normalize(self.strategy_sum, self.tree.mask)

    def step(self) -> None:
        policy = self.current_policy()
        tree = self.tree
        reach = tree.reaches(tree.edge_probabilities(policy))
        own = reach[tree.representatives, tree.owners]
        # Perfect recall makes own reach identical across an information set.
        # Count it once, independently of how many hidden deals or samples reach it.
        self.strategy_sum += own[:, None] * policy
        if self.method == "full":
            self.regrets += regret_delta(tree, policy)
        else:

            def choose(probabilities):
                return self.random.choices(
                    range(len(probabilities)), weights=probabilities
                )[0]

            delta = sampled_delta(tree, policy, 0, choose)
            delta += sampled_delta(tree, policy, 1, choose)
            self.regrets += delta
        self.iterations += 1
        if (
            not np.isfinite(self.regrets).all()
            or not np.isfinite(self.strategy_sum).all()
        ):
            raise FloatingPointError("Non-finite CFR accumulators")

    def train(self, iterations: int) -> None:
        if type(iterations) is not int or iterations < 0:
            raise ValueError("Iterations must be a nonnegative integer")
        for _ in range(iterations):
            self.step()
