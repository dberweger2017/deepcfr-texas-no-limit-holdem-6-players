"""Alternating Deep CFR with separate advantage memories and a learned average policy."""

from dataclasses import dataclass
from random import Random
from time import perf_counter

import numpy as np
import torch

from src.solver.cfr import normalize
from src.solver.games import InformationSet
from src.solver.neural.encoding import ACTION_SLOT, encode, legal_mask
from src.solver.neural.memory import Reservoir
from src.solver.neural.network import (
    fit,
    new_network,
    predict,
    regret_matching,
    stream_seed,
)
from src.solver.tree import GameTree


@dataclass(frozen=True)
class Config:
    hidden: int = 64
    traversals: int = 256
    advantage_steps: int = 256
    strategy_steps: int = 1024
    batch_size: int = 256
    capacity: int = 100_000
    learning_rate: float = 0.001
    seed: int = 0

    def __post_init__(self):
        for value in (
            self.hidden,
            self.traversals,
            self.advantage_steps,
            self.strategy_steps,
            self.batch_size,
            self.capacity,
        ):
            if type(value) is not int or value < 1:
                raise ValueError(
                    "Network, memory, and training sizes must be positive integers"
                )
        if self.hidden > 512 or self.capacity > 1_000_000 or self.batch_size > 8192:
            raise ValueError("Configuration exceeds the small-game reference limits")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        if (
            type(self.learning_rate) not in (int, float)
            or not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("Learning rate must be finite and positive")


def collect(tree, policy, player, iteration, choose, advantage, strategy):
    """One Algorithm 2 traversal, with public information IDs as replay keys."""

    def visit(node):
        actor = tree.actor[node]
        if actor == -2:
            return tree.terminal_values[node] * (1 - 2 * player)
        children = tree.children[node]
        if actor == -1:
            return visit(children[choose(tree.chance[node])])
        info = tree.info[node]
        slots = [ACTION_SLOT[action] for action in tree.information_sets[info].actions]
        probabilities = policy[info, slots]
        if actor != player:
            strategy(info, iteration, policy[info].copy())
            return visit(children[choose(probabilities)])
        values = np.array([visit(child) for child in children])
        value = probabilities @ values
        target = np.zeros(3)
        target[slots] = values - value
        advantage(info, iteration, target)
        return value

    return visit(0)


class Policy:
    def __init__(self, game, player, model, seed):
        if type(player) is not int or player not in (0, 1):
            raise ValueError("Policy owner must be player 0 or 1")
        self.game, self.player, self.model = game, player, model
        self.random = Random(seed)

    def distribution(self, info: InformationSet):
        if not isinstance(info, InformationSet):
            raise TypeError("A policy accepts player information sets only")
        if info.game != self.game or info.player != self.player:
            raise ValueError("Information set does not belong to this policy")
        features = torch.from_numpy(encode(info)[None, :])
        mask = torch.from_numpy(legal_mask(info)[None, :])
        probabilities = predict(self.model, features, mask, strategy=True)[0]
        return tuple(
            (action, float(probabilities[ACTION_SLOT[action]]))
            for action in info.actions
        )

    def choose_action(self, info: InformationSet):
        actions, probabilities = zip(*self.distribution(info))
        return self.random.choices(actions, weights=probabilities)[0]


class DeepCFR:
    def __init__(self, tree: GameTree, config: Config):
        self.tree, self.config = tree, config
        self.features = torch.from_numpy(
            np.stack([encode(info) for info in tree.information_sets])
        )
        self.mask = torch.from_numpy(
            np.stack([legal_mask(info) for info in tree.information_sets])
        )
        self.slots = np.array(
            [
                [ACTION_SLOT[a] for a in info.actions] + [0] * (3 - len(info.actions))
                for info in tree.information_sets
            ]
        )
        self.advantages = [
            new_network(
                config.hidden,
                stream_seed(config.seed, "initial", player=p),
                zero_output=True,
            )
            .eval()
            .requires_grad_(False)
            for p in (0, 1)
        ]
        self.advantage_memories = [
            Reservoir(
                config.capacity,
                stream_seed(config.seed, "advantage-reservoir", player=p),
            )
            for p in (0, 1)
        ]
        self.strategy_memory = Reservoir(
            config.capacity, stream_seed(config.seed, "strategy-reservoir")
        )
        self.traversal_random = Random(stream_seed(config.seed, "traversals"))
        self.strategy = None
        self.iterations = 0
        self.failed = False
        self.fits = []
        self.played_strategy_sum = np.zeros(self.mask.shape)

    def local_policy(self, probabilities):
        local = (
            probabilities[np.arange(len(self.slots))[:, None], self.slots]
            * self.tree.mask
        )
        self.tree.validate_policy(local)
        return local

    def current_policy(self):
        predictions = np.zeros(self.mask.shape)
        for player, model in enumerate(self.advantages):
            rows = self.tree.owners == player
            predictions[rows] = predict(
                model, self.features[rows], self.mask[rows], strategy=False
            )
        return regret_matching(predictions, self.mask.numpy())

    def _fit(self, memory, iteration, player, strategy, deadline):
        config = self.config
        return fit(
            memory,
            self.features,
            self.mask,
            hidden=config.hidden,
            steps=config.strategy_steps if strategy else config.advantage_steps,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            iteration=iteration,
            seed=stream_seed(
                config.seed,
                "strategy-fit" if strategy else "advantage-fit",
                iteration,
                player,
            ),
            strategy=strategy,
            deadline=deadline,
        )

    def step(self, deadline=float("inf")):
        if self.failed:
            raise RuntimeError("A failed iteration cannot be continued")
        iteration = self.iterations + 1
        self.strategy = None

        def choose(probabilities):
            return self.traversal_random.choices(
                range(len(probabilities)), weights=probabilities
            )[0]

        try:
            for player in (0, 1):
                policy = self.current_policy()
                reach = self.tree.reaches(
                    self.tree.edge_probabilities(self.local_policy(policy))
                )
                opponent = self.tree.owners == 1 - player
                own = reach[self.tree.representatives, self.tree.owners]
                # Diagnostic only: the learner never reads this exact average.
                self.played_strategy_sum[opponent] += (
                    iteration * own[opponent, None] * policy[opponent]
                )
                for _ in range(self.config.traversals):
                    if perf_counter() >= deadline:
                        raise TimeoutError("Traversal exceeded the declared deadline")
                    collect(
                        self.tree,
                        policy,
                        player,
                        iteration,
                        choose,
                        self.advantage_memories[player].add,
                        self.strategy_memory.add,
                    )
                model, metrics = self._fit(
                    self.advantage_memories[player], iteration, player, False, deadline
                )
                self.advantages[player] = model
                self.fits.append(
                    {
                        "iteration": iteration,
                        "player": player,
                        "kind": "advantage",
                        **metrics,
                    }
                )
            self.iterations = iteration
        except BaseException:
            self.failed = True
            raise

    def fit_strategy(self, deadline=float("inf")):
        if self.failed or not self.iterations:
            raise RuntimeError("Strategy fitting requires completed valid iterations")
        self.strategy, metrics = self._fit(
            self.strategy_memory, self.iterations, 0, True, deadline
        )
        return metrics

    def average_policy(self):
        if self.strategy is None:
            raise RuntimeError("Fit the average strategy before evaluating it")
        return self.local_policy(
            predict(self.strategy, self.features, self.mask, strategy=True)
        )

    def empirical_average(self):
        means, _ = self.strategy_memory.means(len(self.features))
        return self.local_policy(normalize(means, self.mask.numpy()))

    def played_average(self):
        return self.local_policy(normalize(self.played_strategy_sum, self.mask.numpy()))

    def policy(self, player, seed):
        if self.strategy is None:
            raise RuntimeError("Fit the average strategy before playing")
        return Policy(self.tree.game, player, self.strategy, seed)
