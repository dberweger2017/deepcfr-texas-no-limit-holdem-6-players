"""Controlled river players using the same declared public joint card law.

These are evaluation adapters. They do not alter the saved blueprint or the
ordinary SearchPlayer used outside a conditional river comparison.
"""

from collections import Counter
from random import Random
from time import monotonic

import numpy as np

from src.arena.endgame_quality import _world
from src.blueprint.river_game import RiverGame, RiverUnsupported
from src.blueprint.search import (
    SearchConfig, SearchUnavailable, _observed_likelihood, _rollout,
)
from src.game.observation import ActionTaken, Observation, replay
from src.game.types import Action, Street


def conditional_opponent_weights(game: RiverGame, blueprint,
                                 view: Observation) -> tuple[tuple[tuple[str, str], ...],
                                                                    np.ndarray]:
    """Q(other hand | hero hand, observed river actions) under a named model."""
    root = game.nodes[0].history
    if view.street != Street.RIVER or view.history[:len(root)] != root:
        raise RiverUnsupported("Observation is outside the declared river root")
    if view.seat not in game.seats or len(view.hole_cards) != 2:
        raise RiverUnsupported("Observer is not an active river player")
    own = game.seats.index(view.seat)
    other = 1 - own
    try:
        holding = game.holdings[own].index(tuple(sorted(view.hole_cards)))
    except ValueError as exc:
        raise RiverUnsupported("Actual holding is outside the declared range") from exc
    weights = (game.joint[holding].copy() if own == 0
               else game.joint[:, holding].copy())
    for index in range(len(root), len(view.history)):
        event = view.history[index]
        if isinstance(event, ActionTaken) and event.seat == game.seats[other]:
            for holding_id, pair in enumerate(game.holdings[other]):
                if weights[holding_id] > 0:
                    prior = replay(view.history[:index], game.seats[other], pair)
                    weights[holding_id] *= max(
                        1e-4, _observed_likelihood(blueprint, prior, event.action),
                    )
    total = float(weights.sum())
    if total <= 0:
        raise SearchUnavailable("No opponent holding remains in the declared law")
    return game.holdings[other], weights / total


class ConditionalRiverRollout:
    """Corrected rollout selection with opponent worlds drawn from Q."""

    def __init__(self, blueprint, game: RiverGame, seed: int,
                 config: SearchConfig = SearchConfig()):
        if config.variant != "corrected":
            raise ValueError("Conditional river control requires corrected rollout")
        self.blueprint = blueprint
        self.game = game
        self.config = config
        self.action_random = Random(seed)
        self.search_random = Random(seed ^ 0x505F4C5552494255)
        self.attempts = self.completed = self.fallbacks = 0
        self.search_seconds = []
        self.worlds_completed = []
        self.coverage = Counter()

    def choose_action(self, view: Observation) -> Action:
        menu, probabilities, _ = self.blueprint.distribution(view)
        fallback = self.action_random.choices(menu, weights=probabilities, k=1)[0].action
        if len(menu) < 2:
            return fallback
        self.attempts += 1
        started = monotonic()
        deadline = started + self.config.max_seconds
        completed_worlds = 0
        try:
            pairs, weights = conditional_opponent_weights(self.game, self.blueprint, view)
            other = next(seat for seat in self.game.seats if seat != view.seat)
            root = self.game.nodes[0].history
            values = [0.0] * len(menu)
            for _ in range(self.config.worlds):
                if monotonic() >= deadline:
                    raise TimeoutError("Conditional rollout reached its deadline")
                pair = self.search_random.choices(pairs, weights=weights, k=1)[0]
                world = _world(root, self.game.board,
                               {view.seat: view.hole_cards, other: pair})
                for event in view.history[len(root):]:
                    if isinstance(event, ActionTaken):
                        world = world.apply(event.action)
                if world.events != view.history:
                    raise RuntimeError("Conditional world differs from public observation")
                for hero_style in self.config.styles:
                    styles = [self.search_random.choice(self.config.styles)
                              for _ in view.players]
                    styles[view.seat] = hero_style
                    seed = self.search_random.getrandbits(64)
                    for index, item in enumerate(menu):
                        values[index] += _rollout(
                            self.blueprint, world.apply(item.action), view.seat,
                            Random(seed), tuple(styles), Street.RIVER, deadline,
                            variant="corrected", coverage=self.coverage,
                        )
                completed_worlds += 1
            selected = menu[max(range(len(menu)),
                                key=lambda index: (values[index], probabilities[index]))]
            view.legal_actions.validate(selected.action)
            self.completed += 1
            return selected.action
        except (TimeoutError, SearchUnavailable, RiverUnsupported):
            self.fallbacks += 1
            return fallback
        finally:
            self.search_seconds.append(monotonic() - started)
            self.worlds_completed.append(completed_worlds)


class FrozenRiverProfile:
    """Play one solved public profile across complete paired evaluation deals."""

    def __init__(self, blueprint, game: RiverGame, profile, seed: int,
                 fallback_config: SearchConfig = SearchConfig()):
        self.game = game
        self.profile = profile
        self.random = Random(seed)
        self.fallback = ConditionalRiverRollout(
            blueprint, game, seed ^ 0x46524F5A454E5249, fallback_config,
        )
        self.hand_id = None
        self.delegated = False
        self.delegations = 0

    def choose_action(self, view: Observation) -> Action:
        if view.hand_id != self.hand_id:
            self.hand_id = view.hand_id
            self.delegated = False
        if self.delegated:
            return self.fallback.choose_action(view)
        node_id = self.game.history_to_node.get(view.history)
        if node_id is None or self.game.nodes[node_id].actor != view.seat:
            self.delegated = True
            self.delegations += 1
            return self.fallback.choose_action(view)
        node = self.game.nodes[node_id]
        try:
            holding = self.game.holdings[self.game.seats.index(view.seat)].index(
                tuple(sorted(view.hole_cards))
            )
        except ValueError as exc:
            raise RiverUnsupported("Evaluation holding is outside the range") from exc
        action = self.random.choices(
            [choice.action for choice in node.menu],
            weights=self.profile[node_id][holding], k=1,
        )[0]
        view.legal_actions.validate(action)
        return action
