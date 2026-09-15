"""Host-side hand execution with one observation and private history per player."""

from collections.abc import Mapping
from dataclasses import dataclass
from random import Random
from typing import Protocol

from src.game.hand import Hand
from src.game.observation import Observation, ObservedHand
from src.game.types import Action, ActionKind


class Policy(Protocol):
    def choose_action(self, observation: Observation) -> Action: ...


@dataclass(frozen=True, slots=True)
class PlayerHistory:
    player_id: str
    hands: tuple[ObservedHand, ...] = ()

    def append(self, observation: Observation) -> "PlayerHistory":
        if observation.player_id != self.player_id:
            raise ValueError("Cannot attach another player's private history")
        record = observation.record()
        if any(hand.events[0].hand_id == observation.hand_id for hand in self.hands):
            raise ValueError("This hand has already been recorded")
        return PlayerHistory(self.player_id, self.hands + (record,))


class RandomPolicy:
    def __init__(self, seed: int):
        self._random = Random(seed)

    def distribution(
        self, observation: Observation
    ) -> tuple[tuple[Action, float], ...]:
        legal = observation.legal_actions
        if observation.finished or observation.actor != observation.seat:
            raise ValueError("A policy needs its own current decision")
        actions = []
        for kind in legal.kinds:
            if kind == ActionKind.RAISE:
                targets = {legal.min_raise_to, legal.max_raise_to}
                for target in sorted(targets):
                    actions.append(
                        (Action(kind, target), 1 / len(legal.kinds) / len(targets))
                    )
            else:
                actions.append((Action(kind), 1 / len(legal.kinds)))
        return tuple(actions)

    def choose_action(self, observation: Observation) -> Action:
        choices = self.distribution(observation)
        actions, weights = zip(*choices)
        return self._random.choices(actions, weights)[0]


def play_hand(
    hand: Hand,
    policies: Mapping[str, Policy],
    histories: Mapping[str, PlayerHistory] | None = None,
    *,
    max_decisions: int = 1000,
) -> Hand:
    if any(identity not in policies for identity in hand.table.player_ids):
        raise ValueError("Every player needs a policy")
    if len({id(policies[p]) for p in hand.table.player_ids}) != len(
        hand.table.player_ids
    ):
        raise ValueError("Each seat needs its own policy instance and private memory")
    histories = histories or {}
    for _ in range(max_decisions):
        if hand.finished:
            return hand
        seat = hand.actor
        identity = hand.table.player_ids[seat]
        history = histories.get(identity, PlayerHistory(identity))
        if history.player_id != identity:
            raise ValueError("History identity does not match its owner")
        observation = hand.observe(seat, history.hands)
        action = policies[identity].choose_action(observation)
        hand = hand.apply(action)
    if not hand.finished:
        raise RuntimeError(f"Hand exceeded {max_decisions} decisions")
    return hand
