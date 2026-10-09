"""Admission and transport boundaries for external evaluations."""

from dataclasses import dataclass
from typing import Protocol

from src.blueprint.abstraction import HU100_SCHEMA, HU200_SCHEMA, HU20_UNCAPPED_SCHEMA
from src.blueprint.solver import HU100_GAME, HU200_GAME, HU20_UNCAPPED_GAME
from src.game.observation import Observation
from src.game.types import Action


@dataclass(frozen=True)
class GameContract:
    stack: int
    small_blind: int = 50
    big_blind: int = 100
    players: int = 2
    rake: int = 0
    ante: int = 0
    reset_each_hand: bool = True
    bet_convention: str = 'street-raise-to'

    def admit(self, policy):
        expected = {HU200_GAME: (20000, HU200_SCHEMA), HU100_GAME: (10000, HU100_SCHEMA), HU20_UNCAPPED_GAME: (2000, HU20_UNCAPPED_SCHEMA)}.get(policy.game)
        if (expected is None or (self.stack, policy.abstraction) != expected
                or type(self.stack) is not int or type(self.small_blind) is not int
                or type(self.big_blind) is not int or (self.small_blind, self.big_blind) != (50, 100)
                or type(self.players) is not int or self.players != 2
                or type(self.rake) is not int or self.rake != 0
                or type(self.ante) is not int or self.ante != 0
                or self.reset_each_hand is not True or self.bet_convention != 'street-raise-to'
                or getattr(policy, 'players', 2) != 2 or getattr(policy, 'raise_cap', None) is not None):
            raise ValueError('External game is incompatible with the fixed trained policy; no stack scaling')


class JsonTransport(Protocol):
    """A POST is never automatically retried: remote actions lack idempotency keys."""
    def post(self, endpoint: str, body: dict) -> dict: ...


class ObservationAdapter(Protocol):
    contract: GameContract
    def observation(self, response: dict, hand_id: str) -> Observation: ...
    def encode_action(self, action: Action, observation: Observation) -> str: ...
