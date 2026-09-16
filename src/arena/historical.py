"""Read-only architecture and features for pinned historical arena opponents."""

from decimal import Decimal

import numpy as np
from torch import nn

from src.game.observation import ActionTaken, BoardDealt, Observation
from src.game.types import ActionKind, Street

ACTIONS = {
    ActionKind.FOLD: 0,
    ActionKind.CHECK: 1,
    ActionKind.CALL: 2,
    ActionKind.RAISE: 3,
}
STREETS = {
    Street.PREFLOP: 0,
    Street.FLOP: 1,
    Street.TURN: 2,
    Street.RIVER: 3,
    Street.SHOWDOWN: 4,
}


class PokerNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_size, 3)
        self.sizing_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        hidden = self.base(features)
        return self.action_head(hidden), 0.1 + 2.9 * self.sizing_head(hidden)


def encode_observation(view: Observation) -> np.ndarray:
    if not isinstance(view, Observation):
        raise TypeError("Historical inference requires a player observation")
    if view.finished or view.actor != view.seat:
        raise ValueError("Historical inference needs its own current decision")
    unit = float(Decimal(view.chip_unit))
    players = len(view.players)
    result = np.zeros(120 + 6 * players)
    for card in view.hole_cards:
        result["cdhs".index(card[1]) * 13 + "23456789TJQKA".index(card[0])] = 1
    for card in view.board:
        result[52 + "cdhs".index(card[1]) * 13 + "23456789TJQKA".index(card[0])] = 1
    result[104 + STREETS[view.street]] = 1
    # Old checkpoints normalize by seat 0's remaining stack in table units.
    scale = view.players[0].stack * unit
    if scale <= 0:
        scale = 1.0
    result[109] = view.pot * unit / scale
    result[110 + view.button] = 1
    result[110 + players + view.actor] = 1
    cursor = 110 + 2 * players
    for player in view.players:
        result[cursor : cursor + 4] = (
            float(not player.folded),
            player.street_bet * unit / scale,
            (player.contributed - player.street_bet) * unit / scale,
            player.stack * unit / scale,
        )
        cursor += 4
    wager = view.big_blind
    previous = np.zeros(5)
    for event in view.history:
        if isinstance(event, BoardDealt):
            wager = 0
        elif isinstance(event, ActionTaken):
            increment = 0
            if event.action.kind == ActionKind.RAISE:
                increment = event.action.raise_to - wager
                wager = event.action.raise_to
            previous[:] = 0
            previous[ACTIONS[event.action.kind]] = 1
            previous[4] = increment * unit / scale
    if sum(not p.folded and p.stack > 0 for p in view.players) <= 1:
        wager = max(p.street_bet for p in view.players if not p.folded)
    result[cursor] = wager * unit / scale
    for kind in view.legal_actions.kinds:
        result[cursor + 1 + ACTIONS[kind]] = 1
    result[cursor + 5 :] = previous
    return result.astype(np.float32)
