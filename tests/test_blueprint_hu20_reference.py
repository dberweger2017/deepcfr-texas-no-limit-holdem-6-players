"""Enumerated hidden-rank reference for the production regret traversal."""

from types import SimpleNamespace
from time import perf_counter

import pytest

import src.blueprint.solver as solver
from src.blueprint.abstraction import Choice
from src.blueprint.solver import PilotConfig
from src.game.types import Street


class FiniteHand:
    hidden = "high"

    def __init__(self, table, phase=0, action=None):
        self.table = table
        self.phase = phase
        self.action = action
        self.actor = 0 if phase == 0 else 1
        self.finished = phase == 2

    @classmethod
    def start(cls, table, *, hand_id, seed):
        return cls(table)

    def observe(self, seat):
        payoff = {("high", "A"): 1, ("high", "B"): -1,
                  ("low", "A"): -1, ("low", "B"): 1}
        stack = 100 + payoff[(self.hidden, self.action)] if self.finished else 100
        return SimpleNamespace(actor=self.actor, seat=seat, street=Street.PREFLOP,
                               players=(SimpleNamespace(stack=stack, starting_stack=100),
                                        SimpleNamespace(stack=100, starting_stack=100)))

    def apply(self, action):
        return FiniteHand(self.table, self.phase+1,
                          action if self.phase == 0 else self.action)


def test_production_external_sampling_matches_enumerated_hidden_game(monkeypatch):
    monkeypatch.setattr(solver, "Hand", FiniteHand)
    monkeypatch.setattr(solver, "choices", lambda view, **kwargs: (
        (Choice("A", "A"), Choice("B", "B")) if view.actor == 0
        else (Choice("end", "end"),)))
    monkeypatch.setattr(solver, "information_key", lambda view, menu, **kwargs:
                        "hero-root" if view.actor == 0 else "opponent-private")
    table = SimpleNamespace(big_blind=1)
    config = PilotConfig(max_nodes=100, max_seconds=10)
    weighted = [0.0, 0.0]
    for hidden, chance in (("high", 0.75), ("low", 0.25)):
        FiniteHand.hidden = hidden
        result = solver._collect_root(table, config, {}, 1, 0, 0, perf_counter()+10)
        assert result.deltas["hero-root"].names == ("A", "B")
        for index, regret in enumerate(result.deltas["hero-root"].regrets):
            weighted[index] += chance * regret
    # Independent enumeration: A has .75*(+1)+.25*(-1)=+.5;
    # B has -.5. Uniform root value is zero, so regrets are +.5 and -.5.
    assert weighted == pytest.approx([0.5, -0.5])
