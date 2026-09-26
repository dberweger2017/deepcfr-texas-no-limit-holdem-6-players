"""Range-complete, simultaneous Linear CFR for a two-player river game."""

from dataclasses import dataclass
import resource
import sys
from time import monotonic

import numpy as np

from src.blueprint.river_game import RiverGame


Profile = dict[int, np.ndarray]


def _regret_match(regrets: np.ndarray) -> np.ndarray:
    positive = np.maximum(regrets, 0.0)
    total = positive.sum(axis=1, keepdims=True)
    return np.divide(positive, total, out=np.full_like(positive, 1 / regrets.shape[1]),
                     where=total > 0)


def _profile(game: RiverGame, regrets: Profile) -> Profile:
    return {node.id: _regret_match(regrets[node.id])
            for node in game.nodes if node.actor is not None}


def _visit(game: RiverGame, node_id: int, traverser: int,
           opponent_reach: np.ndarray, profile: Profile,
           deltas: Profile | None = None, weight: int = 1,
           zero_external: list[int] | None = None,
           best_response: bool = False, check=None) -> np.ndarray:
    if check is not None:
        check()
    node = game.nodes[node_id]
    if node.terminal is not None:
        return game.utility_vectors(node.terminal, traverser, opponent_reach)
    policy = profile[node_id]
    if node.actor == traverser:
        actions = np.stack([
            _visit(game, child, traverser, opponent_reach, profile,
                   deltas, weight, zero_external, best_response, check)
            for child in node.children
        ], axis=1)
        value = actions.max(axis=1) if best_response else np.sum(policy * actions, axis=1)
        if deltas is not None:
            deltas[node_id] += weight * (actions - value[:, None])
            if zero_external is not None:
                law = game.joint if traverser == game.seats[0] else game.joint.T
                zero_external[0] += int(np.count_nonzero(law @ opponent_reach == 0))
        return value
    value = np.zeros(len(game.holdings[game.seats.index(traverser)]), dtype=np.float64)
    for action, child in enumerate(node.children):
        value += _visit(game, child, traverser,
                        opponent_reach * policy[:, action], profile,
                        deltas, weight, zero_external, best_response, check)
    return value


def _stage_average(game: RiverGame, node_id: int, traverser: int,
                   own_reach: np.ndarray, profile: Profile, weight: int,
                   numer: Profile, denom: dict[int, np.ndarray]) -> None:
    node = game.nodes[node_id]
    if node.terminal is not None:
        return
    if node.actor == traverser:
        policy = profile[node_id]
        numer[node_id] += weight * own_reach[:, None] * policy
        denom[node_id] += weight * own_reach
        for action, child in enumerate(node.children):
            _stage_average(game, child, traverser, own_reach * policy[:, action],
                           profile, weight, numer, denom)
    else:
        for child in node.children:
            _stage_average(game, child, traverser, own_reach, profile,
                           weight, numer, denom)


def profile_quality(game: RiverGame, profile: Profile) -> dict:
    """Information-set best responses, summing hidden worlds before maximization."""
    values = []
    gains = []
    for seat in game.seats:
        other = game.seats[1] if seat == game.seats[0] else game.seats[0]
        reach = np.ones(len(game.holdings[game.seats.index(other)]), dtype=np.float64)
        value = float(_visit(game, 0, seat, reach, profile).sum())
        response = float(_visit(game, 0, seat, reach, profile,
                                best_response=True).sum())
        values.append(value)
        gains.append(max(0.0, response - value))
    exploitability = sum(gains) / 2
    return {
        "profile_values_bb": values,
        "best_response_gains_bb": gains,
        "exploitability_bb": exploitability,
        "exploitability_root_pot": exploitability * game.big_blind / game.root_pot,
        "zero_sum_error_bb": abs(sum(values)),
    }


@dataclass(slots=True)
class RiverResult:
    current: Profile
    average: Profile
    completed_sweeps: int
    visited_public_nodes: int
    zero_external_reach_entries: int
    zero_average_denominators: int
    stop_reason: str
    elapsed_seconds: float


class RiverCFR:
    def __init__(self, game: RiverGame):
        self.game = game
        self.regrets: Profile = {}
        self.average_numer: Profile = {}
        self.average_denom: dict[int, np.ndarray] = {}
        for node in game.nodes:
            if node.actor is None:
                continue
            size = len(game.holdings[game.seats.index(node.actor)])
            shape = (size, len(node.menu))
            self.regrets[node.id] = np.zeros(shape, dtype=np.float64)
            self.average_numer[node.id] = np.zeros(shape, dtype=np.float64)
            self.average_denom[node.id] = np.zeros(size, dtype=np.float64)
        self.completed_sweeps = 0
        self.visited_public_nodes = 0
        self.zero_external_reach_entries = 0
        self.last_played: Profile | None = None

    def solve(self, *, max_sweeps: int, deadline: float | None = None,
              rss_limit_bytes: int | None = None) -> RiverResult:
        if type(max_sweeps) is not int or max_sweeps < 1:
            raise ValueError("Need a positive full-sweep limit")
        started = monotonic()
        stop = "sweep_cap"
        for _ in range(max_sweeps):
            cycle = self.completed_sweeps + 1
            profile = _profile(self.game, self.regrets)
            deltas = {key: np.zeros_like(value) for key, value in self.regrets.items()}
            numer = {key: np.zeros_like(value) for key, value in self.average_numer.items()}
            denom = {key: np.zeros_like(value) for key, value in self.average_denom.items()}
            zero_external = [0]
            visited = [0]

            def check():
                visited[0] += 1
                if deadline is not None and monotonic() >= deadline:
                    raise TimeoutError("River CFR reached its wall limit")
                if rss_limit_bytes is not None and visited[0] % 64 == 0:
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    rss = rss if sys.platform == "darwin" else rss * 1024
                    if rss >= rss_limit_bytes:
                        raise MemoryError("River CFR reached its RSS limit")

            try:
                for seat in self.game.seats:
                    other = self.game.seats[1] if seat == self.game.seats[0] else self.game.seats[0]
                    _visit(self.game, 0, seat,
                           np.ones(len(self.game.holdings[self.game.seats.index(other)])),
                           profile, deltas, cycle, zero_external, check=check)
                    _stage_average(
                        self.game, 0, seat,
                        np.ones(len(self.game.holdings[self.game.seats.index(seat)])),
                        profile, cycle, numer, denom,
                    )
            except TimeoutError as exc:
                stop = str(exc)
                break
            # Regrets and own-reach averages use the same pre-publication
            # profile. An interrupted sweep publishes neither accumulator.
            for key in self.regrets:
                self.regrets[key] += deltas[key]
                self.average_numer[key] += numer[key]
                self.average_denom[key] += denom[key]
            self.last_played = profile
            self.completed_sweeps = cycle
            self.visited_public_nodes += visited[0]
            self.zero_external_reach_entries += zero_external[0]
        if self.last_played is None:
            raise TimeoutError("River CFR completed no full sweep")
        average = {}
        missing = 0
        for key, numerator in self.average_numer.items():
            denominator = self.average_denom[key]
            missing += int(np.count_nonzero(denominator == 0))
            average[key] = np.divide(
                numerator, denominator[:, None],
                out=np.full_like(numerator, 1 / numerator.shape[1]),
                where=denominator[:, None] > 0,
            )
        return RiverResult(self.last_played, average, self.completed_sweeps,
                           self.visited_public_nodes, self.zero_external_reach_entries,
                           missing, stop, monotonic() - started)
