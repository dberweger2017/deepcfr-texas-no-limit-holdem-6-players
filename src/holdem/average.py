"""Linear strategy mixtures, conditioned only on the owner's earlier actions."""

from math import exp, fsum, log
from random import Random

from src.game.observation import ActionTaken, Observation, replay
from src.game.types import Action
from src.holdem.actions import bet_candidates
from src.holdem.policy import FrozenProfile
from src.solver.neural.network import stream_seed


def own_path(view: Observation):
    """Reconstruct earlier decisions without showing them later board cards."""
    current = bet_candidates(view)
    path = []
    for index, event in enumerate(view.history):
        if isinstance(event, ActionTaken) and event.seat == view.seat:
            previous = replay(
                view.history[:index], view.seat, view.hole_cards, view.previous_hands
            )
            candidates = bet_candidates(previous)
            if event.action not in candidates.actions:
                raise ValueError(
                    "Own history contains an action outside this abstraction"
                )
            path.append((candidates, candidates.actions.index(event.action)))
    return current, path


class AveragePolicy:
    def __init__(self, profiles: tuple[FrozenProfile, ...]):
        if not profiles or any(p.capacity != profiles[0].capacity for p in profiles):
            raise ValueError(
                "An average requires complete profiles of the same capacity"
            )
        self._profiles = tuple(profiles)

    @property
    def fingerprints(self) -> tuple[str, ...]:
        return tuple(p.fingerprint for p in self._profiles)

    def distribution(self, view: Observation) -> tuple[float, ...]:
        current, path = own_path(view)
        logs, distributions = [], []
        for iteration, profile in enumerate(self._profiles, 1):
            weight = log(iteration)
            for candidates, action in path:
                probability = profile.distribution(candidates)[action]
                if probability == 0:
                    weight = float("-inf")
                    break
                weight += log(probability)
            logs.append(weight)
            distributions.append(profile.distribution(current))
        largest = max(logs)
        if largest == float("-inf"):
            return (1 / len(current.actions),) * len(current.actions)
        weights = [exp(w - largest) for w in logs]
        total = fsum(weights)
        return tuple(
            fsum(w * p[action] for w, p in zip(weights, distributions)) / total
            for action in range(len(current.actions))
        )

    def player(self, seed: int) -> "AveragePlayer":
        return AveragePlayer(self._profiles, seed)


class AveragePlayer:
    """Choose a mixture component once per hand, independently for each player."""

    def __init__(self, profiles: tuple[FrozenProfile, ...], seed: int):
        self._profiles = profiles
        self._choice = Random(stream_seed(seed, "holdem-snapshot-choice"))
        self._actions = Random(stream_seed(seed, "holdem-snapshot-actions"))
        self._key = None
        self._history = ()
        self.iteration = None

    def choose_action(self, view: Observation) -> Action:
        candidates = bet_candidates(view)
        key = (view.hand_id, view.player_id, view.seat_numbers[view.seat])
        if key != self._key:
            self.iteration = self._choice.choices(
                range(1, len(self._profiles) + 1),
                weights=range(1, len(self._profiles) + 1),
            )[0]
            self._key, self._history = key, ()
        if self._history and (
            len(view.history) <= len(self._history)
            or view.history[: len(self._history)] != self._history
        ):
            raise ValueError("Sampled play requires forward progress within one hand")
        probabilities = self._profiles[self.iteration - 1].distribution(candidates)
        action = self._actions.choices(candidates.actions, weights=probabilities)[0]
        self._history = view.history
        return action
