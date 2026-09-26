"""Opt-in river CFR play with a retained on-tree profile and search fallback."""

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from math import fsum
from random import Random
from time import monotonic

from src.blueprint.river_cfr import RiverCFR
from src.blueprint.river_game import RiverGame, RiverUnsupported, river_root_history
from src.blueprint.search import (
    DECK, SearchConfig, SearchPlayer, _observed_likelihood,
)
from src.game.observation import ActionTaken, Observation, replay
from src.game.types import Action, Street


@dataclass(frozen=True, slots=True)
class RiverPlayerConfig:
    max_seconds: float = 30.0
    min_sweeps: int = 32
    max_sweeps: int = 4096
    max_public_nodes: int = 10_000
    raise_cap: int = 2
    extraction: str = "average"
    max_rss_gib: float = 10.5

    def __post_init__(self):
        if not 0 < self.max_seconds <= 60 or self.max_rss_gib <= 0:
            raise ValueError("River solve needs bounded time and memory")
        if any(type(value) is not int or value < 1 for value in
               (self.min_sweeps, self.max_sweeps, self.max_public_nodes)):
            raise ValueError("River solve needs positive work bounds")
        if self.max_sweeps < self.min_sweeps or self.raise_cap < 0:
            raise ValueError("Invalid river sweep or raise cap")
        if self.extraction not in {"average", "current"}:
            raise ValueError("Unknown river strategy extraction")


def active_public_ranges(blueprint, root_history, live, deadline,
                         coverage: Counter | None = None):
    """Complete active-seat marginals from public actions before the river.

    Folded hands are integrated out by assumption rather than conditioned on
    their actual cards. This is the named active-range approximation, not the
    true posterior over private deals.
    """
    root_view = replay(root_history, live[0], ())
    pairs = tuple(combinations((card for card in DECK if card not in root_view.board), 2))
    ranges = {}
    for seat in live:
        weighted = []
        for pair in pairs:
            if monotonic() >= deadline:
                raise TimeoutError("Public river range construction timed out")
            mass = 1.0
            for index, event in enumerate(root_history):
                if isinstance(event, ActionTaken) and event.seat == seat:
                    prior = replay(root_history[:index], seat, pair)
                    mass *= max(1e-4, _observed_likelihood(
                        blueprint, prior, event.action, coverage=coverage,
                    ))
            weighted.append((pair, mass))
        total = fsum(mass for _, mass in weighted)
        if total <= 0:
            raise RiverUnsupported("An active public range has zero mass")
        ranges[seat] = tuple((pair, mass / total) for pair, mass in weighted)
    return ranges


def _eligible(view: Observation) -> bool:
    if view.street != Street.RIVER or len(view.players) != 6 or view.actor != view.seat:
        return False
    if any(isinstance(event, ActionTaken) and event.street == Street.RIVER
           and event.seat == view.seat for event in view.history):
        return False
    try:
        root_history = river_root_history(view.history)
        root = replay(root_history, view.seat, ())
    except (ValueError, RiverUnsupported):
        return False
    live = tuple(p.seat for p in root.players if not p.folded)
    contested = tuple(pot for pot in root.pots if pot.refund_to is None)
    return (len(live) == 2 and view.seat in live
            and not any(root.players[seat].all_in for seat in live)
            and len(contested) == 1 and contested[0].eligible_seats == live)


class RiverCFRPlayer:
    """Solve once at the river root; reuse the complete profile on-tree."""

    def __init__(self, blueprint, seed: int,
                 config: RiverPlayerConfig = RiverPlayerConfig(),
                 fallback_config: SearchConfig = SearchConfig(variant="corrected")):
        self.blueprint = blueprint
        self.config = config
        self.action_random = Random(seed)
        self.fallback = SearchPlayer(blueprint, seed ^ 0x5249564552434652,
                                     fallback_config)
        self.hand_id = None
        self.game = None
        self.profile = None
        self.delegated_hand_id = None
        self.attempts = self.completed = self.delegations = 0
        self.records: list[dict] = []

    def _from_profile(self, view: Observation) -> Action:
        node_id = self.game.history_to_node.get(view.history)
        if node_id is None:
            raise RiverUnsupported("Observed river action is outside retained tree")
        node = self.game.nodes[node_id]
        if node.actor != view.seat:
            raise RiverUnsupported("Retained river node has a different actor")
        holding = tuple(sorted(view.hole_cards))
        index = self.game.holdings[self.game.seats.index(view.seat)].index(holding)
        policy = self.profile[node_id][index]
        action = self.action_random.choices(
            [choice.action for choice in node.menu], weights=policy, k=1,
        )[0]
        view.legal_actions.validate(action)
        return action

    def choose_action(self, view: Observation) -> Action:
        if self.hand_id != view.hand_id:
            self.hand_id = view.hand_id
            self.game = self.profile = self.delegated_hand_id = None
        if self.delegated_hand_id == view.hand_id:
            return self.fallback.choose_action(view)
        if self.game is not None:
            try:
                return self._from_profile(view)
            except (RiverUnsupported, ValueError) as exc:
                self.delegated_hand_id = view.hand_id
                self.delegations += 1
                self.records.append({"hand_id": view.hand_id, "status": "off_tree_delegate",
                                     "reason": str(exc)})
                return self.fallback.choose_action(view)
        if not _eligible(view):
            return self.fallback.choose_action(view)
        self.attempts += 1
        started = monotonic()
        deadline = started + self.config.max_seconds
        coverage = Counter()
        solver = None
        try:
            root_history = river_root_history(view.history)
            root = replay(root_history, view.seat, ())
            live = tuple(p.seat for p in root.players if not p.folded)
            ranges = active_public_ranges(
                self.blueprint, root_history, live, deadline, coverage,
            )
            game = RiverGame(
                root_history, ranges, observed_history=view.history,
                raise_cap=self.config.raise_cap,
                max_public_nodes=self.config.max_public_nodes,
                law_label="active-marginals-ignore-folded-removal-v1",
            )
            solver = RiverCFR(game)
            result = solver.solve(
                max_sweeps=self.config.max_sweeps, deadline=deadline,
                rss_limit_bytes=int(self.config.max_rss_gib * 1024**3),
            )
            if result.completed_sweeps < self.config.min_sweeps:
                raise TimeoutError("River CFR did not complete its required sweeps")
            self.game = game
            self.profile = (result.average if self.config.extraction == "average"
                            else result.current)
            action = self._from_profile(view)
            self.completed += 1
            self.records.append({
                "hand_id": view.hand_id, "status": "completed",
                "seconds": monotonic() - started,
                "sweeps": result.completed_sweeps,
                "public_nodes": len(game.nodes),
                "zero_external_reach_entries": result.zero_external_reach_entries,
                "zero_average_denominators": result.zero_average_denominators,
                "stop_reason": result.stop_reason,
                "law": game.law_label,
                "range_trained_lookups": coverage[("range", "trained")],
                "range_untrained_lookups": coverage[("range", "untrained")],
            })
            return action
        except (TimeoutError, MemoryError, RiverUnsupported, ValueError) as exc:
            self.delegated_hand_id = view.hand_id
            self.delegations += 1
            self.records.append({
                "hand_id": view.hand_id, "status": "delegate",
                "reason": f"{type(exc).__name__}: {exc}",
                "seconds": monotonic() - started,
                "completed_sweeps": solver.completed_sweeps if solver is not None else 0,
                "law": "active-marginals-ignore-folded-removal-v1",
            })
            return self.fallback.choose_action(view)
