"""Opt-in exact-holding HU20 river re-solving under a blueprint range model."""

from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from math import fsum
from random import Random
import resource
import sys
from time import monotonic

import numpy as np

from src.arena.schedule import digest
from src.blueprint.river_cfr import RiverCFR
from src.blueprint.river_game import RiverGame, RiverUnsupported, river_root_history
from src.blueprint.search import DECK, _observed_likelihood
from src.game.observation import ActionTaken, HandStarted, replay
from src.game.types import ActionKind, Street

VERSION = "hu20-full-range-river-average-resolving-v1"
LAW = "hu20-product-blueprint-likelihood-compatible-v1"


def public_identity(history):
    """Hand labels never influence a public solve or prevent profile reuse."""
    return digest([asdict(replace(e, hand_id="") if isinstance(e, HandStarted) else e)
                   for e in history])


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


@dataclass(frozen=True, slots=True)
class HU20RiverConfig:
    sweeps: int = 250
    raise_cap: int = 2
    max_public_nodes: int = 10_000
    watchdog_seconds: float = 120
    rss_limit_bytes: int = 6 * 1024**3
    cache_entries: int = 8

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in
               (self.sweeps, self.max_public_nodes, self.rss_limit_bytes, self.cache_entries)):
            raise ValueError("Need positive river work/cache limits")
        if type(self.raise_cap) is not int or self.raise_cap < 0 or not 0 < self.watchdog_seconds <= 1200:
            raise ValueError("Invalid river tree/watchdog limit")


class HU20RiverGame(RiverGame):
    table_players = 2
    free_fold = False

    def __init__(self, root_history, ranges, **kwargs):
        root = replay(root_history, 0, ())
        start = root_history[0]
        if (len(root.players) != 2 or start.stacks != (2000, 2000)
                or start.small_blind != 50 or start.big_blind != 100
                or start.chip_unit != "0.01"):
            raise RiverUnsupported("HU20 adapter requires its exact reset-stack game")
        super().__init__(root_history, ranges, law_label=LAW, **kwargs)


def public_ranges(blueprint, root_history, check=lambda: None):
    """Exact compatible law conditional on the declared action-likelihood model.

    Zero on-menu likelihood stays zero. Off-menu raises use the existing
    corrected likelihood kernel, explicitly counted; no actions are translated.
    """
    view = replay(root_history, 0, ())
    pairs = tuple(combinations((c for c in DECK if c not in view.board), 2))
    ranges = {}; counts = Counter()
    for seat in (0, 1):
        weighted = []
        for pair in pairs:
            check(); mass = 1.0
            for index, event in enumerate(root_history):
                if isinstance(event, ActionTaken) and event.seat == seat:
                    prior = replay(root_history[:index], seat, pair)
                    mass *= _observed_likelihood(blueprint, prior, event.action, coverage=counts)
            weighted.append((tuple(sorted(pair)), mass))
        total = fsum(mass for _, mass in weighted)
        if total <= 0:
            raise RiverUnsupported("Blueprint likelihood factor has zero support")
        ranges[seat] = tuple((pair, mass / total) for pair, mass in weighted)
    coverage = {"trained": counts[("range", "trained")],
                "missing": counts[("range", "untrained")],
                "off_menu_raise_likelihoods": counts[("range", "off_menu_action")],
                "holdings_per_seat": [len(ranges[s]) for s in (0, 1)],
                "positive_holdings_per_seat": [sum(w > 0 for _, w in ranges[s]) for s in (0, 1)]}
    return ranges, coverage


class RiverProfileCache:
    """Bounded source-specific full-profile cache; no private holding/RNG keys."""
    def __init__(self, blueprint, entries=8):
        self.blueprint = blueprint
        self.identity = digest(getattr(blueprint, "description", {"fixture": type(blueprint).__name__}))
        self.entries = entries
        self.ranges = OrderedDict()
        self.profiles = OrderedDict()
        self.stats = Counter()

    def put(self, store, key, value):
        store[key] = value; store.move_to_end(key)
        while len(store) > self.entries:
            store.popitem(last=False); self.stats["evictions"] += 1


class HU20RiverPlayer:
    def __init__(self, blueprint, seed, config=HU20RiverConfig(), cache=None, deadline=None):
        self.blueprint = blueprint; self.config = config; self.random = Random(seed)
        self.cache = cache if cache is not None else RiverProfileCache(blueprint, config.cache_entries)
        if self.cache.blueprint is not blueprint:
            raise ValueError("River cache belongs to a different immutable blueprint")
        self.hand_id = None; self.solution = None; self.used = {}; self.records = []
        self.deadline = deadline

    def _reset(self, view):
        if self.hand_id != view.hand_id:
            self.hand_id = view.hand_id; self.solution = None; self.used = {}

    def _solve(self, view):
        started = monotonic(); deadline = started + self.config.watchdog_seconds
        if self.deadline is not None:
            deadline = min(deadline, self.deadline)

        def check():
            if monotonic() >= deadline:
                raise TimeoutError("HU20 fixed-sweep watchdog expired")
            if peak_rss() >= self.config.rss_limit_bytes:
                raise MemoryError("HU20 river process RSS guard")

        root = river_root_history(view.history)
        root_id = public_identity(root)
        re_solve = self.solution is not None
        try:
            check()
            if root_id in self.cache.ranges:
                ranges, coverage, range_id = self.cache.ranges[root_id]
                self.cache.ranges.move_to_end(root_id); self.cache.stats["range_hits"] += 1
            else:
                ranges, coverage = public_ranges(self.blueprint, root, check)
                range_id = digest(ranges)
                self.cache.put(self.cache.ranges, root_id, (ranges, coverage, range_id))
            inserted = [(public_identity(view.history[:i]), asdict(e.action))
                        for i, e in enumerate(view.history)
                        if isinstance(e, ActionTaken) and e.street == Street.RIVER
                        and e.action.kind == ActionKind.RAISE]
            constraints = [(k, digest([asdict(c.action) for c in menu]),
                            digest(policy.tolist())) for k, (menu, policy) in sorted(self.used.items())]
            key = digest([VERSION, self.cache.identity, root_id, range_id,
                          self.config.sweeps, self.config.raise_cap, inserted, constraints])
            hit = key in self.cache.profiles
            if hit:
                solution = self.cache.profiles[key]; self.cache.profiles.move_to_end(key)
                self.cache.stats["hits"] += 1
            else:
                game = HU20RiverGame(root, ranges, observed_history=view.history,
                    raise_cap=self.config.raise_cap, max_public_nodes=self.config.max_public_nodes)
                lookup = {public_identity(n.history): n.id for n in game.nodes if n.actor is not None}
                fixed = {}
                for prior, (menu, policy) in self.used.items():
                    if prior not in lookup or game.nodes[lookup[prior]].menu != menu:
                        raise RiverUnsupported("Prior hero action menu changed during re-solving")
                    fixed[lookup[prior]] = policy
                check()
                result = RiverCFR(game, fixed_profile=fixed).solve(max_sweeps=self.config.sweeps,
                    deadline=deadline, rss_limit_bytes=self.config.rss_limit_bytes)
                if result.completed_sweeps != self.config.sweeps or result.stop_reason != "sweep_cap":
                    raise TimeoutError("HU20 solve did not finish its fixed sweeps")
                for policy in result.average.values():
                    policy.setflags(write=False)
                solution = (game, result.average, lookup)
                self.cache.put(self.cache.profiles, key, solution)
                self.cache.stats["solves"] += 1
            self.solution = solution
            self.records.append({"status": "completed", "cache_hit": hit,
                "re_solve": re_solve, "sweeps": self.config.sweeps,
                "public_root": root_id, "range_identity": range_id, "profile_identity": key,
                "model_identity": self.cache.identity, "range_coverage": coverage,
                "inserted_raises": inserted, "frozen_hero_nodes": len(self.used),
                "public_nodes": len(solution[0].nodes), "seconds": monotonic()-started,
                "peak_rss_bytes": peak_rss(), "law": LAW, "river_delegation": False})
        except (ValueError, TimeoutError, MemoryError) as exc:
            self.records.append({"status": "failure", "reason": f"{type(exc).__name__}: {exc}",
                "seconds": monotonic()-started, "river_delegation": False})
            raise

    def distribution(self, view):
        self._reset(view)
        if view.street != Street.RIVER:
            return self.blueprint.distribution(view)
        if view.finished or view.actor != view.seat:
            raise ValueError("HU20 river distribution needs an acting-seat observation")
        identity = public_identity(view.history)
        if self.solution is None or identity not in self.solution[2]:
            self._solve(view)
        else:
            self.cache.stats["retained_queries"] += 1
        game, profile, lookup = self.solution
        node = game.nodes[lookup[identity]]
        if node.actor != view.seat:
            raise RiverUnsupported("Retained node has a different acting seat")
        index = game.holdings[game.seats.index(view.seat)].index(tuple(sorted(view.hole_cards)))
        return node.menu, tuple(profile[node.id][index]), True

    def choose_action(self, view):
        menu, probabilities, _ = self.distribution(view)
        action = self.random.choices(menu, weights=probabilities, k=1)[0].action
        view.legal_actions.validate(action)
        if view.street == Street.RIVER:
            game, profile, lookup = self.solution
            identity = public_identity(view.history)
            self.used[identity] = (menu, profile[lookup[identity]])
        return action
