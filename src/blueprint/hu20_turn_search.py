"""Public-state nested HU20 search, shared by play and hypothetical LBR queries."""

from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from itertools import combinations
from math import exp, fsum, isfinite, log
from random import Random
from time import monotonic

import numpy as np

from src.arena.schedule import digest
from src.blueprint.hu20_river import public_identity
from src.blueprint.abstraction import information_key
from src.blueprint.hu20_turn_solver import PolicyMatrix, SolveFailure
from src.blueprint.hu20_turn_tree import (
    VERSION, betting_line, compile_tree, line_key, round_root, solver_action,
)
from src.blueprint.search import DECK
from src.game.observation import ActionTaken, replay
from src.game.types import ActionKind, Street


@dataclass(frozen=True, slots=True)
class TurnSearchConfig:
    iterations: int = 100
    threads: int = 2
    compress: bool = True
    menu: str = "native"
    opponent_likelihood_floor: float = .01
    decision_seconds: float = 30
    memory_budget_bytes: int = 5 * 1024**3
    max_public_nodes: int = 300_000
    cache_entries: int = 32

    def __post_init__(self):
        for name in ("iterations", "threads", "memory_budget_bytes", "max_public_nodes", "cache_entries"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Invalid {name}")
        if (type(self.compress) is not bool or self.menu not in ("native", "cap2")
                or self.opponent_likelihood_floor not in (0, .01)
                or not isfinite(self.decision_seconds) or not 0 < self.decision_seconds <= 30
                or self.memory_budget_bytes > 10 * 1024**3):
            raise ValueError("Invalid search settings")


@dataclass(frozen=True)
class PublicSolution:
    root: tuple
    request: dict
    profiles: dict
    ranges: dict
    coverage: dict

    def matrix(self, history):
        return self.profiles[line_key(betting_line(self.root, history))]


def observed_likelihood(menu, probabilities, action):
    exact = fsum(p for c, p in zip(menu, probabilities, strict=True) if c.action == action)
    if action.kind != ActionKind.RAISE or any(c.action == action for c in menu):
        return exact
    mass = fsum(p * exp(-abs(log(action.raise_to / c.action.raise_to)))
                for c, p in zip(menu, probabilities, strict=True)
                if c.action.kind == ActionKind.RAISE)
    return max(.01, mass)


class HU20TurnSearchPolicy:
    """No mutable live-hand state: a public prefix reconstructs every prior lock."""

    def __init__(self, blueprint, solver, config=TurnSearchConfig()):
        self.blueprint = blueprint
        self.abstraction = getattr(blueprint, "abstraction", None)
        self.raise_cap = getattr(blueprint, "raise_cap", None)
        self.game = getattr(blueprint, "game", None)
        self.solver = solver
        self.config = config
        self.description = {"version": VERSION, "base": blueprint.description,
                            "config": asdict(config),
                            "external_identity": getattr(solver, "expected_sha256", None)}
        self.identity = digest(self.description)
        self.cache = OrderedDict()
        self.range_cache = OrderedDict()
        self.records = []
        self.stats = Counter()

    def _key(self, history, bot_seat):
        return digest([self.identity, bot_seat, public_identity(history)])

    def _put(self, cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.config.cache_entries:
            cache.popitem(last=False)
            self.stats["evictions"] += 1

    @staticmethod
    def _check(deadline):
        if monotonic() >= deadline:
            raise SolveFailure("timeout", "Public policy preparation deadline")

    def _ranges(self, root, bot_seat, deadline):
        key = self._key(root, bot_seat)
        if key in self.range_cache:
            self.range_cache.move_to_end(key)
            return self.range_cache[key]
        public = replay(root, 0, ())
        pairs = tuple(combinations((c for c in DECK if c not in public.board), 2))
        weights = {s: np.ones(len(pairs), dtype=np.float64) for s in (0, 1)}
        counts = Counter()
        for index, event in enumerate(root):
            if not isinstance(event, ActionTaken):
                continue
            prior = root[:index]
            matrix = None
            if event.street == Street.TURN:
                try:
                    matrix = self._resolve(prior, bot_seat, deadline).matrix(prior)
                except SolveFailure as exc:
                    counts["turn_conditioning_fallback:" + exc.cause] += 1
            for j, pair in enumerate(pairs):
                self._check(deadline)
                if matrix is not None and tuple(sorted(pair)) in matrix.holding_indices:
                    likelihood = observed_likelihood(matrix.menu, matrix.row(pair), event.action)
                    counts["turn_solution_factors"] += 1
                else:
                    view = replay(prior, event.seat, pair)
                    menu, p, trained = self.blueprint.distribution(view)
                    likelihood = observed_likelihood(menu, p, event.action)
                    counts["trained" if trained else "missing"] += 1
                    if hasattr(self.blueprint, "zero_mass") and trained:
                        counts["zero_average_mass"] += int(information_key(view, menu,
                            schema=self.blueprint.abstraction) in self.blueprint.zero_mass)
                    counts["off_menu_factors"] += int(event.action not in [c.action for c in menu])
                if likelihood == 0:
                    counts["zero_likelihood:own" if event.seat == bot_seat else "zero_likelihood:opponent"] += 1
                if event.seat != bot_seat and likelihood < self.config.opponent_likelihood_floor:
                    likelihood = self.config.opponent_likelihood_floor
                    counts["floored_opponent_factors"] += 1
                weights[event.seat][j] *= likelihood
            total = float(weights[event.seat].sum())
            if total <= 0:
                who = "own" if event.seat == bot_seat else "opponent"
                self.stats["zero_support:" + who] += 1
                raise SolveFailure("zero_support_" + who, "Public action range has no support")
            weights[event.seat] /= total
        ranges = {s: tuple((tuple(sorted(h)), float(w)) for h, w in zip(pairs, weights[s], strict=True))
                  for s in (0, 1)}
        # A product law is normalized only over disjoint private holdings.
        supported = [[set(h) for h, w in ranges[s] if w > 0] for s in (0, 1)]
        if not any(not a.intersection(b) for a in supported[0] for b in supported[1]):
            raise SolveFailure("zero_joint_support", "Empty compatible private law")
        counts.update({"holdings_per_seat": len(pairs),
                       "positive_holdings:0": int((weights[0] > 0).sum()),
                       "positive_holdings:1": int((weights[1] > 0).sum())})
        result = ranges, dict(counts)
        self._put(self.range_cache, key, result)
        return result

    def _base_matrix(self, history, holdings):
        view = replay(history, replay(history, 0, ()).actor, holdings[0])
        menu = self.blueprint.distribution(view)[0]
        rows = []
        for pair in holdings:
            other, p, _ = self.blueprint.distribution(replay(history, view.seat, pair))
            if other != menu:
                raise SolveFailure("invalid_base", "Base action menu depends on private cards")
            rows.append(p)
        p = np.asarray(rows, dtype=np.float64)
        p.setflags(write=False)
        return PolicyMatrix(menu, tuple(holdings), p)

    def _resolve(self, history, bot_seat, deadline):
        self._check(deadline)
        key = self._key(history, bot_seat)
        if key in self.cache:
            result = self.cache[key]
            self.cache.move_to_end(key)
            self.stats["public_cache_hits"] += 1
            if isinstance(result, SolveFailure):
                raise result
            return result
        root = round_root(history)
        previous = [(i, e) for i, e in enumerate(history) if i >= len(root)
                    and isinstance(e, ActionTaken)]
        if previous:
            prior = history[:previous[-1][0]]
            try:
                solution = self._resolve(prior, bot_seat, deadline)
                solution.matrix(history)
                self._put(self.cache, key, solution)
                self.stats["on_tree_queries"] += 1
                return solution
            except KeyError:
                pass
            except SolveFailure:
                # Prior failed decisions stay base-policy locks, never vanish.
                pass
        started = monotonic()
        try:
            ranges, coverage = self._ranges(root, bot_seat, deadline)
            fixed = {}
            for index, event in previous:
                if event.seat != bot_seat:
                    continue
                prior = history[:index]
                try:
                    matrix = self._resolve(prior, bot_seat, deadline).matrix(prior)
                except SolveFailure:
                    holdings = [h for h, w in ranges[bot_seat] if w > 0]
                    matrix = self._base_matrix(prior, holdings)
                fixed[line_key(betting_line(root, prior))] = matrix
            request = compile_tree(root, history, menu=self.config.menu,
                max_nodes=self.config.max_public_nodes, deadline=deadline,
                fixed_menus={k: tuple(c.action for c in m.menu) for k, m in fixed.items()})
            request.update(threads=self.config.threads, compress=self.config.compress,
                           max_iterations=self.config.iterations,
                           memory_budget_bytes=self.config.memory_budget_bytes,
                           ranges=[[{"hand": list(h), "weight": w} for h, w in ranges[s]]
                                   for s in request["seat_map"]], locks=[])
            for node in request["nodes"]:
                matrix = fixed.get(line_key(node["line"]))
                if matrix is None or node["terminal"]:
                    continue
                policy = np.zeros((len(node["actions"]), len(matrix.holdings)))
                # Amounts are absolute raise-to; reconstruct the prior node for jam/kind.
                prior = next(history[:i] for i, e in previous
                             if e.seat == bot_seat and line_key(betting_line(root, history[:i])) == line_key(node["line"]))
                prior_view = replay(prior, bot_seat, ())
                old = {line_key([solver_action(prior_view, c.action)])[0]: i
                       for i, c in enumerate(matrix.menu)}
                for a, token in enumerate(node["actions"]):
                    if line_key([token])[0] in old:
                        policy[a] = matrix.probabilities[:, old[line_key([token])[0]]]
                request["locks"].append({"line": node["line"], "board": request["board"],
                    "player": node["player"], "actions": node["actions"],
                    "holdings": [list(h) for h in matrix.holdings], "strategy": policy.ravel().tolist()})
            self._check(deadline)
            profiles = self.solver.solve(request, deadline)
            self._check(deadline)
            solution = PublicSolution(root, request, profiles, ranges, coverage)
            solution.matrix(history)
            self._put(self.cache, key, solution)
            self.stats["solves"] += 1
            self.records.append({"status": "completed", "public_history": public_identity(history),
                "street": request["initial_street"], "range_coverage": coverage,
                "frozen_hero_nodes": len(fixed), "seconds": monotonic() - started,
                "request_identity": digest({k: v for k, v in request.items() if k != "seconds"})})
            return solution
        except (TimeoutError, MemoryError, ValueError, KeyError) as exc:
            failure = exc if isinstance(exc, SolveFailure) else SolveFailure(
                "timeout" if isinstance(exc, TimeoutError) else "memory_refusal" if isinstance(exc, MemoryError)
                else "invalid_response", str(exc))
            self._put(self.cache, key, failure)
            self.records.append({"status": "failure", "cause": failure.cause,
                "public_history": public_identity(history), "seconds": monotonic() - started})
            raise failure

    def distribution(self, view, *, query_kind="probe"):
        if view.finished or view.actor != view.seat:
            raise ValueError("Search policy requires an acting-seat observation")
        if view.street not in (Street.TURN, Street.RIVER):
            return self.blueprint.distribution(view)
        started = monotonic()
        try:
            solution = self._resolve(view.history, view.seat,
                                     started + self.config.decision_seconds)
            matrix = solution.matrix(view.history)
            probabilities = matrix.row(view.hole_cards)
            for choice in matrix.menu:
                view.legal_actions.validate(choice.action)
            self.stats[query_kind + ":search"] += 1
            if query_kind == "play":
                self.records.append({"status": "decision", "street": view.street.value,
                    "public_history": public_identity(view.history),
                    "seconds": monotonic() - started, "fallback": False})
            return matrix.menu, probabilities, True
        except (SolveFailure, ValueError, KeyError) as exc:
            cause = exc.cause if isinstance(exc, SolveFailure) else "invalid_response"
            self.stats[query_kind + ":fallback:" + cause] += 1
            self.records.append({"status": "fallback", "cause": cause, "query_kind": query_kind,
                "street": view.street.value, "public_history": public_identity(view.history),
                "seconds": monotonic() - started})
            menu, p, trained = self.blueprint.distribution(view)
            for choice in menu:
                view.legal_actions.validate(choice.action)
            return menu, p, trained


class HU20TurnSearchPlayer:
    def __init__(self, policy, seed):
        self.policy = policy
        self.random = Random(seed)

    def choose_action(self, view):
        menu, p, _ = self.policy.distribution(view, query_kind="play")
        action = self.random.choices(menu, weights=p, k=1)[0].action
        view.legal_actions.validate(action)
        return action
