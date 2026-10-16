"""River-only joint-range re-solving for the HU200 K50 blueprint.

This is ordinary re-solving with hero locks, not a safe re-solving gadget.
The external solver has no opponent counterfactual-value constraints.
"""

from collections import Counter
from functools import wraps
from itertools import combinations
from math import fsum
from random import Random
from time import monotonic
import traceback

import numpy as np

from src.arena.schedule import digest
from src.blueprint.abstraction import Choice, HU200_EQUITY_SCHEMA, choices
from src.blueprint.hu20_turn_search import (
    HU20TurnSearchPolicy, TurnSearchConfig, observed_likelihood,
)
from src.blueprint.hu20_turn_solver import PolicyMatrix, SolveFailure
from src.blueprint.hu20_turn_tree import betting_line, compile_tree, line_key
from src.blueprint.search import DECK
from src.blueprint.river_ledger import DecisionLedger, FamilyPeak, decode_matrix, encode_matrix
from src.game.observation import ActionTaken, replay
from src.game.types import Street


VERSION = "hu200-k50-river-joint-range-locks-v1"


def measured(phase):
    def decorate(method):
        @wraps(method)
        def call(self, *args, **kwargs):
            started = monotonic()
            try:
                return method(self, *args, **kwargs)
            finally:
                self.phase_seconds[phase] += monotonic() - started
        return call
    return decorate


class HU200RiverSearchPolicy(HU20TurnSearchPolicy):
    """Full holding matrices and fallback locks share the HU20 live ledger.

Earlier streets always use the blueprint. All holding likelihoods use that
same policy, including its configured translation, without opponent labels or
actual opponent cards. The existing off-menu raise kernel is explicit model
error; exact insertion applies to the solved river betting tree.
"""

    def __init__(self, blueprint, solver, config=None, *, ledger=None, replay_ledger=False):
        config = config or TurnSearchConfig(
            iterations=50, threads=6, compress=False,
            opponent_likelihood_floor=0, decision_seconds=120,
            memory_budget_bytes=2 * 1024**3, cache_entries=2,
        )
        if (config.menu != "native" or config.opponent_likelihood_floor != 0
                or config.decision_seconds is None):
            raise ValueError("HU200 river requires full native menu, zero floor and a watchdog")
        super().__init__(blueprint, solver, config)
        self.description.update(version=VERSION, search_streets=["river"],
                                resolving="joint-range-cfr-with-hero-locks; no safety guarantee")
        self.identity = digest(self.description)
        self.conditional_cache = {}
        self.ledger = DecisionLedger(ledger, replay=replay_ledger) if ledger is not None else None
        if replay_ledger and self.ledger is None:
            raise ValueError("Replay requires an explicit decision ledger")
        self.decision_records = []
        self.phase_seconds = Counter()

    @staticmethod
    def _validate_blueprint(blueprint):
        if getattr(blueprint, "abstraction", None) != HU200_EQUITY_SCHEMA:
            raise ValueError("HU200 river search requires the HU200 K50 menu-v1 schema")

    @measured("tree")
    def _compile_tree(self, root, history, **kwargs):
        request = compile_tree(root, history, stack_chips=20000, **kwargs)
        request["action_values"] = True
        return request

    def _observe_live_turn(self, history, bot_seat):
        # There is no turn solver or turn-policy state in this adapter.
        pass

    def _try_ancestor(self, history, bot_seat, previous):
        index, event = previous[-1]
        prior = history[:index]
        # A solution requested before this event cannot have inserted it at
        # this prefix. If it is absent from the full native menu, the old
        # solution.matrix(history) must fail. Do not infer non-reuse merely
        # from a likely disagreement with blueprint opponent probabilities.
        # Existing candidates remain authoritative even in unusual probe use.
        if ((bot_seat, prior) in self.live_solutions
                or self._key(prior, bot_seat) in self.cache):
            return True
        public = replay(prior, event.seat, ())
        if event.action not in [c.action for c in choices(public, raise_cap=None, free_fold=False)]:
            self.stats["certified_off_menu_ancestor_skips"] += 1
            return False
        return True

    def _start_live_hand(self, hand_id):
        if hand_id != self.live_hand_id:
            self.conditional_cache.clear()
        super()._start_live_hand(hand_id)

    def _supported_holdings(self, solution, view):
        key = (id(solution), view.history, view.seat)
        if key in self.conditional_cache:
            return self.conditional_cache[key]
        weights = {seat: dict(solution.ranges[seat]) for seat in (0, 1)}
        locks = {line_key(lock["line"]): lock for lock in solution.request["locks"]}
        for i, event in enumerate(view.history):
            if i < len(solution.root) or not isinstance(event, ActionTaken):
                continue
            prefix = line_key(betting_line(solution.root, view.history[:i]))
            lock = locks.get(prefix)
            if lock is None:
                matrix = (self._played_matrix(view.history[:i], view.seat) if event.seat == view.seat
                          else self._opponent_lock(view.history[:i], event.action, None))
                if matrix is None:
                    matrix = solution.matrix(view.history[:i])
                column = next(j for j, c in enumerate(matrix.menu) if c.action == event.action)
                lookup = {h: float(row[column]) for h, row in zip(
                    matrix.holdings, matrix.probabilities, strict=True)}
            else:
                node = next(n for n in solution.request["nodes"] if line_key(n["line"]) == prefix)
                action_index = next(j for j, a in enumerate(node["native_actions"])
                    if a["kind"] == event.action.kind and a["raise_to"] == event.action.raise_to)
                values = np.asarray(lock["strategy"]).reshape(len(lock["actions"]), -1)[action_index]
                lookup = {tuple(sorted(h)): float(p) for h, p in zip(lock["holdings"], values, strict=True)}
            for holding in weights[event.seat]:
                weights[event.seat][holding] *= lookup.get(holding, 0)
            total = fsum(weights[event.seat].values())
            if total <= 0:
                raise SolveFailure("zero_conditional_support", "Observed river history has zero likelihood")
            weights[event.seat] = {h: w / total for h, w in weights[event.seat].items()}
        rival = [(set(h), w) for h, w in weights[1 - view.seat].items() if w > 0]
        # Support is structural: subtracting blocker masses can invent tiny
        # positive residuals for an impossible hand, or erase real tiny support.
        supported = frozenset(h for h, w in weights[view.seat].items()
            if w > 0 and any(set(h).isdisjoint(other) for other, _ in rival))
        if not supported:
            raise SolveFailure("zero_joint_support", "Current river law has no compatible holding pair")
        self.conditional_cache[key] = supported
        return supported

    @measured("opponent_locks")
    def _opponent_lock(self, history, observed_action, deadline):
        """Represent blueprint likelihood at observed river prefixes exactly.

        The root solver takes equal street contributions. Keeping that root
        and locking observed prefixes gives the same conditional law at the
        actual decision, without inventing an asymmetric reset-game state.
        An off-menu likelihood is a declared kernel, not a probability density
        over all possible integer bets. Only its observed event is inserted.
        """
        public = replay(history, 0, ())
        pairs = tuple(tuple(sorted(h)) for h in combinations(
            (c for c in DECK if c not in public.board), 2))
        menu = None
        rows = []
        for pair in pairs:
            self._check(deadline)
            choices, probabilities, _ = self.blueprint.distribution(replay(history, public.actor, pair))
            if menu is None:
                menu = choices
            if choices != menu:
                raise SolveFailure("invalid_base", "Blueprint menu varies by holding")
            if observed_action in [c.action for c in choices]:
                rows.append(probabilities)
            else:
                likelihood = observed_likelihood(choices, probabilities, observed_action)
                if not 0 <= likelihood <= 1:
                    raise SolveFailure("invalid_base", "Off-menu action likelihood is outside [0,1]")
                rows.append(tuple((1 - likelihood) * p for p in probabilities) + (likelihood,))
        if observed_action not in [c.action for c in menu]:
            menu = menu + (Choice("observed-likelihood", observed_action),)
        values = np.asarray(rows, dtype=np.float64)
        values.setflags(write=False)
        return PolicyMatrix(menu, pairs, values)

    @measured("ranges")
    def _ranges(self, root, bot_seat, deadline):
        key = self._key(root, bot_seat)
        if key in self.range_cache:
            self.range_cache.move_to_end(key)
            return self.range_cache[key]
        board = replay(root, 0, ()).board
        pairs = tuple(tuple(sorted(pair)) for pair in combinations(
            (card for card in DECK if card not in board), 2))
        weights = {seat: np.ones(len(pairs), dtype=np.float64) for seat in (0, 1)}
        coverage = Counter()
        for index, event in enumerate(root):
            if not isinstance(event, ActionTaken):
                continue
            prior = root[:index]
            for j, holding in enumerate(pairs):
                self._check(deadline)
                menu, probabilities, known = self.blueprint.distribution(
                    replay(prior, event.seat, holding))
                likelihood = observed_likelihood(menu, probabilities, event.action)
                coverage["trained" if known else "missing"] += 1
                coverage["off_menu_factors"] += int(event.action not in [c.action for c in menu])
                coverage["zero_likelihood:" + str(event.seat)] += int(likelihood == 0)
                weights[event.seat][j] *= likelihood
            total = fsum(weights[event.seat])
            if total <= 0:
                raise SolveFailure("zero_support_" + ("own" if event.seat == bot_seat else "opponent"),
                                   "Blueprint history likelihood has no support")
            weights[event.seat] /= total
        ranges = {seat: tuple(zip(pairs, map(float, weights[seat]), strict=True)) for seat in (0, 1)}
        supported = [[set(h) for h, w in ranges[seat] if w > 0] for seat in (0, 1)]
        if not any(a.isdisjoint(b) for a in supported[0] for b in supported[1]):
            raise SolveFailure("zero_joint_support", "No compatible joint holding law")
        coverage.update(holdings_per_seat=len(pairs), **{
            "positive_holdings:" + str(seat): len(supported[seat]) for seat in (0, 1)})
        result = ranges, dict(coverage)
        self._put(self.range_cache, key, result)
        return result

    def distribution(self, view, *, query_kind="probe"):
        if view.street != Street.RIVER:
            if view.finished or view.actor != view.seat:
                raise ValueError("Search policy requires an acting-seat observation")
            if query_kind == "play":
                self._start_live_hand(view.hand_id)
            return self.blueprint.distribution(view)
        if query_kind != "play":
            return super().distribution(view, query_kind=query_kind)
        self._start_live_hand(view.hand_id)
        key = digest([view.hand_id, self._key(view.history, view.seat)])
        if self.ledger is not None and self.ledger.replay:
            row = self.ledger.read(key)
            matrix = decode_matrix(row["matrix"])
            supported = frozenset(tuple(h) for h in row["supported"])
            self._remember_play(view, matrix, row["fallback"],
                                cause=row["cause"], supported=supported)
            fallback = row["fallback"] or tuple(sorted(view.hole_cards)) not in supported
            self.stats["play:fallback:" + (row["cause"] or "unsupported_holding") if fallback
                       else "play:search"] += 1
            self.decision_records.append(dict(row["telemetry"], replayed=True, ledger_key=key))
            if fallback:
                return self.blueprint.distribution(view)
            return matrix.menu, matrix.row(view.hole_cards), True
        started = monotonic()
        self.phase_seconds.clear()
        native_before = len(getattr(self.solver, "records", ()))
        with FamilyPeak() as memory:
            try:
                result = super().distribution(view, query_kind=query_kind)
            except (OSError, ArithmeticError, RuntimeError, TypeError, StopIteration, MemoryError) as exc:
                cause = "memory_refusal" if isinstance(exc, MemoryError) else "policy_error:" + type(exc).__name__
                self.records.append(dict(status="preparation_error", cause=cause,
                    reason=str(exc), traceback=traceback.format_exc()))
                public = replay(view.history, 0, ())
                pairs = tuple(tuple(sorted(h)) for h in combinations(
                    (c for c in DECK if c not in public.board), 2))
                self._remember_play(view, self._base_matrix(view.history, pairs), True, cause=cause)
                self._pin_live(view, failure=SolveFailure(cause, "River policy preparation failed"))
                self.stats["play:fallback:" + cause] += 1
                result = self.blueprint.distribution(view)
        matrix, policy_fallback, _, info = self.played[view.seat, view.history]
        fallback = policy_fallback
        cause = info["cause"]
        if tuple(sorted(view.hole_cards)) not in info["supported"]:
            fallback, cause = True, "unsupported_holding"
        pinned = self.live_solutions.get((view.seat, view.history))
        request = pinned.request if hasattr(pinned, "request") else None
        telemetry = dict(public_history=key, seconds=monotonic() - started,
            sampled_peak_family_rss_bytes=memory.peak, memory_sample_seconds=memory.sample_seconds,
            minimum_host_available_bytes=memory.minimum_available_bytes,
            tree_nodes=len(request["nodes"]) if request is not None else None,
            fallback=fallback, fallback_cause=cause, replayed=False,
            model_ev_chips=None, model_ev_unavailable="native action values not exported",
            changed_action=None)
        telemetry["phase_seconds"] = dict(self.phase_seconds)
        telemetry["native_seconds"] = fsum(r.get("seconds", 0) for r in
            getattr(self.solver, "records", ())[native_before:])
        telemetry["preparation_errors"] = [r for r in self.records if r["status"] == "preparation_error"]
        if not fallback and hasattr(pinned, "matrix"):
            native = pinned.matrix(view.history)
            if (native.action_values_chips is not None and np.isfinite(
                    native.action_values_chips[native.holding_indices[tuple(sorted(view.hole_cards))]]).all()):
                values = native.action_values_chips[native.holding_indices[tuple(sorted(view.hole_cards))]]
                base_menu, base_p, _ = self.blueprint.distribution(view)
                by_action = {c.action: float(v) for c, v in zip(native.menu, values, strict=True)}
                search_ev = fsum(p * v for p, v in zip(result[1], values, strict=True))
                base_ev = fsum(p * by_action[c.action] for c, p in zip(base_menu, base_p, strict=True))
                telemetry.update(model_ev_chips=search_ev, blueprint_ev_chips=base_ev,
                    model_ev_delta_chips=search_ev - base_ev, model_ev_unavailable=None,
                    model_ev_scope="conditional own holding; solved opponent continuation; native utility origin",
                    action_values_chips=list(map(float, values)))
            elif native.action_values_chips is not None:
                telemetry["model_ev_unavailable"] = "native conditional normalization unavailable"
        self.decision_records.append(telemetry)
        if self.ledger is not None:
            self.ledger.write(key, dict(matrix=encode_matrix(matrix), fallback=policy_fallback,
                cause=info["cause"], supported=sorted(info["supported"]), telemetry=telemetry))
        return result


class HU200RiverSearchPlayer:
    """Use the same uniform draw for the played and blueprint comparison action."""

    def __init__(self, policy, seed):
        self.policy = policy
        self.random = Random(seed)
        self.last_decision = None

    @staticmethod
    def _draw(menu, probabilities, uniform):
        target = uniform * fsum(probabilities)
        cumulative = 0.0
        for choice, probability in zip(menu, probabilities, strict=True):
            cumulative += probability
            if target < cumulative:
                return choice.action
        return menu[-1].action

    def choose_action(self, view):
        started = monotonic()
        menu, probabilities, _ = self.policy.distribution(view, query_kind="play")
        uniform = self.random.random()
        action = self._draw(menu, probabilities, uniform)
        if view.street == Street.RIVER:
            base_menu, base_probabilities, _ = self.policy.blueprint.distribution(view)
            baseline = self._draw(base_menu, base_probabilities, uniform)
            self.last_decision = dict(self.policy.decision_records[-1],
                changed_action=action != baseline, blueprint_action=baseline,
                search_action=action, total_player_seconds=monotonic() - started)
        else:
            self.last_decision = None
        view.legal_actions.validate(action)
        return action
