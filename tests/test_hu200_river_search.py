"""HU200 river information boundary, exact wagers and live policy locks."""

from dataclasses import replace
from itertools import combinations
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

from src.arena.endgame_quality import _world
from src.blueprint.abstraction import HU200_EQUITY_SCHEMA
from src.blueprint.hu200_river_search import HU200RiverSearchPlayer, HU200RiverSearchPolicy
from src.blueprint.hu20_turn_search import TurnSearchConfig
from src.blueprint.hu20_turn_solver import SolveFailure
from src.blueprint.hu20_turn_tree import compile_tree
from src.game.hand import Hand, Table
from src.game.observation import ActionTaken
from src.game.types import Action, ActionKind, Street
from src.blueprint.search import DECK
from tests.test_hu20_turn_search import FakeSolver, Uniform


class Blueprint(Uniform):
    abstraction = HU200_EQUITY_SCHEMA


def river_fixture(street=Street.RIVER):
    hand = Hand.start(Table(("a", "b"), (20000, 20000), button=0),
                      hand_id="hu200-river-correctness", seed=31)
    hand = hand.apply(Action(ActionKind.RAISE, 4000))
    while hand.observe(hand.actor).street != street:
        view = hand.observe(hand.actor)
        hand = hand.apply(Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds
                                 else ActionKind.CALL))
    return hand


def test_turn_is_blueprint_and_hu20_adapter_stays_depth_restricted():
    solver = FakeSolver()
    policy = HU200RiverSearchPolicy(Blueprint(), solver)
    hand = river_fixture(Street.TURN)
    view = hand.observe(hand.actor)
    assert policy.distribution(view, query_kind="play") == Blueprint().distribution(view)
    assert solver.requests == []
    with pytest.raises(ValueError):
        compile_tree(hand.events, hand.events, stack_chips=20000)
    river = river_fixture()
    with pytest.raises(ValueError):
        compile_tree(river.events, river.events)


def test_external_table_defaults_preserve_exact_public_betting():
    hand = river_fixture()
    external = (replace(hand.events[0], seat_numbers=(), table_seats=(), capacity=None),) + hand.events[1:]
    native = compile_tree(hand.events, hand.events, stack_chips=20000)
    adapted = compile_tree(external, external, stack_chips=20000)
    assert adapted["nodes"] == native["nodes"]
    assert external[0].capacity is None
    corrupted = list(external)
    i = next(i for i, e in enumerate(corrupted) if isinstance(e, ActionTaken))
    corrupted[i] = replace(corrupted[i], paid=corrupted[i].paid + 1)
    with pytest.raises((ValueError, RuntimeError)):
        compile_tree(tuple(corrupted), tuple(corrupted), stack_chips=20000)


def test_full_range_uses_each_public_factor_and_no_turn_solve():
    class Recorded(Blueprint):
        def __init__(self):
            self.seen = []

        def distribution(self, view):
            self.seen.append((view.seat, view.hole_cards, view.history))
            return super().distribution(view)

    base, solver = Recorded(), FakeSolver()
    policy = HU200RiverSearchPolicy(base, solver)
    hand = river_fixture()
    ranges, coverage = policy._ranges(hand.events, hand.actor, monotonic() + 120)
    factors = [e for e in hand.events if isinstance(e, ActionTaken)]
    assert len(base.seen) == 1081 * len(factors)
    assert not solver.requests
    assert coverage["holdings_per_seat"] == 1081
    for rows in ranges.values():
        assert len(rows) == 1081
        assert sum(w for _, w in rows) == pytest.approx(1)
        assert all(set(h).isdisjoint(hand.observe(hand.actor).board) for h, _ in rows)


def test_exact_pressure_relocks_earlier_hero_and_hidden_deals_share_solution():
    hand, solver = river_fixture(), FakeSolver()
    policy = HU200RiverSearchPolicy(Blueprint(), solver)
    hero = hand.actor
    view = hand.observe(hero)
    first = policy.distribution(view, query_kind="play")
    request = solver.requests[-1]
    assert request["initial_street"] == "river"
    hand = hand.apply(Action(ActionKind.CHECK))
    amount = hand.observe(hand.actor).pot * 3 // 2
    hand = hand.apply(Action(ActionKind.RAISE, amount))
    second = policy.distribution(hand.observe(hero), query_kind="play")
    inserted = solver.requests[-1]
    assert len(solver.requests) == 2

    assert any(a.get("amount") == amount for n in inserted["nodes"]
               if not n["terminal"] for a in n["actions"])
    assert len(inserted["locks"]) == 2
    lock = next(lock for lock in inserted["locks"] if lock["line"] == [])
    assert lock["actions"] == request["nodes"][0]["actions"]
    assert np.allclose(lock["strategy"], first[1][0])
    # Querying another compatible hidden deal cannot trigger a different solve.
    alternate = _world(hand.events, hand.observe(hero).board, {hero: view.hole_cards})
    assert policy.distribution(alternate.observe(hero)) == second
    assert len(solver.requests) == 2


@pytest.mark.parametrize("amount", (8000, 7333, 16000))
def test_only_provably_off_menu_cold_ancestors_are_skipped(amount):
    class Legacy(HU200RiverSearchPolicy):
        def _try_ancestor(self, history, bot_seat, previous):
            return True

    hand = river_fixture()
    first = hand.observe(hand.actor)
    current = hand.apply(Action(ActionKind.CHECK)).apply(Action(ActionKind.RAISE, amount))
    policies = []
    for cls in (Legacy, HU200RiverSearchPolicy):
        solver = FakeSolver()
        policy = cls(Blueprint(), solver)
        policy._start_live_hand(first.hand_id)
        holdings = tuple(tuple(sorted(h)) for h in combinations(
            (c for c in DECK if c not in first.board), 2))
        matrix = policy._base_matrix(first.history, holdings)
        policy._remember_play(first, matrix, True, cause="earlier-blueprint-river-action")
        policy.distribution(current.observe(first.seat), query_kind="play")
        policies.append(policy)
    old, optimized = policies
    assert old.solver.requests[-1] == optimized.solver.requests[-1]
    np.testing.assert_array_equal(old.played[first.seat, current.events][0].probabilities,
                                  optimized.played[first.seat, current.events][0].probabilities)
    if amount == 7333:
        assert len(optimized.solver.requests) == len(old.solver.requests) - 1
        assert optimized.stats["certified_off_menu_ancestor_skips"] == 1
    else:
        assert len(optimized.solver.requests) == len(old.solver.requests)


def test_zero_likelihood_river_observation_falls_back():
    class NoRiverCheck(Blueprint):
        def distribution(self, view):
            menu, p, known = super().distribution(view)
            if view.street == Street.RIVER:
                p = tuple(0 if c.action.kind == ActionKind.CHECK else 1 for c in menu)
                p = tuple(x / sum(p) for x in p)
            return menu, p, known

    hand = river_fixture().apply(Action(ActionKind.CHECK))
    policy = HU200RiverSearchPolicy(NoRiverCheck(), FakeSolver())
    view = hand.observe(hand.actor)
    assert policy.distribution(view, query_kind="play") == NoRiverCheck().distribution(view)
    assert policy.stats["play:fallback:zero_conditional_support"] == 1


def test_structural_zero_support_and_tiny_positive_support_are_distinct():
    hand = river_fixture()
    view = hand.observe(hand.actor)
    a, b, c, d, e, f, g = [card for card in DECK if card not in view.board][:7]
    own = tuple(sorted((a, b)))
    rival = ((tuple(sorted((a, c))), .3), (tuple(sorted((a, d))), .6),
             (tuple(sorted((b, e))), .1))
    policy = HU200RiverSearchPolicy(Blueprint(), FakeSolver())
    solution = SimpleNamespace(root=view.history, request={"locks": []}, ranges={
        view.seat: ((own, 1.),), 1 - view.seat: rival})
    with pytest.raises(SolveFailure, match="compatible holding"):
        policy._supported_holdings(solution, view)
    solution.ranges[1 - view.seat] += ((tuple(sorted((f, g))), 1e-300),)
    assert policy._supported_holdings(solution, view) == frozenset((own,))


def test_failed_live_decision_stays_blueprint_after_cache_eviction():
    class Failed(FakeSolver):
        def solve(self, request, deadline):
            self.requests.append(request)
            raise SolveFailure("timeout", "fixture timeout")

    hand, solver = river_fixture(), Failed()
    policy = HU200RiverSearchPolicy(Blueprint(), solver,
        replace(HU200RiverSearchPolicy(Blueprint(), solver).config, cache_entries=1))
    view = hand.observe(hand.actor)
    expected = Blueprint().distribution(view)
    assert policy.distribution(view, query_kind="play") == expected
    policy.cache.clear()
    assert policy.distribution(view, query_kind="play") == expected
    assert len(solver.requests) == 1
    assert policy.stats["play:fallback:timeout"] == 2
    assert policy.played[view.seat, view.history][1]


def test_support_allocation_error_is_counted_blueprint_fallback():
    class NoMemory(HU200RiverSearchPolicy):
        def _supported_holdings(self, solution, view):
            raise MemoryError("fixture allocation failure")

    hand = river_fixture()
    view = hand.observe(hand.actor)
    policy = NoMemory(Blueprint(), FakeSolver())
    assert policy.distribution(view, query_kind="play") == Blueprint().distribution(view)
    assert policy.stats["play:fallback:memory_refusal"] == 1
    assert policy.decision_records[-1]["fallback_cause"] == "memory_refusal"


@pytest.mark.parametrize("failed", (False, True))
def test_fresh_player_reproduces_persisted_matrices_and_timeouts(tmp_path, failed):
    class FirstSolver(FakeSolver):
        def solve(self, request, deadline):
            if failed:
                self.requests.append(request)
                raise SolveFailure("timeout", "recorded availability failure")
            return super().solve(request, deadline)

    class NoRecompute(FakeSolver):
        def solve(self, request, deadline):
            raise AssertionError("Ledger replay must not ask a live solver")

    hand = river_fixture()
    view = hand.observe(hand.actor)
    policy = HU200RiverSearchPolicy(Blueprint(), FirstSolver(), ledger=tmp_path)
    first = HU200RiverSearchPlayer(policy, seed=3)
    action = first.choose_action(view)
    replayed = HU200RiverSearchPlayer(HU200RiverSearchPolicy(
        Blueprint(), NoRecompute(), ledger=tmp_path, replay_ledger=True), seed=3)
    assert replayed.choose_action(view) == action
    assert replayed.last_decision["changed_action"] == first.last_decision["changed_action"]
    assert replayed.last_decision["fallback"] == failed
    assert replayed.last_decision["replayed"]
    assert (hand.apply(action).events == hand.apply(replayed.last_decision["search_action"]).events)


def test_rejects_range_floor_reduced_menu_and_wrong_schema():
    for config in (TurnSearchConfig(menu="cap2"), TurnSearchConfig()):
        with pytest.raises(ValueError):
            HU200RiverSearchPolicy(Blueprint(), FakeSolver(), config)
    with pytest.raises(ValueError):
        HU200RiverSearchPolicy(Uniform(), FakeSolver())
