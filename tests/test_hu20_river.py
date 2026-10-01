"""HU20 native/oracle agreement and full-profile information boundaries."""
from itertools import combinations
import numpy as np
import pytest

from src.blueprint.abstraction import choices
from src.blueprint.hu20_river import (HU20RiverConfig, HU20RiverGame, HU20RiverPlayer,
    RiverProfileCache, public_identity, public_ranges)
from src.blueprint.river_cfr import RiverCFR, profile_quality
from src.blueprint.river_game import RiverGame, river_root_history
from src.blueprint.search import DECK
from src.game.hand import Hand, Table
from src.game.observation import ActionTaken
from src.game.types import Action, ActionKind, Street
from src.arena.endgame_quality import TinyRiverGame, _world


class Uniform:
    description = {"fixture": "uniform-hu20"}
    def distribution(self, view):
        menu = choices(view, raise_cap=None, free_fold=False)
        return menu, (1/len(menu),)*len(menu), False


def river(seed=301, raise_to=None):
    hand = Hand.start(Table(('a', 'b'), (2000, 2000), button=0), hand_id='hu20', seed=seed)
    raised = False
    while hand.observe(hand.actor).street != Street.RIVER:
        view = hand.observe(hand.actor)
        if view.street == Street.FLOP and raise_to is not None and not raised:
            action = Action(ActionKind.RAISE, raise_to); raised = True
        else:
            action = Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds else ActionKind.CALL)
        hand = hand.apply(action)
    return hand


def small_ranges(root):
    from src.game.observation import replay
    board = replay(root, 0, ()).board
    pairs = tuple(combinations([c for c in DECK if c not in board][:6], 2))[:4]
    return {s: tuple((tuple(sorted(p)), float(i+1)) for i, p in enumerate(pairs)) for s in (0, 1)}


def tiny_player_world(monkeypatch):
    hand = river(); root = river_root_history(hand.events); ranges = small_ranges(root)
    # Choose compatible stipulated pairs; ranges never depend on actual holding.
    pairs = [p for p, _ in ranges[0]]
    # The first four pairs share a card; include independent pairs for this fixture.
    from src.game.observation import replay
    available = [c for c in DECK if c not in replay(root, 0, ()).board]
    rows = tuple((tuple(sorted(available[i:i+2])), 1.0) for i in (0, 2, 4, 6))
    ranges = {0: rows, 1: rows}
    monkeypatch.setattr('src.blueprint.hu20_river.public_ranges',
        lambda source, history, check: (ranges, {'fixture': True}))
    world = _world(root, hand.observe(0).board, {0: rows[0][0], 1: rows[1][0]})
    return world, root, ranges


def test_hu20_native_terminal_and_scalar_best_responses(monkeypatch):
    hand, root, ranges = tiny_player_world(monkeypatch)
    game = HU20RiverGame(root, ranges)
    import src.arena.endgame_quality as oracle
    monkeypatch.setattr(oracle, 'choices', lambda view, **kw: choices(view, free_fold=False, **kw))
    tiny = TinyRiverGame(root, ranges)
    assert len(tiny.nodes) == len(game.nodes)
    profile = tiny.uniform_profile()
    actual = profile_quality(game, profile); expected = tiny.quality(profile)
    assert actual['profile_values_bb'] == pytest.approx(expected['values_bb'], abs=1e-10)
    assert actual['best_response_gains_bb'] == pytest.approx(expected['individual_deviation_gains_bb'], abs=1e-10)
    for node in game.nodes:
        if node.terminal is None: continue
        for d, (i, j) in enumerate(tiny.deals):
            value = node.terminal.base + node.terminal.win0*(game.first_wins[i,j]>0) + node.terminal.tie0*(game.ties[i,j]>0)
            assert value == pytest.approx(tiny.payoffs[node.id][d,0], abs=1e-10)
    with pytest.raises(ValueError): RiverGame(root, ranges)


def test_public_likelihood_factors_and_exact_card_compatibility():
    root = river_root_history(river().events)
    ranges, coverage = public_ranges(Uniform(), root)
    assert coverage['holdings_per_seat'] == [1081, 1081]
    assert coverage['missing'] > 0
    game = HU20RiverGame(root, ranges)
    assert game.joint.sum() == pytest.approx(1)
    for i in (0, 45, 100):
        for j in (0, 45, 100):
            assert (game.joint[i,j] > 0) == (not set(game.holdings[0][i]) & set(game.holdings[1][j]))
    class Impossible(Uniform):
        def distribution(self, view):
            menu, p, trained = super().distribution(view)
            # Every candidate assigns zero to the actual preflop call.
            return menu, tuple(float(c.action.kind == ActionKind.FOLD) for c in menu), True
    with pytest.raises(ValueError, match='zero support'): public_ranges(Impossible(), root)


def test_profile_cache_exact_holdings_and_hidden_world_invariance(monkeypatch):
    hand, root, ranges = tiny_player_world(monkeypatch); source = Uniform()
    cache = RiverProfileCache(source); config = HU20RiverConfig(sweeps=8)
    view = hand.observe(hand.actor)
    p = HU20RiverPlayer(source, 5, config, cache)
    first = p.distribution(view)
    game, profile, lookup = p.solution
    own = game.holdings[game.seats.index(view.seat)].index(tuple(sorted(view.hole_cards)))
    assert first[1] == tuple(profile[lookup[public_identity(view.history)]][own])
    # One cached full-range profile serves another own holding and hidden deal.
    rows = [pair for pair, _ in ranges[0]]
    other = _world(root, view.board, {view.seat: rows[2], 1-view.seat: rows[3]})
    q = HU20RiverPlayer(source, 5, config, cache); q.distribution(other.observe(view.seat))
    assert q.solution is p.solution and cache.stats['solves'] == cache.stats['hits'] == 1
    hidden = _world(root, view.board, {view.seat: view.hole_cards, 1-view.seat: rows[3]})
    assert hidden.observe(view.seat) == view
    r = HU20RiverPlayer(source, 5, config, cache)
    assert r.choose_action(hidden.observe(view.seat)) == p.choose_action(view)
    # Different hand ID still shares the same public solve.
    from dataclasses import replace
    changed = replace(view, hand_id='other', history=(replace(view.history[0], hand_id='other'),)+view.history[1:])
    q.distribution(changed)
    assert cache.stats['solves'] == 1


def test_retained_play_and_later_exact_off_tree_resolve_freezes_all_rows(monkeypatch):
    hand, _, _ = tiny_player_world(monkeypatch); source = Uniform()
    player = HU20RiverPlayer(source, 1, HU20RiverConfig(sweeps=1))
    first = hand.observe(hand.actor); action = player.choose_action(first)
    assert action.kind == ActionKind.CHECK
    old_id = public_identity(first.history); old_policy = player.used[old_id][1].copy()
    hand = hand.apply(action); opponent = hand.observe(hand.actor)
    wager = opponent.legal_actions.min_raise_to + 1
    assert Action(ActionKind.RAISE, wager) not in [c.action for c in choices(opponent, raise_cap=2, free_fold=False)]
    hand = hand.apply(Action(ActionKind.RAISE, wager)); later = hand.observe(hand.actor)
    player.choose_action(later)
    assert len(player.records) == 2 and player.records[-1]['re_solve']
    game, profile, lookup = player.solution
    np.testing.assert_array_equal(profile[lookup[old_id]], old_policy)
    assert player.records[-1]['frozen_hero_nodes'] == 1
    assert any(c.action.raise_to == wager for n in game.nodes for c in n.menu)
    assert not any(r['river_delegation'] for r in player.records)


def test_on_tree_raise_retains_solution_without_second_solve(monkeypatch):
    hand, _, _ = tiny_player_world(monkeypatch); source = Uniform()
    p = HU20RiverPlayer(source, 1, HU20RiverConfig(sweeps=1))
    hand = hand.apply(p.choose_action(hand.observe(hand.actor)))
    hand = hand.apply(Action(ActionKind.RAISE, hand.observe(hand.actor).legal_actions.min_raise_to))
    p.choose_action(hand.observe(hand.actor))
    assert len(p.records) == 1 and p.cache.stats['solves'] == 1


def test_fixed_average_constraints_determinism_and_watchdog_failure(monkeypatch):
    hand, root, ranges = tiny_player_world(monkeypatch); game = HU20RiverGame(root, ranges)
    a = RiverCFR(game).solve(max_sweeps=10); b = RiverCFR(game).solve(max_sweeps=10)
    for key in a.average: np.testing.assert_array_equal(a.average[key], b.average[key])
    fixed = {0: a.average[0]}
    result = RiverCFR(game, fixed_profile=fixed).solve(max_sweeps=3)
    np.testing.assert_array_equal(result.average[0], fixed[0])
    fixed[0] = np.zeros_like(fixed[0])
    with pytest.raises(ValueError): RiverCFR(game, fixed_profile=fixed)
    p = HU20RiverPlayer(Uniform(), 1, HU20RiverConfig(sweeps=2, rss_limit_bytes=1))
    with pytest.raises(MemoryError): p.choose_action(hand.observe(hand.actor))
    assert p.records[-1]['status'] == 'failure' and not p.records[-1]['river_delegation']
