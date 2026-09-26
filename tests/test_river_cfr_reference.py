"""Independent native settlement and scalar CFR checks on river poker games."""

from itertools import product

import numpy as np
import pytest

from src.blueprint.river_cfr import RiverCFR, profile_quality
from src.blueprint.river_game import RiverGame, river_root_history
from src.blueprint.search import DECK
from src.game.hand import Hand, Table
from src.game.observation import ActionTaken, HandFinished
from src.game.types import Action, ActionKind, Street


def _river(seed=1234, stacks=(10_000,) * 6, survivors=2):
    hand = Hand.start(Table(tuple(f"player-{i}" for i in range(6)), stacks,
                            button=seed % 6), hand_id=f"river-{seed}", seed=seed)
    while hand.observe(hand.actor).street != Street.RIVER:
        view = hand.observe(hand.actor)
        live = sum(not player.folded for player in view.players)
        kind = (ActionKind.FOLD if view.street == Street.PREFLOP and live > survivors
                and ActionKind.FOLD in view.legal_actions.kinds else
                ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds
                else ActionKind.CALL)
        hand = hand.apply(Action(kind))
    return hand


def _ranges(view, size=4):
    remaining = [card for card in DECK if card not in view.board]
    seats = [player.seat for player in view.players if not player.folded]
    result = {}
    for position, seat in enumerate(seats):
        # Deliberate overlaps and a zero-mass holding exercise the joint law.
        pairs = [tuple(sorted((remaining[(position * 7 + 2 * index) % 31],
                               remaining[(position * 7 + 2 * index + 1) % 31])))
                 for index in range(size)]
        masses = [1, 2, 3, 0][:size]
        result[seat] = tuple((pair, float(mass)) for pair, mass in zip(pairs, masses))
    return result


def _native_world(root, board, holes):
    start = root[0]
    used = set(board)
    cards = {}
    for seat in range(len(start.stacks)):
        if seat in holes:
            cards[seat] = holes[seat]
            used.update(holes[seat])
    remaining = iter(card for card in DECK if card not in used)
    for seat in range(len(start.stacks)):
        if seat not in cards:
            cards[seat] = (next(remaining), next(remaining))
    order = tuple((start.button + offset) % len(start.stacks)
                  for offset in range(1, len(start.stacks) + 1))
    dealt = tuple(cards[seat][round_] for round_ in range(2) for seat in order)
    unused = tuple(card for card in DECK if card not in set(dealt + board))
    table = Table(start.player_ids, start.stacks, start.button, start.small_blind,
                  start.big_blind, start.chip_unit)
    hand = Hand.from_deck(table, hand_id=start.hand_id, deck=dealt + board + unused)
    for event in root:
        if isinstance(event, ActionTaken):
            hand = hand.apply(event.action)
    assert hand.events == root
    return hand


def _native_terminal_value(game, root, terminal_id, pair0, pair1):
    hand = _native_world(root, game.board,
                         {game.seats[0]: pair0, game.seats[1]: pair1})
    history = game.nodes[terminal_id].history
    for event in history[len(root):]:
        if isinstance(event, ActionTaken):
            hand = hand.apply(event.action)
    assert hand.finished
    finish = next(event for event in hand.events if isinstance(event, HandFinished))
    first = game.seats[0]
    return ((finish.stacks[first] - game.root_stacks[first] - game.root_pot / 2)
            / game.big_blind)


def _fixture(seed=1234, stacks=(10_000,) * 6):
    hand = _river(seed, stacks)
    view = hand.observe(hand.actor)
    root = river_root_history(view.history)
    return RiverGame(root, _ranges(view)), root


def test_joint_law_card_collisions_and_native_terminal_ledger():
    game, root = _fixture()
    assert game.joint.sum() == pytest.approx(1)
    assert np.count_nonzero(game.joint == 0) > 0
    for node in game.nodes:
        if node.terminal is None:
            continue
        for i, pair0 in enumerate(game.holdings[0]):
            for j, pair1 in enumerate(game.holdings[1]):
                if game.joint[i, j] == 0:
                    continue
                expected = _native_terminal_value(game, root, node.id, pair0, pair1)
                actual = (node.terminal.base
                          + node.terminal.win0 * (game.first_wins[i, j] > 0)
                          + node.terminal.tie0 * (game.ties[i, j] > 0))
                assert actual == pytest.approx(expected, abs=1e-12)


def test_one_sweep_agrees_with_independent_deal_enumeration():
    game, root = _fixture()
    solver = RiverCFR(game)
    solver.solve(max_sweeps=1)
    profile = {node.id: np.full_like(solver.regrets[node.id], 1 / len(node.menu))
               for node in game.nodes if node.actor is not None}
    for seat in game.seats:
        own = game.seats.index(seat)
        other = 1 - own
        accumulated = {node.id: np.zeros_like(profile[node.id])
                       for node in game.nodes if node.actor == seat}
        for own_id, other_id in product(range(len(game.holdings[own])),
                                        range(len(game.holdings[other]))):
            i, j = (own_id, other_id) if own == 0 else (other_id, own_id)
            chance = game.joint[i, j]
            if chance == 0:
                continue

            def visit(node_id, opponent_reach):
                node = game.nodes[node_id]
                if node.terminal is not None:
                    value = _native_terminal_value(
                        game, root, node_id, game.holdings[0][i], game.holdings[1][j],
                    )
                    return chance * opponent_reach * (value if own == 0 else -value)
                if node.actor == seat:
                    values = [visit(child, opponent_reach) for child in node.children]
                    policy = profile[node_id][own_id]
                    accumulated[node_id][own_id] += values
                    return sum(p * value for p, value in zip(policy, values))
                return sum(visit(child, opponent_reach * profile[node_id][other_id, action])
                           for action, child in enumerate(node.children))

            visit(0, 1.0)
        for node_id, action_values in accumulated.items():
            expected = action_values - np.sum(profile[node_id] * action_values,
                                              axis=1, keepdims=True)
            np.testing.assert_allclose(solver.regrets[node_id], expected, atol=1e-9,
                                       rtol=0)


def test_average_profile_improves_small_river_exploitability():
    game, _ = _fixture()
    solver = RiverCFR(game)
    first = solver.solve(max_sweeps=100)
    initial = profile_quality(game, first.average)["exploitability_root_pot"]
    final = solver.solve(max_sweeps=3900)
    average = profile_quality(game, final.average)
    current = profile_quality(game, final.current)
    # This is a development trend check. The proposed 1e-3 acceptance limit
    # still needs a frozen work budget and separate reference-fixture run.
    assert average["exploitability_root_pot"] < 0.01
    assert average["exploitability_root_pot"] < initial
    assert average["zero_sum_error_bb"] < 1e-10
    assert current["exploitability_bb"] >= 0
    assert final.completed_sweeps == 4000
