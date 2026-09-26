"""Independent native settlement and scalar CFR checks on river poker games."""

from dataclasses import replace
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from time import monotonic

import numpy as np
import pytest

from src.blueprint.river_cfr import RiverCFR, profile_quality
import src.blueprint.river_cfr as river_cfr_module
from src.blueprint.river_game import RiverGame, river_root_history
from src.blueprint.river_player import RiverCFRPlayer, RiverPlayerConfig
from src.blueprint.abstraction import choices
from src.blueprint.search import DECK, SearchConfig
from src.arena.endgame_quality import TinyRiverGame, fixture_hand
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


def test_linear_average_uses_own_reach_and_timeout_discards_partial_sweep():
    game, _ = _fixture()
    solver = RiverCFR(game)
    expected_numer = {key: np.zeros_like(value)
                      for key, value in solver.average_numer.items()}
    expected_denom = {key: np.zeros_like(value)
                      for key, value in solver.average_denom.items()}

    def reference_average(profile, weight):
        for seat in game.seats:
            def visit(node_id, reach):
                node = game.nodes[node_id]
                if node.actor is None:
                    return
                if node.actor == seat:
                    policy = profile[node_id]
                    expected_numer[node_id][:] += weight * reach[:, None] * policy
                    expected_denom[node_id][:] += weight * reach
                    for action, child in enumerate(node.children):
                        visit(child, reach * policy[:, action])
                else:
                    for child in node.children:
                        visit(child, reach)
            visit(0, np.ones(len(game.holdings[game.seats.index(seat)])))

    for weight in (1, 2):
        profile = {}
        for node in game.nodes:
            if node.actor is None:
                continue
            regrets = solver.regrets[node.id]
            positive = np.maximum(regrets, 0)
            total = positive.sum(axis=1, keepdims=True)
            profile[node.id] = np.divide(positive, total,
                                          out=np.full_like(positive, 1 / len(node.menu)),
                                          where=total > 0)
        reference_average(profile, weight)
        solver.solve(max_sweeps=1)
    for key in expected_numer:
        np.testing.assert_allclose(solver.average_numer[key], expected_numer[key], atol=1e-12)
        np.testing.assert_allclose(solver.average_denom[key], expected_denom[key], atol=1e-12)
    prior = {key: value.copy() for key, value in solver.regrets.items()}
    with pytest.raises(TimeoutError):
        fresh = RiverCFR(game)
        fresh.solve(max_sweeps=3, deadline=monotonic())
    result = solver.solve(max_sweeps=1, deadline=monotonic())
    assert result.completed_sweeps == 2
    for key in prior:
        np.testing.assert_array_equal(solver.regrets[key], prior[key])


def test_second_player_interruption_discards_both_players_staged_work(monkeypatch):
    game, _ = _fixture()
    solver = RiverCFR(game)
    before = solver.solve(max_sweeps=1)
    saved = {
        "regrets": {key: value.copy() for key, value in solver.regrets.items()},
        "numer": {key: value.copy() for key, value in solver.average_numer.items()},
        "denom": {key: value.copy() for key, value in solver.average_denom.items()},
        "visited": solver.visited_public_nodes,
        "zero_external": solver.zero_external_reach_entries,
    }
    original_visit = river_cfr_module._visit
    interrupted = []

    def interrupt_after_second_player_infoset(*args, **kwargs):
        value = original_visit(*args, **kwargs)
        node_id, traverser = args[1:3]
        if (not interrupted and traverser == game.seats[1]
                and game.nodes[node_id].actor == traverser
                and np.any(args[5][node_id])):
            assert any(np.any(delta) for key, delta in args[5].items()
                       if game.nodes[key].actor == game.seats[0])
            interrupted.append(node_id)
            raise TimeoutError("Deterministic second-player interruption")
        return value

    monkeypatch.setattr(river_cfr_module, "_visit", interrupt_after_second_player_infoset)
    after = solver.solve(max_sweeps=1)
    assert interrupted
    assert after.stop_reason == "Deterministic second-player interruption"
    assert after.completed_sweeps == before.completed_sweeps == 1
    assert solver.visited_public_nodes == saved["visited"]
    assert solver.zero_external_reach_entries == saved["zero_external"]
    for name, actual in (("regrets", solver.regrets),
                         ("numer", solver.average_numer),
                         ("denom", solver.average_denom)):
        for key, value in actual.items():
            np.testing.assert_array_equal(value, saved[name][key])
    for key, value in after.current.items():
        np.testing.assert_array_equal(value, before.current[key])


def test_reference_runner_retains_an_unmet_sweep_milestone(tmp_path, monkeypatch):
    from scripts import evaluate_river_quality as runner

    checkpoint = tmp_path / "unused-checkpoint"
    checkpoint.write_bytes(b"frozen-test-input")
    plan = {
        "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "reference_sweeps": [1, 2, 4],
        "max_wall_seconds": 600,
        "max_rss_gib": 10.5,
        "min_free_gib": 0,
    }
    original_solve = runner.RiverCFR.solve
    completed = []

    def partial_solve(self, **kwargs):
        if not completed:
            result = original_solve(self, **kwargs)
            completed.append(result)
            return result
        return replace(completed[0], stop_reason="Test deadline")

    monkeypatch.setattr(runner.RiverCFR, "solve", partial_solve)
    out = tmp_path / "partial-run"
    fixtures = Path(__file__).resolve().parents[1] / "configs/blueprint/river-reference-fixtures.json"
    result = runner.run(plan, fixtures, checkpoint, out)
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    assert result["status"] == "failed"
    assert len(rows) == 2
    case, failure = rows
    assert case["status"] == "incomplete"
    assert case["unmet_requested_sweeps"] == 2
    assert case["completed_sweeps"] == 1
    assert [(item["requested_sweeps"], item["completed_sweeps"],
             item["milestone_reached"], item["stop_reason"])
            for item in case["work_quality"]] == [
                (1, 1, True, "sweep_cap"), (2, 1, False, "Test deadline"),
            ]
    assert failure["phase"] == "reference" and failure["completed_sweeps"] == 1


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


def test_legal_all_in_showdown_uses_exact_native_settlement():
    game, root = _fixture(1234, (500,) * 6)
    assert any(node.terminal and node.terminal.win0 for node in game.nodes)
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


def test_odd_chip_board_tie_matches_native_button_order():
    board = ("As", "Ks", "Qs", "Js", "Ts")
    table = Table(tuple(f"odd-{seat}" for seat in range(6)), (100,) * 6,
                  button=0, small_blind=1, big_blind=2)
    available = [card for card in DECK if card not in board]
    hand = Hand.from_deck(table, hand_id="odd-river",
                          deck=tuple(available[:12]) + board + tuple(available[12:]))
    while hand.observe(hand.actor).street == Street.PREFLOP:
        view = hand.observe(hand.actor)
        live = sum(not player.folded for player in view.players)
        kind = (ActionKind.FOLD if live > 3 and ActionKind.FOLD in view.legal_actions.kinds
                else ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds
                else ActionKind.CALL)
        hand = hand.apply(Action(kind))
    flop_raised = False
    while hand.observe(hand.actor).street == Street.FLOP:
        view = hand.observe(hand.actor)
        if not flop_raised and ActionKind.RAISE in view.legal_actions.kinds:
            hand = hand.apply(Action(ActionKind.RAISE, 5))
            flop_raised = True
        else:
            hand = hand.apply(Action(ActionKind.CHECK if ActionKind.CHECK in
                                     view.legal_actions.kinds else ActionKind.CALL))
    assert flop_raised
    turn_raised = turn_folded = False
    while hand.observe(hand.actor).street == Street.TURN:
        view = hand.observe(hand.actor)
        if not turn_raised and ActionKind.RAISE in view.legal_actions.kinds:
            hand = hand.apply(Action(ActionKind.RAISE, view.legal_actions.min_raise_to))
            turn_raised = True
        elif turn_raised and not turn_folded and ActionKind.FOLD in view.legal_actions.kinds:
            hand = hand.apply(Action(ActionKind.FOLD))
            turn_folded = True
        else:
            hand = hand.apply(Action(ActionKind.CHECK if ActionKind.CHECK in
                                     view.legal_actions.kinds else ActionKind.CALL))
    view = hand.observe(hand.actor)
    assert view.pot % 2 == 1
    root = river_root_history(view.history)
    game = RiverGame(root, _ranges(view))
    for node in game.nodes:
        if node.terminal is None or node.terminal.tie0 == 0:
            continue
        for i, pair0 in enumerate(game.holdings[0]):
            for j, pair1 in enumerate(game.holdings[1]):
                if game.joint[i, j] == 0:
                    continue
                assert game.ties[i, j] > 0
                expected = _native_terminal_value(game, root, node.id, pair0, pair1)
                assert node.terminal.base + node.terminal.tie0 == pytest.approx(expected)


def test_observed_off_menu_river_raise_is_added_at_exact_size():
    hand = _river()
    root = river_root_history(hand.events)
    view = hand.observe(hand.actor)
    raise_to = view.legal_actions.min_raise_to + 1
    assert raise_to <= view.legal_actions.max_raise_to
    action = Action(ActionKind.RAISE, raise_to)
    assert action not in [choice.action for choice in choices(view)]
    hand = hand.apply(action)
    later = hand.observe(hand.actor)
    game = RiverGame(root, _ranges(later), observed_history=later.history)
    root_node = game.nodes[0]
    assert action in [choice.action for choice in root_node.menu]
    assert later.history in game.history_to_node


class _UniformBlueprint:
    def distribution(self, view):
        menu = choices(view)
        return menu, (1 / len(menu),) * len(menu), False


def test_play_adapter_retains_profile_and_delegates_unsupported_root():
    hand = _river()
    player = RiverCFRPlayer(
        _UniformBlueprint(), 35,
        RiverPlayerConfig(max_seconds=3, min_sweeps=1, max_sweeps=1),
    )
    view = hand.observe(hand.actor)
    action = player.choose_action(view)
    view.legal_actions.validate(action)
    assert player.attempts == player.completed == 1
    assert player.game.law_label == "active-marginals-ignore-folded-removal-v1"
    hand = hand.apply(action)
    if not hand.finished:
        opponent = hand.observe(hand.actor)
        if ActionKind.RAISE in opponent.legal_actions.kinds:
            raise_to = opponent.legal_actions.min_raise_to
            hand = hand.apply(Action(ActionKind.RAISE, raise_to))
            if not hand.finished and hand.actor == view.seat:
                later = hand.observe(hand.actor)
                second = player.choose_action(later)
                later.legal_actions.validate(second)
                assert player.attempts == 1
                assert player.delegations == 0
    unsupported = _river(1235, survivors=3)
    view = unsupported.observe(unsupported.actor)
    other = RiverCFRPlayer(
        _UniformBlueprint(), 40,
        fallback_config=SearchConfig(max_seconds=0.01, worlds=2,
                                     variant="corrected"),
    )
    action = other.choose_action(view)
    view.legal_actions.validate(action)
    assert other.attempts == 0


def test_play_adapter_delegates_after_later_off_menu_raise():
    hand = _river()
    player = RiverCFRPlayer(
        _UniformBlueprint(), 35,
        RiverPlayerConfig(max_seconds=3, min_sweeps=1, max_sweeps=1),
        SearchConfig(max_seconds=0.01, worlds=2, variant="corrected"),
    )
    first = hand.observe(hand.actor)
    action = player.choose_action(first)
    assert action.kind == ActionKind.RAISE
    hand = hand.apply(action)
    opponent = hand.observe(hand.actor)
    raise_to = opponent.legal_actions.min_raise_to + 1
    off_tree = Action(ActionKind.RAISE, raise_to)
    assert off_tree not in [choice.action for choice in choices(opponent)]
    hand = hand.apply(off_tree)
    later = hand.observe(hand.actor)
    assert later.seat == first.seat
    decision = player.choose_action(later)
    later.legal_actions.validate(decision)
    assert player.attempts == 1
    assert player.delegations == 1
    assert player.records[-1]["status"] == "off_tree_delegate"


def test_play_adapter_cannot_distinguish_unseen_opponent_hands():
    hand = _river()
    root = river_root_history(hand.events)
    first = hand.observe(hand.actor)
    hero = tuple(first.hole_cards)
    available = [card for card in DECK if card not in first.board and card not in hero]
    opponents = [(available[0], available[1]), (available[2], available[3])]
    other_seat = next(player.seat for player in first.players
                      if not player.folded and player.seat != first.seat)
    worlds = [_native_world(root, first.board,
                            {first.seat: hero, other_seat: pair})
              for pair in opponents]
    assert worlds[0].observe(first.seat) == worlds[1].observe(first.seat)
    players = [RiverCFRPlayer(_UniformBlueprint(), 734,
                              RiverPlayerConfig(max_seconds=3, min_sweeps=1,
                                                max_sweeps=1)) for _ in worlds]
    actions = [player.choose_action(world.observe(first.seat))
               for player, world in zip(players, worlds)]
    assert actions[0] == actions[1]
    for node_id in players[0].profile:
        np.testing.assert_array_equal(players[0].profile[node_id],
                                      players[1].profile[node_id])


_CASES = json.loads((Path(__file__).resolve().parents[1]
                    / "configs/blueprint/river-reference-fixtures.json").read_text())["cases"]
_DEVELOPMENT_CASES = json.loads((Path(__file__).resolve().parents[1]
                                / "configs/blueprint/river-development-m4.json").read_text())[
                                    "full_range_cases"]
_AMENDMENT_CASES = json.loads((Path(__file__).resolve().parents[1]
                              / "configs/blueprint/river-range-amendment-m4.json").read_text())[
                                  "full_range_cases"]


@pytest.mark.parametrize("case,pot_bb", zip(_DEVELOPMENT_CASES, (2, 4, 8, 12, 32)),
                         ids=[case["id"] for case in _DEVELOPMENT_CASES])
def test_frozen_development_roots_are_legal_and_distinct(case, pot_bb):
    from scripts.evaluate_river_development import _shape_ranges

    hand = fixture_hand(case)
    view = hand.observe(hand.actor)
    assert view.pot / view.big_blind == pot_bb
    assert list(view.board) == case["board"]
    ranges = _shape_ranges(_ranges(view), case["range_shape"])
    game = RiverGame(river_root_history(view.history), ranges)
    assert game.root_pot == view.pot
    assert len(game.nodes) > 1
    assert game.joint.sum() == pytest.approx(1)


@pytest.mark.parametrize("case,pot_bb", zip(_AMENDMENT_CASES, (2, 12, 32)),
                         ids=[case["id"] for case in _AMENDMENT_CASES])
def test_frozen_range_amendment_is_legal_and_nonuniform(case, pot_bb):
    from itertools import combinations
    from scripts.evaluate_river_development import _shape_ranges, _range_summary

    hand = fixture_hand(case)
    view = hand.observe(hand.actor)
    assert view.pot / view.big_blind == pot_bb
    root = river_root_history(view.history)
    RiverGame(root, _ranges(view))
    pairs = tuple(combinations((card for card in DECK if card not in view.board), 2))
    seat = next(player.seat for player in view.players if not player.folded)
    shaped = _shape_ranges({seat: tuple((pair, 1.0) for pair in pairs)},
                           case["range_shape"])
    assert sum(weight for _, weight in shaped[seat]) == pytest.approx(1)
    assert _range_summary(shaped)[str(seat)]["effective_holdings"] < len(pairs)


@pytest.mark.parametrize("case", _CASES, ids=[case["id"] for case in _CASES])
def test_declared_river_fixtures_have_independent_quality_oracle(case):
    hand = fixture_hand(case)
    view = hand.observe(hand.actor)
    assert list(view.board) == case["board"]
    root = river_root_history(view.history)
    ranges = _ranges(view)
    tiny = TinyRiverGame(root, ranges)
    quality = tiny.quality(tiny.uniform_profile())
    assert len(tiny.seats) == case["survivors"]
    assert abs(sum(quality["values_bb"])) < 1e-9
    assert all(gain >= 0 for gain in quality["individual_deviation_gains_bb"])
    assert quality["nash_conv_bb"] == pytest.approx(sum(
        quality["individual_deviation_gains_bb"]), abs=1e-12)
    if case["survivors"] == 2:
        game = RiverGame(root, ranges)
        assert len(game.nodes) == len(tiny.nodes)
        for node, reference in zip(game.nodes, tiny.nodes):
            assert node.actor == reference.actor
            if node.actor is None:
                for deal_index, ids in enumerate(tiny.deals):
                    i, j = ids
                    expected = tiny.payoffs[node.id][deal_index, 0]
                    actual = (node.terminal.base
                              + node.terminal.win0 * (game.first_wins[i, j] > 0)
                              + node.terminal.tie0 * (game.ties[i, j] > 0))
                    assert actual == pytest.approx(expected, abs=1e-9)
        batched = profile_quality(game, tiny.uniform_profile())
        assert batched["profile_values_bb"] == pytest.approx(quality["values_bb"], abs=1e-9)
        assert batched["best_response_gains_bb"] == pytest.approx(
            quality["individual_deviation_gains_bb"], abs=1e-9)
