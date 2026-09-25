"""Local CFR keeps private information private and updates regrets correctly."""

from collections import Counter
from itertools import combinations, permutations
from random import Random
from time import monotonic

import pytest

from scripts.evaluate_blueprint_search import run
from src.blueprint.abstraction import choices
from src.blueprint.artifact import save_training
from src.blueprint.local_cfr import (
    LocalCFRConfig, LocalCFRPlayer, _LocalSolver, _eligible, _flop_root,
    _external_sampling_cycle, _publish, _record_delta, _root_ranges,
    _sample_holes, _world,
)
from src.blueprint.search import DECK
from src.blueprint.solver import BlueprintTrainer, PilotConfig
from src.game.hand import Hand, Table
from src.game.observation import ActionTaken, replay
from src.game.types import Action, ActionKind, Street


class UniformBlueprint:
    def distribution(self, view):
        menu = choices(view)
        return menu, (1 / len(menu),) * len(menu), False


def _three_way_flop(deck=DECK):
    table = Table(tuple(f"player-{seat}" for seat in range(6)), (10_000,) * 6)
    hand = Hand.from_deck(table, hand_id="local-cfr-test", deck=deck)
    while hand.observe(hand.actor).street == Street.PREFLOP:
        view = hand.observe(hand.actor)
        live = sum(not player.folded for player in view.players)
        action = (
            Action(ActionKind.FOLD)
            if live > 3 and ActionKind.FOLD in view.legal_actions.kinds
            else Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds else ActionKind.CALL)
        )
        hand = hand.apply(action)
    assert _eligible(hand.observe(hand.actor))
    return hand


def test_linear_regret_update_matches_enumerated_three_player_hidden_game():
    # Three distinct private ranks are dealt to three seats. The two opponents
    # call with the highest rank and otherwise fold. Enumerate every deal,
    # independently of the regret-update implementation.
    values = []
    for action in ("check", "bet"):
        outcomes = []
        for hero, left, right in permutations((0, 1, 2)):
            if hero != 1:
                continue
            if action == "check":
                payoff = 2 if hero > max(left, right) else -1
            else:
                callers = sum(card == 2 for card in (left, right))
                payoff = 2 if callers == 0 else -2 * callers
            outcomes.append(payoff)
        values.append(sum(outcomes) / len(outcomes))
    assert values == [-1, -2]
    nodes = {}
    deltas = {}
    policy = (0.25, 0.75)
    expected = sum(p * value for p, value in zip(policy, values))
    assert _record_delta(deltas, ("hero", 1), ("check", "bet"), policy,
                         tuple(values), 3, 0.5) == pytest.approx(expected)
    _publish(nodes, deltas)
    node = nodes[("hero", 1)]
    assert node.regrets == pytest.approx([3 * (value - expected) for value in values])
    assert node.average == pytest.approx([3 * 0.5 * p for p in policy])
    assert node.policy() == (1.0, 0.0)


def test_shared_external_sampling_core_converges_in_three_player_private_game():
    # Three private ranks are dealt without replacement. Each seat chooses a
    # hidden bit; matching its private rank is a strict best response even
    # though the neighboring seats' hidden choices also affect the payoff.
    # This runs the production cycle, delta publication and regret matching,
    # and checks the resulting strategy against the enumerated pure solution.
    nodes = {}
    random = Random(907)

    def visit(world, traverser, deltas, weight, seat=0, actions=(), own_reach=1.0):
        if seat == 3:
            mine = actions[traverser]
            correct = world[traverser] % 2
            neighbors = (actions[(traverser + 1) % 3], actions[(traverser - 1) % 3])
            return (2 if mine == correct else 0) + sum(
                0.25 if mine == other else -0.25 for other in neighbors
            )
        key = (seat, world[seat])  # No opponent rank or hidden choice.
        node = nodes.get(key)
        policy = node.policy() if node is not None else (0.5, 0.5)
        if seat != traverser:
            chosen = random.choices((0, 1), weights=policy, k=1)[0]
            return visit(world, traverser, deltas, weight, seat + 1,
                         actions + (chosen,), own_reach)
        values = tuple(
            visit(world, traverser, deltas, weight, seat + 1,
                  actions + (choice,), own_reach * policy[choice])
            for choice in (0, 1)
        )
        return _record_delta(deltas, key, ("0", "1"), policy, values,
                             weight, own_reach)

    deals = tuple(permutations((0, 1, 2)))
    for cycle in range(1, 1001):
        deltas = _external_sampling_cycle(
            (0, 1, 2), cycle, lambda _: random.choice(deals), visit,
        )
        _publish(nodes, deltas)
    assert set(nodes) == set((seat, rank) for seat in range(3) for rank in range(3))
    for (seat, rank), node in nodes.items():
        assert node.average_policy()[rank % 2] > 0.98, (seat, rank, node)


def test_eligibility_requires_three_players_at_flop_root():
    table = Table(tuple(f"player-{seat}" for seat in range(6)), (10_000,) * 6)
    hand = Hand.from_deck(table, hand_id="four-way-root", deck=DECK)
    while hand.observe(hand.actor).street == Street.PREFLOP:
        view = hand.observe(hand.actor)
        live = sum(not player.folded for player in view.players)
        kind = (ActionKind.FOLD if live > 4 and ActionKind.FOLD in view.legal_actions.kinds
                else ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds
                else ActionKind.CALL)
        hand = hand.apply(Action(kind))
    root = hand.observe(hand.actor)
    assert sum(not player.folded for player in root.players) == 4
    hand = hand.apply(Action(ActionKind.FOLD))
    current = hand.observe(hand.actor)
    assert sum(not player.folded for player in current.players) == 3
    assert not _eligible(current)


def test_flop_root_has_all_six_public_ranges_and_collision_free_worlds():
    view = _three_way_flop().observe(1)
    root_events, _ = _flop_root(view)
    ranges = _root_ranges(
        UniformBlueprint(), view, root_events, Random(17), 24,
        monotonic() + 5, Counter(),
    )
    assert set(ranges) == set(range(6))
    assert len(ranges[view.seat]) == len(tuple(combinations(
        (card for card in DECK if card not in view.board), 2,
    )))
    assert any(set(pair) == set(view.hole_cards) for pair, _ in ranges[view.seat])
    for seat in ranges:
        assert sum(weight for _, weight in ranges[seat]) == pytest.approx(1)
        assert all(not set(pair).intersection(view.board) for pair, _ in ranges[seat])
    for seed in range(10):
        random = Random(seed)
        holes = _sample_holes(ranges, random, (view.seat, view.hole_cards))
        world = _world(view, root_events, holes, random)
        assert world.events == root_events
        assert world.observe(view.seat).hole_cards == view.hole_cards
        assert len(set(card for pair in holes.values() for card in pair)) == 12


def test_observed_off_menu_raise_is_available_at_its_exact_size():
    hand = _three_way_flop()
    prior = hand.observe(hand.actor)
    target = prior.legal_actions.min_raise_to + 37
    assert target < prior.legal_actions.max_raise_to
    assert all(item.action != Action(ActionKind.RAISE, target) for item in choices(prior))
    hand = hand.apply(Action(ActionKind.RAISE, target))
    view = hand.observe(hand.actor)
    solver = _LocalSolver(
        UniformBlueprint(), view, Random(3), LocalCFRConfig(range_samples=8),
        monotonic() + 5, Counter(),
    )
    root = _world(
        view, solver.root_events,
        _sample_holes(solver.root_ranges, Random(5), (view.seat, view.hole_cards)),
        Random(6),
    )
    first = root.observe(root.actor)
    assert Action(ActionKind.RAISE, target) in [item.action for item in solver._menu(first)]
    assert any(isinstance(event, ActionTaken) and event.action.raise_to == target
               for event in view.history)


def test_leaf_choice_ignores_hole_card_deal_order():
    view = _three_way_flop().observe(1)
    solver = _LocalSolver(
        UniformBlueprint(), view, Random(3), LocalCFRConfig(range_samples=8),
        monotonic() + 5, Counter(),
    )
    alternate = replay(view.history, view.seat, view.hole_cards[::-1])
    assert solver._leaf_key(view) == solver._leaf_key(alternate)
    menu = solver._menu(view)
    assert solver._action_key(view, menu) == solver._action_key(alternate, menu)
    other_pair = next(pair for pair, _ in solver.root_ranges[view.seat]
                      if set(pair) != set(view.hole_cards))
    other = replay(view.history, view.seat, other_pair)
    assert solver._action_key(view, menu) != solver._action_key(other, menu)


def test_decision_after_second_flop_raise_delegates_to_rollout_search():
    hand = _three_way_flop()
    for _ in range(2):
        view = hand.observe(hand.actor)
        hand = hand.apply(Action(ActionKind.RAISE, view.legal_actions.min_raise_to))
    view = hand.observe(hand.actor)
    assert view.street == Street.FLOP
    assert not _eligible(view)
    player = LocalCFRPlayer(UniformBlueprint(), 7)
    view.legal_actions.validate(player.choose_action(view))
    assert player.attempts == 0
    assert player.other.attempts == 1


def test_local_player_is_legal_and_independent_of_unseen_deal():
    first = _three_way_flop().observe(1)
    alternate = list(DECK)
    alternate[1], alternate[2] = alternate[2], alternate[1]
    second = _three_way_flop(tuple(alternate)).observe(1)
    assert first == second
    config = LocalCFRConfig(max_seconds=5, range_samples=8, min_cycles=2, max_cycles=2)
    left = LocalCFRPlayer(UniformBlueprint(), 23, config)
    right = LocalCFRPlayer(UniformBlueprint(), 23, config)
    action = left.choose_action(first)
    first.legal_actions.validate(action)
    assert action == right.choose_action(second)
    assert left.completed == right.completed == 1
    assert left.cycles == right.cycles == [2]
    assert left.leaf_choices > 0


def test_timeout_uses_seeded_blueprint_fallback():
    view = _three_way_flop().observe(1)
    player = LocalCFRPlayer(
        UniformBlueprint(), 53,
        LocalCFRConfig(max_seconds=1e-9, range_samples=8),
    )
    menu, probabilities, _ = UniformBlueprint().distribution(view)
    expected = Random(53).choices(menu, weights=probabilities, k=1)[0].action
    assert player.choose_action(view) == expected
    assert player.attempts == player.fallbacks == 1
    assert player.completed == 0
    assert player.attempt_records[0]["reason"].startswith("TimeoutError:")


def test_paired_runner_accepts_local_candidate_and_corrected_baseline(tmp_path):
    table = Table(tuple(f"player-{seat}" for seat in range(6)), (10_000,) * 6)
    checkpoint = tmp_path / "checkpoint.json.gz"
    digest = save_training(BlueprintTrainer(table, PilotConfig()), checkpoint)
    config = {
        "search": {"max_seconds": 0.2, "worlds": 1, "range_samples": 8},
        "local_cfr": {"max_seconds": 0.2, "range_samples": 8,
                      "min_cycles": 1, "max_cycles": 1},
        "execution": {"max_wall_seconds": 30, "max_rss_gib": 4,
                      "min_free_gib": 0.01},
        "comparisons": {"styles": {
            "scenarios": [{"name": "six", "stacks": [10_000] * 6}],
            "candidate": "blueprint_local_cfr", "baseline": "blueprint_search",
            "opponents": ["tight_passive"], "blocks": 1,
            "root_seed": 2026092601, "split": "validation",
        }},
    }
    result = run(checkpoint, digest, config, tmp_path / "pilot", tensorboard=True)
    assert result["status"] == "valid"
    telemetry = result["comparisons"]["styles"]["telemetry"]
    assert telemetry["baseline"]["search_attempts"] > 0
    assert telemetry["candidate"]["delegated_search_attempts"] > 0
    assert telemetry["candidate"]["search_attempts"] >= 0
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = next((tmp_path / "pilot" / "tensorboard").glob("events.out.tfevents.*"))
    assert "styles/six/paired_bb_per_100" in EventAccumulator(str(events)).Reload().Tags()["scalars"]
