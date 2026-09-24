"""Search samples only compatible hidden worlds and returns legal actions."""

from random import Random
from time import monotonic

import pytest

from scripts.evaluate_blueprint_search import run
from src.blueprint.abstraction import choices
from src.blueprint.artifact import save_training
from src.blueprint.search import SearchConfig, SearchPlayer, _sample_world, public_ranges
from src.blueprint.solver import BlueprintTrainer, PilotConfig
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind, Street

DECK = tuple(rank + suit for suit in "cdhs" for rank in "23456789TJQKA")


class UniformBlueprint:
    def distribution(self, view):
        menu = choices(view)
        return menu, (1 / len(menu),) * len(menu), False


class CardSensitiveBlueprint:
    def distribution(self, view):
        menu = choices(view)
        if view.street != Street.FLOP:
            return menu, (1 / len(menu),) * len(menu), True
        raises = [item for item in menu if item.action.kind == ActionKind.RAISE]
        raise_mass = 0.9 if "As" in view.hole_cards else 0.1
        other = len(menu) - len(raises)
        weights = tuple(
            raise_mass / len(raises) if item in raises else (1 - raise_mass) / other
            for item in menu
        ) if raises else (1 / len(menu),) * len(menu)
        return menu, weights, True


def _postflop_hand(deck=DECK):
    table = Table(("first", "second"), (10_000, 10_000))
    hand = Hand.from_deck(table, hand_id="search-test", deck=deck)
    while hand.observe(hand.actor).street == Street.PREFLOP:
        view = hand.observe(hand.actor)
        kind = ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds else ActionKind.CALL
        hand = hand.apply(Action(kind))
    assert hand.observe(hand.actor).street == Street.FLOP
    return hand


def _facing_off_menu_bet(deck=DECK):
    hand = _postflop_hand(deck)
    view = hand.observe(hand.actor)
    legal = view.legal_actions
    target = legal.min_raise_to + 37
    assert target < legal.max_raise_to
    assert target not in {item.action.raise_to for item in choices(view) if item.action.kind == ActionKind.RAISE}
    return hand.apply(Action(ActionKind.RAISE, target))


def test_ranges_exclude_visible_cards_and_world_replays_actual_off_menu_bet():
    view = _facing_off_menu_bet().observe(0)
    assert view.actor == view.seat
    ranges = public_ranges(UniformBlueprint(), view, Random(3), 24, monotonic() + 5)
    known = set(view.hole_cards + view.board)
    assert set(ranges) == {1}
    assert abs(sum(weight for _, weight in ranges[1]) - 1) < 1e-12
    assert all(not known.intersection(pair) for pair, _ in ranges[1])
    for seed in range(10):
        world = _sample_world(view, ranges, Random(seed))
        assert world.events == view.history
        assert world.observe(view.seat).hole_cards == view.hole_cards
        assert world.observe(view.seat).board == view.board


def test_public_action_updates_private_hand_belief():
    view = _facing_off_menu_bet().observe(0)
    ranges = public_ranges(CardSensitiveBlueprint(), view, Random(3), 1326, monotonic() + 5)
    posterior = sum(weight for pair, weight in ranges[1] if "As" in pair)
    prior = sum("As" in pair for pair, _ in ranges[1]) / len(ranges[1])
    assert posterior > 2 * prior


def test_six_player_worlds_keep_all_private_cards_disjoint():
    table = Table(tuple(f"player-{i}" for i in range(6)), (10_000,) * 6)
    hand = Hand.from_deck(table, hand_id="six-search", deck=DECK)
    while hand.observe(hand.actor).street == Street.PREFLOP:
        current = hand.observe(hand.actor)
        kind = ActionKind.CHECK if ActionKind.CHECK in current.legal_actions.kinds else ActionKind.CALL
        hand = hand.apply(Action(kind))
    current = hand.observe(hand.actor)
    hand = hand.apply(Action(ActionKind.RAISE, current.legal_actions.min_raise_to + 37))
    view = hand.observe(hand.actor)
    ranges = public_ranges(UniformBlueprint(), view, Random(41), 48, monotonic() + 5)
    for seed in range(6):
        world = _sample_world(view, ranges, Random(seed))
        holdings = [card for player in world._state.players_state for card in player.hand]
        assert len(set(holdings)) == 12
        assert world.events == view.history


def test_search_is_legal_and_hidden_world_independent():
    first = _facing_off_menu_bet().observe(0)
    alternate = list(DECK)
    # Seat 0 keeps the same hole cards, and the flop remains unchanged.
    alternate[0], alternate[10] = alternate[10], alternate[0]
    second = _facing_off_menu_bet(tuple(alternate)).observe(0)
    assert first == second
    config = SearchConfig(max_seconds=5, worlds=1, range_samples=8, styles=("blueprint",))
    left = SearchPlayer(UniformBlueprint(), 11, config)
    right = SearchPlayer(UniformBlueprint(), 11, config)
    action = left.choose_action(first)
    first.legal_actions.validate(action)
    assert action == right.choose_action(second)
    assert left.completed == right.completed == 1


@pytest.mark.parametrize("street", [Street.TURN, Street.RIVER])
def test_search_completes_on_later_streets(street):
    hand = _postflop_hand()
    while hand.observe(hand.actor).street != street:
        view = hand.observe(hand.actor)
        kind = ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds else ActionKind.CALL
        hand = hand.apply(Action(kind))
    view = hand.observe(hand.actor)
    player = SearchPlayer(
        UniformBlueprint(), 59,
        SearchConfig(max_seconds=5, worlds=1, range_samples=8, styles=("blueprint",)),
    )
    view.legal_actions.validate(player.choose_action(view))
    assert player.completed == 1
    assert player.by_street[(street.value, "completed")] == 1


def test_time_limit_uses_unchanged_blueprint_action():
    view = _facing_off_menu_bet().observe(0)
    search = SearchPlayer(UniformBlueprint(), 12, SearchConfig(max_seconds=1e-9))
    menu, weights, _ = UniformBlueprint().distribution(view)
    expected = Random(12).choices(menu, weights=weights, k=1)[0].action
    assert search.choose_action(view) == expected
    assert search.fallbacks == 1


def test_paired_search_comparison_loads_one_pinned_checkpoint(tmp_path):
    table = Table(tuple(f"player-{i}" for i in range(6)), (10_000,) * 6)
    checkpoint = tmp_path / "checkpoint.json.gz"
    digest = save_training(BlueprintTrainer(table, PilotConfig()), checkpoint)
    config = {
        "search": {"max_seconds": 0.2, "worlds": 1, "range_samples": 8,
                   "styles": ["blueprint"]},
        "execution": {"max_wall_seconds": 30, "max_rss_gib": 4},
        "comparisons": {
            "random": {
                "scenarios": [{"name": "six", "stacks": [10_000] * 6}],
                "candidate": "blueprint_search", "baseline": "blueprint_live",
                "opponents": ["check_call"], "blocks": 1,
                "root_seed": 918, "split": "validation",
            }
        },
    }
    with pytest.raises(ValueError, match="hash mismatch"):
        run(checkpoint, "0" * 64, config, tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()
    result = run(checkpoint, digest, config, tmp_path / "comparison")
    assert result["status"] == "valid"
    report = result["comparisons"]["random"]
    assert report["report"]["completed_hands"] == 12
    assert report["telemetry"]["search_attempts"] > 0
    assert report["telemetry"]["search_completed"] > 0
