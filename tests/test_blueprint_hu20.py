"""The versioned short-stack game keeps legacy artifacts and menus intact."""

import gzip
import json

import pytest

from src.blueprint.abstraction import HU20_SCHEMA, choices, information_key
from src.arena.catalog import Checkpoint
from src.blueprint.artifact import (HU20_FORMAT, FrozenBlueprint, export_policy,
                                    load_training, save_training)
from src.blueprint.lookup import TableDistribution
from src.blueprint.solver import BlueprintTrainer, HU20_GAME, PilotConfig
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind, Street

DECK = tuple(rank + suit for suit in "cdhs" for rank in "23456789TJQKA")


def table(button=0):
    ids = ("hero", "villain") if button == 0 else ("villain", "hero")
    return Table(ids, (2000, 2000), button=button)


def config(seed=7):
    return PilotConfig(seed=seed, abstraction=HU20_SCHEMA, game=HU20_GAME,
                       max_nodes=10000, max_entries=100000, max_seconds=30)


def coupled(button, villain_cards=("Kc", "Kd")):
    holes = {"hero": ("Ac", "Ad"), "villain": villain_cards}
    board = ("2c", "3d", "4h", "5s", "9c")
    order = tuple((button + offset + 1) % 2 for offset in range(2))
    ids = table(button).player_ids
    dealt = tuple(holes[ids[seat]][round_] for round_ in range(2) for seat in order)
    rest = tuple(card for card in DECK if card not in set(dealt + board))
    return Hand.from_deck(table(button), hand_id="same-public-game", deck=dealt + board + rest)


def test_native_headsup_blinds_and_action_order():
    for button in (0, 1):
        hand = Hand.start(table(button), hand_id=f"hu-{button}", seed=31)
        view = hand.observe(hand.actor)
        assert view.player_id == "hero"
        assert hand.actor == button
        assert view.players[button].street_bet == 50
        assert view.players[1-button].street_bet == 100
        hand = hand.apply(Action(ActionKind.CALL))
        assert hand.actor == 1-button
        hand = hand.apply(Action(ActionKind.CHECK))
        assert hand.observe(hand.actor).street == Street.FLOP
        assert hand.actor == 1-button


def test_training_menu_removes_only_free_fold(monkeypatch):
    trainer = BlueprintTrainer(table(), config())
    seen = []
    import src.blueprint.solver as solver
    original = solver.choices

    def checked(view, **kwargs):
        menu = original(view, **kwargs)
        if ActionKind.CHECK in view.legal_actions.kinds:
            seen.append(menu)
            assert all(item.action.kind != ActionKind.FOLD for item in menu)
        return menu

    monkeypatch.setattr(solver, "choices", checked)
    first_report = trainer.step()
    assert seen
    assert sum(first_report.new_entries_by_street.values()) == first_report.new_entries
    assert sum(first_report.revisited_keys_by_street.values()) == (
        first_report.contributing_infosets-first_report.new_entries)
    hand = Hand.start(table(), hand_id="legacy-menu", seed=31)
    while hand.observe(hand.actor).street == Street.PREFLOP:
        view = hand.observe(hand.actor)
        hand = hand.apply(Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds
                                 else ActionKind.CALL))
    view = hand.observe(hand.actor)
    assert any(item.action.kind == ActionKind.FOLD for item in choices(view))
    assert all(item.action.kind != ActionKind.FOLD for item in choices(view, free_fold=False))


def test_coupled_rotations_share_keys_and_ignore_unseen_hands():
    hands = [coupled(button) for button in (0, 1)]
    for _ in range(12):
        views = [hand.observe(hand.actor) for hand in hands]
        assert views[0].player_id == views[1].player_id
        keys = [information_key(view, choices(view, free_fold=False), schema=HU20_SCHEMA)
                for view in views]
        assert keys[0] == keys[1]
        actions = []
        for view in views:
            menu = choices(view, free_fold=False)
            actions.append(next((item.action for item in menu if item.action.kind == ActionKind.CALL),
                                next((item.action for item in menu if item.action.kind == ActionKind.CHECK),
                                     menu[0].action)))
        assert actions[0] == actions[1]
        hands = [hand.apply(action) for hand, action in zip(hands, actions)]
        if hands[0].finished:
            break

    first = coupled(0).observe(0)
    unseen_changed = coupled(0, ("Qc", "Qd")).observe(0)
    assert first == unseen_changed


@pytest.mark.parametrize("second_action", ["fold", "call"])
def test_rotated_off_menu_raise_fold_and_all_in_histories(second_action):
    hands = [coupled(button) for button in (0, 1)]
    for step in range(5 if second_action == "call" else 2):
        views = [hand.observe(hand.actor) for hand in hands]
        assert views[0].player_id == views[1].player_id
        assert information_key(views[0], choices(views[0], free_fold=False), schema=HU20_SCHEMA) == (
            information_key(views[1], choices(views[1], free_fold=False), schema=HU20_SCHEMA))
        if step == 0:
            action = Action(ActionKind.RAISE, 275)
            assert action not in tuple(item.action for item in choices(views[0], free_fold=False))
        elif step == 1:
            action = Action(ActionKind.FOLD if second_action == "fold" else ActionKind.CALL)
        elif step == 2:
            action = Action(ActionKind.CHECK)
        elif step == 3:
            action = Action(ActionKind.RAISE, views[0].legal_actions.max_raise_to)
        else:
            action = Action(ActionKind.CALL)
        for view in views:
            view.legal_actions.validate(action)
        hands = [hand.apply(action) for hand in hands]
    assert all(hand.finished for hand in hands)
    assert len({next(p.stack for p in hand.observe(0).players if p.player_id == "hero")
                for hand in hands}) == 1


def test_new_artifact_identity_and_resume(tmp_path):
    trainer = BlueprintTrainer(table(), config())
    trainer.step()
    path = tmp_path / "new.json.gz"
    save_training(trainer, path)
    resumed = load_training(path)
    assert resumed.config == trainer.config
    trainer.step()
    resumed.step()
    assert save_training(trainer, tmp_path / "a.gz") == save_training(resumed, tmp_path / "b.gz")
    export_path = tmp_path / "policy.gz"
    digest = export_policy(trainer, export_path)
    policy = FrozenBlueprint(Checkpoint("hu20", str(export_path), digest, HU20_FORMAT), export_path)
    view = Hand.start(table(), hand_id="policy", seed=17).observe(0)
    menu, probabilities, _ = policy.distribution(view)
    assert len(menu) == len(probabilities) and sum(probabilities) == pytest.approx(1)
    assert TableDistribution(trainer).distribution(view)[0] == menu
    with pytest.raises(ValueError, match="format differs"):
        FrozenBlueprint(Checkpoint("wrong", str(export_path), digest, "holdem-blueprint-v1"), export_path)
    with pytest.raises(ValueError, match="20BB table"):
        BlueprintTrainer(Table(("hero", "villain"), (10000, 10000)), config())
    with pytest.raises(ValueError, match="Invalid bounded"):
        PilotConfig(abstraction=HU20_SCHEMA)
    lines = gzip.decompress(path.read_bytes()).splitlines()
    header = json.loads(lines[0])
    header["identity"]["action_menu"] = "incompatible-menu"
    altered = tmp_path / "altered.gz"
    altered.write_bytes(gzip.compress(json.dumps(header).encode()+b"\n"+
                                      b"\n".join(lines[1:])+b"\n", mtime=0))
    with pytest.raises(ValueError, match="abstraction schema"):
        load_training(altered)
