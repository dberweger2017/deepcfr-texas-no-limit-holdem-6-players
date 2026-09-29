"""Three-seat identity, coupled histories, settlement and atomic training."""

import gzip
import json
from dataclasses import replace
from time import time
from pathlib import Path

import pytest

from scripts.evaluate_tp20 import UniformTP20, run
from scripts.tp20_common import density, schedule
from src.arena.catalog import Checkpoint
from src.blueprint.abstraction import TP20_SCHEMA, choices, information_key
from src.blueprint.artifact import (FrozenBlueprint, TP20_FORMAT, export_policy,
                                    load_training, save_training)
from src.blueprint.lookup import TableDistribution
from src.blueprint.solver import (BlueprintTrainer, CollectionLimitExceeded,
                                  PilotConfig, TP20_GAME)
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind, Street

DECK = tuple(r+s for s in "cdhs" for r in "23456789TJQKA")


def config(**kw):
    return PilotConfig(abstraction=TP20_SCHEMA, game=TP20_GAME,
        max_nodes=100000, max_entries=100000, max_seconds=30, **kw)


def table(button=0, stacks=(2000,)*3):
    return Table(tuple(("hero", "left", "right")[(s-button)%3] for s in range(3)),
                 stacks, button=button)


def coupled(button=0, left=("Kc", "Kd"), stacks=(2000,)*3, board=("2c","3d","4h","5s","9c")):
    t = table(button, stacks)
    holes = {"hero": ("Ac","Ad"), "left": left, "right": ("Qc","Qd")}
    order = tuple((button+i+1)%3 for i in range(3))
    dealt = tuple(holes[t.player_ids[s]][r] for r in range(2) for s in order)
    rest = tuple(c for c in DECK if c not in set(dealt+board))
    return Hand.from_deck(t, hand_id="coupled", deck=dealt+board+rest)


def test_three_seat_blinds_and_orders():
    for button in range(3):
        h = coupled(button)
        v = h.observe(h.actor)
        assert h.actor == button
        assert v.players[(button+1)%3].street_bet == 50
        assert v.players[(button+2)%3].street_bet == 100
        assert v.players[button].street_bet == 0
        for seat in (button, (button+1)%3, (button+2)%3):
            assert h.actor == seat
            v = h.observe(seat)
            h = h.apply(Action(ActionKind.CHECK if ActionKind.CHECK in v.legal_actions.kinds
                               else ActionKind.CALL))
        assert h.actor == (button+1)%3
        assert h.observe(h.actor).street == Street.FLOP


@pytest.mark.parametrize("fold", [False, True])
def test_coupled_rotations_with_offmenu_fold_and_allin(fold):
    hands = [coupled(b) for b in range(3)]
    step = 0
    while not hands[0].finished:
        views = [h.observe(h.actor) for h in hands]
        assert len({v.player_id for v in views}) == 1
        assert len({information_key(v, choices(v, free_fold=False), schema=TP20_SCHEMA)
                    for v in views}) == 1
        if step == 0:
            action = Action(ActionKind.RAISE, 275)
            assert action not in [c.action for c in choices(views[0], free_fold=False)]
        elif step == 1 and fold:
            action = Action(ActionKind.FOLD)
        elif views[0].street == Street.FLOP and ActionKind.RAISE in views[0].legal_actions.kinds:
            action = Action(ActionKind.RAISE, views[0].legal_actions.max_raise_to)
        else:
            action = Action(ActionKind.CHECK if ActionKind.CHECK in views[0].legal_actions.kinds
                            else ActionKind.CALL)
        hands = [h.apply(action) for h in hands]
        step += 1
        assert step < 20
        if fold and step > 2 and not hands[0].finished:
            assert all(h.table.capacity == 3 for h in hands)
            v = hands[0].observe(hands[0].actor)
            assert v.players[1].folded
            information_key(v, choices(v, free_fold=False), schema=TP20_SCHEMA)
    ledgers = [dict((p.player_id,p.stack) for p in h.observe(0).players) for h in hands]
    assert ledgers[0] == ledgers[1] == ledgers[2]
    assert sum(ledgers[0].values()) == 6000


def test_hidden_cards_are_not_policy_inputs():
    a, b = coupled(), coupled(left=("Jh","Jd"))
    assert a.observe(0) == b.observe(0)
    source = UniformTP20()
    assert source.distribution(a.observe(0)) == source.distribution(b.observe(0))


def test_sidepots_refunds_and_board_tie_use_native_accounting():
    for board, expected in [(("2c","3d","4h","8s","9c"), (1500,1000,1000)),
                            (("Th","Jh","Qh","Kh","Ah"), (500,1000,2000))]:
        h = coupled(stacks=(500,1000,2000), board=board)
        h = h.apply(Action(ActionKind.RAISE,500))
        h = h.apply(Action(ActionKind.RAISE,1000))
        h = h.apply(Action(ActionKind.CALL))
        assert h.finished
        assert tuple(p.stack for p in h.observe(0).players) == expected
        assert sum(expected) == 3500


def test_no_free_fold_in_actual_training_and_inference(monkeypatch, tmp_path):
    import src.blueprint.solver as solver
    original = solver.choices
    seen = []
    def checked(v, **kw):
        assert kw["free_fold"] is False
        menu = original(v, **kw)
        if ActionKind.CHECK in v.legal_actions.kinds:
            seen.append(1)
            assert all(c.action.kind != ActionKind.FOLD for c in menu)
        return menu
    monkeypatch.setattr(solver, "choices", checked)
    trainer = BlueprintTrainer(table(), config())
    trainer.step()
    assert seen
    path = tmp_path / "p.gz"
    sha = export_policy(trainer, path)
    source = FrozenBlueprint(Checkpoint("tp",str(path),sha,TP20_FORMAT),path)
    h = coupled()
    while h.observe(h.actor).street == Street.PREFLOP:
        v = h.observe(h.actor)
        h = h.apply(Action(ActionKind.CHECK if ActionKind.CHECK in v.legal_actions.kinds else ActionKind.CALL))
    menu, probabilities, found = source.distribution(h.observe(h.actor))
    assert all(c.action.kind != ActionKind.FOLD for c in menu)
    assert sum(probabilities) == pytest.approx(1)
    assert TableDistribution(trainer).distribution(h.observe(h.actor))[0] == menu
    # Deliberately absent table entries use the declared same-menu uniform fallback.
    source.entries.clear()
    assert source.distribution(h.observe(h.actor))[1:] == ((1/len(menu),)*len(menu), False)


def test_checkpoint_resume_identity_and_atomic_limit(tmp_path):
    trainer = BlueprintTrainer(table(), config())
    trainer.step()
    path = tmp_path / "train.gz"
    save_training(trainer,path)
    restored = load_training(path)
    trainer.step(); restored.step()
    assert save_training(trainer,tmp_path/"a.gz") == save_training(restored,tmp_path/"b.gz")
    with pytest.raises(ValueError,match="20BB table"):
        BlueprintTrainer(Table(("a","b"),(2000,2000)),config())
    with pytest.raises(ValueError,match="20BB table"):
        BlueprintTrainer(table(stacks=(2000,2000,1000)),config())
    with pytest.raises(ValueError,match="Invalid bounded"):
        replace(config(),postflop_replicates=4)
    rows = gzip.decompress(path.read_bytes()).splitlines()
    header = json.loads(rows[0]); header["identity"]["players"] = 2
    bad = tmp_path/"bad.gz"
    bad.write_bytes(gzip.compress(json.dumps(header).encode()+b"\n"+b"\n".join(rows[1:])))
    with pytest.raises(ValueError,match="schema"):
        load_training(bad)
    before = save_training(trainer,tmp_path/"before.gz")
    trainer.config = replace(trainer.config,max_nodes=1)
    with pytest.raises(CollectionLimitExceeded):
        trainer.step()
    assert trainer.last_attempt_nodes == 1
    trainer.config = config()
    assert save_training(trainer,tmp_path/"after.gz") == before


def plan():
    return json.loads(Path("configs/blueprint/tp20-m4.json").read_text())


def test_mixed_order_schedule_and_paired_replay(tmp_path):
    p = plan()
    _, blocks, _ = schedule(p,"secondary","mixed-tight-loose",4)
    assert blocks[0].opponents == blocks[1].opponents[::-1]
    assert all(len(set(b.action_seeds)) == 3 for b in blocks)
    a = run(p,tmp_path,"uniform","tp20_uniform","preflight",tmp_path/"a",time()+30,blocks=3)
    b = run(p,tmp_path,"uniform","tp20_uniform","preflight",tmp_path/"b",time()+30,blocks=3)
    assert a["status"] == b["status"] == "complete"
    first = [json.loads(r) for r in (tmp_path/"a"/"hands.jsonl").read_text().splitlines()]
    second = [json.loads(r) for r in (tmp_path/"b"/"hands.jsonl").read_text().splitlines()]
    assert len(first) == 9
    assert [r["public_events_sha256"] for r in first] == [r["public_events_sha256"] for r in second]
    assert [r["candidate_chips"] for r in first] == [r["candidate_chips"] for r in second]
    assert all(sum(r["net_chips"]) == 0 for r in first)


def test_density_is_decision_weighted_and_separates_streets():
    from types import SimpleNamespace
    result = density({"a": SimpleNamespace(visits=9)}, [
        {"street":"flop","key":"a","decisions":1},
        {"street":"flop","key":"absent","decisions":3},
        {"street":"river","key":"a","decisions":5}])
    assert result["flop"]["coverage"] == .25
    assert result["flop"]["visit_quantiles"]["p50"] == 0
    assert result["river"]["coverage"] == 1


def test_mid_traversal_stop_discards_all_staged_updates(tmp_path):
    trainer = BlueprintTrainer(table(), config())
    trainer.step()
    before = save_training(trainer, tmp_path/"before.gz")
    calls = 0
    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= 30
    with pytest.raises(CollectionLimitExceeded, match="cancelled before publication"):
        trainer.step(cancelled=cancelled)
    assert trainer.last_attempt_nodes == 29
    assert save_training(trainer,tmp_path/"after.gz") == before
