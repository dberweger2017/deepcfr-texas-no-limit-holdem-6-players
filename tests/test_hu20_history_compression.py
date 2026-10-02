"""History identity, public boundaries and exact trainer recovery."""
from dataclasses import replace
import gzip
import json
import subprocess
import sys

import pytest

from src.blueprint.abstraction import (
    HU20_COMPRESSED_SCHEMA, HU20_UNCAPPED_SCHEMA, _band, _history,
    choices, compressed_history, information_key,
)
from src.blueprint.artifact import export_policy, load_training, save_training
from src.blueprint.solver import BlueprintTrainer, HU20_UNCAPPED_GAME, PilotConfig
from src.game.hand import Hand, Table
from src.game.observation import ActionTaken, BoardDealt
from src.game.types import Action, ActionKind, Street
from tests.test_blueprint_hu20 import coupled


def config():
    return PilotConfig(seed=2026093001, raise_cap=None, max_nodes=250000,
                       abstraction=HU20_COMPRESSED_SCHEMA, game=HU20_UNCAPPED_GAME)


def river():
    hand = Hand.start(Table(('a', 'b'), (2000, 2000)), hand_id='fixture', seed=7)
    hand = hand.apply(Action(ActionKind.CALL)).apply(Action(ActionKind.CHECK))
    while hand.observe(hand.actor).street != Street.RIVER:
        hand = hand.apply(Action(ActionKind.CHECK))
    return hand.observe(hand.actor)


def test_preflop_exact_keys_and_hidden_worlds():
    for button in (0, 1):
        views = [coupled(button, cards).observe(button) for cards in (('Qc', 'Qd'), ('Jc', 'Jd'))]
        for view in views:
            menu = choices(view, raise_cap=None, free_fold=False)
            assert information_key(view, menu, schema=HU20_COMPRESSED_SCHEMA) == information_key(view, menu, schema=HU20_UNCAPPED_SCHEMA)
        assert views[0] == views[1]
        assert compressed_history(views[0]) == _history(views[0])


def test_current_street_tokens_and_public_start_chips():
    view = river()
    summary = compressed_history(view)
    assert summary[0] == (0, True, None)
    assert summary[1] == (('flop', None), ('turn', None))
    assert summary[2] == (_band(2, (2, 4, 8, 16, 32)), _band(9.5, (.5, 1, 2, 4, 8)))
    assert summary[3] == tuple(token for token in _history(view) if token[0] == 'river')
    # Same earlier public sequence, different own cards: history is card-free.
    assert compressed_history(replace(view, hole_cards=('As', 'Ah'))) == summary
    hand = Hand.start(Table(('a', 'b'), (2000, 2000)), hand_id='bet', seed=7)
    hand = hand.apply(Action(ActionKind.RAISE, 400)).apply(Action(ActionKind.CALL))
    v = hand.observe(hand.actor)
    assert compressed_history(v)[0] == (1, True, 0)
    assert compressed_history(v)[2] == (_band(8, (2,4,8,16,32)), _band(2, (.5,1,2,4,8)))
    hand = hand.apply(Action(ActionKind.RAISE, hand.observe(hand.actor).legal_actions.min_raise_to))
    v = hand.observe(hand.actor)
    assert compressed_history(v)[3] == tuple(token for token in _history(v) if token[0] == 'flop')
    assert compressed_history(v)[2] == compressed_history(hand.observe(hand.actor))[2]


def test_distinct_public_summary_and_order_separation():
    view = river()
    events = list(view.history)
    turn_check = next(i for i,e in enumerate(events) if isinstance(e, ActionTaken) and e.street == Street.TURN)
    events[turn_check] = replace(events[turn_check], action=Action(ActionKind.RAISE, 100), paid=100)
    assert compressed_history(replace(view, history=tuple(events))) != compressed_history(view)
    current = ActionTaken(view.actor, Street.RIVER, Action(ActionKind.CHECK), 0)
    other = replace(current, seat=1-view.actor)
    assert compressed_history(replace(view, history=(*view.history,current,other))) != compressed_history(replace(view, history=(*view.history,other,current)))
    for thresholds in ((2,4,8,16,32),(.5,1,2,4,8)):
        for threshold in thresholds:
            assert _band(threshold, thresholds) != _band(threshold-1e-6, thresholds)
    with pytest.raises(ValueError, match='boundary'):
        compressed_history(replace(view, history=tuple(e for e in view.history if not isinstance(e,BoardDealt) or e.street!=Street.RIVER)))


def test_deterministic_state_fresh_process_resume_and_identity(tmp_path):
    table=Table(('a','b'),(2000,2000))
    a,b=BlueprintTrainer(table,config()),BlueprintTrainer(table,config())
    for _ in range(8):
        a.step(); b.step()
    midpoint=tmp_path/'midpoint.gz'; save_training(a,midpoint)
    for _ in range(8): a.step()
    final=tmp_path/'final.gz'; expected=save_training(a,final)
    output=tmp_path/'fresh.gz'
    subprocess.run([sys.executable,'-c',
        'import sys; from pathlib import Path; from src.blueprint.artifact import load_training,save_training; t=load_training(Path(sys.argv[1])); [t.step() for _ in range(8)]; save_training(t,Path(sys.argv[2]))',str(midpoint),str(output)],check=True)
    assert output.read_bytes()==final.read_bytes()
    assert load_training(output).nodes==a.nodes
    assert save_training(b,tmp_path/'repeat.gz')==save_training(load_training(midpoint),tmp_path/'reload.gz')
    exp=tmp_path/'export.gz'; export_policy(a,exp)
    assert gzip.decompress(exp.read_bytes())
    header,*rows=gzip.decompress(final.read_bytes()).splitlines()
    header=json.loads(header)
    assert header['identity']['history_descriptor']=='hu20-earlier-streets-public-summary-v1'
    header['identity'].pop('history_descriptor')
    final.write_bytes(gzip.compress(json.dumps(header).encode()+b'\n'+b'\n'.join(rows)+b'\n',mtime=0))
    with pytest.raises(ValueError,match='schema'): load_training(final)
    with pytest.raises(ValueError): replace(config(),raise_cap=2)


def test_postflop_hidden_cards_and_button_rotations():
    hands=[coupled(button, cards) for button in (0,1) for cards in (('Qc','Qd'),('Jc','Jd'))]
    for _ in range(8):
        views=[hand.observe(hand.actor) for hand in hands]
        menus=[choices(view,raise_cap=None,free_fold=False) for view in views]
        # Earlier actions are identical public call/check paths, including rotations.
        keys=[information_key(view,menu,schema=HU20_COMPRESSED_SCHEMA) for view,menu in zip(views,menus)]
        assert keys[0]==keys[2] and keys[1]==keys[3]
        if views[0].seat==views[0].button:
            assert len(set(keys))==1
        actions=[next(item.action for item in menu if item.action.kind in (ActionKind.CALL,ActionKind.CHECK)) for menu in menus]
        hands=[hand.apply(action) for hand,action in zip(hands,actions)]
