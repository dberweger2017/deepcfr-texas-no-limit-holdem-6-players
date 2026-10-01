"""Representation isolation, concrete collisions and unchanged recipe recovery."""
from dataclasses import replace
from hashlib import sha256
from itertools import permutations
import gzip
import json

import pytest

from src.arena.catalog import Checkpoint
from src.blueprint.abstraction import (HU20_CARD_V2_SCHEMA, HU20_UNCAPPED_SCHEMA,
    _postflop, _preflop, choices, information_key, _history)
from src.blueprint.cards_v2 import VERSION, postflop_v2
from src.blueprint.artifact import export_policy, save_training, load_training, FrozenBlueprint, HU20_UNCAPPED_FORMAT
from src.blueprint.solver import BlueprintTrainer, PilotConfig, HU20_UNCAPPED_GAME, _seed
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind, Street
from tests.test_blueprint_hu20 import coupled

COLLISIONS = (
    ('7h Td Ts 2s 2c', '4d 5s', '7s Kd'),
    ('Td Tc 9s Th 7s', '3h 5s', 'Ad Ks'),
    ('Qh 9s Ad 3d As', '2h 4c', 'Ks Th'),
    ('4h Jh 3s Jd', '2c 7d', '8d Ah'),
    ('8s 8d 8h', '3h 4d', '5h Ac'),
)


@pytest.mark.parametrize('board,low,high', COLLISIONS)
def test_concrete_pr139_collisions_separate(board, low, high):
    b=tuple(board.split());lo=tuple(low.split());hi=tuple(high.split())
    assert _postflop(lo,b)==_postflop(hi,b)
    assert postflop_v2(lo,b)!=postflop_v2(hi,b)
    assert postflop_v2(lo,b)[0]==_postflop(lo,b)


def test_suit_card_order_and_observation_information_isolation():
    for board,low,high in COLLISIONS:
        b=tuple(board.split());c=tuple(high.split());expected=postflop_v2(c,b)
        assert postflop_v2(c[::-1],b[::-1])==expected
        for suits in permutations('cdhs'):
            mapping=dict(zip('cdhs',suits));convert=lambda cards:tuple(x[0]+mapping[x[1]] for x in cards)
            assert postflop_v2(convert(c),convert(b))==expected
    a,b=coupled(0).observe(0),coupled(0,('Qc','Qd')).observe(0)
    menu=choices(a,raise_cap=None,free_fold=False)
    assert information_key(a,menu,schema=HU20_CARD_V2_SCHEMA)==information_key(b,menu,schema=HU20_CARD_V2_SCHEMA)
    assert _history(a)==_history(b)


def test_contribution_pair_relative_kickers_and_draw_quality():
    # River board play versus a privately upgraded second pair.
    board=tuple('7h Td Ts 2s 2c'.split())
    assert postflop_v2(('4d','5s'),board)[1]==0
    assert postflop_v2(('7s','Kd'),board)[1]==2
    # Same binary flush draw, different private nut potential.
    b=('Kh','8h','2c')
    assert postflop_v2(('Ah','3h'),b)[5]!=postflop_v2(('Qh','3h'),b)[5]
    # One versus two distinct completing ranks, own-card participation.
    assert postflop_v2(('9c','Td'),('8h','7s','2c'))[6][0]==2
    assert postflop_v2(('9c','Td'),('8h','6s','2c'))[6][0]==1
    assert postflop_v2(('Ah','3h'),('Kh','8h','2c','4d','6s'))[6]==(0,0,0,0)
    with pytest.raises(ValueError):postflop_v2(('Ah','Ah'),b)


def test_preflop_partition_menu_history_seed_and_v1_identity_unchanged(tmp_path):
    hand=Hand.start(Table(('a','b'),(2000,2000)),hand_id='same',seed=7)
    while not hand.finished:
        view=hand.observe(hand.actor);menu=choices(view,raise_cap=None,free_fold=False)
        for item in menu:view.legal_actions.validate(item.action)
        if view.street==Street.PREFLOP:
            assert _preflop(view.hole_cards)==_preflop(view.hole_cards[::-1])
        assert information_key(view,menu,schema=HU20_CARD_V2_SCHEMA)!=information_key(view,menu,schema=HU20_UNCAPPED_SCHEMA)
        hand=hand.apply(Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds else ActionKind.CALL))
    config=PilotConfig(seed=7,abstraction=HU20_UNCAPPED_SCHEMA,game=HU20_UNCAPPED_GAME,raise_cap=None)
    v2=replace(config,abstraction=HU20_CARD_V2_SCHEMA)
    assert replace(v2,abstraction=HU20_UNCAPPED_SCHEMA)==config
    assert _seed(config.seed,1,0,0,'deal')==_seed(v2.seed,1,0,0,'deal')
    for cfg in (config,v2):
        t=BlueprintTrainer(Table(('a','b'),(2000,2000)),cfg)
        path=tmp_path/(cfg.abstraction+'.gz');export_policy(t,path)
        header=json.loads(gzip.decompress(path.read_bytes()))
        assert header['identity']['card_descriptor']==(VERSION if cfg==v2 else 'legacy-postflop-descriptor-v1')
        assert header['identity']['action_menu']=='hu20-min-pot-conditional-jam-native-reopening-v1'
    with pytest.raises(ValueError):replace(v2,raise_cap=2)


def test_v2_checkpoint_export_resume_and_identity_rejection(tmp_path):
    cfg=PilotConfig(seed=17,abstraction=HU20_CARD_V2_SCHEMA,game=HU20_UNCAPPED_GAME,raise_cap=None,max_nodes=10000)
    trainer=BlueprintTrainer(Table(('a','b'),(2000,2000)),cfg)
    trainer.step();mid=tmp_path/'mid.gz';save_training(trainer,mid)
    resumed=load_training(mid)
    trainer.step();resumed.step()
    assert save_training(trainer,tmp_path/'a.gz')==save_training(resumed,tmp_path/'b.gz')
    policy=tmp_path/'policy.gz';hash_=export_policy(trainer,policy)
    source=FrozenBlueprint(Checkpoint('v2',str(policy),hash_,HU20_UNCAPPED_FORMAT),policy)
    assert source.abstraction==HU20_CARD_V2_SCHEMA
    view=Hand.start(trainer.table,hand_id='test',seed=11).observe(0)
    if view.actor!=0:view=replace(view,seat=view.actor,hole_cards=('Ah','Kd'))
    menu,p,trained=source.distribution(view)
    assert abs(sum(p)-1)<1e-12
    for c in menu:view.legal_actions.validate(c.action)
    doc=json.loads(gzip.decompress(policy.read_bytes()));doc['identity']['card_descriptor']='legacy-postflop-descriptor-v1'
    policy.write_bytes(gzip.compress(json.dumps(doc).encode(),mtime=0))
    with pytest.raises(ValueError,match='schema'):
        FrozenBlueprint(Checkpoint('wrong',str(policy),sha256(policy.read_bytes()).hexdigest(),HU20_UNCAPPED_FORMAT),policy)
    with pytest.raises(ValueError,match='current export'):export_policy(trainer,tmp_path/'avg.gz',strategy='average')
