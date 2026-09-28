"""One cap setting changes menu availability, while native rules remain authoritative."""

from dataclasses import replace
from hashlib import sha256
import gzip
import json

import pytest

from src.arena.catalog import Checkpoint
from src.blueprint.abstraction import (HU20_SCHEMA, HU20_UNCAPPED_SCHEMA,
                                      choices, information_key)
from src.blueprint.artifact import (FrozenBlueprint, HU20_FORMAT, HU20_UNCAPPED_FORMAT,
                                    export_policy, load_training, save_training)
from src.blueprint.lookup import TableDistribution
from src.blueprint.solver import (BlueprintTrainer, CollectionLimitExceeded,
                                  HU20_GAME, HU20_UNCAPPED_GAME, PilotConfig)
from src.diagnostics.robustness import LocalBestResponse, LBRConfig, ReactiveAttack
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind
from tests.test_blueprint_hu20 import coupled


def config(seed=7, **kwargs):
    return PilotConfig(seed=seed, raise_cap=None, abstraction=HU20_UNCAPPED_SCHEMA,
                       game=HU20_UNCAPPED_GAME, **kwargs)


def test_repeated_minraises_menu_difference_and_terminal_refunds():
    hand = Hand.start(Table(('a', 'b'), (2000, 2000)), hand_id='repeated', seed=91)
    raises = 0
    while not hand.finished:
        view = hand.observe(hand.actor)
        a = choices(view, free_fold=False)
        b = choices(view, raise_cap=None, free_fold=False)
        for item in b:
            view.legal_actions.validate(item.action)
        if raises < 2:
            assert a == b
        if ActionKind.RAISE in view.legal_actions.kinds:
            abstract_raises = [c.action for c in b if c.action.kind == ActionKind.RAISE]
            assert abstract_raises
            assert abstract_raises[0].raise_to == view.legal_actions.min_raise_to
            if raises >= 2:
                assert not any(c.action.kind == ActionKind.RAISE for c in a)
                # The nonraise actions and all sizing computations are shared.
                assert a == tuple(c for c in b if c.action.kind != ActionKind.RAISE)
            action = abstract_raises[0]
            raises += 1
        else:
            assert not any(c.action.kind == ActionKind.RAISE for c in b)
            action = Action(ActionKind.CALL if ActionKind.CALL in view.legal_actions.kinds else ActionKind.CHECK)
        hand = hand.apply(action)
    assert raises > 10
    assert sum(p.stack for p in hand.observe(0).players) == 4000
    assert all(p.contributed == 0 for p in hand.observe(0).players)

    folded = Hand.start(Table(('a','b'), (2000,2000)), hand_id='refund', seed=3)
    folded = folded.apply(Action(ActionKind.RAISE, 300))
    folded = folded.apply(Action(ActionKind.FOLD))
    assert sorted(p.stack for p in folded.observe(0).players) == [1900, 2100]


def test_native_short_allin_does_not_reopen_menu():
    hand = Hand.start(Table(('a','b'), (2000,1000)), hand_id='short', seed=1)
    hand = hand.apply(Action(ActionKind.RAISE, 800))
    hand = hand.apply(Action(ActionKind.RAISE, 1000))
    view = hand.observe(hand.actor)
    assert ActionKind.RAISE not in view.legal_actions.kinds
    assert all(c.action.kind != ActionKind.RAISE for c in choices(view, raise_cap=None, free_fold=False))
    hand = hand.apply(Action(ActionKind.CALL))
    assert hand.finished and sum(p.stack for p in hand.observe(0).players) == 3000


def test_rotations_and_hidden_world_invariance_after_cap():
    hands = [coupled(b) for b in (0,1)]
    for step in range(7):
        views = [h.observe(h.actor) for h in hands]
        menus = [choices(v, raise_cap=None, free_fold=False) for v in views]
        assert menus[0] == menus[1]
        assert information_key(views[0], menus[0], schema=HU20_UNCAPPED_SCHEMA) == information_key(views[1], menus[1], schema=HU20_UNCAPPED_SCHEMA)
        action = next(c.action for c in menus[0] if c.action.kind == ActionKind.RAISE)
        hands = [h.apply(action) for h in hands]
    assert coupled(0).observe(0) == coupled(0, ('Qc','Qd')).observe(0)
    trainer = BlueprintTrainer(Table(('a','b'),(2000,2000)), config())
    source = TableDistribution(trainer)
    a,b = coupled(0).observe(0), coupled(0, ('Qc','Qd')).observe(0)
    assert source.distribution(a) == source.distribution(b)
    assert LocalBestResponse(source, 31, LBRConfig(1,5)).choose_action(a) == LocalBestResponse(source,31,LBRConfig(1,5)).choose_action(b)


def test_free_checks_and_common_attacker_contract():
    hand = Hand.start(Table(('a','b'),(2000,2000)),hand_id='checks',seed=5)
    for _ in range(3):
        v=hand.observe(hand.actor)
        hand=hand.apply(Action(ActionKind.RAISE,v.legal_actions.min_raise_to))
    v=hand.observe(hand.actor)
    assert ReactiveAttack('minraise','menu').choose_action(v).kind == ActionKind.CALL
    assert ReactiveAttack('minraise','native').choose_action(v).kind == ActionKind.RAISE
    hand=hand.apply(Action(ActionKind.CALL))
    v=hand.observe(hand.actor)
    assert ActionKind.CHECK in v.legal_actions.kinds
    for cap in (2,None):
        assert all(c.action.kind != ActionKind.FOLD for c in choices(v,raise_cap=cap,free_fold=False))


def test_identity_rejection_and_no_partial_publication(tmp_path):
    trainer=BlueprintTrainer(Table(('a','b'),(2000,2000)),config(max_nodes=50000,max_seconds=30))
    path=tmp_path/'before.gz';before=save_training(trainer,path)
    calls=0
    def cancel():
        nonlocal calls
        calls+=1
        return calls>100
    with pytest.raises(CollectionLimitExceeded,match='cancelled'):
        trainer.step(cancelled=cancel)
    assert trainer.iteration==0 and trainer.nodes=={}
    assert trainer.last_attempt_nodes==100
    assert save_training(trainer,tmp_path/'after.gz')==before
    assert trainer.last_attempt_work['nodes_by_street']
    resumed=load_training(path)
    assert resumed.config==trainer.config
    assert save_training(resumed,tmp_path/'resumed.gz')==before
    exp=tmp_path/'policy.gz';hash_=export_policy(resumed,exp)
    policy=FrozenBlueprint(Checkpoint('B',str(exp),hash_,HU20_UNCAPPED_FORMAT),exp)
    assert policy.raise_cap is None and policy.game==HU20_UNCAPPED_GAME
    assert policy.identity['raise_cap_semantics'].startswith('none;')
    with pytest.raises(ValueError,match='format differs'):
        FrozenBlueprint(Checkpoint('wrong',str(exp),hash_,HU20_FORMAT),exp)
    with pytest.raises(ValueError):
        replace(trainer.config,raise_cap=2)
    with pytest.raises(ValueError):
        replace(trainer.config,abstraction=HU20_SCHEMA,game=HU20_GAME)
    header,*rows=gzip.decompress(path.read_bytes()).splitlines()
    document=json.loads(header);document['identity']['raise_cap_semantics']='two'
    bad=tmp_path/'bad.gz';bad.write_bytes(gzip.compress(json.dumps(document).encode()+b'\n'+b'\n'.join(rows),mtime=0))
    with pytest.raises(ValueError,match='abstraction schema'):
        load_training(bad)


def test_capped_trainer_bytes_match_reviewed_parent(tmp_path):
    # Independently measured at #114/36a678a, before this menu refactor.
    trainer = BlueprintTrainer(Table(('a','b'),(2000,2000)), PilotConfig(
        seed=71, abstraction=HU20_SCHEMA, game=HU20_GAME,
        max_nodes=250000, max_entries=4000000, max_seconds=300))
    for _ in range(3):
        trainer.step()
    assert save_training(trainer,tmp_path/'capped.gz') == '79048936a05789c186637526e20305883a5a83b5cca94a37872a503f3994f428'


def test_common_lbr_updates_likelihood_for_target_reraises_beyond_attacker_cap():
    hand=Hand.start(Table(('a','b'),(2000,2000)),hand_id='lbr-reopening',seed=13)
    source=TableDistribution(BlueprintTrainer(hand.table,config()))
    for _ in range(3):
        v=hand.observe(hand.actor)
        hand=hand.apply(Action(ActionKind.RAISE,v.legal_actions.min_raise_to))
    v=hand.observe(hand.actor)
    response=LocalBestResponse(source,7,LBRConfig(1,5))
    response.update(v)
    assert not response.zero_likelihood
    assert response.weights.sum()==pytest.approx(1)
    action=response.choose_action(v)
    assert action in [c.action for c in choices(v,free_fold=False)]
    assert action.kind != ActionKind.RAISE


def test_uncapped_production_inference_and_human_menu(tmp_path):
    from scripts.play_hu20_native import play
    from scripts.play_hu20 import replay_history
    from src.blueprint.solver import Node
    from src.arena.registry import load_frozen
    trainer=BlueprintTrainer(Table(('a','b'),(2000,2000)),config())
    # A fixed nonuniform test profile exercises the real loader and session.
    v=Hand.start(trainer.table,hand_id='profile',seed=4).observe(0)
    menu=choices(v,raise_cap=None,free_fold=False)
    key=information_key(v,menu,schema=HU20_UNCAPPED_SCHEMA)
    trainer.nodes[key]=Node(tuple(c.name for c in menu),[0,1,0,0],[0]*len(menu))
    trainer.iteration=1
    policy=tmp_path/'policy.gz';h=export_policy(trainer,policy)
    spec=Checkpoint('B',str(policy),h,HU20_UNCAPPED_FORMAT)
    assert load_frozen(spec,policy).distribution(v)[1] == TableDistribution(trainer).distribution(v)[1]
    visible=[]
    def choose(prompt):
        return next(line.split('.')[0].strip() for line in reversed(visible)
                    if line.startswith('  ') and ('check' in line or 'call' in line))
    history=tmp_path/'human.jsonl'
    result=play(policy,h,history,max_hands=2,input_fn=choose,output=visible.append)
    assert result['hands']==replay_history(history)==2
    assert HU20_UNCAPPED_GAME in visible[0]
    assert not any('Bot cards:' in line for line in visible)
