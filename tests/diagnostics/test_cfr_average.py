"""Independent accumulator math, diagnostic identity and information boundary."""
from dataclasses import replace
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter

import pytest

from src.arena.catalog import Checkpoint
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA, choices, information_key
from src.blueprint.artifact import FrozenBlueprint, HU20_UNCAPPED_FORMAT, export_policy, save_training
from src.blueprint.solver import BlueprintTrainer, Node, PilotConfig, HU20_UNCAPPED_GAME, _collect_root
from src.diagnostics.cfr_average import FORMAT, DiagnosticAverage, extract, audit, normalized
from src.game.hand import Hand, Table
from src.game.types import Street


def fixture(tmp_path):
    table=Table(('a','b'),(2000,2000));config=PilotConfig(seed=7,raise_cap=None,
        abstraction=HU20_UNCAPPED_SCHEMA,game=HU20_UNCAPPED_GAME)
    trainer=BlueprintTrainer(table,config);trainer.iteration=10
    view=Hand.start(table,hand_id='math',seed=17).observe(0);menu=choices(view,raise_cap=None,free_fold=False)
    key=information_key(view,menu,schema=config.abstraction);n=len(menu)
    trainer.nodes[key]=Node(tuple(c.name for c in menu),[1]+[0]*(n-1),[0]+[2]*(n-1),3)
    trainer.nodes['f'*32]=Node(('check',),[0],[0],0)
    checkpoint=tmp_path/'checkpoint.gz';current=tmp_path/'current.gz'
    ch=save_training(trainer,checkpoint);ph=export_policy(trainer,current)
    spec={'name':'fixture','seed':7,'iteration':10,'checkpoint_sha256':ch,'sha256':ph}
    return trainer,view,checkpoint,current,spec


def test_extraction_verifies_all_accumulators_and_guard_stays(tmp_path):
    trainer,view,checkpoint,current,spec=fixture(tmp_path);output=tmp_path/'average.gz'
    before=checkpoint.read_bytes();result=extract(checkpoint,spec,output)
    assert result['counts']=={'entries':2,'positive_mass':1,'visits':3,'zero_mass':1}
    checked=audit(checkpoint,current,output,spec,result['sha256'])
    assert checked['all_nodes_verified']==2 and checked['maximum_tv']==1
    average=DiagnosticAverage(output,result['sha256'])
    menu,p,trained=average.distribution(view)
    assert trained and p[0]==0 and sum(p)==pytest.approx(1)
    assert checkpoint.read_bytes()==before
    with pytest.raises(ValueError,match='separately collected'):export_policy(trainer,tmp_path/'not-production.gz',strategy='average')
    with pytest.raises(ValueError):FrozenBlueprint(Checkpoint('avg',str(output),result['sha256'],HU20_UNCAPPED_FORMAT),output)
    with pytest.raises(FileExistsError):extract(checkpoint,spec,output)
    assert extract(checkpoint,spec,tmp_path/'same-bytes.gz')['sha256']==result['sha256']


@pytest.mark.parametrize('average,visits', [([-1,2],1),([float('nan'),0],1),([1,1],0),([100,0],1),([1,0],True)])
def test_invalid_accumulators_are_rejected(average,visits):
    with pytest.raises(ValueError):normalized(average,visits,10)


def test_tampered_hash_or_normalization_is_rejected(tmp_path):
    _,_,checkpoint,current,spec=fixture(tmp_path);output=tmp_path/'average.gz'
    with pytest.raises(ValueError,match='hash'):extract(checkpoint,{**spec,'checkpoint_sha256':'0'*64},output)
    result=extract(checkpoint,spec,output)
    data=gzip.decompress(output.read_bytes()).decode().splitlines();row=json.loads(data[1]);row[2]=[1]+[0]*(len(row[2])-1)
    data[1]=json.dumps(row);output.write_bytes(gzip.compress(('\n'.join(data)+'\n').encode(),mtime=0))
    from src.diagnostics.saved_hu20 import file_hash
    with pytest.raises(ValueError,match='Normalized'):audit(checkpoint,current,output,spec,file_hash(output))


def test_observation_only_inference_and_zero_mass_is_distinct(tmp_path):
    trainer,view,checkpoint,current,spec=fixture(tmp_path)
    key=information_key(view,choices(view,raise_cap=None,free_fold=False),schema=trainer.config.abstraction)
    trainer.nodes[key].average=[0]*len(trainer.nodes[key].names)
    spec['checkpoint_sha256']=save_training(trainer,checkpoint);output=tmp_path/'average.gz';result=extract(checkpoint,spec,output)
    source=DiagnosticAverage(output,result['sha256'])
    menu,p,trained=source.distribution(view);assert trained and key in source.zero_mass
    assert p==(1/len(menu),)*len(menu)
    other=replace(view,hole_cards=('Ac','Ad'))
    assert source.distribution(other)[2] is False
    assert source.policy(17).choose_action(view)==source.policy(17).choose_action(view)
    # Equal own observations remain independent of external hidden simulator data.
    assert source.distribution(replace(view,hand_id='another-hidden-world'))==source.distribution(view)


def test_unrevealed_holding_and_future_deck_cannot_change_average_query(tmp_path):
    from src.blueprint.search import DECK
    _,_,checkpoint,_,spec=fixture(tmp_path);output=tmp_path/'average.gz';result=extract(checkpoint,spec,output)
    source=DiagnosticAverage(output,result['sha256']);deck=list(DECK);other=list(deck)
    other[0],other[4]=other[4],other[0];other[2],other[5]=other[5],other[2];other[6:]=reversed(other[6:])
    views=[Hand.from_deck(Table(('a','b'),(2000,2000)),hand_id='same-visible',deck=d).observe(0) for d in (deck,other)]
    assert views[0]==views[1]
    assert source.distribution(views[0])==source.distribution(views[1])
    assert source.policy(19).choose_action(views[0])==source.policy(19).choose_action(views[1])


def test_collection_weights_own_reach_not_opponent_probability(monkeypatch):
    import src.blueprint.solver as solver
    root_key='0'*32;left='1'*32;right='2'*32;opponent='3'*32
    menu=(SimpleNamespace(name='a',action=0),SimpleNamespace(name='b',action=1))
    class Tree:
        def __init__(self,path=()):self.path=path;self.actor=1 if len(path)==1 else 0;self.finished=len(path)==3;self.table=SimpleNamespace(big_blind=100)
        def apply(self,action):return Tree((*self.path,action))
        def observe(self,seat):
            key=root_key if not self.path else opponent if len(self.path)==1 else left if self.path[0]==0 else right
            return SimpleNamespace(street=Street.PREFLOP,history=(),key=key,players=(SimpleNamespace(stack=2000,starting_stack=2000),)*2)
    monkeypatch.setattr(solver.Hand,'start',lambda *args,**kwargs:Tree())
    monkeypatch.setattr(solver,'choices',lambda *args,**kwargs:menu)
    monkeypatch.setattr(solver,'information_key',lambda view,*args,**kwargs:view.key)
    frozen={root_key:Node(('a','b'),[1,3],[0,0]),left:Node(('a','b'),[3,1],[0,0]),
            right:Node(('a','b'),[3,1],[0,0]),opponent:Node(('a','b'),[99,1],[0,0])}
    config=PilotConfig(raise_cap=None,abstraction=HU20_UNCAPPED_SCHEMA,game=HU20_UNCAPPED_GAME)
    first=_collect_root(Table(('a','b'),(2000,2000)),config,frozen,4,0,0,perf_counter()+5)
    assert first.deltas[root_key].average==[1,3]
    assert first.deltas[left].average==[.75,.25]  # 4 × .25 own reach × [.75,.25]
    assert first.deltas[right].average==[2.25,.75]
    second=_collect_root(Table(('a','b'),(2000,2000)),config,frozen,8,0,0,perf_counter()+5)
    accumulated=[a+b for a,b in zip(first.deltas[left].average,second.deltas[left].average)]
    assert accumulated==[2.25,.75] and normalized(accumulated,2,8)[0]==(.75,.25)
