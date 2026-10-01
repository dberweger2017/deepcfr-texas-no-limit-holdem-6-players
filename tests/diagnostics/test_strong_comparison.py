"""Outcome-blind selection, freeze checks and paired checkpoint arithmetic."""
import json
from pathlib import Path

import pytest

from scripts.compare_strong_hu20 import DecisionSample, checkpoint_changes, verify_freeze
from src.arena.schedule import digest
from src.game.hand import Hand, Table


def test_selection_is_coordinate_only_and_stratified():
    hand=Hand.start(Table(('a','b'),(2000,2000)),hand_id='sample',seed=91);view=hand.observe(0)
    sample=DecisionSample(92,2)
    for block in range(6):sample.consider(view,(),(),bool(block%2),0,block,0)
    selected=sample.selected();expected=sorted(range(6),key=lambda b:digest([92,b,0,0]))[:2]
    assert [c['block'] for _,c in selected]==expected
    again=DecisionSample(92,2)
    for block in reversed(range(6)):again.consider(view,(),(),False,0,block,0)
    assert [c['block'] for _,c in again.selected()]==expected
    assert next(r for r in sample.counts() if r['street']=='river')['selected']==0


def test_paired_lineages_and_modes_are_separate():
    specs=[];panels=[]
    for mode in ('restricted','native'):
        for seed in (1,2,3):
            for checkpoint in (100000000,500000000):
                name=f'{seed}-{checkpoint}'
                if mode=='restricted':specs.append({'name':name,'seed':seed,'milestone':checkpoint})
                base=100 if mode=='restricted' else -100
                panels.append({'policy':name,'mode':mode,'blocks':[0,1],
                    'paired_block_chips':[base+(10 if checkpoint==500000000 else 0),base+(20 if checkpoint==500000000 else 0)]})
    result=checkpoint_changes(panels,specs)
    assert len(result['seed_changes'])==6 and len(result['three_lineage_changes'])==2
    assert all(r['500m_minus_100m']['bb_per_100']==15 for r in result['three_lineage_changes'])
    panels[0]['blocks']=[3,4]
    with pytest.raises(ValueError,match='pairing'):checkpoint_changes(panels,specs)


def test_freeze_is_verified_before_loading_models():
    config=json.loads(Path('configs/diagnostics/strong-rollout-hu20-v1.json').read_text())
    p=Path('docs/reports/strong-rollout-hu20-v1-artifacts/opponent-freeze.json')
    assert verify_freeze(p,config)['quality_goal_passed'] is False
    config['equity_worlds']+=1
    with pytest.raises(ValueError,match='Configuration'):verify_freeze(p,config)


def test_full_runner_keeps_input_root_separate_from_metadata(tmp_path,monkeypatch):
    from scripts.compare_strong_hu20 import run
    from src.blueprint.abstraction import choices
    class Uniform:
        description={'fixture':True}
        def distribution(self,view):
            menu=choices(view,raise_cap=None,free_fold=False)
            return menu,tuple(1/len(menu) for _ in menu),False
    input_root=tmp_path/'inputs';input_root.mkdir()
    def load(spec,root):
        assert root==input_root
        return Uniform()
    monkeypatch.setattr('scripts.compare_strong_hu20.load_policy',load)
    config=json.loads(Path('configs/diagnostics/strong-rollout-hu20-v1.json').read_text())
    config={**config,'particles':8,'equity_worlds':4}
    plan={'models':[{'name':'fixture','seed':1,'milestone':100000000,'bytes':1,'sha256':'fixture'}],
          'modes':['restricted'],'blocks':2,'decision_sample_root':93,'deal_root':94,
          'decisions_per_street_position':1,'selection_worlds':2,'max_seconds':30}
    result=run(plan,config,{},tmp_path/'out',input_root)
    assert result['status']=='complete' and result['hands']==4
    assert result['inputs'][0]['name']=='fixture' and result['decision_diagnostics']
