import gzip
import json
from pathlib import Path
from time import time
import pytest
from scripts import evaluate_hu20_cfr_plus_direct as direct
from scripts.audit_hu20_k50_first_pass_direct import replay
from src.blueprint.abstraction import choices, information_key, HU20_UNCAPPED_SCHEMA


class Uniform:
    visits={}
    def distribution(self,view):
        menu=choices(view,raise_cap=None,free_fold=False)
        return menu,tuple(1/len(menu) for _ in menu),False
    def information_key(self,view,menu):
        return information_key(view,menu,schema=HU20_UNCAPPED_SCHEMA)


def played(tmp_path,monkeypatch):
    monkeypatch.setattr(direct,'load',lambda spec,root:Uniform())
    spec=lambda name:dict(name=name,strategy='average',seed=11,abstraction=HU20_UNCAPPED_SCHEMA)
    plan=dict(root=42,blocks=33,started_at=time(),max_seconds=300,
              pairs=[dict(candidate=spec('candidate'),reference=spec('reference'))])
    result=direct.run(plan,tmp_path,tmp_path/'out',1)
    assert result['status']=='complete'
    return plan,tmp_path/'out'


def test_independent_replay_checks_both_schema_keys_and_coverage(tmp_path,monkeypatch):
    plan,out=played(tmp_path,monkeypatch)
    cells,receipt=replay(plan,out,tmp_path)
    assert len(cells)==66 and receipt['hands_replayed']==66
    assert receipt['policy_owned_key_checks'] and receipt['decisions_checked']>66
    assert receipt['coverage']['candidate']['unique_stored_reached_keys']==0
    assert sum(v['missing_reached_keys'] for v in receipt['coverage']['candidate']['streets'].values())>0


def test_independent_replay_rejects_raw_accounting_corruption(tmp_path,monkeypatch):
    plan,out=played(tmp_path,monkeypatch)
    path=out/'direct-lineage-1.hands.jsonl.gz'
    with gzip.open(path,'rt') as source:rows=list(map(json.loads,source))
    rows[0]['target_chips']+=1
    with gzip.open(path,'wt') as target:
        for row in rows:target.write(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='accounting'):
        replay(plan,out,tmp_path)


def test_independent_replay_rejects_foreign_rival_key(tmp_path,monkeypatch):
    plan,out=played(tmp_path,monkeypatch)
    path=out/'direct-lineage-1.hands.jsonl.gz'
    with gzip.open(path,'rt') as source:rows=list(map(json.loads,source))
    decision=next(a for row in rows for a in row['actions'] if a['logical_player'])
    decision['reference_decision']['key']='0'*32
    with gzip.open(path,'wt') as target:
        for row in rows:target.write(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='schema key'):
        replay(plan,out,tmp_path)
