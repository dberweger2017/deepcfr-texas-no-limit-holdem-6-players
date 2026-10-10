from dataclasses import replace
import json
import subprocess

import pytest

from scripts import evaluate_hu100_direct as direct
from tests.test_native_hu100_baseline import model_fixture
from src.arena.registry import PolicyRegistry
from src.arena.schedule import build_schedule
from src.blueprint.average import TranslationOptions
from src.policies.files import file_hash


def specs(tmp_path):
    _,model,path=model_fixture(tmp_path)
    return [{'name':name,'path':str(path),'sha256':file_hash(path),
        'format':'holdem-hu100-stored-cfr-average-research-v1','bytes':path.stat().st_size,
        'entries':1,'iteration':2,'source_checkpoint_sha256':model.description['source_checkpoint_sha256']}
        for name in ('candidate-model','reference-model')]


def test_two_registry_models_have_private_instances_and_menu_actions(tmp_path):
    models=specs(tmp_path);plan=direct.make_plan(models,16,2026100922201,'fixture')
    registry=PolicyRegistry(plan);direct.validate(registry,models,plan)
    schedule=build_schedule(plan)
    rows=[];decisions=[];context={}
    def seed_factory(b,r,arm,role):
        seed=direct.action_seed(plan.root_seed,b,r,role);context[seed]={'block':b.index,'rotation':r,'role':role,'seed':seed};return seed
    def factory(name,seed):return direct.Probe(registry.make_policy(name,seed),registry.models[name],{**context[seed],'policy':name},decisions.append)
    assert direct.run_schedule(plan,schedule,lambda row,timing:rows.append(row),factory=factory,seed_factory=seed_factory,arms=('candidate',))
    assert len(rows)==32 and all(r['status']=='completed' and sum(r['net_chips'])==0 for r in rows)
    assert {r['role'] for r in decisions}=={0,1}
    assert len({direct.action_seed(plan.root_seed,b,r,p) for b in schedule for r in (0,1) for p in (0,1)})==64
    for row in decisions:
        assert any(c['action']==row['action'] and p>0 for c,p in zip(row['menu'],row['probabilities']))


def test_full_replay_and_exact_reproduction(tmp_path,monkeypatch):
    models=specs(tmp_path)
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    real=direct.subprocess.check_output
    monkeypatch.setattr(direct.subprocess,'check_output',lambda cmd,**kw:'' if cmd[:2]==['git','status'] else real(cmd,**kw))
    registry,_=direct.execute(models,32,2026100922201,'fixture',tmp_path/'play',source)
    result=direct.audit(tmp_path/'play',tmp_path/'audit.json',report=True)
    assert result['hands_replayed']==64 and result['settlement_sum_zero'] and result['all_native_menu_actions']
    registry.models[models[0]['name']].configure_translation(TranslationOptions())
    _,repeat=direct.execute(models,32,2026100922201,'fixture',tmp_path/'repeat',source,registry,tmp_path/'play')
    assert repeat['reproduced_all_hands_and_decisions']
    # Reused registry validation resets translation, without changing policies.
    from pathlib import Path
    Path(models[0]['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError):direct.execute(models,32,2026100922201,'fixture',tmp_path/'corrupt',source,registry)


def test_sorted_buffers_equal_general_loader_and_reject_wrong_order():
    from src.blueprint.compact_policy import CompactBuilder
    left,right=CompactBuilder(),CompactBuilder()
    for i in range(65540):
        key=f'{i:032x}'
        for builder in (left,right):builder.add(key,('check','jam'),(.25,.75),i,i%2)
    normal,flags,counts=left.build()
    sorted_entries,sorted_flags,sorted_counts=right.build_sorted()
    for i in (0,1,65535,65536,65539):
        key=f'{i:032x}'
        assert normal[key]==sorted_entries[key] and counts[key]==sorted_counts[key]
        assert (key in flags)==(key in sorted_flags)
    bad=CompactBuilder()
    for key in ('00000000000000000000000000000001','00000000000000000000000000000000'):
        bad.add(key,('check',),(1.,),0,0)
    with pytest.raises(ValueError,match='strictly sorted'):bad.build_sorted()
