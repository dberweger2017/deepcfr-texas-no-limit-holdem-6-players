"""Owner envelope, pressure and retained-source regressions; disposable fixtures only."""
import json
from pathlib import Path
from time import time

import pytest

from scripts import native_hu_followup_limits as limits
from scripts import hu20_scaling_supervise as supervisor
from scripts import compare_native_hu_recovery as recovery


def approval(tmp_path):
    p=tmp_path/'approval.json'
    p.write_bytes(Path('configs/native-hu100-followup-20261008.json').read_bytes())
    return p


@pytest.mark.parametrize('field,value',[('rss_gib',11),('swap_gib',1),('hard_deadline','2026-10-08T11:00:00Z'),
                                       ('body','altered'),('verification_attempts',2)])
def test_followup_requires_exact_approval(tmp_path,field,value):
    p=approval(tmp_path);a=json.loads(p.read_text());a[field]=value;p.write_text(json.dumps(a))
    with pytest.raises(ValueError):limits.envelope(p)


def test_original_envelope_does_not_inherit_new_ceiling(tmp_path):
    assert limits.envelope()['rss_gib']==5.5
    assert limits.envelope(approval(tmp_path))['rss_gib']==10


def test_entire_family_includes_deep_descendants_and_excludes_other_job():
    ps=[(n,n-1,100) for n in range(1,21)]+[(99,0,10000)]
    assert sum(limits.family_rss(ps,{1}))==20*100*1024


@pytest.mark.parametrize('level,percent,admission,unsafe',[(1,81,True,False),(1,70,True,True),
                                                        (2,80,False,True),(1,14,False,True),(1,70,False,False)])
def test_system_pressure_is_independent_of_family_rss(level,percent,admission,unsafe):
    assert limits.unsafe_memory({'pressure_level':level,'free_percent':percent,'physical_bytes':16*1024**3},admission)==unsafe


def test_unsafe_system_pressure_refuses_child_before_admission(monkeypatch,tmp_path):
    from tests.test_native_hu_supervisor import environment
    environment(monkeypatch)
    monkeypatch.setattr(supervisor,'memory_snapshot',lambda:{'pressure_level':2,'free_percent':80,'physical_bytes':16*1024**3})
    monkeypatch.setattr(supervisor.subprocess,'Popen',lambda *_a,**_kw:pytest.fail('unsafe child launched'))
    r=supervisor.run([{'name':'never','command':['unused']}],tmp_path/'guard',time()+20,
                     disk_gib=.001,rss_gib=10,system_memory_guard=True)
    assert r['status']=='incomplete' and 'memory' in r['attempts'][0]['guard_failure']


def test_verifier_revision_can_differ_only_with_original_plan_bound_qualification(monkeypatch,tmp_path):
    from tests.test_native_hu_recovery_equivalence import fixture
    args=fixture(tmp_path,monkeypatch);q=tmp_path/'training-q.json'
    q.write_text(json.dumps({'status':'verified','source':'a'*40,'binary_sha256':'b'*64,
                            'independent_review':'passed','checks':[{'status':'passed'}]}))
    plan=tmp_path/'plan.json';p=json.loads(plan.read_text());p.update(source='a'*40,qualification_sha256=recovery.file_hash(q));plan.write_text(json.dumps(p))
    monkeypatch.setattr(recovery,'isolated_inspect',lambda *_a,**_kw:{'status':'verified'})
    result=recovery.compare(**args,training_qualification=q,isolated_audits=True)
    assert result['executed_training_source']=='a'*40 and result['verifier_source']!='a'*40
    p['qualification_sha256']='changed';plan.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='qualification'):recovery.compare(**args,training_qualification=q,isolated_audits=True)


def test_second_retained_verification_claim_is_permanently_refused(monkeypatch,tmp_path):
    from tests.test_native_hu_launch_claim import fixture,put
    from scripts import verify_native_hu_launch as launch
    plan,state,_=fixture(tmp_path,monkeypatch);a=approval(tmp_path)
    monkeypatch.setattr(launch,'memory_snapshot',lambda:{'pressure_level':1,'free_percent':81,'physical_bytes':16*1024**3})
    p=json.loads(plan.read_text());p.pop('plan_sha256');p.update(stage='verification',limits={'rss_gib':10},
        followup_approval_path=str(a),followup_approval_sha256=launch.file_hash(a));p['plan_sha256']=launch.digest(p);put(plan,p)
    launch.verify(plan);s=json.loads(state.read_text());assert s['verification_attempt_claimed']
    s['active_attempt']=None;put(state,s)
    with pytest.raises(ValueError,match='One approved'):launch.verify(plan)
