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


def verification_fixture(tmp_path,monkeypatch):
    from tests.test_native_hu_launch_claim import fixture,put
    from scripts import verify_native_hu_launch as launch
    plan,state,_=fixture(tmp_path,monkeypatch);a=approval(tmp_path)
    monkeypatch.setattr(launch,'memory_snapshot',lambda:{'pressure_level':1,'free_percent':81,'physical_bytes':16*1024**3})
    inputs={}
    for name in ('reference-02','recovery-02'):
        root=tmp_path/name;root.mkdir();original=root/'plan.json'
        put(original,{'campaign_swap_baseline':'used = 0M'})
        inputs[str(original)]={'bytes':original.stat().st_size,'sha256':launch.file_hash(original)}
    put(state,{'active_attempt':None,'swap_baseline':'used = 0M'})
    p=json.loads(plan.read_text());p.pop('plan_sha256');p.update(stage='verification',limits={'rss_gib':10},
        retained_inputs=inputs,followup_approval_path=str(a),followup_approval_sha256=launch.file_hash(a))
    p['plan_sha256']=launch.digest(p);put(plan,p)
    return plan,state


def test_second_retained_verification_claim_is_permanently_refused(monkeypatch,tmp_path):
    from tests.test_native_hu_launch_claim import put
    from scripts import verify_native_hu_launch as launch
    plan,state=verification_fixture(tmp_path,monkeypatch)
    launch.verify(plan);s=json.loads(state.read_text());assert s['verification_attempt_claimed']
    s['active_attempt']=None;put(state,s)
    with pytest.raises(ValueError,match='One approved'):launch.verify(plan)


def test_verification_claim_race_is_rechecked_under_lock(monkeypatch,tmp_path):
    from tests.test_native_hu_launch_claim import put
    from scripts import verify_native_hu_launch as launch
    plan,state=verification_fixture(tmp_path,monkeypatch)
    original_open=Path.open
    def raced_open(path,*a,**kw):
        if path.name==state.name+'.launch-lock':
            put(state,{'active_attempt':None,'swap_baseline':'used = 0M','verification_attempt_claimed':True})
        return original_open(path,*a,**kw)
    monkeypatch.setattr(Path,'open',raced_open)
    with pytest.raises(ValueError,match='retain lock'):launch.verify(plan)
    assert not (tmp_path/'LAUNCH.json').exists()
    assert state.with_name(state.name+'.launch-lock').exists()


@pytest.mark.parametrize('where',['state','reference'])
def test_reset_baseline_refused_before_claim(monkeypatch,tmp_path,where):
    from tests.test_native_hu_launch_claim import put
    from scripts import verify_native_hu_launch as launch
    plan,state=verification_fixture(tmp_path,monkeypatch)
    if where=='state':put(state,{'active_attempt':None,'swap_baseline':'used = 500M'})
    else:
        original=tmp_path/'reference-02/plan.json';put(original,{'campaign_swap_baseline':'used = 500M'})
        p=json.loads(plan.read_text());p.pop('plan_sha256');p['retained_inputs'][str(original)]={'bytes':original.stat().st_size,'sha256':launch.file_hash(original)}
        p['plan_sha256']=launch.digest(p);put(plan,p)
    with pytest.raises(ValueError,match='baseline'):launch.verify(plan)
    assert not (tmp_path/'LAUNCH.json').exists()


def test_preparation_rejects_reset_baseline_without_creating_attempt(monkeypatch,tmp_path):
    from scripts import prepare_native_hu_followup as prep
    from tests.test_native_hu_launch_claim import put
    import hashlib
    root=tmp_path/'original';root.mkdir();binary=tmp_path/'binary';binary.write_bytes(b'native')
    names=['qualification.json','reference-02/plan.json','recovery-02/plan.json',
           'reference-02/training/HU20-2026100601-1000000000.json.gz',
           'recovery-02/training/HU20-2026100601-1000000000.json.gz','inputs/historical-500M.json.gz']
    names += [f'{phase}/{name}' for phase in ('reference-02','recovery-02') for name in ('checkpoints.jsonl','current.json.gz','average.jsonl.gz')]
    members=[]
    for name in names:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
        if name.endswith('plan.json'):put(p,{'campaign_swap_baseline':'used = 0M'})
        if name=='qualification.json':put(p,{'source':prep.TRAINING_SOURCE,'binary_sha256':prep.file_hash(binary)})
        members.append({'path':'research/'+name,'bytes':p.stat().st_size,'sha256':prep.file_hash(p)})
    manifest={'members':members};put(root/'archive-member-index.json',manifest)
    monkeypatch.setattr(prep,'MANIFEST_SHA',hashlib.sha256(json.dumps(manifest,sort_keys=True,indent=2).encode()).hexdigest())
    monkeypatch.setattr(prep,'qualification',lambda *_:('new-source',{}))
    monkeypatch.setattr(prep,'time',lambda:prep.DEADLINE-2000)
    out=tmp_path/'new-attempt'
    with pytest.raises(ValueError,match='baseline'):prep.prepare_verification(out,binary,tmp_path/'q',root,approval(tmp_path),'used = 500M')
    assert not out.exists()


def audit_environment(monkeypatch,tmp_path,percentages):
    snapshots=iter(percentages)
    monkeypatch.setattr(recovery,'memory_snapshot',lambda:{'pressure_level':1,'free_percent':next(snapshots),'physical_bytes':16*1024**3})
    monkeypatch.setattr(recovery,'system',lambda args:'AC Power' if args[0]=='pmset' else 'used = 0M')
    from types import SimpleNamespace
    monkeypatch.setattr(recovery.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))
    return {'approval':approval(tmp_path),'swap_baseline':'used = 0M','deadline':time()+60}


def test_each_isolated_audit_has_fresh_admission(monkeypatch,tmp_path):
    guard=audit_environment(monkeypatch,tmp_path,[81,70])
    calls=[]
    monkeypatch.setattr(recovery.subprocess,'check_output',lambda *a,**kw:calls.append(a) or '{"status":"verified"}')
    assert recovery.isolated_inspect(tmp_path/'cp',tmp_path/'cur',tmp_path/'avg',20,audit_guard=guard)['status']=='verified'
    with pytest.raises(ValueError,match='admission refused'):
        recovery.isolated_inspect(tmp_path/'cp',tmp_path/'cur',tmp_path/'avg',20,audit_guard=guard)
    assert len(calls)==1


def test_first_isolated_audit_never_spawns_without_headroom(monkeypatch,tmp_path):
    guard=audit_environment(monkeypatch,tmp_path,[70])
    monkeypatch.setattr(recovery.subprocess,'check_output',lambda *_a,**_kw:pytest.fail('unsafe audit child launched'))
    with pytest.raises(ValueError,match='admission refused'):
        recovery.isolated_inspect(tmp_path/'cp',tmp_path/'cur',tmp_path/'avg',20,audit_guard=guard)
