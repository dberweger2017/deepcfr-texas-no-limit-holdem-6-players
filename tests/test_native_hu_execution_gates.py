"""Receipt/growth admission fixtures; no campaign execution or external services."""
import json
from types import SimpleNamespace
import pytest
from scripts import prepare_native_hu_execution as execution
from scripts.prepare_native_hu_campaign import file_hash


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value)+'\n')


def test_pilot_refuses_recovery_mismatch_before_preparing_any_commands(monkeypatch,tmp_path):
    binary=tmp_path/'binary'; binary.write_bytes(b'fixture')
    monkeypatch.setattr(execution,'qualification',lambda *_a:('source',{}))
    eq=tmp_path/'equivalence.json'; put(eq,{'status':'mismatch'})
    out=tmp_path/'pilot'
    with pytest.raises(ValueError,match='equivalence'):
        execution.prepare_pilot(out,binary,tmp_path/'qualification',eq,'used = 0M',execution.DEADLINE)
    assert not out.exists()


def test_equivalence_cannot_reset_campaign_swap_baseline(tmp_path):
    binary=tmp_path/'binary'; binary.write_bytes(b'fixture')
    eq=tmp_path/'equivalence.json'
    put(eq,{'status':'verified','seed':execution.SEED,'source':'source','binary_sha256':file_hash(binary),
            'campaign_swap_baseline':'used = 0M','files':{}})
    with pytest.raises(ValueError,match='baseline'): execution.checked_equivalence(eq,'source',binary,'used = 512M')
    binary.write_bytes(b'replaced binary')
    with pytest.raises(ValueError): execution.checked_equivalence(eq,'source',binary,'used = 0M')


@pytest.mark.parametrize('phase',['training','export-audit'])
def test_growth_refuses_incomplete_pilot_guard(monkeypatch,tmp_path,phase):
    binary=tmp_path/'binary'; binary.write_bytes(b'fixture'); eq=tmp_path/'equivalence.json'; put(eq,{})
    pilot=tmp_path/'pilot'; pilot.mkdir()
    put(pilot/'plan.json',{'stage':'pilot','source':'source','binary_sha256':file_hash(binary),
        'campaign_swap_baseline':'used = 0M','equivalence_sha256':file_hash(eq)})
    put(pilot/'audit-10000000.json',{'status':'verified','stack_bb':100,
        'native_state':{'completed_nodes':10000001,'coverage_start':[0,0,0],'traverser_visits_by_street':[1,1,1,1]}})
    for name in ('training','export-audit'): put(pilot/f'{name}-guard/campaign.json',{'status':'incomplete' if name==phase else 'complete'})
    monkeypatch.setattr(execution,'qualification',lambda *_a:('source',{}))
    monkeypatch.setattr(execution,'checked_equivalence',lambda *_a:{'files':{}})
    out=tmp_path/'growth'
    with pytest.raises(ValueError,match='guards'):
        execution.prepare_growth(out,binary,tmp_path/'q',eq,pilot,'used = 0M',execution.DEADLINE,tmp_path/'capacity')
    assert not out.exists()


def capacity_fixture(tmp_path,monkeypatch):
    pilot=tmp_path/'pilot'; pilot.mkdir()
    (pilot/'checkpoints.jsonl').write_text(json.dumps({'diagnostics':{'entries':1000},'write_seconds':1.0,'checkpoint_bytes':100000})+'\n')
    put(pilot/'audit-10000000.json',{'status':'verified'})
    put(pilot/'export-audit-guard/campaign.json',{'attempts':[{'started':0,'finished':2}]})
    monkeypatch.setattr('shutil.disk_usage',lambda *_a:SimpleNamespace(free=100*1024**3))
    return pilot,{'pilot_telemetry_sha256':file_hash(pilot/'checkpoints.jsonl'),
        'pilot_audit_sha256':file_hash(pilot/'audit-10000000.json'),'measurement_files':{},
        'forecast_entry_ceiling':2000,'save_reserve_seconds':4,'export_audit_reserve_seconds':8,
        'archive_reserve_seconds':44,'serialization_rss_bytes':32*2000+64*1024**2,
        'disk_reserve_bytes':400000*23+1000000,'retained_non_growth_bytes':1000000}


def test_capacity_arithmetic_admits_reserved_window(monkeypatch,tmp_path):
    pilot,c=capacity_fixture(tmp_path,monkeypatch); execution.validate_capacity(c,pilot,execution.DEADLINE-52)


@pytest.mark.parametrize('field,value',[
    ('save_reserve_seconds',3.9),('export_audit_reserve_seconds',7.9),('archive_reserve_seconds',43.9),
    ('serialization_rss_bytes',1),('disk_reserve_bytes',1),('forecast_entry_ceiling',1000)])
def test_capacity_rejects_underreserved_costs(monkeypatch,tmp_path,field,value):
    pilot,c=capacity_fixture(tmp_path,monkeypatch); c[field]=value
    with pytest.raises(ValueError): execution.validate_capacity(c,pilot,execution.DEADLINE-52)


def test_capacity_rejects_consuming_closeout_deadline(monkeypatch,tmp_path):
    pilot,c=capacity_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError): execution.validate_capacity(c,pilot,execution.DEADLINE-51)
