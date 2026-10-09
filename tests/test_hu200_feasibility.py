"""Real admission boundaries for the 60-minute M1 pilot."""
from scripts import run_hu200_feasibility as p


def test_actual_m1_guards_keep_fixed_swap_baseline():
    s=dict(pressure=1,free_percent=70,swap_bytes=1_500_000_000,ac=True,disk_free_bytes=30*p.GIB)
    assert p.violation(s,1_500_000_000,1*p.GIB) is None
    assert p.violation({**s,'swap_bytes':2_100_000_000},1_500_000_000,0)=='swap limit'
    assert p.violation({**s,'swap_bytes':3_000_000_001},3_000_000_000,0)=='swap limit'
    assert p.violation({**s,'pressure':2},1_500_000_000,0)=='system pressure'
    assert p.violation({**s,'ac':False},1_500_000_000,0)=='AC power'
    assert p.violation({**s,'disk_free_bytes':p.DISK_FLOOR},1_500_000_000,0)=='disk floor'
    assert p.violation(s,1_500_000_000,p.HARD)=='hard family RSS'


def test_quote_reserves_closeout_and_caps_entries_by_memory():
    last={'diagnostics':{'entries':500_000},'completed_nodes':1_000_000,'write_seconds':2}
    q=p.quote(last,'next',20_000_000,2,12)
    assert q['entry_ceiling']==10_000_000 and q['upper_seconds']>=p.CLOSEOUT
    large=p.quote(last,'next',100_000_000,2,12)
    assert large['forecast_family_bytes']<=p.SOFT
    assert large['entry_ceiling']==(p.SOFT-100_000_000)//110


def guarded_fixture(tmp_path, monkeypatch, remaining, *, kernel_peak=0, failed=False):
    import psutil
    s=dict(pressure=1,free_percent=70,available_bytes=8*p.GIB,swap_bytes=1_500_000_000,ac=True,disk_free_bytes=30*p.GIB,at=1)
    monkeypatch.setattr(p,'host',lambda out:s)
    monkeypatch.setattr(p,'monotonic',lambda:100.)
    g=p.Guard(tmp_path,100.);g.deadline=100+remaining;g.failed=failed
    class Child:
        returncode=0
        def poll(self):return 0
    def spawn(command,**kwargs):
        kwargs['stderr'] # Owned process log includes kernel peak even if sampling misses it.
        kwargs['stdout'].write(f'{kernel_peak} maximum resident set size\n');kwargs['stdout'].flush()
        return Child()
    monkeypatch.setattr(p.subprocess,'Popen',spawn)
    return g


def test_archival_uses_reserved_window_after_science_latch(tmp_path, monkeypatch):
    g=guarded_fixture(tmp_path,monkeypatch,599,failed=True)
    receipt=g.run('archive',['fixture'])
    assert receipt['status']=='complete' and g.failed
    import pytest
    with pytest.raises(RuntimeError,match='Terminal guard latch'):g.run('more-training',['fixture'])


def test_kernel_proven_hard_peak_latches_failure(tmp_path,monkeypatch):
    import json
    import pytest
    g=guarded_fixture(tmp_path,monkeypatch,1800,kernel_peak=5*p.GIB)
    with pytest.raises(RuntimeError,match='Kernel command peak'):g.run('fixture',['fixture'])
    r=json.loads((tmp_path/'operations/fixture/receipt.json').read_text())
    assert r['status']=='failed' and g.failed


def test_kernel_soft_peak_is_returned_for_terminal_stop(tmp_path,monkeypatch):
    g=guarded_fixture(tmp_path,monkeypatch,1800,kernel_peak=int(3.1*p.GIB))
    assert g.run('fixture',['fixture'],train=True)['kernel_soft_limit_exceeded']


def controller_fixture(tmp_path,monkeypatch, *, archive_error=False, export_soft=False):
    import json
    out=tmp_path/'pilot';binary=tmp_path/'binary';binary.write_bytes(b'fixture')
    review=tmp_path/'review.json';review.write_text(json.dumps({'status':'clear','source':'fixture-source'}))
    monkeypatch.setattr(p,'BINARY',binary);monkeypatch.setattr(p,'LOCK',tmp_path/'lock')
    monkeypatch.setattr(p.psutil,'process_iter',lambda *a:[])
    monkeypatch.setattr(p,'monotonic',lambda:100.)
    s=dict(pressure=1,free_percent=70,available_bytes=8*p.GIB,swap_bytes=1_500_000_000,ac=True,disk_free_bytes=30*p.GIB,at=1)
    monkeypatch.setattr(p,'host',lambda out:s)
    def checked(cmd,**kwargs):
        if cmd[:2]==['sysctl','-n']: return 'Apple M1' if cmd[-1]=='machdep.cpu.brand_string' else str(16*p.GIB)
        if cmd[:2]==['git','status']:return ''
        return 'fixture-source'
    monkeypatch.setattr(p.subprocess,'check_output',checked)
    ops=[]
    class Guard:
        def __init__(self,*a):self.failed=False;self.soft_stopped=False;self.swap0=s['swap_bytes'];self.deadline=3700
        def run(self,name,cmd,**kwargs):
            ops.append(name)
            if name.startswith('train'):
                (out/'HU200-1000000.telemetry.jsonl').write_text(json.dumps({'completed_nodes':500000,'stop_requested':True}))
                return {'returncode':0 if export_soft else 3,'seconds':1,'soft_stop_requested':not export_soft,'kernel_soft_limit_exceeded':False}
            if name=='archive' and archive_error:raise RuntimeError('archive fixture failure')
            return {'returncode':0,'seconds':1,'soft_stop_requested':export_soft and name.startswith('export'),'kernel_soft_limit_exceeded':False}
    monkeypatch.setattr(p,'Guard',Guard)
    p.run(out,tmp_path/'archive.zip',review)
    return ops,json.loads((out/'closeout.json').read_text())


def test_soft_stop_permits_required_audit_closeout_but_no_smoke(tmp_path,monkeypatch):
    ops,close=controller_fixture(tmp_path,monkeypatch)
    assert ops==['source-snapshot','train-1000000','export-1000000','audit-1000000','archive']
    assert close['archive_status']=='complete' and close['workers_exited']


def test_archive_failure_still_writes_truthful_closeout(tmp_path,monkeypatch):
    ops,close=controller_fixture(tmp_path,monkeypatch,archive_error=True)
    assert close['archive_status']=='failed' and 'archive fixture failure' in close['archive_failure']


def test_export_soft_stop_blocks_next_training_and_smoke(tmp_path,monkeypatch):
    ops,close=controller_fixture(tmp_path,monkeypatch,export_soft=True)
    assert ops==['source-snapshot','train-1000000','export-1000000','audit-1000000','archive']


def test_kernel_soft_export_latches_new_science_but_permits_preservation(tmp_path,monkeypatch):
    import pytest
    g=guarded_fixture(tmp_path,monkeypatch,1800,kernel_peak=int(3.5*p.GIB))
    assert g.run('export-fixture',['fixture'])['kernel_soft_limit_exceeded']
    with pytest.raises(RuntimeError,match='Terminal guard latch'):g.run('train-next',['fixture'],train=True)
    with pytest.raises(RuntimeError,match='Terminal guard latch'):g.run('smoke',['fixture'])
    assert g.run('audit-fixture',['fixture'])['status']=='complete'
    assert g.run('archive',['fixture'])['status']=='complete'


def test_actual_smoke_worker_replays_and_reproduces_all_hands(tmp_path):
    import json
    from src.blueprint.artifact import save_training
    from src.diagnostics.cfr_average import extract
    from src.blueprint.abstraction import HU200_SCHEMA
    from src.policies.files import file_hash
    from tests.test_native_hu100_preparation import fixture
    _,_,trainer=fixture(200)
    checkpoint=tmp_path/'fixture.gz';average=tmp_path/'average.gz';save_training(trainer,checkpoint)
    extract(checkpoint,dict(seed=123,iteration=2,checkpoint_sha256=file_hash(checkpoint)),average,expected_schema=HU200_SCHEMA)
    p.smoke_worker(tmp_path,average)
    result=json.loads((tmp_path/'smoke.json').read_text())
    assert result['status']=='verified' and result['hands']==320
    assert len((tmp_path/'smoke-hands.jsonl').read_text().splitlines())==320
