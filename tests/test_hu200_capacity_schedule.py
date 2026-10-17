"""Queue dependencies, shared reservations, global stops and byte equivalence."""
from concurrent.futures import ThreadPoolExecutor
import os
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts import schedule_hu200_capacity as schedule


def test_cpu_reservations_admit_two_helpers_but_never_nine_cpus():
    active=[]
    for seed in schedule.SEEDS:
        cpus=schedule.allocate(active,'train',seed)
        assert len(cpus)==2
        active.append(dict(cpus=cpus))
    assert schedule.allocate(active,'audit')==[6]
    active.append(dict(cpus=[6]))
    assert schedule.allocate(active,'audit')==[7]
    active.append(dict(cpus=[7]))
    assert schedule.allocate(active,'audit') is None
    assert schedule.allocate(active,'train',schedule.SEEDS[0]) is None


def test_disk_reservations_include_all_writers_and_unchanged_plan():
    assert schedule.fits(100_000_000_000,200_000_000_000,46_000_000_000,23_000_000_000)
    assert not schedule.fits(150_000_000_000,200_000_000_000,46_000_000_000,23_000_000_000)
    assert not schedule.fits(10_000_000_000,80_000_000_000,23_000_000_000,20_000_000_000)


@pytest.mark.skipif(not (schedule.QUALIFIED/'trainer-first').exists(),reason='Qualified bb128 source pins')
def test_continuation_initialization_preserves_absolute_deadline(monkeypatch,tmp_path):
    committed=Path('docs/reports/hu200-capacity-cygnus-artifacts')
    for name in ('admission.json','training-clock.json'):
        (tmp_path/name).write_bytes((committed/name).read_bytes())
    original=(tmp_path/'training-clock.json').read_bytes()
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    schedule.put(tmp_path/'scheduling-handoff.json',dict(new_source=source,
        original_clock=schedule.pin(tmp_path/'training-clock.json')))
    (tmp_path/'guards').mkdir()
    monkeypatch.setattr(schedule.Queue,'build',lambda self:None)
    q=schedule.Queue(tmp_path)
    assert q.deadline==schedule.read(committed/'training-clock.json')['deadline_epoch']
    assert (tmp_path/'training-clock.json').read_bytes()==original
    assert q.audit_limit==2


def test_seed_waits_for_own_audit_cleanup_and_all_inputs(monkeypatch,tmp_path):
    q=schedule.Queue.__new__(schedule.Queue)
    q.base,q.tasks,q.done=tmp_path,{},set()
    monkeypatch.setattr(schedule,'checked_audit',lambda p:{'native':{}})
    monkeypatch.setattr(schedule,'native_record',lambda *a:{})
    q.build()
    seed=schedule.SEEDS[0]
    first=q.tasks[f'train-{seed}-2000000000']
    second=q.tasks[f'train-{seed}-5000000000']
    assert not q.ready(first)
    q.done.add('prefix-finalize')
    assert q.ready(first)
    q.done.add(f'audit-{seed}-2000000000')
    assert not q.ready(second)
    q.done.add(f'retire-{seed}-2000000000')
    assert not q.ready(second)
    q.done.add('restore')
    assert q.ready(second)
    other=q.tasks[f'train-{schedule.SEEDS[1]}-5000000000']
    assert not q.ready(other)


@pytest.mark.parametrize('field,value,reason',[
    ('package_c',85,'CPU'),('available_bytes',1,'RAM'),
    ('swap_used_bytes',300*1024**2,'Swap'),('free_disk_bytes',49_999_999_999,'Disk')])
def test_resource_thresholds_do_not_relax(field,value,reason):
    row=dict(package_c=40,available_bytes=100*schedule.GIB,swap_used_bytes=0,free_disk_bytes=200_000_000_000)
    row[field]=value
    assert reason in schedule.resource_breach(row,{},[],0,1,100)
    row[field]=dict(package_c=40,available_bytes=100*schedule.GIB,swap_used_bytes=0,free_disk_bytes=200_000_000_000)[field]
    assert schedule.resource_breach(row,{},[],0,100,100)=='Original wall-time cap'


def test_helper_rss_and_combined_rss_are_both_guarded():
    row=dict(package_c=40,available_bytes=100*schedule.GIB,swap_used_bytes=0,free_disk_bytes=200_000_000_000)
    assert schedule.resource_breach(row,{'audit':schedule.GIB},[dict(name='audit',rss_gib=.5)],0,1,100)=='Per-job family RSS cap'
    assert schedule.resource_breach(row,{'train':117*schedule.GIB},[dict(name='train',rss_gib=200)],0,1,100)=='Aggregate family RSS cap'


@pytest.mark.skipif(not Path('/proc/self/stat').exists(),reason='Linux administrative process handoff')
def test_handoff_waits_for_all_children_and_keeps_original_clock(tmp_path):
    import sys
    import time
    folder=tmp_path/'results/2026101302/twin500m'
    folder.mkdir(parents=True)
    receipt=folder/'audit.json'
    receipt.write_text(json.dumps(dict(status='verified',audit=dict(status='verified',all_nodes_verified=1),
                                      native=dict(diagnostics=dict(entries=1)),files={})))
    guard=tmp_path/'guards/twin500m-audit-2026101302'
    guard.mkdir(parents=True)
    (guard/'plan.json').write_text(json.dumps(dict(jobs=[dict(command=['audit','--base',str(folder)])])))
    (guard/'resources.jsonl').write_text('{"monotonic":10}\n{"monotonic":11}\n')
    clock=tmp_path/'training-clock.json'
    clock.write_text(json.dumps(dict(deadline_epoch=time.time()+30,start_utc='unchanged')))
    original=clock.read_bytes()
    # The idle fixture parent deliberately leaves both successful children as
    # zombies; a live second child must prevent early administrative retirement.
    code='import subprocess,time,sys; subprocess.Popen([sys.executable,"-c","pass"]); subprocess.Popen([sys.executable,"-c","import time;time.sleep(1)"]);time.sleep(20)'
    parent=subprocess.Popen([sys.executable,'-c',code,'scripts.run_hu200_capacity_training','train',str(tmp_path)])
    started=time.monotonic()
    try:
        schedule.handoff(tmp_path,parent.pid)
        assert time.monotonic()-started>=.9
        assert parent.wait(timeout=2)==-9
        assert clock.read_bytes()==original
        checked=schedule.read(tmp_path/'scheduling-handoff.json')
        assert checked['status']=='successful-audit-boundary'
        assert checked['original_clock']['sha256']==schedule.file_hash(clock)
        assert checked['completed_guard_elapsed_seconds']==1
    finally:
        if parent.poll() is None:parent.kill();parent.wait()


@pytest.mark.parametrize('mode',['cpu','vrm','audit-failure'])
def test_one_guard_broadcasts_to_trainers_and_audits_before_termination(monkeypatch,tmp_path,mode):
    q=schedule.Queue.__new__(schedule.Queue)
    q.base=tmp_path
    q.root=tmp_path/'guard'
    q.root.mkdir()
    q.deadline=10000
    q.admission=dict(swap_baseline_bytes=0)
    q.done=set()
    q.tasks={}
    q.active={}
    q.next_counter=float('inf')
    tick=[0]
    events=[]
    for i,kind in enumerate(['train','train','audit','audit']):
        name=f'{kind}{i}'
        stop=tmp_path/(name+'.stop')
        class Process:
            def __init__(self,pid,stop):self.pid,self.stop,self.code=pid,stop,None
            def poll(self):
                if mode=='audit-failure' and self.pid==103:return 1
                return 3 if self.stop.exists() else self.code
        proc=Process(100+i,stop)
        q.tasks[name]=dict(name=name,kind=kind,deps={'never'})
        q.active[name]=dict(name=name,kind=kind,proc=proc,identity='pin',cpus=[i],rss_gib=1,
                            log=(tmp_path/(name+'.log')).open('w'),stop_file=str(stop))
    monkeypatch.setattr(schedule,'motherboard_sensors',lambda:(tmp_path/'vrm',tmp_path/'fan'))
    monkeypatch.setattr(schedule.sensors,'package_sensor',lambda:tmp_path/'cpu')
    monkeypatch.setattr(schedule.sensors,'sample',lambda p:dict(package_c=85 if mode=='cpu' else 40,
        available_bytes=100*schedule.GIB,swap_used_bytes=0))
    monkeypatch.setattr(schedule.sensors,'read_int',lambda p:700 if p.name=='fan' else 95000 if mode=='vrm' else 60000)
    monkeypatch.setattr(schedule.sensors,'group_rss',lambda p:1024)
    monkeypatch.setattr(schedule,'owned_bytes',lambda p:0)
    monkeypatch.setattr(schedule.shutil,'disk_usage',lambda p:SimpleNamespace(free=200_000_000_000))
    monkeypatch.setattr(schedule.time,'time',lambda:tick[0])
    monkeypatch.setattr(schedule.time,'monotonic',lambda:tick[0])
    monkeypatch.setattr(schedule.time,'sleep',lambda seconds:tick.__setitem__(0,tick[0]+seconds))
    monkeypatch.setattr(schedule,'process',lambda pid:dict(start='pin'))
    def kill(pid,sig):
        assert all(Path(j['stop_file']).exists() for j in q.active.values())
        events.append(('signal',pid))
    monkeypatch.setattr(schedule.os,'killpg',kill)
    monkeypatch.setattr(schedule.sensors,'terminate_group',lambda proc:events.append(('terminate',proc.pid)))
    q.decide_batch=lambda *a:None
    q.publish=lambda:None
    q.reservation=lambda job:0
    with pytest.raises(RuntimeError,match='CPU|VRM|exit1|exit 1'):
        q.run()
    assert [e[0] for e in events[:4]]==['signal']*4
    assert {e[1] for e in events[:4]}=={100,101,102,103}
    assert schedule.read(tmp_path/'training-failed.json')['automatic_retry'] is False
    assert not (tmp_path/'training-clock.json').exists()
    if mode=='vrm':assert tick[0]==30


@pytest.mark.skipif(not hasattr(os,'uname') or os.uname().nodename!='cygnus-x1',
                    reason='Qualified bb128 CPU reservations and50GB disk floor')
def test_real_queue_overlaps_only_independent_jobs_and_pins_cpu_limits(monkeypatch,tmp_path):
    import sys
    from types import SimpleNamespace
    # This process-overlap fixture writes only small metadata. Resource-budget
    # behavior has separate tests; host disk pressure must not disable its jobs.
    monkeypatch.setattr(schedule.shutil,'disk_usage',lambda p:SimpleNamespace(free=200_000_000_000))
    q=schedule.Queue.__new__(schedule.Queue)
    q.base,q.root=tmp_path,tmp_path/'guard'
    q.root.mkdir()
    q.deadline=schedule.time.time()+30
    q.admission=dict(swap_baseline_bytes=0)
    q.active,q.done,q.tasks={},set(),{}
    q.audit_limit,q.batch_decided,q.batch,q.batch_started=2,True,[],None
    q.next_counter=float('inf')
    for seed in schedule.SEEDS[:2]:
        q.task(f'train{seed}','train',[],seed=seed,label='fixture')
        q.task(f'audit{seed}','audit',[f'train{seed}'],seed=seed,label='fixture')
    def command(task):
        folder=q.folder(task);folder.mkdir(parents=True,exist_ok=True)
        seconds=2.2 if task['name']==f'train{schedule.SEEDS[1]}' else .1
        path=folder/(task['kind']+'-env.json')
        code='import json,os,time;from pathlib import Path;Path('+repr(str(path))+').write_text(json.dumps(dict(cpus=sorted(os.sched_getaffinity(0)),threads=os.environ["RAYON_NUM_THREADS"])));time.sleep('+str(seconds)+')'
        return [sys.executable,'-c',code]
    q.command=command
    q.publish=lambda:None
    q.decide_batch=lambda *a:None
    monkeypatch.setattr(schedule,'native_record',lambda *a:{})
    monkeypatch.setattr(schedule,'checked_audit',lambda *a:{})
    monkeypatch.setattr(schedule,'motherboard_sensors',lambda:(tmp_path/'vrm',tmp_path/'fan'))
    monkeypatch.setattr(schedule.sensors,'package_sensor',lambda:tmp_path/'cpu')
    monkeypatch.setattr(schedule.sensors,'sample',lambda p:dict(package_c=40,available_bytes=100*schedule.GIB,swap_used_bytes=0))
    monkeypatch.setattr(schedule.sensors,'read_int',lambda p:700 if p.name=='fan' else 60000)
    for target in schedule.TARGETS:schedule.put(tmp_path/f'milestone-{target}.json',{})
    schedule.put(tmp_path/'scheduling-handoff.json',dict(new_source='fixture-only'))
    q.run()
    assert len(q.done)==4
    assert schedule.read(q.root/'result.json')['status']=='passed'
    rows=[json.loads(line) for line in (q.root/'resources.jsonl').read_text().splitlines()]
    assert any(any(j['kind']=='train' for j in r['active_jobs'].values()) and
               any(j['kind']=='audit' for j in r['active_jobs'].values()) for r in rows)
    for r in rows:
        cores=[c for j in r['active_jobs'].values() for c in j['cpus']]
        assert len(cores)==len(set(cores)) and len(cores)<=8
    for seed in schedule.SEEDS[:2]:
        folder=tmp_path/'results'/str(seed)/'fixture'
        assert schedule.read(folder/'train-env.json')['threads']=='2'
        assert schedule.read(folder/'audit-env.json')['threads']=='1'


@pytest.mark.skipif(not (schedule.QUALIFIED/'trainer-first').exists(),reason='Qualified bb128 binary only')
def test_parallel_native_exports_and_audits_equal_serial_bytes(tmp_path):
    from tests.diagnostics.test_cfr_average import fixture
    from scripts.audit_hu200_capacity_sorted import audit_sorted
    from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
    from src.policies.files import file_hash
    source=tmp_path/'fixture'
    source.mkdir()
    _,_,checkpoint,_,spec=fixture(source)
    def check(name,threads):
        folder=tmp_path/name
        folder.mkdir()
        current,average=folder/'current.gz',folder/'average.gz'
        env=dict(os.environ,RAYON_NUM_THREADS=str(threads),OMP_NUM_THREADS=str(threads))
        subprocess.run([str(schedule.QUALIFIED/'trainer-first'),'export',str(checkpoint),
                        '--current',str(current),'--average',str(average),'--zero-mass','uniform'],
                       env=env,check=True,capture_output=True)
        checked=audit_sorted(checkpoint,current,average,{**spec,'sha256':file_hash(current)},
            file_hash(average),expected_schema=HU20_UNCAPPED_SCHEMA)
        return current.read_bytes(),average.read_bytes(),checked
    original=check('serial',2)
    with ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(check,name,1) for name in ('parallel-a','parallel-b')]
        assert all(f.result()==original for f in futures)
