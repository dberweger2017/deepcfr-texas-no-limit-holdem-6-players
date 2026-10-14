"""One-use PR278 M4 training control; science is the unchanged native CLI."""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
from scripts.guard_m4_research import host, sustained
from scripts.guard_hu100_k50_v052 import family, stop as stop_owned
from src.blueprint.equity_buckets import EQUITY_K50_SHA256
from src.policies.files import file_hash

BASE = ROOT / 'results/hu200-v055-confirmation'
BINARY = ROOT / 'native/hu20-trainer/target/release/hu20-trainer'
TABLES = BASE / 'tables'
SOURCE = '9c449929e4b41e8c60383e682bea1e0e1e4c01d5'
CAP = 85_373_862
SEEDS = (2026101421, 2026101422)
GIB = 2**30
RESERVE = 8100


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.new')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path); stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=file_hash(path))


def remaining_seconds(deadline, now):
    return deadline - now - RESERVE


def phase(seed, name, stop_file=None):
    if (BASE/'ABORT.json').exists():
        raise RuntimeError('Resource/failure latch prohibits dependent compute')
    put(BASE/'phase.json', dict(seed=seed, name=name, at=time.time(), stop_file=str(stop_file) if stop_file else None))


def rows(path):
    if not Path(path).exists(): return []
    result=[]
    for line in Path(path).read_text().splitlines():
        try: result.append(json.loads(line))
        except json.JSONDecodeError: break
    return result


def keys(path, count):
    digest=hashlib.sha256(); previous=b''; seen=0; tick=time.time()
    with gzip.open(path,'rb') as stream:
        header=json.loads(next(stream))
        for line in stream:
            if line[:2]!=b'["' or line[34:36]!=b'",': raise ValueError('Unexpected serialized key framing')
            key=line[2:34]
            if len(key)!=32 or key<=previous: raise ValueError('Unsorted or duplicate serialized key')
            digest.update(key); previous=key; seen+=1
    if seen!=count: raise ValueError('Serialized key count mismatch')
    result=dict(path=str(path),entries=seen,sha256=digest.hexdigest(),seconds=time.time()-tick,
                method='SHA256 of concatenated sorted 32-character serialized keys',iteration=header['iteration'])
    put(Path(path).with_suffix('.keys.json'),result)
    return result


def verify(row):
    record=pin(row['path'])
    if (record['sha256'],record['bytes'])!=(row['checkpoint_sha256'],row['checkpoint_bytes']):
        raise ValueError('Checkpoint bytes disagree with native receipt')
    if row['binary_sha256']!=read(BASE/'build-receipt.json')['binary_sha256']:
        raise ValueError('Native executable changed')
    with gzip.open(record['path'],'rt') as stream: h=json.loads(next(stream))
    if h['native_state']['completed_nodes']!=row['completed_nodes'] or h['iteration']!=row['iteration']:
        raise ValueError('Checkpoint/header counters disagree')
    return record


def curve(row):
    d=row['diagnostics']
    return dict(requested_nodes=row['requested_nodes'],actual_nodes=row['completed_nodes'],iterations=row['iteration'],
        entries=d['entries'],status=row['status'],mean_traverser_visits_per_key=d['traverser_visits']/d['entries'],
        zero_visit_share=d['visits_histogram']['0']/d['entries'],zero_average_mass_share=d['zero_average_mass_keys']/d['entries'],
        rss_bytes_after_save=row['process_rss_bytes_after_save'],write_seconds=row['write_seconds'],
        frozen_table=row.get('frozen_table'),checkpoint_bytes=row['checkpoint_bytes'],checkpoint_sha256=row['checkpoint_sha256'])


def execute(seed,name,args,stop_file=None,allowed=(0,),on_save=None,telemetry=None):
    phase(seed,name,stop_file)
    dest=BASE/f'seed-{seed}'/'commands'/name; dest.mkdir(parents=True,exist_ok=False)
    args=list(map(str,args)); tick=time.time()
    put(dest/'intent.json',dict(args=args,started=tick,deadline=read(BASE/'baseline.json')['deadline'],
        export_audit_reserve_seconds=7200,terminal_save_reserve_seconds=900))
    done=[]
    with (dest/'log.txt').open('x') as log:
        child=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        put(dest/'pid.json',dict(pid=child.pid,create_time=psutil.Process(child.pid).create_time()))
        try:
            while True:
                for row in (rows(telemetry)[len(done):] if telemetry else []):
                    if on_save: on_save(row,done)
                    done.append(row)
                rc=child.poll()
                if rc is not None:
                    if telemetry and len(rows(telemetry))>len(done): continue
                    break
                time.sleep(5)
        except BaseException:
            if stop_file and child.poll() is None:
                put(stop_file,dict(reason='verification failure',at=time.time()))
                child.wait()
            raise
    put(dest/'receipt.json',dict(returncode=rc,started=tick,finished=time.time(),wall_seconds=time.time()-tick))
    if rc not in allowed: raise RuntimeError(f'{name} returned {rc}')
    if (BASE/'ABORT.json').exists(): raise RuntimeError('Campaign resource/failure latch after '+name)
    return done


def train_args(seed,target,dest,cap,seconds):
    if seconds<=0: raise ValueError('No training time remains before reserves')
    return [BINARY,'train','--stack-bb',200,'--seed',seed,'--roots-per-seat',1,
        '--average-rule','opponent-sampled','--nodes',target,'--max-entries',cap,
        '--card-buckets',TABLES,'--out',dest/'{nodes}.gz','--telemetry',dest/'telemetry.jsonl',
        '--max-seconds',seconds,'--stop-file',dest/'stop.request']


def prepare_tables(seed):
    phase(seed,'prepare-tables')
    owners=read(BASE/'input-owners.json')
    if any(owners[str(pr)]['state']!='MERGED' for pr in (163,250,265)):
        raise ValueError('Input owners must be merged before copying')
    source=Path.home()/'Local/hu200-k50-first-20261013/results/hu200-k50-first/tables'
    TABLES.mkdir(exist_ok=False)
    result={}
    for street,expected in EQUITY_K50_SHA256.items():
        src=source/f'{street}-k50.bin'; dst=TABLES/src.name
        original=pin(src)
        if original['sha256']!=expected: raise ValueError('Source K50 hash differs')
        shutil.copy2(src,dst); dst.chmod(0o444); copied=pin(dst)
        if copied['sha256']!=expected or copied['bytes']!=original['bytes']: raise ValueError('Copied K50 hash differs')
        result[street]=dict(original=original,copy=copied,mode=oct(dst.stat().st_mode & 0o777),
            retrieval_command=['cp',str(src),str(dst)])
    put(BASE/'inputs.json',dict(owners=owners,tables=result,
        pr163_archive_id='1Gl1JYWN0F7KFJbwoKtcA0zB_rUsWWZ6T',
        pr163_archive_sha256='03d4828f030082f213c8ce82e29fa0fa8fd97288bd8035ad0056de85bc0816ca'))


def worker(seed,build):
    if build:
        execute(seed,'build',[sys.executable,'-m','scripts.build_native_trainer','--receipt',BASE/'build-receipt.json'])
        prepare_tables(seed)
    else:
        if pin(BINARY)['sha256']!=read(BASE/'build-receipt.json')['binary_sha256']: raise ValueError('Binary pin changed')
        for street,sha in EQUITY_K50_SHA256.items():
            if pin(TABLES/f'{street}-k50.bin')['sha256']!=sha: raise ValueError('Table pin changed')
    dest=BASE/f'seed-{seed}'; s1=dest/'stage1'; s1.mkdir(parents=True,exist_ok=False)
    baseline=read(BASE/'baseline.json')
    saves=execute(seed,'stage1-train',[*train_args(seed,3_000_000_000,s1,CAP,remaining_seconds(baseline['deadline'],time.time())),
        '--milestones','500000000,1000000000,1500000000,2000000000,2500000000'],
        stop_file=s1/'stop.request',allowed=(0,3),telemetry=s1/'telemetry.jsonl')
    if not saves: raise ValueError('Stage 1 produced no complete save')
    verified=[dict(telemetry=r,pin=verify(r)) for r in saves]
    terminal=saves[-1]; cap=terminal['diagnostics']['entries']
    phase(seed,'stage1-key-verification')
    digest=keys(terminal['path'],cap)
    cap_reached=cap>=CAP and not terminal['stop_requested']
    parent=dict(status='entry-cap' if cap_reached else 'incomplete',actual_nodes=terminal['completed_nodes'],
        actual_iterations=terminal['iteration'],actual_entries=cap,checkpoint=verified[-1]['pin'],keys=digest,
        saves=verified,nominal_cap=CAP)
    put(dest/'stage1-verified.json',parent)
    if not cap_reached:
        put(dest/'incomplete.json',dict(stage=1,terminal=parent,at=time.time())); return
    frozen=dest/'frozen'; frozen.mkdir(exist_ok=False)
    frozen_verified=[]
    def on_save(row,done):
        record=verify(row)
        if row['diagnostics']['entries']!=cap or row['frozen_table']['state']['entries']!=cap:
            raise ValueError('Frozen table key count changed')
        if row['frozen_table']['state']['since_iteration']!=terminal['iteration']:
            raise ValueError('Freeze lineage differs')
        actual_keys=keys(record['path'],cap)
        if actual_keys['sha256']!=digest['sha256']: raise ValueError('Frozen serialized key list changed')
        frozen_verified.append(dict(telemetry=row,pin=record,keys=actual_keys))
        put(dest/'learning-curve.json',dict(stage1=[curve(r) for r in saves],frozen=[curve(r['telemetry']) for r in frozen_verified]))
        # PR265 permits removal of the earlier 2B/10B only after the successor's
        # full bytes/count/key checks. Stage-1 terminal, 5B and final are retained.
        if done and done[-1]['requested_nodes'] in (2_000_000_000,10_000_000_000):
            old=Path(done[-1]['path'])
            if old.parent!=frozen or old.is_symlink() or pin(old)['sha256']!=done[-1]['checkpoint_sha256']:
                raise ValueError('Uncertain predecessor; retain')
            old.unlink()
            put(old.with_suffix('.removal.json'),dict(path=str(old),sha256=done[-1]['checkpoint_sha256'],
                verified_successor=record,verified_keys=actual_keys,at=time.time(),authorization='PR265 intermediate frozen retention rule'))
    seconds=remaining_seconds(baseline['deadline'],time.time())
    execute(seed,'frozen-train',[*train_args(seed,20_000_000_000,frozen,cap,seconds),
        '--resume',terminal['path'],'--resume-sha256',parent['checkpoint']['sha256'],
        '--freeze-at-cap','--recovery','--menu','v1','--milestones','2000000000,5000000000,10000000000'],
        stop_file=frozen/'stop.request',allowed=(0,3),on_save=on_save,telemetry=frozen/'telemetry.jsonl')
    if not frozen_verified: raise ValueError('No verified frozen terminal')
    final=frozen_verified[-1]; row=final['telemetry']
    complete=row['completed_nodes']>=20_000_000_000
    put(dest/'frozen-complete.json',dict(status='reached-20B' if complete else 'incomplete',final=final,saves=frozen_verified))
    checkpoint=Path(row['path']); current=checkpoint.with_suffix('.current.gz'); average=checkpoint.with_suffix('.average.gz')
    execute(seed,'export-final',[BINARY,'export',checkpoint,'--current',current,'--average',average,'--zero-mass','uniform'])
    execute(seed,'audit-final',[sys.executable,__file__,'audit','--seed',seed])
    put(dest/'science-complete.json',dict(status='reached-20B-every-row-audited' if complete else 'incomplete-every-row-audited',
        seed=seed,finished=time.time(),within_clock=time.time()<=baseline['deadline'],final=final,
        exports=read(checkpoint.with_suffix('.audit.json'))))
    phase(seed,'science-complete')


def audit_final(seed):
    from src.blueprint.abstraction import HU200_EQUITY_SCHEMA
    from src.diagnostics.cfr_average import audit
    dest=BASE/f'seed-{seed}'; final=read(dest/'frozen-complete.json')['final']; checkpoint=Path(final['pin']['path'])
    current=checkpoint.with_suffix('.current.gz'); average=checkpoint.with_suffix('.average.gz')
    start=time.time(); cp=verify(final['telemetry']); avg=pin(average); cur=pin(current)
    spec=dict(name=f'hu200-v055-seed-{seed}',seed=seed,iteration=final['telemetry']['iteration'],checkpoint_sha256=cp['sha256'],sha256=cur['sha256'])
    result=audit(checkpoint,current,average,spec,avg['sha256'],expected_schema=HU200_EQUITY_SCHEMA)
    if result['all_nodes_verified']!=final['telemetry']['diagnostics']['entries']: raise ValueError('Every-row audit incomplete')
    put(checkpoint.with_suffix('.audit.json'),dict(result,actual_nodes=final['telemetry']['completed_nodes'],
        seconds=time.time()-start,keys=final['keys'],files=dict(checkpoint=cp,current=cur,average=avg)))


def supervise(seed,build):
    BASE.mkdir(parents=True,exist_ok=True)
    if subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()!='Apple M4': raise ValueError('M4 only')
    if int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))!=16*GIB: raise ValueError('16 GiB only')
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=SOURCE: raise ValueError('Wrong current-main source')
    held=[]
    for name in ('.hu20-m4-research.lock','.hu100-m4-research.lock'):
        handle=(Path.home()/'Local'/name).open('a+'); fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB); held.append(handle)
    if (BASE/'ABORT.json').exists(): raise RuntimeError('Previous failure; no retry')
    if build:
        if (BASE/'baseline.json').exists(): raise RuntimeError('First build already attempted')
        sample=host(BASE)
        if sample['disk_free_bytes']<45*GIB: raise ValueError('Admission disk below 45 GiB')
        put(BASE/'baseline.json',dict(sample,deadline=sample['at']+43200,source=SOURCE,controller=pin(__file__),
            clock_origin='detached first-build supervisor, immediately before build worker',locks='both exclusive nonblocking flocks held'))
    baseline=read(BASE/'baseline.json')
    if seed==SEEDS[1]:
        first=read(BASE/f'seed-{SEEDS[0]}'/'science-complete.json')
        if first['status']!='reached-20B-every-row-audited': raise ValueError('Seed 1 incomplete; no dependent seed')
        if baseline['deadline']-time.time()<5.5*3600:
            put(BASE/'seed2-not-started.json',dict(reason='less than 5.5 hours remain',at=time.time())); return
    claim=BASE/f'seed-{seed}-claim.json'
    if claim.exists(): raise RuntimeError('Seed already attempted; no retry')
    put(claim,dict(pid=os.getpid(),create_time=psutil.Process().create_time(),at=time.time(),build=build,controller=pin(__file__)))
    known={}; since={}; last=None; child=None; reason=[]; count=peak=0; started=time.time()
    caffeinate=subprocess.Popen(['/usr/bin/caffeinate','-is','-w',str(os.getpid())]); known[caffeinate.pid]=psutil.Process(caffeinate.pid).create_time()
    try:
        with (BASE/f'seed-{seed}-worker.log').open('x') as log,(BASE/f'seed-{seed}-resources.jsonl').open('x') as stream:
            while True:
                sample=host(BASE); readings=[]
                for p in family(known):
                    try: readings.append(dict(pid=p.pid,create_time=p.create_time(),rss=p.memory_info().rss,cpu_seconds=sum(p.cpu_times()[:2])))
                    except (psutil.NoSuchProcess,psutil.ZombieProcess): pass
                sample.update(processes=readings,family_rss_bytes=sum(p['rss'] for p in readings))
                gap=None if last is None else sample['monotonic']-last
                reset=gap is None or gap>10 or gap<0
                if reset: since.clear()
                active,stop=sustained(sample,baseline,since); last=sample['monotonic']
                current=read(BASE/'phase.json') if (BASE/'phase.json').exists() else {}
                if current.get('stop_file') and time.time()>=baseline['deadline']-RESERVE: stop.append('training-wall-time-reserve')
                if time.time()>=baseline['deadline']: stop.append('12-hour-deadline')
                sample.update(phase=current,breaches=active,held_seconds={k:last-v for k,v in since.items()},continuity_reset=reset,stop=stop)
                stream.write(json.dumps(sample,sort_keys=True)+'\n'); stream.flush(); put(BASE/'guard-status.json',sample)
                count+=1; peak=max(peak,sample['family_rss_bytes'])
                if '12-hour-deadline' in stop:
                    put(BASE/'ABORT.json',dict(reasons=stop,at=time.time())); stop_owned(known); raise RuntimeError('Hard 12-hour deadline')
                if stop and set(stop)-set(reason):
                    reason=sorted(set(reason)|set(stop)); put(BASE/'stop-observation.json',dict(reasons=reason,sample=sample))
                    resource=bool(set(reason)-{'training-wall-time-reserve'})
                    if resource: put(BASE/'ABORT.json',dict(reasons=reason,at=time.time()))
                    target=current.get('stop_file')
                    if child is not None and target:
                        if not Path(target).exists(): put(target,dict(reasons=reason,at=time.time()))
                    else:
                        if child is not None: stop_owned(known)
                        raise RuntimeError('Guard stopped outside native training')
                if child is None:
                    args=[sys.executable,__file__,'worker','--seed',str(seed)]+(['--build'] if build else [])
                    child=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT); known[child.pid]=psutil.Process(child.pid).create_time()
                elif child.poll() is not None:
                    if child.returncode: raise RuntimeError('Worker exited '+str(child.returncode))
                    break
                time.sleep(5)
    except BaseException as exc:
        put(BASE/f'seed-{seed}-failure.json',dict(failure=repr(exc),at=time.time()))
        if not (BASE/'ABORT.json').exists(): put(BASE/'ABORT.json',dict(failure=repr(exc),at=time.time()))
        stop_owned(known); raise
    finally:
        if caffeinate.poll() is None and psutil.Process(caffeinate.pid).create_time()==known.get(caffeinate.pid):
            caffeinate.terminate(); caffeinate.wait()
        put(BASE/f'seed-{seed}-supervisor-receipt.json',dict(started=started,finished=time.time(),reasons=reason,
            samples=count,peak_family_rss_bytes=peak,returncode=child.returncode if child else None))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('supervise','worker','audit'))
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--build',action='store_true')
    args=parser.parse_args()
    if args.operation=='audit': audit_final(args.seed)
    elif args.operation=='worker': worker(args.seed,args.build)
    else: supervise(args.seed,args.build)
