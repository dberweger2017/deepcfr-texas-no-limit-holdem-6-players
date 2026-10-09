"""One fresh, reviewed M1 HU200 feasibility pilot; no campaign or live transport."""
import argparse
from dataclasses import asdict
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from time import monotonic, sleep, time
from zipfile import ZipFile, ZIP_STORED

import psutil

from src.policies.files import file_hash

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT/'native/hu20-trainer/target/release/hu20-trainer'
GIB = 1024**3
SOFT, HARD = 3*GIB, 4*GIB
DISK_FLOOR = int(15.5*GIB)
SEED = 2026100905
ENDPOINTS = (1_000_000, 20_000_000, 100_000_000)
CLOSEOUT = 600
LOCK = Path('/tmp/deepcfr-m1-research.lock')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')


def host(out):
    def call(*args): return subprocess.check_output(args, text=True, timeout=5)
    free = re.search(r'System-wide memory free percentage:\s*(\d+)%', call('memory_pressure','-Q'))
    swap = re.search(r'used = ([\d.]+)([MG])', call('sysctl','vm.swapusage'))
    if free is None or swap is None: raise ValueError('Unreadable host resource sample')
    return dict(at=time(), pressure=int(call('sysctl','-n','kern.memorystatus_vm_pressure_level')),
                free_percent=int(free[1]), available_bytes=psutil.virtual_memory().available,
                swap_bytes=int(float(swap[1])*(1024**2 if swap[2]=='M' else GIB)),
                ac='AC Power' in call('pmset','-g','batt'), disk_free_bytes=__import__('shutil').disk_usage(out).free)


def violation(sample, swap0, rss):
    if rss >= HARD: return 'hard family RSS'
    if sample['pressure'] != 1 or sample['free_percent'] < 15: return 'system pressure'
    if sample['swap_bytes'] > 3_000_000_000 or sample['swap_bytes']-swap0 > 512*1024**2: return 'swap limit'
    if sample['disk_free_bytes'] <= DISK_FLOOR: return 'disk floor'
    if not sample['ac']: return 'AC power'
    return None


def quote(last, operation, target, train_seconds, tool_seconds):
    """Conservatively scale entries linearly with nodes, capped by M1 memory."""
    entries=last['diagnostics']['entries']
    factor=target/last['completed_nodes']
    ceiling=min(math.ceil(entries*factor), math.floor((SOFT-100_000_000)/110))
    factor=ceiling/entries
    memory=110*ceiling+100_000_000
    # Full tools are linear in entries; native training is scaled in additional nodes.
    seconds=2*(train_seconds*max(0,target-last['completed_nodes'])/last['completed_nodes']
               +tool_seconds*factor+last['write_seconds']*factor)+CLOSEOUT
    return dict(target=target, entry_ceiling=min(ceiling, math.floor((SOFT-100_000_000)/110)),
                forecast_family_bytes=memory, upper_seconds=seconds,
                tool_save_reserve_seconds=2*(tool_seconds*factor+last['write_seconds']*factor)+CLOSEOUT,
                disk_required_bytes=ceiling*400+GIB, operation=operation,
                limitation='linear node-to-entry bound capped by memory; capacity stop may precede endpoint')


class Guard:
    def __init__(self, out, started):
        self.out, self.started = out, started
        self.deadline=started+3600
        self.swap0=host(out)['swap_bytes']
        self.failed=False

    def run(self, name, command, *, train=False, allowance=None):
        if self.failed and name != 'archive': raise RuntimeError('Terminal guard latch; no further science')
        admission=host(self.out)
        problem=violation(admission,self.swap0,0)
        if problem: raise RuntimeError('Admission: '+problem)
        if allowance and (allowance['forecast_family_bytes']>SOFT
                          or admission['available_bytes']<allowance['forecast_family_bytes']+GIB
                          or admission['disk_free_bytes']-allowance['disk_required_bytes']<=DISK_FLOOR
                          or self.deadline-monotonic()<allowance['upper_seconds']):
            return None
        if self.deadline-monotonic()<=(5 if name=='archive' else CLOSEOUT): return None
        directory=self.out/'operations'/name;directory.mkdir(parents=True)
        write(directory/'intent.json',dict(command=list(map(str,command)),admission=admission,allowance=allowance))
        stop=self.out/(name+'.stop')
        if train: command=[*command,'--stop-file',str(stop)]
        begun=monotonic();peak=0;failure=None;child=None;soft=False
        try:
            with (directory/'log.txt').open('x') as log, (directory/'resources.jsonl').open('x') as samples:
                child=subprocess.Popen(['/usr/bin/time','-l',*map(str,command)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                parent=psutil.Process(os.getpid())
                while child.poll() is None:
                    rss=0
                    for p in [parent,*parent.children(recursive=True)]:
                        try: rss+=p.memory_info().rss
                        except psutil.NoSuchProcess: pass
                    peak=max(peak,rss);s=host(self.out)
                    samples.write(json.dumps(dict(**s,family_rss_bytes=rss,elapsed_seconds=monotonic()-self.started))+'\n');samples.flush()
                    problem=violation(s,self.swap0,rss)
                    if problem: raise RuntimeError('Guard: '+problem)
                    remaining=self.deadline-monotonic()
                    if remaining<=0: raise TimeoutError('60-minute hard deadline')
                    if train and (rss>=SOFT or remaining<=CLOSEOUT+600):
                        if not stop.exists(): write(stop,dict(reason='soft RSS or save/tool reserve',at=time()))
                        soft=True
                    elif not train and name!='archive' and remaining<=CLOSEOUT:
                        raise TimeoutError('Tool closeout reserve exhausted')
                    sleep(.5)
                if child.returncode not in ((0,3) if train else (0,)):
                    raise RuntimeError(f'{name} exited {child.returncode}')
        except BaseException as exc:
            failure=repr(exc);self.failed=True
            raise
        finally:
            if child is not None and child.poll() is None:
                # Kill only the owned process group; preserve atomic saves/temporary files.
                try: os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=5)
                except (ProcessLookupError,subprocess.TimeoutExpired):
                    try: os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
                    except ProcessLookupError: pass
            text=(directory/'log.txt').read_text() if (directory/'log.txt').exists() else ''
            kernel=re.search(r'(\d+)\s+maximum resident set size',text)
            kernel_peak=int(kernel[1]) if kernel else None
            if kernel_peak is not None and kernel_peak>=HARD:
                failure=failure or 'Kernel command peak exceeded hard family RSS'
                self.failed=True
            receipt=dict(soft_stop_requested=soft,kernel_soft_limit_exceeded=kernel_peak is not None and kernel_peak>=SOFT,status='failed' if failure else 'complete',failure=failure,seconds=monotonic()-begun,
                         peak_family_rss_bytes=peak,kernel_command_peak_rss_bytes=int(kernel[1]) if kernel else None,
                         returncode=child.returncode if child else None,finished=time())
            write(directory/'receipt.json',receipt)
        if failure: raise RuntimeError(failure)
        return receipt


def audit_worker(out, checkpoint):
    from src.blueprint.abstraction import HU200_SCHEMA
    from src.diagnostics.cfr_average import audit
    with gzip.open(checkpoint,'rt') as f: h=json.loads(f.readline())
    current=checkpoint.with_name(checkpoint.name+'.current.gz');average=checkpoint.with_name(checkpoint.name+'.average.gz')
    spec=dict(seed=h['config']['seed'],iteration=h['iteration'],checkpoint_sha256=file_hash(checkpoint),sha256=file_hash(current))
    result=audit(checkpoint,current,average,spec,file_hash(average),expected_schema=HU200_SCHEMA)
    write(checkpoint.with_name(checkpoint.name+'.audit.json'),result)


def smoke_worker(out, checkpoint):
    from collections import Counter
    from statistics import mean, stdev
    from scipy.stats import t
    from src.blueprint.abstraction import HU200_SCHEMA
    from src.blueprint.average import AveragePolicy
    from src.arena.policies import make_policy
    from src.game.hand import Hand, Table
    from src.game.types import Action, ActionKind
    from src.arena.schedule import canonical
    model=AveragePolicy(checkpoint,file_hash(checkpoint),expected_schema=HU200_SCHEMA)
    summaries=[]
    with (out/'smoke-hands.jsonl').open('x') as raw:
        for oi,opponent in enumerate(('random','check_call','tight_aggressive','loose_aggressive','pot_pressure')):
            values=[];coverage=Counter();visits=Counter()
            for block in range(32):
                chips=[]
                for seat in (0,1):
                    table=Table(('player-0','player-1'),(20000,20000),button=block%2)
                    seed=20261009050000+oi*1000+block
                    hand=Hand.start(table,hand_id=f'smoke/{opponent}/{block}/{seat}',seed=seed)
                    player=model.policy(seed+100000+seat);other=make_policy(opponent,seed+200000+seat)
                    rows=[]
                    for _ in range(1000):
                        if hand.finished: break
                        view=hand.observe(hand.actor)
                        action=(player if hand.actor==seat else other).choose_action(view)
                        view.legal_actions.validate(action)
                        if hand.actor==seat:
                            menu,p,known,info=model.distribution_with_telemetry(view)
                            coverage[(view.street.value,info['reason'])]+=1
                            if known:
                                v=model.visits[info['exact_key']];visits[(view.street.value,'0' if v==0 else '1-9' if v<10 else '10+')]+=1
                        rows.append([hand.actor,action.kind.value,action.raise_to]);hand=hand.apply(action)
                    if not hand.finished: raise RuntimeError('Smoke hand decision bound')
                    final=hand.events[-1].stacks
                    if sum(final)!=40000: raise ValueError('Smoke settlement conservation')
                    # Independent action replay and exact policy reproduction on the fixed seed.
                    replay=Hand.start(table,hand_id=hand.observe(0).hand_id,seed=seed)
                    again=model.policy(seed+100000+seat);again_other=make_policy(opponent,seed+200000+seat)
                    for actor,kind,target in rows:
                        if replay.actor!=actor: raise ValueError('Replay actor')
                        action=Action(ActionKind(kind),target)
                        if (again if actor==seat else again_other).choose_action(replay.observe(actor))!=action:
                            raise ValueError('Smoke policy reproduction')
                        replay=replay.apply(action)
                    if canonical(replay.events)!=canonical(hand.events): raise ValueError('Smoke event/settlement replay')
                    raw.write(json.dumps(dict(opponent=opponent,block=block,seat=seat,seed=seed,actions=rows,final_stacks=final))+'\n')
                    chips.append(final[seat]-20000)
                values.append(sum(chips)/2) # With 100-chip BB, chips/hand numerically equals BB/100.
            margin=float(t.ppf(.975,31))*stdev(values)/math.sqrt(32)
            summaries.append(dict(opponent=opponent,hands=64,bb_per_100=mean(values),descriptive_95_interval=[mean(values)-margin,mean(values)+margin],
                                  coverage={str(k):v for k,v in coverage.items()},known_visit_bands={str(k):v for k,v in visits.items()}))
    write(out/'smoke.json',dict(status='verified',hands=320,scope='feasibility smoke, no strength qualification',opponents=summaries,
                              model_sha256=file_hash(checkpoint),replay='all actions, events and settlements',reproduction='all selected actions'))


def archive_worker(out, destination):
    # Own open-PR evidence is explicitly authorized for archival by this task.
    pr=json.loads(subprocess.check_output(['gh','pr','view','--json','number,state,url'],cwd=ROOT,text=True))
    if pr['state']!='OPEN': raise ValueError('Expected this pilot owning open PR')
    write(out/'owning-pr-before-archive.json',pr)
    paths={str(p.relative_to(out)):p for p in out.rglob('*') if p.is_file() and 'operations/archive' not in str(p.relative_to(out))}
    paths['runtime/hu20-trainer']=BINARY
    manifest=dict(source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),owning_pr=pr,
                  original_root=str(out),remote_bytes_downloaded=False,members=[])
    for name,p in sorted(paths.items()):
        s=p.stat();manifest['members'].append(dict(path=name,bytes=s.st_size,sha256=file_hash(p),original_path=str(p),mtime_ns=s.st_mtime_ns))
    if host(out)['disk_free_bytes']-sum(m['bytes'] for m in manifest['members'])-GIB<=DISK_FLOOR: raise ValueError('Archive disk admission')
    encoded=json.dumps(manifest,sort_keys=True).encode()
    with ZipFile(destination,'x',compression=ZIP_STORED,allowZip64=True) as z:
        for m in manifest['members']:
            p=paths[m['path']];s=p.stat()
            if s.st_size!=m['bytes'] or s.st_mtime_ns!=m['mtime_ns']: raise ValueError('Archive source changed')
            z.write(p,m['path'])
        z.writestr('ARCHIVE-MANIFEST.json',encoded)
    with ZipFile(destination) as z:
        for m in manifest['members']:
            digest=hashlib.sha256();size=0
            with z.open(m['path']) as f:
                for chunk in iter(lambda:f.read(1024**2),b''): digest.update(chunk);size+=len(chunk)
            if size!=m['bytes'] or digest.hexdigest()!=m['sha256']: raise ValueError('Archive readback')
    write(out/'archive.json',dict(path=str(destination),bytes=destination.stat().st_size,sha256=file_hash(destination),
                                manifest_path='ARCHIVE-MANIFEST.json',manifest_sha256=hashlib.sha256(encoded).hexdigest(),
                                verified_members=len(paths),members=manifest['members'],remote_bytes_downloaded=False))


def run(out, destination, review):
    out.mkdir(parents=True,exist_ok=False)
    if subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()!='Apple M1': raise ValueError('M1 only')
    if int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))!=16*GIB: raise ValueError('16 GiB M1 only')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip(): raise ValueError('Commit reviewed source before pilot')
    reviewed=json.loads(review.read_text());source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if reviewed['status']!='clear' or reviewed['source']!=source: raise ValueError('Independent correctness review must bind exact source')
    with LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        inventory=[]
        for p in psutil.process_iter(['pid','cmdline']):
            cmd=' '.join(p.info['cmdline'] or [])
            if p.pid!=os.getpid() and re.search(r'hu20-trainer (train|bench-train)|scripts\.(run_hu100|run_native_hu|train_)',cmd):
                inventory.append(dict(pid=p.pid,command=cmd))
        if inventory: write(out/'ownership-refusal.json',inventory);raise ValueError('Competing research worker')
        started=monotonic();g=Guard(out,started)
        initial=host(out)
        if initial['available_bytes']<4*GIB or violation(initial,g.swap0,0): raise ValueError('M1 admission requires 4 GiB available and normal guards')
        write(out/'baseline.json',dict(source=source,binary_sha256=file_hash(BINARY),review_sha256=file_hash(review),host=initial,
                                      seed=SEED,endpoints=ENDPOINTS,started=time(),hard_seconds=3600,closeout_reserve=CLOSEOUT,
                                      soft_family_bytes=SOFT,hard_family_bytes=HARD,ownership_inventory=inventory,lock=str(LOCK)))
        records=[];stop_reason=None
        try:
            write(out/'independent-review.json',reviewed)
            g.run('source-snapshot',['git','archive','--format=tar','--output',out/'source.tar','HEAD',
                  'native/hu20-trainer','native/hu20-buckets','src/blueprint','src/game','src/arena',
                  'src/policies/files.py','src/diagnostics/cfr_average.py','scripts/run_hu200_feasibility.py',
                  'docs/hu200-feasibility.md','requirements-dev.txt','requirements.txt'])
            allowance=None;previous=None;previous_actual=0
            for target in ENDPOINTS:
                checkpoint=out/f'HU200-{target}.gz';telemetry=out/f'HU200-{target}.telemetry.jsonl'
                cmd=[BINARY,'train','--stack-bb','200','--seed',str(SEED),'--average-rule','opponent-sampled','--nodes',str(target),
                     '--out',checkpoint,'--telemetry',telemetry,'--max-entries',str(allowance['entry_ceiling'] if allowance else 1_000_000),
                     '--max-seconds',str(max(1,g.deadline-monotonic()-max(CLOSEOUT+600,allowance['tool_save_reserve_seconds'] if allowance else 0)))]
                if previous:cmd += ['--resume',previous,'--resume-sha256',file_hash(previous)]
                train=g.run(f'train-{target}',cmd,train=True,allowance=allowance)
                if train is None:stop_reason='measured admission refused';break
                row=json.loads(telemetry.read_text().splitlines()[-1])
                export=g.run(f'export-{target}',[BINARY,'export',checkpoint,'--current',str(checkpoint)+'.current.gz','--average',str(checkpoint)+'.average.gz','--zero-mass','uniform'])
                if export is None:stop_reason='export reserve exhausted';break
                audit=g.run(f'audit-{target}',[sys.executable,'-m','scripts.run_hu200_feasibility','audit','--out',out,'--checkpoint',checkpoint])
                if audit is None:stop_reason='audit reserve exhausted';break
                tools=export['seconds']+audit['seconds']
                records.append(dict(target=target,telemetry=row,train=train,export=export,audit=audit))
                previous=checkpoint
                if train['returncode']==3 or train['soft_stop_requested'] or train['kernel_soft_limit_exceeded']:
                    stop_reason='native capacity/time/soft stop';break
                if target!=ENDPOINTS[-1]:
                    next_target=ENDPOINTS[ENDPOINTS.index(target)+1]
                    # Scale the recent training segment to the table's total node count.
                    train_cost=max(0,train['seconds']-row['write_seconds'])*row['completed_nodes']/(row['completed_nodes']-previous_actual)
                    allowance=quote(row,'next-endpoint',next_target,train_cost,tools)
                    write(out/f'admission-{next_target}.json',allowance)
                previous_actual=row['completed_nodes']
            if previous and stop_reason is None and not g.failed and g.deadline-monotonic()>CLOSEOUT+180:
                g.run('smoke',[sys.executable,'-m','scripts.run_hu200_feasibility','smoke','--out',out,'--checkpoint',str(previous)+'.average.gz'])
        except BaseException as exc:
            stop_reason=repr(exc)
        write(out/'science.json',dict(source=source,records=records,stop_reason=stop_reason,seconds=monotonic()-started,
                                     status='partial' if stop_reason else 'complete',no_live_match=True))
        archive_status='not-started';archive_failure=None
        try:
            archived=g.run('archive',[sys.executable,'-m','scripts.run_hu200_feasibility','archive','--out',out,'--destination',destination])
            archive_status='complete' if archived is not None else 'time-refused'
        except BaseException as exc:
            archive_status='failed';archive_failure=repr(exc)
        finally:
            write(out/'closeout.json',dict(seconds=monotonic()-started,within_cap=monotonic()-started<=3600,
                  guard_failed=g.failed,workers_exited=True,originals_retained=True,
                  archive_status=archive_status,archive_failure=archive_failure))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('run','audit','smoke','archive'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--destination',type=Path);p.add_argument('--review',type=Path)
    a=p.parse_args()
    if a.mode=='run':run(a.out,a.destination,a.review)
    elif a.mode=='audit':audit_worker(a.out,a.checkpoint)
    elif a.mode=='smoke':smoke_worker(a.out,a.checkpoint)
    else:archive_worker(a.out,a.destination)


if __name__=='__main__':main()
