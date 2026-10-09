"""One M1-only translation comparison with outcome-blind timing admission."""
import argparse
from dataclasses import asdict
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
from time import monotonic, sleep, time
from zipfile import ZipFile, ZIP_DEFLATED

import psutil

from scripts.evaluate_hu200_diagnosis import OPPONENTS, seed
from scripts.run_hu200_diagnosis import MODELS, BINARY, BINARY_SHA
from scripts.run_hu200_feasibility import host, write, GIB, LOCK, SOFT, HARD
from src.arena.schedule import digest
from src.blueprint.action_translation import TranslationOptions
from src.blueprint.average import average_rule, zero_mass_rule
from src.policies.files import file_hash

ROOT=Path(__file__).resolve().parents[1]
FINAL_ROOT=2026100908
TIMING_ROOT=2026100909
DEFAULT_DISK_FLOOR=int(15.5*GIB)


def problem(sample,swap0,rss,disk_floor):
    if rss>=HARD:return 'hard family RSS'
    if rss>=SOFT:return 'soft family RSS'
    if sample['pressure']!=1 or sample['free_percent']<15:return 'system pressure'
    if sample['swap_bytes']>3_000_000_000 or sample['swap_bytes']-swap0>512*1024**2:return 'swap limit'
    if sample['disk_free_bytes']<=disk_floor:return 'disk floor'
    if not sample['ac']:return 'AC power'
    return None


class Guard:
    def __init__(self,out,started,initial,disk_floor):
        self.out=out;self.started=started;self.deadline=started+3600;self.swap0=initial['swap_bytes']
        self.disk_floor=disk_floor;self.failed=False

    def run(self,name,command):
        if self.failed and name!='archive':raise RuntimeError('Terminal science latch')
        admission=host(self.out)
        issue=problem(admission,self.swap0,0,self.disk_floor)
        if issue:raise RuntimeError('Admission: '+issue)
        if self.deadline-monotonic()<=(5 if name=='archive' else 600):raise TimeoutError('Closeout reserve exhausted')
        directory=self.out/'operations'/name;directory.mkdir(parents=True,exist_ok=False)
        write(directory/'intent.json',dict(command=list(map(str,command)),admission=admission))
        begun=monotonic();peak=0;failure=None;child=None
        original={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
        def interrupt(signum,frame):raise RuntimeError(f'Supervisor interrupted by signal {signum}')
        for sig in original:signal.signal(sig,interrupt)
        try:
            with (directory/'log.txt').open('x') as log,(directory/'resources.jsonl').open('x') as samples:
                child=subprocess.Popen(['/usr/bin/time','-l',*map(str,command)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                parent=psutil.Process(os.getpid())
                while child.poll() is None:
                    rss=0
                    for p in [parent,*parent.children(recursive=True)]:
                        try:rss+=p.memory_info().rss
                        except psutil.NoSuchProcess:pass
                    peak=max(peak,rss);s=host(self.out)
                    samples.write(json.dumps(dict(**s,family_rss_bytes=rss,elapsed_seconds=monotonic()-self.started))+'\n');samples.flush()
                    issue=problem(s,self.swap0,rss,self.disk_floor)
                    if issue:raise RuntimeError('Guard: '+issue)
                    if self.deadline-monotonic()<=0:raise TimeoutError('60-minute hard deadline')
                    if name!='archive' and self.deadline-monotonic()<=600:raise TimeoutError('Science reserve exhausted')
                    sleep(.5)
                if child.returncode:raise RuntimeError(f'{name} exited {child.returncode}')
        except BaseException as exc:
            failure=repr(exc);self.failed=True;raise
        finally:
            if child is not None and child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=5)
                except (ProcessLookupError,subprocess.TimeoutExpired):
                    try:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
                    except ProcessLookupError:pass
            for sig,handler in original.items():signal.signal(sig,handler)
            text=(directory/'log.txt').read_text() if (directory/'log.txt').exists() else ''
            kernel=re.search(r'(\d+)\s+maximum resident set size',text);kernel_peak=int(kernel[1]) if kernel else None
            if kernel_peak is not None and kernel_peak>=SOFT:
                failure=failure or 'Kernel command peak exceeded RSS limit';self.failed=True
            receipt=dict(status='failed' if failure else 'complete',failure=failure,seconds=monotonic()-begun,
                peak_family_rss_bytes=peak,kernel_command_peak_rss_bytes=kernel_peak,
                returncode=child.returncode if child else None,finished=time())
            write(directory/'receipt.json',receipt)
        if failure:raise RuntimeError(failure)
        return receipt


def indexed_model():
    index=json.loads((ROOT/'docs/reports/hu200-feasibility-artifacts/model-index.json').read_text())
    m=next(m for m in index['models'] if m['target']==100_000_000);spec=m['files']['average'];path=MODELS/spec['path']
    if path.stat().st_size!=spec['bytes'] or file_hash(path)!=spec['sha256']:raise ValueError('Indexed model bytes differ')
    with gzip.open(path,'rt') as f:metadata=json.loads(f.readline())
    h=metadata['checkpoint_header']
    if (metadata['source_checkpoint_sha256']!=m['files']['checkpoint']['sha256'] or h['iteration']!=m['iteration']
            or h['config']['seed']!=index['seed'] or h['config']['game']!=index['game'] or h['abstraction']!=index['schema']
            or h['native_state']['completed_nodes']!=m['actual_nodes'] or average_rule(h)!='opponent-sampled'
            or zero_mass_rule(metadata)!='uniform' or m['audit']['status']!='verified'
            or m['audit']['average_sha256']!=spec['sha256']):raise ValueError('Indexed HU200 identity differs')
    return dict(**m,path=str(path),sha256=spec['sha256'],header=metadata,restoration=index['archive'])


def freshness():
    current={name:{seed(root,'deal',op,b) for op in OPPONENTS for b in range(n)}
             for name,root,n in [('timing',TIMING_ROOT,32),('final',FINAL_ROOT,2048)]}
    if current['timing'] & current['final']:raise ValueError('Timing/final seed collision')
    prior={seed(r,'deal',op,b) for r,n in [(2026100906,2048),(2026100907,32)] for op in OPPONENTS for b in range(n)}
    # #216 smoke's explicit seed schedule and trainer lineage; HU100 schedules use a separate codec.
    prior.update(20261009050000+oi*1000+b for oi in range(5) for b in range(32));prior.add(2026100905)
    from scripts.report_native_hu100_learning_curves import frozen_schedule
    from scripts.run_hu100_action_translation import PRIOR_ROOTS
    cfg=json.loads((ROOT/'configs/arena/hu100-learning-curves-v1.json').read_text())
    old_roots=[*PRIOR_ROOTS,(2026100820521,16),(2026100820512,2048),(2026100810011,16),(2026100810012,2048),
               (cfg['pilot_root'],16),(cfg['final_root'],2048)]
    for r,n in old_roots:
        doc=frozen_schedule(cfg,n,r)
        prior.update(b['deal_seeds'][0] for p in doc['panels'].values() for b in p['blocks'])
    if any(s & prior for s in current.values()):raise ValueError('Prior deal seed collision')
    return dict(status='verified',timing_root=TIMING_ROOT,final_root=FINAL_ROOT,prior_seed_count=len(prior),
                prior_hu200_roots=[2026100905,2026100906,2026100907],prior_hu100_roots=old_roots,
                timing_final_and_prior_deal_seeds_disjoint=True)


def sample_quote(costs,parity_seconds,remaining,evidence_bytes,free_bytes,disk_floor):
    options=[]
    for n in (2048,1024,512,256,128,64,32):
        # One shared compact model load. Scale the complete supervised parity cost conservatively.
        upper=2*(costs['model_load_seconds']+(costs['play_replay_seconds']+parity_seconds)*n/32)+600
        disk=2*evidence_bytes*n/32+GIB
        options.append(dict(blocks=n,fixed_reload_seconds=costs['model_load_seconds'],
            scalable_calibration_seconds=costs['play_replay_seconds']+parity_seconds,
            upper_seconds=upper,required_disk_bytes=disk,admitted=upper<remaining and free_bytes-disk>disk_floor))
    return options


def seal(root,destination):
    if destination.exists():raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True);members=[]
    with ZipFile(destination,'x',compression=ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.is_relative_to(root/'operations/archive'):continue
            stat=p.stat();name=str(p.relative_to(root));members.append(dict(path=name,bytes=stat.st_size,
                sha256=file_hash(p),original_path=str(p.resolve()),mtime_ns=stat.st_mtime_ns));z.write(p,name)
        raw=json.dumps(dict(members=members,original_root=str(root),
            restoration='Verify whole ZIP and manifest; extract to fresh ignored root; verify all selected members'),sort_keys=True).encode()
        z.writestr('ARCHIVE-MANIFEST.json',raw)
    with ZipFile(destination) as z:
        for m in members:
            h=hashlib.sha256();size=0
            with z.open(m['path']) as f:
                while chunk:=f.read(1024**2):size+=len(chunk);h.update(chunk)
            if size!=m['bytes'] or h.hexdigest()!=m['sha256']:raise ValueError('Archive readback failed: '+m['path'])
        if z.read('ARCHIVE-MANIFEST.json')!=raw:raise ValueError('Manifest readback failed')
    write(root/'archive-receipt.json',dict(status='locally-verified',path=str(destination),bytes=destination.stat().st_size,
        sha256=file_hash(destination),manifest_path='ARCHIVE-MANIFEST.json',manifest_sha256=hashlib.sha256(raw).hexdigest(),
        verified_members=len(members),upload_status='pending-owner-or-later-agent-handoff',remote_bytes_downloaded=False,
        cloud_acceptance_claimed=False,originals_retained=True,models_and_runtime_reused_without_copy=True))


def run(root,destination):
    disk_floor=DEFAULT_DISK_FLOOR
    with LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if root.exists():raise FileExistsError(root)
        if (platform.node()!='dberweger-m1' or subprocess.check_output(['sysctl','-n','hw.model'],text=True).strip()!='MacBookPro17,1'):
            raise RuntimeError('M1-only host identity')
        if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
            raise RuntimeError('Committed clean source required')
        owner=[]
        for p in psutil.process_iter(['pid','cmdline','create_time']):
            if p.pid in (os.getpid(),os.getppid()):continue
            argv=p.info['cmdline'] or []
            if (any(a.endswith('hu20-trainer') for a in argv) and 'train' in argv
                    or any(a.startswith(('scripts.run_hu200','scripts.evaluate_hu200','scripts.run_hu100_seed')) for a in argv)):
                owner.append(p.info)
        if owner:raise RuntimeError('Competing research process: '+repr(owner))
        initial=host(ROOT)
        if problem(initial,initial['swap_bytes'],0,disk_floor) or initial['available_bytes']<4*GIB:
            raise RuntimeError('Initial M1 admission refused: '+repr(initial))
        root.mkdir(parents=True);started=monotonic();guard=Guard(root,started,initial,disk_floor)
        source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        write(root/'baseline.json',dict(source=source,host=platform.node(),initial=initial,started_at=time(),cap_seconds=3600,
            lock=str(LOCK),owner_pid=os.getpid(),competing_research_processes=owner,python=sys.version,
            limits=dict(soft_family_gib=3,hard_family_gib=4,disk_floor_bytes=disk_floor,swap_growth_bytes=512*1024**2)))
        status='incomplete';failure=None
        try:
            for number in (216,217):
                parent=json.loads(subprocess.check_output(['gh','pr','view',str(number),'--json','number,state,mergedAt,url'],text=True))
                if parent['state']!='MERGED':raise RuntimeError('Input/evaluator parent PR is no longer merged')
                write(root/f'parent-{number}-live-status.json',parent)
            guard.run('source-snapshot',['git','archive','--format=tar','--output',root/'source.tar','HEAD',
                'scripts','src','tests/test_hu100_action_translation.py','tests/test_hu200_action_translation.py',
                'tests/test_hu200_translation_campaign.py','docs/hu200-action-translation.md',
                'docs/reports/hu200-action-translation-artifacts','docs/reports/hu200-feasibility-artifacts/model-index.json',
                'requirements-dev.txt','requirements-monitoring.txt','requirements.txt',
                'configs/arena/hu100-learning-curves-v1.json'])
            tick=monotonic();m=indexed_model()
            if file_hash(BINARY)!=BINARY_SHA:raise ValueError('Pinned native runtime differs')
            plan=dict(source=source,model=m,final_root=FINAL_ROOT,timing_root=TIMING_ROOT,target_blocks=2048,
                primary='pot_pressure',translation=asdict(TranslationOptions()),
                protocol_sha256=file_hash(ROOT/'docs/hu200-action-translation.md'),input_verification_seconds=monotonic()-tick,
                binary_path=str(BINARY),binary_sha256=BINARY_SHA,disk_floor_bytes=disk_floor)
            write(root/'freshness.json',freshness());write(root/'timing-plan.json',plan)
            guard.run('timing',[sys.executable,'-m','scripts.evaluate_hu200_translation','worker',
                '--plan',root/'timing-plan.json','--stage','timing','--out',root/'timing'])
            costs=json.loads((root/'timing/costs.json').read_text());parity=0.
            for arm in ('off','on'):
                p=guard.run('timing-native-'+arm,[BINARY,'parity',root/'timing'/arm/'native-fixtures.jsonl']);parity+=p['seconds']
            evidence=sum(p.stat().st_size for p in (root/'timing').rglob('*') if p.is_file())
            options=sample_quote(costs,parity,guard.deadline-monotonic(),evidence,shutil.disk_usage(root).free,disk_floor)
            chosen=next((o for o in options if o['admitted']),None)
            write(root/'sample-admission.json',dict(options=options,chosen=chosen,elapsed_seconds=monotonic()-started,
                remaining_seconds=guard.deadline-monotonic(),calibration_payoffs_inspected=False,
                calibration_native_supervised_seconds=parity,calibration_bytes=evidence))
            if chosen is None:raise RuntimeError('No fixed sample fits measured remaining budget')
            plan={**plan,'blocks':chosen['blocks'],'sample_frozen_at':time()};write(root/'frozen-plan.json',plan)
            guard.run('final',[sys.executable,'-m','scripts.evaluate_hu200_translation','worker',
                '--plan',root/'frozen-plan.json','--stage','final','--out',root/'final'])
            for arm in ('off','on'):
                guard.run('final-native-'+arm,[BINARY,'parity',root/'final'/arm/'native-fixtures.jsonl'])
            guard.run('verify-report',[sys.executable,'-m','scripts.evaluate_hu200_translation','report',
                '--plan',root/'frozen-plan.json','--out',root])
            status='complete'
        except BaseException as exc:
            failure=repr(exc);write(root/'failure.json',dict(failure=failure,status=status,at=time()))
        finally:
            write(root/'science-closeout.json',dict(status=status,failure=failure,elapsed_seconds=monotonic()-started,
                source=source,model_reused=True,no_training=True))
            try:
                own=json.loads(subprocess.check_output(['gh','pr','view','--json','number,state,url'],text=True))
                write(root/'owning-pr-before-archive.json',own)
                guard.run('archive',[sys.executable,'-m','scripts.run_hu200_translation','archive',
                    '--out',root,'--destination',destination])
            finally:
                write(root/'closeout.json',dict(status=status,failure=failure,seconds=monotonic()-started,
                    within_cap=monotonic()-started<3600,science_latched=guard.failed,ownership_released=True,
                    upload_pending=True,originals_retained=True))
        if status!='complete':raise RuntimeError(failure)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('run','archive'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args()
    if a.command=='run':run(a.out,a.destination)
    else:seal(a.out,a.destination)


if __name__=='__main__':main()
