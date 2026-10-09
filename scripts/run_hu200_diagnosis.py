"""One-use, M1-only HU200 comparison with blind measured sample admission."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import monotonic, time
from zipfile import ZipFile, ZIP_DEFLATED

import psutil

from scripts.evaluate_hu200_diagnosis import indexed_models
from scripts.run_hu200_feasibility import Guard, host, violation, write, GIB, LOCK, DISK_FLOOR
from src.arena.schedule import digest
from src.policies.files import file_hash

ROOT=Path(__file__).resolve().parents[1]
PARENT=Path('/Users/dberweger/Local/deepcfr-hu200-slumbot')
MODELS=PARENT/'results/hu200-pilot-20261009'
BINARY=PARENT/'native/hu20-trainer/target/release/hu20-trainer'
BINARY_SHA='c72e900bc6e47e1c2ae78f7b16d87afbd2e9397adb946a86149e6ab4160a6878'


class EvaluationGuard(Guard):
    def run(self,name,command,**kwargs):
        # Every new science step must respect a prior soft breach, including tools.
        if self.soft_stopped and name!='archive':
            raise RuntimeError('Terminal soft resource latch')
        return super().run(name,command,**kwargs)


def sample_quote(costs, parity_seconds, remaining, evidence_bytes, free_bytes):
    fixed=sum(c['model_load_seconds'] for c in costs)
    scalable=sum(c['play_replay_seconds'] for c in costs)+parity_seconds
    options=[]
    for n in (2048,1024,512,256,128,64,32):
        seconds=2*(fixed+scalable*n/32)+600
        disk=2*evidence_bytes*n/32+GIB
        options.append(dict(blocks=n,fixed_load_seconds=fixed,scalable_calibration_seconds=scalable,
            upper_seconds=seconds,required_disk_bytes=disk,
            admitted=seconds<remaining and free_bytes-disk>DISK_FLOOR))
    return options


def seal(root,destination):
    if destination.exists():raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    members=[]
    with ZipFile(destination,'x',compression=ZIP_DEFLATED,compresslevel=1) as z:
        for path in sorted(root.rglob('*')):
            # Archive monitor writes during seal; its final log/receipt is indexed separately.
            if not path.is_file() or path.is_relative_to(root/'operations/archive'):
                continue
            name=str(path.relative_to(root));spec=dict(path=name,bytes=path.stat().st_size,sha256=file_hash(path))
            z.write(path,name);members.append(spec)
        raw=json.dumps(dict(members=members,original_root=str(root),
            restore='Verify whole ZIP and embedded manifest; extract to fresh ignored root; verify member SHA256'),sort_keys=True).encode()
        z.writestr('ARCHIVE-MANIFEST.json',raw)
    import hashlib
    with ZipFile(destination) as z:
        for m in members:
            h=hashlib.sha256();size=0
            with z.open(m['path']) as f:
                while chunk:=f.read(1024**2):size+=len(chunk);h.update(chunk)
            if size!=m['bytes'] or h.hexdigest()!=m['sha256']:
                raise ValueError('Archive member readback failed: '+m['path'])
        if z.read('ARCHIVE-MANIFEST.json')!=raw:raise ValueError('Archive manifest readback failed')
    write(root/'archive-receipt.json',dict(status='locally-verified',path=str(destination),bytes=destination.stat().st_size,
        sha256=file_hash(destination),manifest_path='ARCHIVE-MANIFEST.json',manifest_sha256=hashlib.sha256(raw).hexdigest(),
        verified_members=len(members),upload_status='pending-owner-or-later-agent-handoff',cloud_acceptance_claimed=False,
        remote_bytes_downloaded=False,originals_retained=True))


def run(root,destination):
    # Lock check precedes clock/admission; root reuse never starts a fresh baseline.
    with LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if root.exists():raise FileExistsError(root)
        if platform.node()!='dberweger-m1' or subprocess.check_output(['sysctl','-n','hw.model'],text=True).strip()!='MacBookPro17,1':
            raise RuntimeError('M1-only host identity')
        owner=[]
        for p in psutil.process_iter(['pid','cmdline','create_time']):
            if p.pid in (os.getpid(),os.getppid()):continue
            argv=p.info['cmdline'] or []
            if (any(a.endswith('hu20-trainer') for a in argv) and 'train' in argv
                    or any(a in ('scripts.run_hu200_feasibility','scripts.run_hu100_seed_qualification','scripts.evaluate_hu200_diagnosis') for a in argv)):
                owner.append(p.info)
        if owner:raise RuntimeError('Competing research process: '+repr(owner))
        root.mkdir(parents=True);initial=host(root)
        if violation(initial,initial['swap_bytes'],0) or initial['available_bytes']<4*GIB:
            raise RuntimeError('Initial M1 admission refused')
        started=monotonic();source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        guard=EvaluationGuard(root,started);guard.swap0=initial['swap_bytes']
        write(root/'baseline.json',dict(source=source,host=platform.node(),initial=initial,started_at=time(),cap_seconds=3600,
            lock=str(LOCK),owner_pid=os.getpid(),competing_research_processes=owner,python=sys.version,
            limits=dict(soft_family_gib=3,hard_family_gib=4,disk_floor_bytes=DISK_FLOOR,swap_growth_bytes=512*1024**2)))
        status='incomplete';failure=None;plan=None
        try:
            parent=json.loads(subprocess.check_output(['gh','pr','view','216','--json','number,state,mergedAt,url'],text=True))
            if parent['state']!='MERGED':raise RuntimeError('Parent PR is no longer merged')
            write(root/'parent-live-status.json',parent)
            guard.run('source-snapshot',['git','archive','--format=tar','--output',root/'source.tar','HEAD',
                'scripts','src','tests/test_hu200_diagnosis.py','docs/hu200-playing-diagnosis.md',
                'docs/reports/hu200-feasibility-artifacts/model-index.json','requirements-dev.txt','requirements.txt'])
            # One linear hash/header read; no model copies or old training re-audits.
            tick=monotonic();models=indexed_models(MODELS)
            if file_hash(BINARY)!=BINARY_SHA:raise ValueError('Pinned native runtime differs')
            plan=dict(source=source,models=models,final_root=2026100906,timing_root=2026100907,
                target_blocks=2048,primary_family=5,protocol_sha256=file_hash(ROOT/'docs/hu200-playing-diagnosis.md'),
                input_verification_seconds=monotonic()-tick,binary_path=str(BINARY),binary_sha256=BINARY_SHA)
            write(root/'timing-plan.json',plan)
            costs=[];parity=0.;evidence=0
            for m in models:
                out=root/'timing'/str(m['target'])
                guard.run('timing-'+str(m['target']),[sys.executable,'-m','scripts.evaluate_hu200_diagnosis','worker',
                    '--plan',root/'timing-plan.json','--stage','timing','--target',m['target'],'--out',out])
                costs.append(json.loads((out/'costs.json').read_text()))
                p=guard.run('timing-native-'+str(m['target']),[BINARY,'parity',out/'native-fixtures.jsonl'])
                parity+=p['seconds'];evidence+=sum(x.stat().st_size for x in out.iterdir() if x.is_file())
            options=sample_quote(costs,parity,guard.deadline-monotonic(),evidence,shutil.disk_usage(root).free)
            chosen=next((o for o in options if o['admitted']),None)
            write(root/'sample-admission.json',dict(options=options,chosen=chosen,
                elapsed_seconds=monotonic()-started,remaining_seconds=guard.deadline-monotonic(),
                calibration_payoffs_inspected=False,calibration_native_seconds=parity,calibration_bytes=evidence))
            if chosen is None:raise RuntimeError('No fixed sample fits measured remaining budget')
            plan={**plan,'blocks':chosen['blocks'],'sample_frozen_at':time()};write(root/'frozen-plan.json',plan)
            for m in models:
                out=root/'final'/str(m['target'])
                guard.run('final-'+str(m['target']),[sys.executable,'-m','scripts.evaluate_hu200_diagnosis','worker',
                    '--plan',root/'frozen-plan.json','--stage','final','--target',m['target'],'--out',out])
                guard.run('final-native-'+str(m['target']),[BINARY,'parity',out/'native-fixtures.jsonl'])
            guard.run('verify-report',[sys.executable,'-m','scripts.evaluate_hu200_diagnosis','report','--plan',root/'frozen-plan.json','--out',root])
            status='complete'
        except BaseException as exc:
            failure=repr(exc)
            write(root/'failure.json',dict(failure=failure,status=status,at=time()))
        finally:
            write(root/'science-closeout.json',dict(status=status,failure=failure,elapsed_seconds=monotonic()-started,
                source=source,models_reused=True,no_training=True))
            own=json.loads(subprocess.check_output(['gh','pr','view','--json','number,state,url'],text=True))
            write(root/'owning-pr-before-archive.json',own)
            try:
                guard.run('archive',[sys.executable,'-m','scripts.run_hu200_diagnosis','archive','--out',root,'--destination',destination])
            finally:
                write(root/'closeout.json',dict(status=status,failure=failure,seconds=monotonic()-started,
                    within_cap=monotonic()-started<3600,soft_latched=guard.soft_stopped,hard_latched=guard.failed,
                    ownership_released=True,upload_pending=True,originals_retained=True))
        if status!='complete':raise RuntimeError(failure)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('run','archive'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args()
    if a.command=='run':run(a.out,a.destination)
    else:seal(a.out,a.destination)


if __name__=='__main__':main()
