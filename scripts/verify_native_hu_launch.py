"""Fail-closed PR197 prelaunch identity/admission check and durable launch claim.

This performs lightweight checks only, never launches training. Called first by
each prepared script. The operator writes a current admission.json beside its
plan after inspecting #188's worker-side closeout and the whole M4 workload.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from time import time

from scripts.prepare_native_hu_campaign import digest, file_hash
from scripts.prepare_native_hu_execution import (DEADLINE, qualification, checked_equivalence,
                                               validate_capacity, verify_pilot_prerequisites)

NOT_BEFORE=DEADLINE-6*3600  # October 8 04:00 Madrid /02:00 UTC


def verify(plan_path):
    plan_path=plan_path.resolve(); root=plan_path.parent
    p=json.loads(plan_path.read_text())
    content={k:v for k,v in p.items() if k != 'plan_sha256'}
    if digest(content) != p['plan_sha256']:
        raise ValueError('Prepared plan changed')
    if not NOT_BEFORE <= time() < p['hard_deadline'] <= DEADLINE:
        raise ValueError('Campaign deadline has passed or exceeds owner ceiling')
    if platform.system() != 'Darwin': raise ValueError('PR197 uses the free M4 macOS worker only')
    chip=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True,timeout=10).strip()
    if 'Apple M4' not in chip: raise ValueError('Refuse training on M1 or other workers')
    binary=Path(p['binary'])
    qpath=Path(p['qualification_path'])
    if file_hash(qpath) != p['qualification_sha256']: raise ValueError('Qualification receipt changed')
    source,q=qualification(qpath,binary)
    if source != p['source'] or file_hash(binary) != p['binary_sha256']:
        raise ValueError('Prepared source/binary changed before launch')
    if not p.get('campaign_swap_baseline'): raise ValueError('One campaign-wide swap baseline required')
    for name,spec in p['phase_jobs'].items():
        if digest(json.loads((root/f'{name}-jobs.json').read_text())) != spec['sha256']:
            raise ValueError('Prepared jobs changed')
        if (root/f'{name}-guard').exists(): raise ValueError('Attempt already executed; reconcile receipts without retry')
    for field in ('equivalence','capacity'):
        if field+'_path' in p and file_hash(Path(p[field+'_path'])) != p[field+'_sha256']:
            raise ValueError(f'{field} receipt changed')
    if 'equivalence_path' in p:
        checked_equivalence(Path(p['equivalence_path']),source,binary,p['campaign_swap_baseline'])
    if 'parent_path' in p and file_hash(Path(p['parent_path'])) != p['parent_sha256']:
        raise ValueError('Resume parent changed before execution')
    if p['stage']=='growth':
        verify_pilot_prerequisites(Path(p['pilot_root']),p['pilot_prerequisites'])
        validate_capacity(p['capacity_plan'],Path(p['pilot_root']),p['hard_deadline'])
    admission=json.loads((root/'admission.json').read_text())
    if (admission.get('status')!='admitted' or admission.get('m4_idle') is not True
        or admission.get('pr188_state')!='MERGED' or admission.get('worker_closeout')!='complete'
        or not admission.get('closeout_evidence') or not 0 <= time()-admission['observed_at'] <= 120):
        raise ValueError('Fresh M4 idle and PR188 worker-side closeout evidence required')
    for name,spec in admission['closeout_evidence'].items():
        path=Path(name)
        if path.stat().st_size != spec['bytes'] or file_hash(path) != spec['sha256']:
            raise ValueError('PR188 closeout evidence changed')
    live=json.loads(subprocess.check_output(['gh','pr','view','188','--repo',
        'dberweger2017/deepcfr-texas-no-limit-holdem-6-players','--json','state'],text=True,timeout=20))
    if live['state']!='MERGED': raise ValueError('PR188 still owns M4 until merged and finished')
    listing=subprocess.check_output(['ps','-axo','pid=,ppid=,command='],text=True,timeout=10)
    for line in listing.splitlines():
        fields=line.split(None,2)
        if len(fields)==3 and 'hu20-o-10b-lbr-20261007' in fields[2]:
            raise ValueError('PR188 worker-side process is still alive')
    # Exclusive durable intent precedes execution. Lost acknowledgement leaves a
    # claim to reconcile, never a reason to launch again into a different root.
    state_path=Path(admission['campaign_state_path'])
    state=json.loads(state_path.read_text())
    if state.get('active_attempt') is not None:
        raise ValueError('Another campaign attempt is active or uncertain; reconcile it first')
    claim={'status':'launch-intent','pid':os.getpid(),'source':source,'plan_sha256':p['plan_sha256'],
        'plan_path':str(plan_path),'admission_sha256':file_hash(root/'admission.json'),'at':time()}
    lock=state_path.with_name(state_path.name+'.launch-lock')
    with lock.open('x') as f: json.dump(claim,f); f.write('\n'); f.flush(); os.fsync(f.fileno())
    # Re-read after lock acquisition to close two concurrent preparers' race.
    state=json.loads(state_path.read_text())
    if state.get('active_attempt') is not None:
        raise ValueError('Campaign became active before claim; retain lock evidence')
    with (root/'LAUNCH.json').open('x') as f: json.dump(claim,f); f.write('\n'); f.flush(); os.fsync(f.fileno())
    state['active_attempt']=claim; state.setdefault('launch_attempts',[]).append(claim)
    temporary=state_path.with_name(state_path.name+'.tmp')
    temporary.write_text(json.dumps(state,sort_keys=True,indent=2)+'\n'); os.replace(temporary,state_path)
    return claim


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True); args=parser.parse_args()
    print(json.dumps(verify(args.plan)))


if __name__=='__main__': main()
