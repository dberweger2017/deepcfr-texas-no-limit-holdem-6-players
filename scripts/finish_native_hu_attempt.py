"""Reconcile a terminal PR197 attempt and retain its durable launch/closeout receipts."""
import argparse
import json
import os
from pathlib import Path
from time import time

from scripts.prepare_native_hu_campaign import file_hash


def finish(state_path):
    state_path=state_path.resolve(); state=json.loads(state_path.read_text())
    active=state.get('active_attempt')
    if active is None: raise ValueError('No active launch to reconcile')
    root=Path(active['plan_path']).parent
    plan=json.loads(Path(active['plan_path']).read_text())
    records=[]; failed=False
    for phase in sorted(plan['phase_jobs'],key=lambda name:name!='training'):
        path=root/f'{phase}-guard/campaign.json'
        if not path.exists():
            if failed: break  # The fail-fast script cannot start phases after a failure.
            raise ValueError('Required phase has not closed out')
        record=json.loads(path.read_text())
        if record['status'] not in ('complete','incomplete','interrupted'):
            raise ValueError('Owned guard is still running or uncertain')
        if record.get('cleanup_failure') or any(a.get('cleanup_failure') for a in record['attempts']):
            raise ValueError('Resolve recorded owned-process cleanup failure first')
        for attempt in record['attempts']:
            if attempt.get('pid'):
                try: os.kill(attempt['pid'],0)
                except ProcessLookupError: pass
                else: raise ValueError('Recorded worker PID still exists; inspect identity without stopping it')
        records.append({'path':str(path),'sha256':file_hash(path),'status':record['status']})
        failed=record['status']!='complete'
        if failed: break
    lock=state_path.with_name(state_path.name+'.launch-lock')
    if not lock.is_file(): raise ValueError('Missing campaign launch lock; reconcile provenance')
    claim=json.loads(lock.read_text())
    if claim != active: raise ValueError('Campaign launch lock belongs to another attempt')
    closeout={'at':time(),'launch':active,'guards':records,'status':'incomplete' if failed else 'complete',
              'launch_lock_sha256':file_hash(lock)}
    with (root/'CLOSEOUT.json').open('x') as f:
        json.dump(closeout,f,sort_keys=True,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    state['active_attempt']=None; state.setdefault('closed_attempts',[]).append(closeout)
    tmp=state_path.with_name(state_path.name+'.tmp')
    tmp.write_text(json.dumps(state,sort_keys=True,indent=2)+'\n'); os.replace(tmp,state_path)
    # Only the owned bookkeeping lock is removed. Its full contents and hash are
    # retained in LAUNCH/CLOSEOUT and campaign state; no research payload is removed.
    lock.unlink()
    return closeout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-state',type=Path,required=True); args=parser.parse_args()
    print(json.dumps(finish(args.campaign_state)))


if __name__=='__main__': main()
