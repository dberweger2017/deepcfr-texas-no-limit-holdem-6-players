"""Self-contained M4 coordinator; never calls or waits for the M1."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import sleep,time

from scripts.hu20_scaling_common import specification
from scripts.hu20_scaling_runtime import validate_inputs
from scripts.tp20_common import interruptible
from scripts.train_hu20 import write_json
from src.blueprint.windowed import _hash


def coordinate(plan,planpath):
    root=Path(plan['root']);recordpath=root/'campaign.json'
    with recordpath.open('x') as f:json.dump({'status':'starting','pid':os.getpid()},f)
    interruptible()
    record={'attempt_id':plan['attempt_id'],'status':'validation','pid':os.getpid(),
            'started':plan['started'],'deadline':plan['deadline'],'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'plan_sha256':_hash(planpath),'attempts':[],'failure':None,'phase_order':plan['phase_order']}
    active=None
    def phase(name,command,end):
        nonlocal active
        if time()>=end-120:raise TimeoutError(f'No safe launch reserve for {name}')
        folder=root/f'{name}-supervisor';jobs=root/f'{name}-jobs.json'
        write_json(jobs,[{'name':name,'command':command,'deadline':end}])
        record['status']=name;write_json(recordpath,record)
        args=[sys.executable,'-m','scripts.hu20_scaling_supervise','--jobs',str(jobs),
              '--out',str(folder),'--deadline',str(end),'--swap-baseline',plan['swap_baselines']['m4'],
              '--coordinator-pid',str(os.getpid()),'--require-ac']
        with (root/f'{name}-launcher.log').open('w') as log:
            active=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            row={'phase':name,'supervisor_pid':active.pid,'started':time()};record['attempts'].append(row);write_json(recordpath,record)
            while active.poll() is None:sleep(3)
            row.update(exit_code=active.returncode,finished=time())
            active=None;write_json(recordpath,record)
            if row['exit_code']!=0:raise RuntimeError(f'{name} failed; inspect retained attempt')
    try:
        validate_inputs(plan)
        if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
            raise ValueError('Deployed frozen source must be clean')
        for seed in plan['training_seeds']:
            phase(f'train-{seed}',[sys.executable,'-m','scripts.train_hu20_scaling','--plan',str(planpath),
                  '--parent',str(root/'inputs'/f'resume-{seed}.json'),'--out',str(root/'training'/f'B-{seed}'),
                  '--deadline',str(plan['training_deadline'])],plan['training_deadline'])
        models=plan['baseline_models'][:]
        for seed in plan['training_seeds']:
            folder=root/'training'/f'B-{seed}';r=json.loads((folder/'result.json').read_text())
            if r['status']!='complete' or r['completed_nodes']<100000000:raise ValueError('Missing fixed final state')
            for m in r['milestones']:
                models.append(specification(seed,m['requested_total_nodes'],m['iteration'],
                     Path(m['checkpoint_path']),Path(m['policy_path']),m['checkpoint_sha256'],m['policy_sha256']))
        models+=plan['reference_models'];write_json(Path(plan['coordinator_models']),models)
        for name in ['primary','diagnostic']:
            if name=='diagnostic':
                required=plan['resource_forecast']['diagnostic_seconds']+plan['audit_reserve_seconds']+plan['report_reserve_seconds']
                if time()+required>plan['deadline']:
                    record['diagnostic_admission']={'status':'pending','reason':'Outcome-free forecast exceeds remaining reserve','required_seconds':required,'checked_at':time()}
                    write_json(recordpath,record);break
            phaseplan={**plan,'panel_filter':name};path=root/f'{name}-plan.json';write_json(path,phaseplan)
            phase(f'evaluate-{name}',[sys.executable,'-m','scripts.evaluate_hu20_scaling','--plan',str(path),
                 '--models',plan['coordinator_models'],'--host','m4','--out',str(root/f'evaluation-{name}'),
                 '--deadline',str(plan['evaluation_deadline']),'--swap-baseline',plan['swap_baselines']['m4']],plan['evaluation_deadline'])
            phase(f'audit-{name}',[sys.executable,'-m','scripts.report_hu20_scaling','--plan',str(path),
                  '--evaluation',str(root/f'evaluation-{name}'),'--out',str(root/f'audit-{name}')],plan['deadline']-plan['report_reserve_seconds'])
        record['status']='ready_for_report'
    except Exception as exc:
        record.update(status='incomplete',failure=f'{type(exc).__name__}: {exc}')
    finally:
        if active and active.poll() is None:
            # The child owns a separate session; stop only our recorded attempt.
            folder=root/f"{record['attempts'][-1]['phase']}-supervisor"
            statepath=folder/'campaign.json'
            if statepath.exists():
                state=json.loads(statepath.read_text())
                for a in state['attempts']:
                    if 'pid' in a and a['status']=='running':
                        try:os.killpg(a['pid'],signal.SIGTERM)
                        except ProcessLookupError:pass
            try:active.wait(timeout=70)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid,signal.SIGTERM);active.wait(timeout=10)
        record['finished']=time();write_json(recordpath,record)
    # Reporting is part of the M4 job and proceeds even after an operational failure.
    if time()<plan['deadline']-120:
        try:
            phase('final-report',[sys.executable,'-m','scripts.finish_hu20_recovery','--plan',str(planpath)],plan['deadline']-360)
            report=json.loads((root/'report/results.json').read_text())
            record.update(status='reported-'+report['status'],report_finished=time())
        except Exception as exc:
            record.update(status='report-failed',report_failure=f'{type(exc).__name__}: {exc}')
    write_json(recordpath,record)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);a=p.parse_args();path=a.plan.resolve()
    plan=json.loads(path.read_text());os.chdir(plan['source'])
    result=coordinate(plan,path);print(json.dumps({k:v for k,v in result.items() if k!='attempts'}))
    raise SystemExit(0 if result['status'].startswith('reported-') else 1)
