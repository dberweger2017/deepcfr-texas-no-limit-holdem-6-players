"""One predeclared Phase 2 seed, scheduled saves, full averages and every-row audits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.audit_nplayer_policy import audit, sha

SEEDS = (2026101541, 2026101542, 2026101543)
TARGETS = (200_000_000, 500_000_000, 1_000_000_000)


def run(seed, binaries, tables, out):
    stop = os.environ.get('NPLAYER_STOP_FILE')
    if not stop or Path(stop).exists():
        raise RuntimeError('Need a live on-host supervisor and fresh stop file')
    if seed not in SEEDS:
        raise ValueError('seed outside fixed campaign')
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('source checkout must be clean')
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    train, policy = binaries/'nplayer_train', binaries/'nplayer_policy'
    admission = {'seed':seed,'targets':TARGETS,'players':3,'stack_bb':20,'blinds':[50,100],
        'last_average_street':3,'max_entries':150_000_000,'freeze_at_cap':False,
        'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'native_tree':subprocess.check_output(['git','rev-parse','HEAD:native'],text=True).strip(),
        'train_binary_sha256':sha(train),'policy_binary_sha256':sha(policy),
        'segment_training_seconds':1800,'supervisor_stop_file':stop}
    (out/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
    records=[]
    def execute(name, command):
        if Path(stop).exists(): raise RuntimeError('supervisor requested stop; no continuation')
        command=list(map(str,command))
        (out/f'{name}.command.json').write_text(json.dumps(command,indent=2)+'\n')
        tick=time.monotonic()
        with (out/f'{name}.log').open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        elapsed=time.monotonic()-tick
        record={'name':name,'seconds':elapsed,'returncode':result.returncode}
        records.append(record)
        (out/'operations.json').write_text(json.dumps(records,indent=2)+'\n')
        if result.returncode or Path(stop).exists(): raise RuntimeError(f'{name} failed/stopped; no continuation')
        parsed=[json.loads(line) for line in (out/f'{name}.log').read_text().splitlines() if line.startswith('{')]
        record['report']=parsed[-1]
        (out/'operations.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(record),flush=True)
        return record['report']
    previous=None
    previous_counts={'decisions':[0]*4,'missing':[0]*4}
    milestones=[]
    started=time.monotonic()
    for target in TARGETS:
        label=f'{target//1_000_000}m'
        checkpoint=out/f'{label}.gz'
        command=[train,'--players',3,'--stack-bb',20,'--small-blind',50,'--big-blind',100,
            '--menu','v1','--seed',seed,'--nodes',target,'--max-entries',150_000_000,
            '--max-seconds',1800,'--last-average-street',3,'--card-buckets',tables,
            '--out',checkpoint,'--telemetry',out/f'{label}.telemetry.jsonl','--stop-file',stop]
        if previous:
            command += ['--resume',previous,'--resume-sha256',sha(previous)]
        report=execute(f'train-{label}',command)
        if report['stop_reason']!='nodes' or report['frozen_at'] is not None:
            raise RuntimeError(f'training stopped before scheduled milestone: {report["stop_reason"]}')
        export=out/f'{label}-average.gz'
        execute(f'export-{label}',[policy,'export',checkpoint,report['sha256'],'average',export])
        if Path(stop).exists(): raise RuntimeError('supervisor stop before audit')
        tick=time.monotonic()
        audited=audit(checkpoint,export)
        audited['elapsed_seconds']=time.monotonic()-tick
        for street,row in enumerate(audited['by_street']):
            d=report['counts']['decisions'][street]-previous_counts['decisions'][street]
            m=report['counts']['missing'][street]-previous_counts['missing'][street]
            row['segment_decisions']=d;row['segment_missing_decisions']=m
            row['segment_missing_share']=m/d if d else None
        (out/f'{label}-audit.json').write_text(json.dumps(audited,indent=2)+'\n')
        milestones.append({'target':target,'checkpoint':str(checkpoint),'checkpoint_sha256':report['sha256'],
            'average':str(export),'average_sha256':audited['policy_sha256'],'entries':report['entries'],
            'nodes':report['counts']['nodes'],'iteration':report['iteration'],'audit':audited})
        (out/'milestones.json').write_text(json.dumps(milestones,indent=2)+'\n')
        print(json.dumps({'milestone':label,'audit':audited}),flush=True)
        previous=checkpoint;previous_counts=report['counts']
    result={'status':'passed','seed':seed,'elapsed_seconds':time.monotonic()-started,'milestones':milestones}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',required=True,type=int,choices=SEEDS)
    p.add_argument('--binaries',required=True,type=Path)
    p.add_argument('--tables',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();run(a.seed,a.binaries.resolve(),a.tables.resolve(),a.out)
