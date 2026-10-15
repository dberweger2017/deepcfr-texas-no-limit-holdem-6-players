"""Execute only the admitted 200M three-player smoke, inside the on-host watchdog."""
from __future__ import annotations
import argparse
import filecmp
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from scripts.audit_nplayer_policy import audit
from scripts.check_nplayer_policy_replay import check


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main(out):
    root=Path.cwd().parent
    assert json.loads((root/'size-main/result.json').read_text())['status']=='passed'
    assert json.loads((root/'key-compat/receipt.json').read_text())['status']=='passed'
    assert json.loads((root/'policy-tests/result.json').read_text())['status']=='passed'
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    train=Path('native/hu20-trainer/target/release/examples/nplayer_train').resolve()
    policy=train.with_name('nplayer_policy')
    admission={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'native_tree':subprocess.check_output(['git','rev-parse','HEAD:native'],text=True).strip(),'train_binary_sha256':sha(train),'policy_binary_sha256':sha(policy),'players':3,'stack_bb':20,'seed':2026101531,'nodes':200_000_000,'scope':0,'max_entries':30_000_000,'play_seed':2026101532,'play_hands':128}
    (out/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
    started=time.monotonic();results=[]
    def run(name,command):
        (out/f'{name}.command.json').write_text(json.dumps(list(map(str,command)),indent=2)+'\n')
        start=time.monotonic()
        with (out/f'{name}.log').open('w') as f:subprocess.run(list(map(str,command)),stdout=f,stderr=subprocess.STDOUT,check=True)
        parsed=[json.loads(s) for s in (out/f'{name}.log').read_text().splitlines() if s.startswith('{')]
        result={'name':name,'wall_seconds':time.monotonic()-start,'report':parsed[-1]};results.append(result)
        (out/'operations.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(result),flush=True);return result['report']
    for name,nodes,resume in [('fresh',200_000_000,None),('prefix',100_000_000,None),('resumed',200_000_000,'prefix')]:
        command=[train,'--players',3,'--stack-bb',20,'--seed',admission['seed'],'--nodes',nodes,'--max-entries',admission['max_entries'],'--max-seconds',900,'--last-average-street',0,'--card-buckets',root/'tables','--out',out/f'{name}.gz','--telemetry',out/f'{name}.telemetry.jsonl']
        if resume:command+=['--resume',out/f'{resume}.gz','--resume-sha256',sha(out/f'{resume}.gz')]
        report=run(name,command);assert report['stop_reason']=='nodes',report
    assert sha(out/'fresh.gz')==sha(out/'resumed.gz') and filecmp.cmp(out/'fresh.gz',out/'resumed.gz',shallow=False)
    (out/'recovery.json').write_text(json.dumps({'status':'passed','fresh_sha256':sha(out/'fresh.gz'),'resumed_sha256':sha(out/'resumed.gz'),'prefix_sha256':sha(out/'prefix.gz'),'full_byte_comparison':True},indent=2)+'\n')
    export=out/'scoped-average.gz'
    run('export',[policy,'export',out/'fresh.gz',sha(out/'fresh.gz'),'scoped-average',export])
    audit_start=time.monotonic();audit_result=audit(out/'fresh.gz',export);audit_result['elapsed_seconds']=time.monotonic()-audit_start
    (out/'export-audit.json').write_text(json.dumps(audit_result,indent=2)+'\n');print(json.dumps(audit_result),flush=True)
    for i in range(2):
        trace=out/f'play-{i}.jsonl'
        result=run(f'play-{i}',[policy,'play',root/'tables',admission['play_hands'],admission['play_seed'],trace,*([export,sha(export)]*3)])
        assert result['distinct_policy_instances']==3 and all(v>0 for v in result['seat_decisions'])
        reference=check(trace);(out/f'play-{i}-oracle.json').write_text(json.dumps(reference,indent=2)+'\n')
    assert filecmp.cmp(out/'play-0.jsonl',out/'play-1.jsonl',shallow=False)
    summary={'status':'passed','elapsed_seconds':time.monotonic()-started,'checkpoint_sha256':sha(out/'fresh.gz'),'policy_sha256':sha(export),'trace_sha256':sha(out/'play-0.jsonl'),'trace_reproduction_byte_identical':True}
    (out/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)
