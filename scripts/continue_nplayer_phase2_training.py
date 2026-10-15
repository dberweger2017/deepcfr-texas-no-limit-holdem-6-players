"""Continue the two already-authorized seeds after seed 1 finishes successfully.

One-shot continuation only: fixed seeds, no retries, no extensions, no scheduler.
The training checkout stays pinned while the controller can be copied outside it.
"""
import argparse
import hashlib
import re
import json
import os
from pathlib import Path
import signal
import subprocess
import time

SEEDS=(2026101541,2026101542,2026101543)


def assert_idle():
    names={'hu20-trainer','nplayer_train','nplayer_arena','nplayer_policy','multi_known','nplayer_size','cargo','rustc'}
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args=(proc/'cmdline').read_bytes().split(b'\0')
            decoded=[os.fsdecode(arg) for arg in args if arg]
            if int(proc.name)==os.getpid(): continue
            native=decoded and Path(decoded[0]).name in names
            python=decoded and 'python' in Path(decoded[0]).name and any(
                re.search(r'(^|[./])(train_|evaluate_|run_hu|run_nplayer|play_slumbot)',arg) for arg in decoded[1:])
            if native or python:
                raise RuntimeError(f'another research process is still active: PID {proc.name}')
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            continue


def passed(root,seed):
    guard=json.loads((root/'guards'/f'seed-{seed}'/'result.json').read_text())
    if guard['status']!='passed':
        raise RuntimeError('prior seed failed; campaign stops')
    result=json.loads((root/'training'/str(seed)/'result.json').read_text())
    if result['status']!='passed' or result['seed']!=seed:
        raise RuntimeError('prior seed failed; campaign stops')
    if [m['target'] for m in result['milestones']]!=[200_000_000,500_000_000,1_000_000_000]:
        raise RuntimeError('prior seed missing scheduled endpoints')


def run(root,source):
    root=root.resolve();repo=root/'repo'
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()!=source:
        raise RuntimeError('training checkout source changed')
    first=json.loads((root/'training'/str(SEEDS[0])/'admission.json').read_text())
    def check_binaries():
        for name,key in [('nplayer_train','train_binary_sha256'),('nplayer_policy','policy_binary_sha256')]:
            with (root/'target/release/examples'/name).open('rb') as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=first[key]:
                    raise RuntimeError('training/export binary changed')
    check_binaries()
    receipt=root/'training-continuation.json'
    if receipt.exists():raise RuntimeError('continuation already admitted')
    receipt.write_text(json.dumps({'source':source,'seeds':SEEDS,'started_unix':time.time(),
        'waits_for':SEEDS[0],'per_seed_supervisor_seconds':10800,'no_retries':True},indent=2)+'\n')
    end=time.monotonic()+10800
    while not (root/'guards'/f'seed-{SEEDS[0]}'/'result.json').exists():
        if time.monotonic()>=end:raise RuntimeError('prior-seed wait budget exhausted')
        time.sleep(2)
    passed(root,SEEDS[0])
    # A terminal receipt is written immediately before the previous lock releases.
    time.sleep(2)
    for seed in SEEDS[1:]:
        assert_idle()
        check_binaries()
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()!=source:
            raise RuntimeError('training checkout source changed')
        command=[str(repo/'.venv/bin/python'),'-m','scripts.machine_qualification.vrm_watchdog',
            '--output',str(root/'guards'/f'seed-{seed}'),'--stop-file',str(root/'stops'/f'seed-{seed}'),
            '--seconds','10800','--max-rss-gib','80','--',str(repo/'.venv/bin/python'),
            '-m','scripts.run_nplayer_blueprint_seed','--seed',str(seed),
            '--binaries',str(root/'target/release/examples'),'--tables',str(root/'tables'),
            '--out',str(root/'training'/str(seed))]
        with (root/f'seed-{seed}-launch.log').open('x') as log:
            proc=subprocess.Popen(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
            try:
                code=proc.wait()
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=10)
        if code:raise RuntimeError(f'seed {seed} guard failed; no next seed')
        passed(root,seed)
        print(json.dumps({'seed':seed,'status':'passed'}),flush=True)
    (root/'training-result.json').write_text(json.dumps({'status':'passed','seeds':SEEDS},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--source',required=True)
    a=p.parse_args()
    def interrupt(number,frame):raise KeyboardInterrupt(f'controller signal {number}')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupt)
    run(a.root,a.source)
