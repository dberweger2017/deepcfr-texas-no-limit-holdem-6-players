"""Offline PR278 db-ocean cache preparation; never invokes the play client."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path.home()/'deepcfr-slumbot'
REPO = BASE/'repo-completed-hands'
SOURCE = 'ff6cc77aa9510854eef49114355f272d6c465570'


def put(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def start_ticks(pid):
    raw=Path(f'/proc/{pid}/stat').read_text()
    return int(raw.rsplit(')',1)[1].split()[19])


def sample(pid=None):
    values={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,_,value=line.partition(':')
        if key in ('MemTotal','MemFree','MemAvailable','SwapTotal','SwapFree'):
            values[key+'_bytes']=int(value.split()[0])*1024
    result=dict(at=time.time(),host=values)
    if pid:
        try:
            rss={}
            for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                key,_,value=line.partition(':')
                if key in ('VmRSS','RssAnon','RssFile','VmSwap'):rss[key+'_bytes']=int(value.split()[0])*1024
            result['worker']=dict(pid=pid,start_ticks=start_ticks(pid),**rss)
        except FileNotFoundError: result['worker']=None
    return result


def worker(seed,sha):
    sys.path.insert(0,str(REPO))
    from src.blueprint.average import AveragePolicy
    from src.blueprint.abstraction import HU200_EQUITY_SCHEMA
    from src.blueprint.equity_buckets import EQUITY_K50_SHA256
    from src.blueprint.policy_cache import folder
    from src.policies.files import file_hash
    root=BASE/f'preparation-v055-seed-{seed}'
    model=BASE/f'inputs/model-seed-{seed}/final.average.gz'; cache=BASE/f'cache-seed-{seed}'
    tables=BASE/'inputs/tables'
    if file_hash(model)!=sha:raise ValueError('Server model SHA256 differs')
    table_pins={s:file_hash(tables/f'{s}-k50.bin') for s in EQUITY_K50_SHA256}
    if table_pins!=EQUITY_K50_SHA256:raise ValueError('Server table hashes differ')
    if cache.exists():raise ValueError('Preserve existing cache; do not replace/retry')
    started=time.time()
    policy=AveragePolicy(model,sha,expected_schema=HU200_EQUITY_SCHEMA,sorted_rows=True,card_buckets=tables,cache=cache)
    description=policy.description; entries=len(policy.entries);del policy;gc.collect()
    verified=AveragePolicy(model,sha,expected_schema=HU200_EQUITY_SCHEMA,sorted_rows=True,card_buckets=tables,cache=cache,cache_read_only=True)
    if len(verified.entries)!=entries:raise ValueError('Read-only cache entry count differs')
    manifest=folder(cache,sha,HU200_EQUITY_SCHEMA)/'manifest.json'; data=json.loads(manifest.read_text())
    put(root/'cache-receipt.json',dict(status='verified',seed=seed,model=dict(path=str(model),bytes=model.stat().st_size,sha256=sha),
        cache=str(cache),cache_folder=str(manifest.parent),manifest_sha256=file_hash(manifest),manifest=data,
        backing_bytes=sum(row['bytes'] for row in data['files'].values()),entries=entries,description=description,
        table_sha256=table_pins,source=SOURCE,seconds=time.time()-started,finished=time.time(),
        all_cache_payloads_verified_by_read_only_load=True,no_hands_played=True,no_credentials_accessed=True))


def supervise(seed,sha):
    if subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()!=SOURCE:
        raise ValueError('Prepared completed-hands source changed')
    root=BASE/f'preparation-v055-seed-{seed}'; root.mkdir(exist_ok=False)
    initial=sample()
    put(root/'admission.json',dict(initial,source=SOURCE,minimum_available_bytes=2_000_000_000,model_sha256=sha))
    if initial['host']['MemAvailable_bytes']<2_000_000_000:raise ValueError('Available memory below 2 GB; no cache build')
    started=time.time(); child=None;identity=None;failure=None;count=0;peak=0;minimum=initial['host']['MemAvailable_bytes']
    try:
        with (root/'worker.log').open('x') as log,(root/'resources.jsonl').open('x') as stream:
            child=subprocess.Popen([sys.executable,__file__,'worker','--seed',str(seed),'--sha256',sha],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            identity=start_ticks(child.pid);put(root/'worker-identity.json',dict(pid=child.pid,start_ticks=identity))
            while True:
                row=sample(child.pid);stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush();count+=1
                minimum=min(minimum,row['host']['MemAvailable_bytes']);peak=max(peak,(row.get('worker') or {}).get('VmRSS_bytes',0))
                if row['host']['MemAvailable_bytes']<2_000_000_000:
                    raise RuntimeError('Available memory below 2 GB; stop only owned cache worker')
                if child.poll() is not None:break
                time.sleep(5)
            if child.returncode:raise RuntimeError('Cache worker returned '+str(child.returncode))
    except BaseException as exc:
        failure=repr(exc);put(root/'failure.json',dict(failure=failure,at=time.time()))
        if child is not None and child.poll() is None and start_ticks(child.pid)==identity:
            os.kill(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if start_ticks(child.pid)==identity:os.kill(child.pid,signal.SIGKILL)
                child.wait()
        raise
    finally:
        put(root/'supervisor-receipt.json',dict(failure=failure,samples=count,peak_worker_rss_bytes=peak,
            minimum_available_bytes=minimum,started=started,finished=time.time(),wall_seconds=time.time()-started,
            returncode=child.returncode if child else None,existing_sessions_and_paths_untouched=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('supervise','worker'))
    parser.add_argument('--seed',type=int,choices=(2026101421,2026101422),required=True)
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args()
    if args.operation=='supervise':supervise(args.seed,args.sha256)
    else:worker(args.seed,args.sha256)
