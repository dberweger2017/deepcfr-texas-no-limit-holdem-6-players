"""Bounded frozen-table engineering gates; no playing-strength evaluation.

Build main and the candidate with scripts.build_native_trainer first, retaining
main's executable separately. Real K50 verification is opt-in and reads the
specified tables without copying or modifying them. Every trainer operation has
a 240-second deadline, a 5-GiB family RSS ceiling and a 32-GiB free-disk floor.

python -m scripts.check_frozen_table --baseline BINARY --out NEW_DIRECTORY
    [--real-tables DIRECTORY]
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import psutil

from src.blueprint.equity_buckets import EQUITY_K50_SHA256
from src.policies.files import file_hash

BINARY = Path('native/hu20-trainer/target/release/hu20-trainer').resolve()
EXPECTED_20M = '4009bbbce8d95a722e4e63e96dfc6e2b67957095f06c53487a29e4bf3d3acc8b'


def put(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')


def execute(command, directory, name):
    command = list(map(str, command))
    started = time.monotonic()
    samples = []
    environment = dict(os.environ, RAYON_NUM_THREADS='2')
    with (directory/f'{name}.stdout').open('x') as stdout, (directory/f'{name}.stderr').open('x') as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment)
        process = psutil.Process(child.pid)
        failure = None
        while child.poll() is None:
            elapsed = time.monotonic()-started
            try:
                rss = process.memory_info().rss + psutil.Process().memory_info().rss
            except psutil.NoSuchProcess:
                rss = psutil.Process().memory_info().rss
            free = shutil.disk_usage(directory).free
            samples.append({'seconds':elapsed,'family_rss_bytes':rss,'disk_free_bytes':free})
            if elapsed > 240 or rss > 5*2**30 or free < 32*2**30:
                failure = 'engineering gate resource ceiling'
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
                break
            time.sleep(0.2)
        receipt = {'command':command,'seconds':time.monotonic()-started,'exit_code':child.wait(),
                   'family_peak_rss_bytes':max((s['family_rss_bytes'] for s in samples),default=0),
                   'sample_period_seconds':0.2,'samples':samples,'failure':failure}
    put(directory/f'{name}.operation.json',receipt)
    if receipt['exit_code'] or failure:
        raise RuntimeError(f'{name}: {failure or receipt["exit_code"]}')
    return {k:v for k,v in receipt.items() if k != 'samples'}


def header(path):
    with gzip.open(path,'rt') as stream: return json.loads(stream.readline())


def key_digest(path):
    """Streaming digest of sorted serialized keys and exact stored visits, without a key set in RAM."""
    digest, count, visits = hashlib.sha256(), 0, 0
    previous = b''
    with gzip.open(path,'rb') as stream:
        stream.readline()
        for line in stream:
            key = line[2:34]
            if len(key) != 32 or key <= previous: raise ValueError('checkpoint keys are not sorted/unique')
            previous = key
            digest.update(key)
            count += 1
            visits += int(line.rsplit(b',',1)[1].rstrip(b']\n'))
    return {'sorted_key_sha256':digest.hexdigest(),'entries':count,'stored_visits':visits}


def record(path):
    h = header(path)
    return {'file':path.name,'sha256':file_hash(path),'bytes':path.stat().st_size,
            'iteration':h['iteration'],'nodes':h['native_state']['completed_nodes'],
            'frozen_table':h.get('frozen_table')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--real-tables',type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    summary={'baseline_sha256':file_hash(args.baseline),'candidate_sha256':file_hash(BINARY),
             'main_revision':'ac19e29a','small':[],'real_model':None}
    put(args.out/'plan.json',{'node_targets':{'small':1_000_000,'real':20_000_000,'real_frozen_segment':20_200_000},
        'operation_seconds_limit':240,'family_rss_limit_bytes':5*2**30,'disk_floor_bytes':32*2**30,
        'seed':2026101302,'real_recipe_entry_cap':85373862,'expected_real_sha256':EXPECTED_20M,
        'no_matches_or_strength_claim':True})
    try:
        for bb in (20,100,200):
            common=['train','--stack-bb',bb,'--seed',5,'--average-rule','opponent-sampled','--recovery','--nodes',1_000_000]
            paths=[]
            for name, binary, flags in [('main',args.baseline,[]),('candidate',BINARY,[]),('enabled',BINARY,['--freeze-at-cap'])]:
                path=args.out/f'hu{bb}-{name}.gz'; paths.append(path)
                execute([binary,*common,*flags,'--out',path],args.out,f'hu{bb}-{name}')
            pins=[record(p) for p in paths]
            if len({p['sha256'] for p in pins}) != 1: raise ValueError(f'HU{bb} v1 checkpoint byte parity failed')
            frozen=args.out/f'hu{bb}-frozen.gz'
            telemetry=args.out/f'hu{bb}-frozen.telemetry.jsonl'
            operation=execute([BINARY,*common,'--freeze-at-cap','--max-entries',20000,'--milestones','500000',
                '--out',args.out/f'hu{bb}-frozen-{{nodes}}.gz','--telemetry',telemetry],args.out,f'hu{bb}-frozen')
            frozen=args.out/f'hu{bb}-frozen-1000000.gz'
            pins.append(record(frozen))
            receipts=[json.loads(s) for s in telemetry.read_text().splitlines()]
            if len({r['diagnostics']['entries'] for r in receipts}) != 1: raise ValueError('small frozen table grew')
            summary['small'].append({'bb':bb,'checkpoints':pins,'frozen_operation':operation,
                'receipts':receipts})
        if args.real_tables:
            tables={street:{'path':str(args.real_tables/f'{street}-k50.bin'),
                'bytes':(args.real_tables/f'{street}-k50.bin').stat().st_size,
                'sha256':file_hash(args.real_tables/f'{street}-k50.bin')} for street in EQUITY_K50_SHA256}
            if any(tables[s]['sha256'] != v for s,v in EQUITY_K50_SHA256.items()): raise ValueError('pinned K50 table hash differs')
            put(args.out/'real-table-inputs.json',tables)
            fresh=args.out/'hu200-k50-fresh20m.gz'
            telemetry=args.out/'hu200-k50-fresh20m.telemetry.jsonl'
            common=[BINARY,'train','--stack-bb',200,'--seed',2026101302,'--roots-per-seat',1,
                '--average-rule','opponent-sampled','--card-buckets',args.real_tables,'--freeze-at-cap']
            fresh_op=execute([*common,'--nodes',20_000_000,'--max-entries',85373862,'--out',fresh,'--telemetry',telemetry],args.out,'real-fresh')
            fresh_pin=record(fresh)
            if fresh_pin['sha256'] != EXPECTED_20M or fresh_pin['bytes'] != 270143830:
                raise ValueError('#250 fresh 20M hash/size differs')
            initial=key_digest(fresh)
            frozen=args.out/'hu200-k50-frozen20m2.gz'
            frozen_op=execute([*common,'--nodes',20_200_000,'--max-entries',initial['entries'],
                '--resume',fresh,'--resume-sha256',EXPECTED_20M,'--out',frozen,
                '--telemetry',args.out/'hu200-k50-frozen20m2.telemetry.jsonl'],args.out,'real-frozen')
            final=key_digest(frozen)
            if initial['sorted_key_sha256'] != final['sorted_key_sha256'] or initial['entries'] != final['entries']:
                raise ValueError('real frozen key set changed')
            h=header(frozen); state=h['frozen_table']; coverage=h['native_state']
            if coverage['coverage_start'][2]+sum(coverage['traverser_visits_by_street']) != final['stored_visits']+state['discarded_traverser_visits']:
                raise ValueError('real frozen coverage invariant failed')
            summary['real_model']={'fresh':fresh_pin,'frozen':record(frozen),'initial_keys':initial,'final_keys':final,
                'fresh_operation':fresh_op,'frozen_operation':frozen_op,'tables':tables,'coverage_verified':True}
        summary['status']='passed'
    except Exception as problem:
        summary['status']='failed'; summary['failure']=str(problem); raise
    finally: put(args.out/'summary.json',summary)


if __name__ == '__main__': main()
