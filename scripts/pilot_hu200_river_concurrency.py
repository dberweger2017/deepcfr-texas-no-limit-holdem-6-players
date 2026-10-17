"""Measure fixed24 throughput at two, then optionally three, unchanged searches."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import shutil
from time import monotonic

import numpy as np
import psutil

from scripts.check_hu200_river_transport import same_numbers
from scripts.evaluate_hu100_direct import put
from scripts.pilot_hu200_river_search import MODEL_SHA256, measure_spot
from src.blueprint.abstraction import HU200_EQUITY_SCHEMA
from src.blueprint.action_translation import TranslationOptions
from src.blueprint.average import AveragePolicy
from src.blueprint.river_ledger import DecisionLedger, FamilyPeak
from src.blueprint.river_evidence import COMPACT_RETENTION_RULE
from src.policies.files import file_hash


class PilotStopped(BaseException):
    """Abort without converting a sibling's correctness failure to fallback."""


def worker(args, indices, barrier, stop):
    try:
        return worker_body(args, indices, barrier, stop)
    except BaseException:
        stop.set()
        barrier.abort()
        raise


def worker_body(args, indices, barrier, stop):
    started = monotonic()
    base = AveragePolicy(args.model,MODEL_SHA256,expected_schema=HU200_EQUITY_SCHEMA,
        card_buckets=args.tables,cache=args.cache,cache_read_only=True,
        translation=TranslationOptions(512,128))
    setup = monotonic()-started
    plan = json.loads(args.plan.read_text())
    barrier.wait(timeout=300)
    started = monotonic()
    rows = []
    checked_at = 0.

    def cancel_check():
        nonlocal checked_at
        now = monotonic()
        if now-checked_at >= .02:
            checked_at = now
            if stop.is_set():
                raise PilotStopped('Sibling pilot worker failed')

    for index in indices:
        if stop.is_set():
            raise PilotStopped('No further spot after pilot failure')
        row = measure_spot(index,plan['sample'][index],base,args.binary,args.out,
                           compact_profiles=True,cancel_check=cancel_check,
                           profile_retention=COMPACT_RETENTION_RULE if args.sample_native_bodies else None)
        baseline = json.loads((args.sequential/str(index)/'decision.json').read_text())
        for key in ('search_action','blueprint_action','fallback','changed_action'):
            if row[key] != baseline[key]:
                # Action dataclasses have already been encoded by measure_spot.
                raise ValueError('Concurrent same-seed output differs: '+key)
        for key in ('action_values_chips','model_ev_chips','blueprint_ev_chips','model_ev_delta_chips'):
            same_numbers(row[key],baseline[key])
        paths = list((args.out/str(index)/'ledger').glob('*.json'))
        old_paths = list((args.parity/str(index)/'ledger').glob('*.json'))
        if len(paths) != 1 or len(old_paths) != 1:
            raise ValueError('Ambiguous concurrent or parity ledger')
        current = DecisionLedger(paths[0].parent,replay=True).read(paths[0].stem)
        old = DecisionLedger(old_paths[0].parent,replay=True).read(old_paths[0].stem)
        for key in ('menu','holdings'):
            if current['matrix'][key] != old['matrix'][key]:
                raise ValueError('Concurrent policy identity differs')
        same_numbers(current['matrix']['probabilities'],old['matrix']['probabilities'])
        if any(current[key] != old[key] for key in ('fallback','cause','supported')):
            raise ValueError('Concurrent played support/fallback differs')
        rows.append(row)
        print(json.dumps(dict(workers=args.workers,index=index,seconds=row['complete_seconds'])),flush=True)
    return dict(setup_seconds=setup,started=started,finished=monotonic(),rows=rows)


def run(args):
    root = Path(__file__).resolve().parents[1]
    declaration = json.loads((root/'docs/reports/hu200-river-search-artifacts/optimization-declaration.json').read_text())
    if file_hash(args.plan) != declaration['full_plan_sha256'] or file_hash(args.binary) != declaration['external_binary_sha256']:
        raise ValueError('Concurrent sample or executable differs')
    parity = json.loads((args.parity/'summary.json').read_text())
    sequential = json.loads((args.sequential/'summary.json').read_text())
    input_pins = {}
    for name, folder in (('parity',args.parity),('sequential',args.sequential),
                          *([('predecessor',args.predecessor)] if args.workers == 3 and args.predecessor else [])):
        inputs = json.loads((folder/'inputs.json').read_text())
        if (inputs['plan_sha256'] != file_hash(args.plan)
                or inputs['binary_sha256'] != file_hash(args.binary)):
            raise ValueError('Concurrent admission inputs differ: '+name)
        if args.sample_native_bodies and inputs.get('profile_retention') != COMPACT_RETENTION_RULE:
            raise ValueError('Retention concurrency evidence differs: '+name)
        input_pins[name] = dict(inputs_sha256=file_hash(folder/'inputs.json'),
                               summary_sha256=file_hash(folder/'summary.json'))
    if (parity['status'] != 'exact-parity' or parity['decisions'] != 24
            or not all(sequential[k] for k in ('latency_target_met','memory_target_met',
                'telemetry_target_met','all_search_work_completed'))
            or sequential['latency_target_seconds'] != 10):
        raise ValueError('Concurrency requires the completed amended gate')
    args.out.mkdir(parents=True,exist_ok=False)
    capacity = min(13*1024**3,16*1024**3-3_000_000_000)
    if args.workers == 2:
        projected_rss = sequential['peak_family_rss_bytes']*2
        reserve = 2_000_000_000
        projected_available = sequential['minimum_host_available_bytes']-reserve
    else:
        previous = json.loads((args.predecessor/'summary.json').read_text()) if args.predecessor else {}
        if previous.get('workers') != 2 or not previous.get('memory_target_met') or not previous.get('outputs_match_parity'):
            raise ValueError('Three workers require successful two-worker evidence')
        projected_rss = previous['peak_family_rss_bytes']*1.5
        incremental = max(0,sequential['minimum_host_available_bytes']-previous['minimum_host_available_bytes'])
        reserve = incremental+500_000_000
        projected_available = previous['minimum_host_available_bytes']-reserve
    current_available = psutil.virtual_memory().available
    projected_available = min(projected_available,current_available-reserve)
    memory_admitted = projected_rss <= capacity and projected_available >= 3_000_000_000
    receipts = [json.loads(p.read_text()) for p in args.sequential.glob('*/native/*/receipt.json')]
    if args.sample_native_bodies:
        scratch = max(sum(pin['bytes'] for pin in r['profile_retention']['generated_files'].values())
            +r['profile_retention']['request_uncompressed']['bytes'] for r in receipts)
    else:
        scratch = max(sum(p.stat().st_size for p in Path(r['path']).iterdir() if p.is_file()) for r in receipts)
    retained = sum(p.stat().st_size for p in args.sequential.rglob('*') if p.is_file())
    required_disk = 16*1024**3 + retained + args.workers*scratch + 512*1024**2
    disk_free = shutil.disk_usage(args.out).free
    admitted = memory_admitted and disk_free >= required_disk
    put(args.out/'inputs.json',dict(plan_sha256=file_hash(args.plan),binary_sha256=file_hash(args.binary),
        evidence=input_pins,profile_retention=COMPACT_RETENTION_RULE if args.sample_native_bodies else None,source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    put(args.out/'admission.json',dict(admitted=admitted,workers=args.workers,
        memory_admitted=memory_admitted,disk_admitted=disk_free >= required_disk,
        disk_free_bytes=disk_free,required_disk_free_bytes=required_disk,
        per_worker_transient_disk_bytes=scratch,retained_fixed24_bytes=retained,
        projected_family_rss_bytes=projected_rss,capacity_bytes=capacity,
        projected_host_available_bytes=projected_available,
        fresh_host_available_bytes=current_available,
        rule='two: twice sequential RSS and2GB host reserve; three:1.5x two-worker RSS and observed incremental host-use plus0.5GB reserve',
        plan_sha256=file_hash(args.plan),binary_sha256=file_hash(args.binary),no_strength_hands=True))
    if not admitted:
        put(args.out/'summary.json',dict(status='not-admitted',workers=args.workers,
            memory_target_met=False,no_strength_hands=True))
        return
    context = mp.get_context('spawn')
    setup_started = monotonic()
    with FamilyPeak() as memory, context.Manager() as manager:
        barrier = manager.Barrier(args.workers)
        stop = manager.Event()
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=context) as pool:
            futures = [pool.submit(worker,args,list(range(w,24,args.workers)),barrier,stop)
                       for w in range(args.workers)]
            try:
                results = [f.result() for f in as_completed(futures)]
            except BaseException:
                stop.set()
                barrier.abort()
                for future in futures:
                    future.cancel()
                raise
    rows = sorted([r for worker_result in results for r in worker_result['rows']],key=lambda r:r['sample_index'])
    if [r['sample_index'] for r in rows] != list(range(24)):
        raise ValueError('Concurrent fixed24 incomplete')
    times = [r['complete_seconds'] for r in rows]
    wall = max(r['finished'] for r in results)-min(r['started'] for r in results)
    result = dict(status='complete',workers=args.workers,decisions=24,
        seconds=dict(p50=float(np.percentile(times,50)),p95=float(np.percentile(times,95)),max=max(times)),
        decision_work_wall_seconds=wall,decisions_per_second=24/wall,
        throughput_scope='includes per-spot parity validation, evidence writes, and contention; excludes fixed input verification',
        setup_and_work_wall_seconds=monotonic()-setup_started,
        worker_setup_seconds=[r['setup_seconds'] for r in results],
        peak_family_rss_bytes=memory.peak,minimum_host_available_bytes=memory.minimum_available_bytes,
        memory_target_met=memory.peak<=16*1024**3-3_000_000_000 and memory.minimum_available_bytes>=3_000_000_000,
        outputs_match_parity=True,fallback_decisions=sum(r['fallback'] for r in rows),
        no_strength_hands=True,per_search_threads=6,per_search_iterations=50)
    put(args.out/'summary.json',result)
    print(json.dumps(result),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan','model','cache','tables','binary','out','sequential','parity'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--predecessor',type=Path)
    parser.add_argument('--workers',type=int,choices=(2,3),required=True)
    parser.add_argument("--sample-native-bodies",action="store_true")
    args = parser.parse_args()
    if subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip() != 'Apple M4':
        raise ValueError('Concurrency pilot requires the authorized M4')
    if not any('scripts.guard_m4_research' in p.cmdline() for p in psutil.Process().parents()):
        raise ValueError('Concurrency pilot requires the resource guard')
    with (Path.home()/'Local/.hu20-m4-research.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == '__main__':
    main()
