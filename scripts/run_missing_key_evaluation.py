"""Detached HU200 fallback evaluation with server-side resource admission.

Run workers at nice 10, up to four, without changing any shared model/cache bytes.
Pilot output contains costs and SDs only. Final readout is separate and requires
all hand replays and exact action/decision/settlement reproduction to pass.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import monotonic, sleep

import psutil
from scripts.evaluate_hu100_direct import PANEL_OPPONENTS, audit, execute, put
from src.policies.files import file_hash


def breached(sample, baseline_swap, low_since, now):
    low_since = (now if low_since is None else low_since) if sample['available_bytes'] < 2_000_000_000 else None
    reason = ('memory-below-2GB-for-60s' if low_since is not None and now - low_since >= 60 else
              'swap-growth-above-1GB' if sample['swap_used_bytes'] - baseline_swap > 1_000_000_000 else
              'disk-below-20GB' if sample['disk_free_bytes'] < 20_000_000_000 else None)
    return reason, low_since


def worker(a):
    specs = json.loads(a.specs.read_text())
    started = monotonic()
    registry, primary = execute(specs, a.blocks, a.root, 'missing-key-cc', a.out/'play', a.source,
        stack_bb=200, opponent=a.opponent, cache=a.policy_cache, cache_read_only=True)
    replay = audit(a.out/'play', a.out/'replay.json', report=False, cost_only=a.pilot)
    _, repeated = execute(specs, a.blocks, a.root, 'missing-key-cc', a.out/'reproduction', a.source,
        registry, a.out/'play', stack_bb=200, opponent=a.opponent, cache=a.policy_cache, cache_read_only=True)
    # Readout is deferred to the controller's separate --readout step, after every
    # worker verifies. The pilot never produces a center, CI or arm outcome.
    put(a.out/'verified.json', {'status': 'verified', 'seconds': monotonic()-started,
        'primary': primary, 'reproduction': repeated, 'replay_seconds': replay['seconds'],
        'blocks': a.blocks, 'root': a.root, 'opponent': a.opponent,
        **({'paired_sd_bb_per_100': replay['paired_sd_bb_per_100']} if a.pilot else {})})


def guard(root, processes, stream, baseline_swap, low_since, started):
    now = monotonic()
    rss = pss = 0
    for p in processes:
        if p.poll() is not None: continue
        try:
            parent = psutil.Process(p.pid)
            for member in [parent, *parent.children(recursive=True)]:
                mem = member.memory_full_info(); rss += mem.rss; pss += getattr(mem, 'pss', 0)
        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
    sample = {'elapsed_seconds': now-started, 'available_bytes': psutil.virtual_memory().available,
              'swap_used_bytes': psutil.swap_memory().used, 'disk_free_bytes': psutil.disk_usage(root).free,
              'owned_family_rss_bytes': rss, 'owned_family_pss_bytes': pss,
              'active_workers': sum(p.poll() is None for p in processes)}
    reason, low_since = breached(sample, baseline_swap, low_since, now)
    stream.write(json.dumps({**sample, 'stop': reason}, sort_keys=True)+'\n'); stream.flush()
    return reason, low_since, sample


def terminate(processes):
    for p in processes:
        if p.poll() is None: os.killpg(p.pid, signal.SIGTERM)
    deadline = monotonic()+10
    while any(p.poll() is None for p in processes) and monotonic() < deadline: sleep(.2)
    for p in processes:
        if p.poll() is None: os.killpg(p.pid, signal.SIGKILL)
    for p in processes: p.wait()


def campaign(a):
    plan = json.loads(a.plan.read_text())
    if plan['source'] != a.source or plan['workers'] > 4 or plan['workers'] < 1:
        raise ValueError('Foreign source or worker limit')
    if file_hash(a.specs) != plan['specs_sha256']: raise ValueError('Model specs changed')
    if not a.pilot:
        receipt = json.loads(a.publication.read_text())
        if receipt['plan_sha256'] != file_hash(a.plan) or a.plan.read_text() not in receipt['body']:
            raise ValueError('Canonical plan publication differs')
        body = subprocess.check_output(['gh', 'pr', 'view', receipt['pr_url'], '--json', 'body'], text=True)
        if json.loads(body)['body'] != receipt['body']: raise ValueError('Published plan changed before admission')
    a.out.mkdir(parents=True, exist_ok=False)
    baseline_swap = psutil.swap_memory().used
    started = monotonic(); low_since = None; processes = []; handles = []; results = {}; samples = []
    queue = list(plan['jobs']); failure = None
    with (a.out/'guards.jsonl').open('x') as stream:
        try:
            while queue or any(p.poll() is None for p in processes):
                reason, low_since, sample = guard(a.out, processes, stream, baseline_swap, low_since, started)
                samples.append(sample)
                if reason: raise RuntimeError(reason)
                if monotonic()-started > plan['wall_seconds_cap']: raise RuntimeError('declared-wall-cap')
                for name, p in [(p.job_name, p) for p in processes]:
                    if p.poll() is not None and name not in results:
                        if p.returncode != 0: raise RuntimeError('worker-failed:'+name)
                        results[name] = json.loads((a.out/name/'verified.json').read_text())
                while queue and sum(p.poll() is None for p in processes) < plan['workers']:
                    job = queue.pop(0); name = job['name']
                    log = (a.out/(name+'.log')).open('x'); handles.append(log)
                    cmd = [sys.executable, '-m', 'scripts.run_missing_key_evaluation', '--worker',
                        '--specs', str(a.specs), '--source', a.source, '--blocks', str(job['blocks']),
                        '--root', str(job['root']), '--out', str(a.out/name), '--policy-cache', str(a.policy_cache)]
                    if job.get('opponent'): cmd += ['--opponent', job['opponent']]
                    if a.pilot: cmd += ['--pilot']
                    p = subprocess.Popen(['nice', '-n', '10', *cmd], stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                                                    'MKL_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
                    p.job_name = name; processes.append(p)
                if any(p.poll() is None for p in processes): sleep(5)
            for p in processes:
                if p.returncode != 0: raise RuntimeError('worker-failed:'+p.job_name)
                results[p.job_name] = json.loads((a.out/p.job_name/'verified.json').read_text())
        except BaseException as error:
            failure = str(error); terminate(processes)
        finally:
            for handle in handles: handle.close()
    put(a.out/'campaign.json', {'status': 'stopped' if failure else 'verified', 'failure': failure,
        'seconds': monotonic()-started, 'pilot': a.pilot, 'source': a.source,
        'plan_sha256': file_hash(a.plan), 'baseline_swap_bytes': baseline_swap,
        'guard_summary': {'readings': len(samples), 'min_available_bytes': min(s['available_bytes'] for s in samples),
            'max_swap_growth_bytes': max(s['swap_used_bytes']-baseline_swap for s in samples),
            'min_disk_free_bytes': min(s['disk_free_bytes'] for s in samples),
            'max_owned_family_rss_bytes': max(s['owned_family_rss_bytes'] for s in samples),
            'max_owned_family_pss_bytes': max(s['owned_family_pss_bytes'] for s in samples),
            'max_active_workers': max(s['active_workers'] for s in samples)}, 'jobs': results})
    if failure: raise SystemExit(1)


def readout(a):
    campaign = json.loads((a.out/'campaign.json').read_text())
    if campaign['status'] != 'verified' or campaign['pilot']:
        raise ValueError('All final jobs must verify before readout')
    # Run this inside the same resource controller as a worker; every retained
    # hand has already replayed and reproduced before outcomes become available.
    contrasts = {}
    for name in campaign['jobs']:
        contrasts[name] = audit(a.out/name/'play', a.out/name/'readout.json', report=True)
    direct = contrasts['direct']; panel = {k:v for k,v in contrasts.items() if k != 'direct'}
    passed = direct['ci95'][0] > 0 and not any(v['ci95'][1] < -20 for v in panel.values())
    put(a.out/'readout.json', {'rule_passed': passed, 'contrasts': contrasts,
        'rule': 'direct paired lower nominal 95% > 0 and no scripted paired upper nominal 95% < -20 BB/100'})


def guarded_readout(a):
    started = monotonic(); baseline = psutil.swap_memory().used; low = None; samples = []
    with (a.out/'readout.log').open('x') as log, (a.out/'readout-guards.jsonl').open('x') as stream:
        cmd = ['nice', '-n', '10', sys.executable, '-m', 'scripts.run_missing_key_evaluation',
               '--readout', '--out', str(a.out)]
        reason, low, sample = guard(a.out, [], stream, baseline, low, started); samples.append(sample)
        if reason: raise RuntimeError(reason)
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                reason, low, sample = guard(a.out, [process], stream, baseline, low, started); samples.append(sample)
                if reason: raise RuntimeError(reason)
                sleep(5)
            if process.returncode != 0: raise RuntimeError('readout-failed')
        except BaseException:
            terminate([process]); raise
    put(a.out/'readout-guard-summary.json', {'readings': len(samples), 'seconds': monotonic()-started,
        'min_available_bytes': min(s['available_bytes'] for s in samples),
        'max_swap_growth_bytes': max(s['swap_used_bytes']-baseline for s in samples),
        'min_disk_free_bytes': min(s['disk_free_bytes'] for s in samples)})


def guarded_command(a):
    a.out.mkdir(parents=True, exist_ok=False)
    started = monotonic(); baseline = psutil.swap_memory().used; low = None; samples = []; failure = None
    with (a.out/'command.log').open('x') as log, (a.out/'guards.jsonl').open('x') as stream:
        reason, low, sample = guard(a.out, [], stream, baseline, low, started); samples.append(sample)
        if reason: raise RuntimeError(reason)
        process = subprocess.Popen(['nice', '-n', '10', *a.guard_command], stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                reason, low, sample = guard(a.out, [process], stream, baseline, low, started); samples.append(sample)
                if reason: raise RuntimeError(reason)
                sleep(5)
            if process.returncode != 0: raise RuntimeError('command-failed')
        except BaseException as error:
            failure = str(error); terminate([process])
    put(a.out/'receipt.json', {'status': 'stopped' if failure else 'verified', 'failure': failure,
        'readings': len(samples), 'seconds': monotonic()-started, 'command': a.guard_command,
        'baseline_swap_bytes': baseline, 'min_available_bytes': min(s['available_bytes'] for s in samples),
        'max_swap_growth_bytes': max(s['swap_used_bytes']-baseline for s in samples),
        'min_disk_free_bytes': min(s['disk_free_bytes'] for s in samples)})
    if failure: raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--specs',type=Path);p.add_argument('--policy-cache',type=Path)
    p.add_argument('--source');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--plan',type=Path);p.add_argument('--publication',type=Path)
    p.add_argument('--worker',action='store_true');p.add_argument('--readout',action='store_true')
    p.add_argument('--guarded-readout',action='store_true')
    p.add_argument('--pilot',action='store_true');p.add_argument('--blocks',type=int);p.add_argument('--root',type=int)
    p.add_argument('--opponent',choices=PANEL_OPPONENTS)
    p.add_argument('--guard-command', nargs=argparse.REMAINDER)
    a=p.parse_args()
    if a.guard_command: guarded_command(a)
    elif a.guarded_readout: guarded_readout(a)
    elif a.worker: worker(a)
    elif a.readout: readout(a)
    else: campaign(a)


if __name__ == '__main__': main()
