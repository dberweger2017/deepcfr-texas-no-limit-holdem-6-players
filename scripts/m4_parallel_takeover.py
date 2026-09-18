"""One-time M4 handoff; deploy under ignored results to preserve trainer source."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

# This file runs from results on the pinned training checkout.
sys.path.insert(0, str(Path.cwd()))
from scripts.local_fullgame import GIB, guarded_run, resource_failure, used_bytes, write_json
from src.arena.artifacts import source_fingerprint


def process(pid):
    result = subprocess.run(
        ['ps', '-p', str(pid), '-o', 'ppid=,pgid=,rss=,state=,etime=,command='],
        capture_output=True, text=True, check=False,
    )
    if not result.stdout.strip():
        return None
    parent, group, rss, state, elapsed, command = result.stdout.strip().split(None, 5)
    days, sep, elapsed = elapsed.rpartition('-')
    parts = [int(p) for p in elapsed.split(':')]
    seconds = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
    if sep:
        seconds += int(days) * 86400
    return dict(parent=int(parent), group=int(group), rss=int(rss)*1024,
                state=state, seconds=seconds, command=command)


def train_command(out, seed):
    return [sys.executable, '-m', 'scripts.train_holdem', '--plan',
            str(out/'plan.json'), '--seed', str(seed), '--out', str(out/f'seed-{seed}')]


def run(out, supervisor, worker):
    seeds = (2026091802, 2026091803)
    manifest = json.loads((out/'manifest.json').read_text())
    if source_fingerprint() != manifest['source_sha256']:
        raise RuntimeError('Pinned training source changed')
    if (out/f'seed-{seeds[1]}').exists() or (out/'parallel-handoff.json').exists():
        raise RuntimeError('Second seed or handoff already exists')
    parent, first = process(supervisor), process(worker)
    if not parent or '-m scripts.local_fullgame --plan' not in parent['command']:
        raise RuntimeError('Unexpected original supervisor')
    if (not first or first['parent'] != supervisor or first['group'] != worker
            or f'--seed {seeds[0]}' not in first['command']):
        raise RuntimeError('Unexpected existing worker')
    if shutil.disk_usage(out).free < 20*GIB:
        raise RuntimeError('Need 20 GiB free for second seed')
    record = dict(state='preparing', supervisor_pid=os.getpid(), old_supervisor_pid=supervisor,
                  existing_worker_pid=worker, authorization='Owner requested concurrent planned seeds',
                  combined_rss_limit_bytes=9*GIB, source_sha256=manifest['source_sha256'],
                  utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    owned = False
    stopped = False
    children = {}
    handles = []
    live = {}
    status = dict(state='running', phase='concurrent_training', seeds=[], promoted=False)
    try:
        os.kill(supervisor, signal.SIGSTOP)
        stopped = True
        # Confirm the parent is stopped before taking ownership or starting seed 2.
        for _ in range(50):
            current = process(supervisor)
            if current and 'T' in current['state']:
                break
            time.sleep(.02)
        else:
            raise RuntimeError('Original supervisor did not stop')
        first = process(worker)
        if not first or first['parent'] != supervisor:
            raise RuntimeError('Worker changed during handoff')
        for name in ('status.json', f'seed-{seeds[0]}.json'):
            shutil.copyfile(out/name, out/(name+'.before-parallel'))
        prior = json.loads((out/f'seed-{seeds[0]}.json').read_text())
        if prior['pid'] != worker or prior['status'] != 'running':
            raise RuntimeError('Original worker record changed')
        write_json(out/'parallel-handoff.json', record)
        # Kill only the stopped coordinator; its separate worker group survives.
        os.kill(supervisor, signal.SIGKILL)
        owned = True
        live[seeds[0]] = dict(pid=worker, start=time.monotonic()-first['seconds'],
                              prior=prior, adopted=True)
        second_command = train_command(out, seeds[1])
        log = (out/f'seed-{seeds[1]}.log').open('xb')
        handles.append(log)
        child = subprocess.Popen(second_command, stdout=log, stderr=log, start_new_session=True)
        children[seeds[1]] = child
        live[seeds[1]] = dict(pid=child.pid, start=time.monotonic(),
                              prior=dict(command=second_command, limit_seconds=21600), adopted=False)
        record.update(state='transferred', second_worker_pid=child.pid,
                      first_elapsed_seconds=first['seconds'])
        write_json(out/'parallel-handoff.json', record)
        while live:
            total_rss = 0
            stored, free = used_bytes(out), shutil.disk_usage(out).free
            for seed, item in list(live.items()):
                pid = item['pid']
                state = process(pid)
                exit_code = children[seed].poll() if seed in children else None
                finished = (state is None or 'Z' in state['state'] or exit_code is not None)
                elapsed = time.monotonic()-item['start']
                rss = 0 if finished else state['rss']
                total_rss += rss
                row = item['prior']
                row.update(pid=pid, seconds=elapsed, rss_bytes=rss,
                           peak_sampled_rss_bytes=max(row.get('peak_sampled_rss_bytes', 0), rss),
                           status='running', supervisor_pid=os.getpid(), adopted=item['adopted'])
                reason = resource_failure(elapsed, 21600, rss, free, stored)
                if finished:
                    result_path = out/f'seed-{seed}'/'result.json'
                    complete = result_path.exists() and json.loads(result_path.read_text()).get('complete')
                    if not complete or (seed in children and exit_code != 0):
                        raise RuntimeError(f'Seed {seed} failed: exit={exit_code}, complete={complete}')
                    row.update(status='complete', returncode=exit_code)
                    status['seeds'].append(seed)
                    del live[seed]
                elif reason:
                    raise RuntimeError(f'Seed {seed}: {reason}')
                write_json(out/f'seed-{seed}.json', row)
            if total_rss > 9*GIB:
                raise RuntimeError('combined_process_memory_limit')
            status.update(active_seeds=list(live), combined_rss_bytes=total_rss)
            write_json(out/'status.json', status)
            if live:
                time.sleep(5)
        remaining = 1800.0
        for seed in seeds:
            status.update(phase='verification_and_final_test', active_seeds=[seed])
            write_json(out/'status.json', status)
            verified = guarded_run(
                [sys.executable, '-m', 'scripts.local_fullgame', '--verify-job',
                 str(out/f'seed-{seed}'), '--out', str(out/f'final-{seed}')],
                out/f'final-{seed}.log', seconds=remaining,
            )
            remaining -= verified['seconds']
        summaries = [json.loads((out/f'final-{s}'/'final-summary.json').read_text()) for s in seeds]
        write_json(out/'summary.json', dict(seeds=summaries, promoted=False,
                   competence_check_passed=all(s['competence_check_passed'] for s in summaries)))
        status.update(state='complete', active_seeds=[])
    except BaseException as exc:
        status.update(state='failed', error=repr(exc))
        if owned:
            for item in live.values():
                pid = item['pid']
                state = process(pid)
                if state and 'Z' not in state['state']:
                    try:
                        os.killpg(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            time.sleep(5)
            for item in live.values():
                pid = item['pid']
                state = process(pid)
                if state and 'Z' not in state['state']:
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                item['prior'].update(status='failed', error=repr(exc))
            for seed, item in live.items():
                write_json(out/f'seed-{seed}.json', item['prior'])
        elif stopped:
            os.kill(supervisor, signal.SIGCONT)
        raise
    finally:
        for handle in handles:
            handle.close()
        if owned:
            write_json(out/'status.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--supervisor', type=int, required=True)
    parser.add_argument('--worker', type=int, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    run(args.out.resolve(), args.supervisor, args.worker)


if __name__ == '__main__':
    main()
