"""Serial M4 evaluation queue; frozen hands, native replay, explicit failures."""

import argparse
from collections import Counter
import fcntl
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.evaluate_hu20_stackoff import recorded_hand, swap_bytes
from scripts.hu20_platform_pilot import peak_rss, write
from src.diagnostics.saved_hu20 import load_saved
from src.diagnostics.selective_stackoff import VERSION


def execute_task(plan, spec, stage, out):
    out.mkdir(parents=True, exist_ok=False)
    before_swap = swap_bytes()
    result = dict(status='running', stage=stage, spec=spec, started=time.time(), attempts=[], hands=0)
    write(out/'attempt.json', result)
    panels = plan[stage+'_panels']
    if VERSION != plan['opponent_version']:
        raise ValueError('Scripted stress opponent version changed')
    last_check = 0
    def guard():
        nonlocal last_check
        if time.time()-last_check < 2:
            return
        last_check = time.time()
        if peak_rss() >= 10.5*2**30:
            raise MemoryError('M4 evaluator RSS')
        if shutil.disk_usage(out).free < 8*2**30:
            raise OSError('M4 evaluator disk')
        if swap_bytes()-before_swap > .5*2**30:
            raise MemoryError('M4 evaluator swap growth')
    try:
        source, visits = load_saved(spec, Path('/'), guard)
        with gzip.open(out/'hands.jsonl.gz', 'wt') as handle:
            for panel in panels:
                attempt = dict(panel=panel['name'], requested_blocks=panel['blocks'], completed_blocks=0)
                result['attempts'].append(attempt)
                for block in range(panel['blocks']):
                    for rotation in (0, 1):
                        guard()
                        row = recorded_hand(source, visits, spec, panel, block, rotation, plan, guard)
                        row['campaign_stage'] = stage
                        row['milestone'] = spec['milestone']
                        row['seed'] = spec['seed']
                        # Additional target all-in/large-call exposure, with
                        # actual state retained in the v1 observation snapshots.
                        row['large_calls'] = sum(a['logical_player']==0 and a['kind']=='call' and
                            a['observation']['call_amount']>=800 for a in row['actions'])
                        row['allin_calls'] = sum(a['logical_player']==0 and a['kind']=='call' and
                            a['observation']['call_amount']==a['observation']['stack'] and
                            a['observation']['stack']>0 for a in row['actions'])
                        handle.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
                        handle.flush()
                        result['hands'] += 1
                        if row['status'] != 'complete':
                            raise RuntimeError(row.get('error', 'Failed frozen hand'))
                    attempt['completed_blocks'] += 1
                    if block % 32 == 0:
                        write(out/'progress.json', dict(hands=result['hands'], current=attempt, heartbeat=time.time()))
        result['status'] = 'complete'
    except Exception as exc:
        result.update(status='incomplete', failure=f'{type(exc).__name__}: {exc}')
    result.update(finished=time.time(), peak_rss_bytes=peak_rss(), swap_growth_bytes=swap_bytes()-before_swap)
    write(out/'result.json', result)
    return result['status'] != 'complete'


def queue(a):
    if sys.platform != 'darwin':
        raise ValueError('M4 evaluation only')
    plan = json.loads(a.plan.read_text())
    a.root.mkdir(parents=True, exist_ok=True)
    lock_path = Path('/tmp/DR_RESEARCH_M4_HEAVY.lock')
    queue_path = a.root/'evaluation-queue'
    queue_path.mkdir(exist_ok=True)
    for parent in plan['parents']:
        path = queue_path/(str(parent['seed'])+'-100000000.json')
        if not path.exists():
            write(path, dict(spec=dict(parent, dual_menu_telemetry=True), destination='M4'))
    attempted = set()
    outputs = a.root/'evaluation'
    outputs.mkdir(exist_ok=True)
    for folder in outputs.iterdir():
        if folder.is_dir():
            attempted.add(folder.name)
    while True:
        state = dict(status='waiting', heartbeat=time.time(), pid=os.getpid())
        done = a.root/'operator-finished.json'
        rentals_done = done.exists() and json.loads(done.read_text())['status']=='training-complete'
        jobs = []
        for path in sorted(queue_path.glob('*.json')):
            task = json.loads(path.read_text())
            if task['destination'] != 'M4':
                # Retrieval back to M4 must precede heavy reads. Do not load
                # policies or run evaluation on the travelling M1.
                continue
            spec = task['spec']
            for stage in ('light', 'broad', 'heldout'):
                if stage != 'light' and not rentals_done:
                    continue
                if spec['milestone'] not in plan[stage+'_totals']:
                    continue
                ident = stage+'-'+str(spec['seed'])+'-'+str(spec['milestone'])
                if ident not in attempted:
                    jobs.append((stage, spec, ident))
        if jobs:
            # Only one scientific M4 subprocess; transport and provider control
            # remain lightweight. Use the shared owner coordination note.
            with lock_path.open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(30)
                    continue
                stage, spec, ident = jobs[0]
                if shutil.disk_usage(a.root).free < 12*2**30:
                    state['status'] = 'disk-paused'
                    write(a.root/'evaluation-supervisor.json', state)
                    time.sleep(60)
                    continue
                command_file = a.root/(ident+'.task.json')
                write(command_file, dict(stage=stage, spec=spec))
                with (a.root/(ident+'.log')).open('w') as log:
                    c = subprocess.Popen([sys.executable, '-m', 'scripts.evaluate_hu20_500m', 'task',
                        '--plan', str(a.plan.resolve()), '--root', str(outputs/ident), '--task', str(command_file)],
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    state.update(status='running', task=ident, child_pid=c.pid)
                    write(a.root/'evaluation-supervisor.json', state)
                    with Path('/tmp/DR_RESEARCH_M4_COORDINATION.txt').open('a') as note:
                        note.write(f'\nDoctor Research #136 evaluator CLAIM {ident}, PID {c.pid}; one heavy worker.\n')
                    # Native per-hand guard checks itself; also guard the initial
                    # large policy load which precedes those checks.
                    before_swap = swap_bytes()
                    stopped = None
                    while c.poll() is None:
                        try:
                            memory = sum(int(v) for v in subprocess.check_output(
                                ['ps', '-o', 'rss=', '-p', str(c.pid)], text=True).split())*1024
                        except subprocess.CalledProcessError:
                            memory = 0
                        if (memory >= 10.5*2**30 or swap_bytes()-before_swap > .5*2**30
                                or shutil.disk_usage(a.root).free < 8*2**30):
                            if stopped is None:
                                c.terminate()
                                stopped = time.time()
                            elif time.time()-stopped > 30:
                                c.kill()
                        state.update(heartbeat=time.time(), child_rss_bytes=memory)
                        write(a.root/'evaluation-supervisor.json', state)
                        time.sleep(2)
                    state.update(exit=c.returncode, finished=time.time())
                    write(outputs/(ident+'.exit.json'), state)
                    attempted.add(ident)
                    with Path('/tmp/DR_RESEARCH_M4_COORDINATION.txt').open('a') as note:
                        note.write(f'\nDoctor Research #136 evaluator RELEASE {ident}, exit {c.returncode}; no heavy child.\n')
                if c.returncode:
                    state['status'] = 'failed-retained-no-retry'
                    write(a.root/'evaluation-supervisor.json', state)
                    return 1
        else:
            expected = sum(len(plan[s+'_totals'])*3 for s in ('light', 'broad', 'heldout'))
            if rentals_done and len(attempted) == expected:
                state['status'] = 'complete'
                write(a.root/'evaluation-supervisor.json', state)
                return 0
            write(a.root/'evaluation-supervisor.json', state)
            time.sleep(30)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('queue', 'task'))
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--task', type=Path)
    a = p.parse_args()
    if a.phase == 'queue':
        raise SystemExit(queue(a))
    task = json.loads(a.task.read_text())
    raise SystemExit(execute_task(json.loads(a.plan.read_text()), task['spec'], task['stage'], a.root))
