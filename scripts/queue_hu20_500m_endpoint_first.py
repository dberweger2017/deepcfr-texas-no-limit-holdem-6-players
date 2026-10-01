"""Operational queue replacement; children retain the frozen scientific executor."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def priority(plan, stage, spec):
    if stage == 'light':
        return (0, spec['milestone'], spec['seed'])
    if stage == 'broad':
        order = [100000000, 500000000] + [
            m for m in sorted(plan['broad_totals']) if m not in (100000000, 500000000)]
        return (1, order.index(spec['milestone']), spec['seed'])
    if stage == 'heldout':
        return (2, spec['seed'], spec['milestone'])
    raise ValueError('Unknown frozen stage')


def pending_jobs(plan, tasks, attempted):
    jobs = []
    seen = set()
    for task in tasks:
        if task['destination'] != 'M4':
            raise ValueError('Model must be retrieved to M4 before evaluation')
        spec = task['spec']
        for stage in ('light', 'broad', 'heldout'):
            if spec['milestone'] not in plan[stage + '_totals']:
                continue
            ident = f"{stage}-{spec['seed']}-{spec['milestone']}"
            if ident in seen:
                raise ValueError('Duplicate model task')
            seen.add(ident)
            if ident not in attempted:
                jobs.append((stage, spec, ident))
    return sorted(jobs, key=lambda job: priority(plan, job[0], job[1]))


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def process(pid):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'stat=', '-o', 'command='],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ''


def swap_bytes():
    import re
    value = subprocess.check_output(['sysctl', 'vm.swapusage'], text=True)
    return int(float(re.search(r'used = ([0-9.]+)M', value).group(1)) * 2**20)


def note(message):
    with Path('/tmp/DR_RESEARCH_M4_COORDINATION.txt').open('a') as handle:
        handle.write('\nDoctor Research #136 endpoint queue ' + message + '\n')


def watch(pid, state, root, before_swap, child=None):
    stopped = None
    while process(pid) and not process(pid).startswith('Z'):
        rss = sum(int(v) for v in subprocess.check_output(
            ['ps', '-o', 'rss=', '-p', f'{pid},{os.getpid()}'], text=True).split()) * 1024
        violated = (rss >= 10.5 * 2**30 or swap_bytes() - before_swap > .5 * 2**30
                    or shutil.disk_usage(root).free < 8 * 2**30)
        if violated:
            if stopped is None:
                os.kill(pid, signal.SIGTERM)
                stopped = time.time()
            elif time.time() - stopped > 30:
                os.kill(pid, signal.SIGKILL)
        state.update(heartbeat=time.time(), child_rss_bytes=rss,
                     status='guard-stopping' if stopped else 'running')
        write(root / 'evaluation-supervisor.json', state)
        if child is not None and child.poll() is not None:
            break
        time.sleep(2)
    if child is not None:
        state['exit'] = child.wait()
    return stopped is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--adopt-queue-pid', type=int, required=True)
    parser.add_argument('--test-file', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads(args.plan.read_text())
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if plan_hash != '74f0c18024a4a409d223276d4781c2464b27fc8d5d32a51ccb22a224332f2219':
        raise ValueError('Frozen scientific plan changed')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != '17b4c9a08ed0765d0fb8f05240c0409b21e43977':
        raise ValueError('Scientific executor source changed')
    if json.loads((root / 'operator-finished.json').read_text())['status'] != 'training-complete':
        raise ValueError('All rentals must be closed before broader work')
    prior = process(args.adopt_queue_pid)
    if 'scripts.evaluate_hu20_500m queue' not in prior:
        raise ValueError('Expected original queue owner is absent')
    handoff = dict(started=time.time(), old_queue_pid=args.adopt_queue_pid,
                   new_queue_pid=os.getpid(), scientific_executor='17b4c9a',
                   plan_sha256=plan_hash,
                   reporting_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   test_source_sha256=hashlib.sha256(args.test_file.read_bytes()).hexdigest(),
                   status='preparing')
    record = root / 'endpoint-order-handoff.json'
    if record.exists():
        raise ValueError('Retain prior handoff; no blind restart')
    write(record, handoff)
    # Freeze only the coordinator long enough to adopt its exact live child.
    # The child is never paused or signalled during an ordinary handoff.
    os.kill(args.adopt_queue_pid, signal.SIGSTOP)
    try:
        state = json.loads((root / 'evaluation-supervisor.json').read_text())
        pid = state.get('child_pid')
        ident = state.get('task')
        if state['status'] != 'running' or not pid or not ident:
            raise ValueError('Expected live task boundary handoff')
        if 'scripts.evaluate_hu20_500m task' not in process(pid):
            raise ValueError('Live scientific child identity changed')
        task_doc = json.loads((root / (ident + '.task.json')).read_text())
        expected = f"{task_doc['stage']}-{task_doc['spec']['seed']}-{task_doc['spec']['milestone']}"
        if ident != expected:
            raise ValueError('Live task coordinates changed')
        handoff.update(status='adopted', adopted_task=ident, adopted_child_pid=pid)
        write(record, handoff)
        os.kill(args.adopt_queue_pid, signal.SIGTERM)
    finally:
        os.kill(args.adopt_queue_pid, signal.SIGCONT)
    with Path('/tmp/DR_RESEARCH_M4_HEAVY.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        note(f'adopted {ident}, child PID {pid}; frozen task continues uninterrupted.')
        state.update(pid=os.getpid(), queue_order='endpoint-first-v1', adopted=True)
        healthy = watch(pid, state, root, swap_bytes())
        result_path = root / 'evaluation' / ident / 'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        state.update(finished=time.time(), exit=None, exit_status_observed=False,
                     completion_evidence='closed result after adopted process exit')
        write(root / 'evaluation' / (ident + '.exit.json'), state)
        note(f'RELEASE adopted {ident}; status {result.get("status", "missing")}, no child exit code observed.')
        if not healthy or result.get('status') != 'complete':
            handoff.update(status='failed-retained-no-retry', finished=time.time())
            write(record, handoff)
            return 1
        # Focused queue tests run only once the current scientific child closes.
        import importlib.util
        module_name = 'scripts.queue_hu20_500m_endpoint_first'
        spec = importlib.util.spec_from_file_location(module_name, __file__)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        import pytest
        with (root / 'endpoint-order-tests.log').open('w') as log:
            from contextlib import redirect_stdout, redirect_stderr
            with redirect_stdout(log), redirect_stderr(log):
                status = pytest.main(['-q', str(args.test_file.resolve())])
        write(root / 'endpoint-order-tests.json', dict(exit=int(status), finished=time.time(), host='M4'))
        if status:
            handoff.update(status='tests-failed-retained', finished=time.time())
            write(record, handoff)
            return 1
        handoff.update(status='boundary-switch-complete', finished=time.time())
        write(record, handoff)
        outputs = root / 'evaluation'
        attempted = {d.name for d in outputs.iterdir() if d.is_dir()}
        for ident in attempted:
            r = outputs / ident / 'result.json'
            if not r.exists() or json.loads(r.read_text()).get('status') != 'complete':
                raise ValueError('Failed/partial task retained; no blind retry')
        tasks = [json.loads(p.read_text()) for p in sorted((root / 'evaluation-queue').glob('*.json'))]
        jobs = pending_jobs(plan, tasks, attempted)
        expected = sum(len(plan[s + '_totals']) * len(plan['parents']) for s in ('light', 'broad', 'heldout'))
        if len(attempted) + len(jobs) != expected:
            raise ValueError('Frozen task count changed')
        write(root / 'endpoint-order-pending.json', [dict(stage=s, seed=m['seed'], milestone=m['milestone'], task=i) for s,m,i in jobs])
        fcntl.flock(lock, fcntl.LOCK_UN)
        for stage, model, ident in jobs:
            # Reporting can take this same lock between scientific children.
            fcntl.flock(lock, fcntl.LOCK_EX)
            if shutil.disk_usage(root).free < 12 * 2**30:
                raise OSError('Disk reserve before next model load')
            task = root / (ident + '.task.json')
            write(task, dict(stage=stage, spec=model))
            with (root / (ident + '.log')).open('w') as log:
                child = subprocess.Popen([sys.executable, '-m', 'scripts.evaluate_hu20_500m', 'task',
                    '--plan', str(args.plan.resolve()), '--root', str(outputs / ident), '--task', str(task)],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state = dict(pid=os.getpid(), status='running', task=ident, child_pid=child.pid,
                             queue_order='endpoint-first-v1')
                note(f'CLAIM {ident}, PID {child.pid}; one heavy scientific worker.')
                healthy = watch(child.pid, state, root, swap_bytes(), child)
                state.update(finished=time.time())
                write(outputs / (ident + '.exit.json'), state)
                note(f'RELEASE {ident}, exit {child.returncode}; no heavy child.')
                if not healthy or child.returncode:
                    state['status'] = 'failed-retained-no-retry'
                    write(root / 'evaluation-supervisor.json', state)
                    return 1
            fcntl.flock(lock, fcntl.LOCK_UN)
            time.sleep(5)
        write(root / 'evaluation-supervisor.json', dict(pid=os.getpid(), status='complete',
              finished=time.time(), queue_order='endpoint-first-v1'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
