"""Owner-authorized M4 guard: observed sustained breaches, with stage baselines."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

import psutil

GIB = 1024**3


def put(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def host(path):
    def output(*args):
        return subprocess.check_output(args, text=True, timeout=10)
    swap = output('sysctl', 'vm.swapusage')
    free = output('memory_pressure', '-Q')
    return {'at': time.time(), 'monotonic': time.monotonic(),
            'swap_bytes': int(float(re.search(r'used = ([\d.]+)M', swap)[1])*1024**2),
            'free_percent': int(re.search(r'free percentage:\s*(\d+)%', free)[1]),
            'pressure_level': int(output('sysctl', '-n', 'kern.memorystatus_vm_pressure_level')),
            'disk_free_bytes': shutil.disk_usage(path).free,
            'swap_raw': swap, 'free_raw': free}


def family(known):
    """Never require access to privileged macOS helpers or confuse reused PIDs."""
    rows, denied = [], []
    for process in psutil.process_iter(['pid', 'ppid', 'create_time'], ad_value=None):
        try:
            row = process.info
            if row['ppid'] is None:
                denied.append(row['pid'])
            else:
                rows.append(row)
        except psutil.AccessDenied:
            denied.append(process.pid)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
    parents = {}
    for row in rows:
        parents.setdefault(row['ppid'], []).append(row)
    for pid, created in list(known.items()):
        try:
            if psutil.Process(pid).create_time() != created:
                known.pop(pid, None)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            known.pop(pid, None)
        except psutil.AccessDenied as exc:
            raise RuntimeError(f'Cannot validate owned PID {pid}') from exc
    pending = [os.getpid(), *known]
    seen = set()
    while pending:
        parent = pending.pop()
        for row in parents.get(parent, []):
            pid = row['pid']
            if pid == os.getpid() or pid in seen:
                continue
            seen.add(pid)
            if row['create_time'] is not None:
                known.setdefault(pid, row['create_time'])
            pending.append(pid)
    processes, readings = [], []
    for pid, created in [(os.getpid(), psutil.Process().create_time()), *list(known.items())]:
        try:
            process = psutil.Process(pid)
            if process.create_time() != created:
                known.pop(pid, None)
                continue
            readings.append({'pid': pid, 'create_time': created,
                             'rss': process.memory_info().rss,
                             'cpu_seconds': sum(process.cpu_times()[:2])})
            processes.append(process)
        except psutil.AccessDenied as exc:
            # An inaccessible owned process makes the resource reading invalid.
            raise RuntimeError(f'Cannot observe owned PID {pid}') from exc
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            known.pop(pid, None)
    return processes, readings, denied


def breaches(sample, baseline):
    return {'family_rss': sample['family_rss_bytes'] > 13*GIB,
            'critical_pressure': sample['pressure_level'] == 4,
            'free_memory': sample['free_percent'] < 8,
            'swap_growth': sample['swap_bytes']-baseline['swap_bytes'] > 4_000_000_000,
            'free_disk': sample['disk_free_bytes'] < 16*GIB}


def sustained(sample, baseline, since):
    active = breaches(sample, baseline)
    for key, breached in active.items():
        if breached:
            since.setdefault(key, sample['monotonic'])
        else:
            since.pop(key, None)
    stop = [key for key in since if key == 'free_disk' or sample['monotonic']-since[key] >= 60]
    return active, stop


def terminate(known):
    alive = []
    for pid, created in list(known.items()):
        try:
            process = psutil.Process(pid)
            if process.create_time() == created:
                process.terminate()
                alive.append(process)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
    _, alive = psutil.wait_procs(alive, timeout=5)
    for process in alive:
        try:
            if process.create_time() == known[process.pid]:
                process.kill()
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
    psutil.wait_procs(alive, timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--allow-exit', type=int, action='append', default=[0])
    parser.add_argument('--preserve', action='store_true', help='allow archival after stage stop')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    identity = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    if identity != 'Apple M4':
        raise ValueError('Only the authorized M4 may run this task')
    args.stage.mkdir(parents=True, exist_ok=True)
    if (args.stage/'STOP.json').exists() and not args.preserve:
        raise ValueError('Stage already stopped; no scientific retry')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    operation = args.stage/'operations'/args.name
    operation.mkdir(parents=True, exist_ok=False)
    baseline_path = args.stage/'baseline.json'
    if not baseline_path.exists():
        put(baseline_path, dict(host(args.stage), identity=identity))
    baseline = json.loads(baseline_path.read_text())
    known = {}
    held_path = args.stage/'guard-held.json'
    since = json.loads(held_path.read_text()) if held_path.exists() else {}
    child, failure, reason = None, None, []
    peak = count = 0
    started = time.time()
    cpu_peaks = {}
    def interrupted(signum, frame):
        raise RuntimeError('Supervisor interrupted: '+str(signum))
    handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    put(operation/'intent.json', {'command': command, 'started': started,
                                'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                                'baseline': str(baseline_path), 'guard_rules': {
                                    'family_rss_bytes': 13*GIB, 'critical_pressure': 4, 'minimum_free_percent': 8,
                                    'swap_growth_bytes': 4_000_000_000, 'minimum_disk_bytes': 16*GIB,
                                    'sustained_seconds': 60, 'sample_seconds': 5}})
    try:
        with (operation/'worker.log').open('x') as log, (operation/'resources.jsonl').open('x') as resources:
            while True:
                processes, readings, denied = family(known)
                sample = dict(host(args.stage), processes=readings, access_denied_other_pids=denied,
                              family_rss_bytes=sum(p['rss'] for p in readings))
                active, stop = sustained(sample, baseline, since)
                held_path.write_text(json.dumps(since, sort_keys=True)+'\n')
                sample.update(breaches=active, held_seconds={k: sample['monotonic']-v for k,v in since.items()}, stop=stop)
                resources.write(json.dumps(sample, sort_keys=True)+'\n'); resources.flush()
                peak = max(peak, sample['family_rss_bytes']); count += 1
                for p in readings:
                    key = str((p['pid'], p['create_time']))
                    cpu_peaks[key] = max(cpu_peaks.get(key, 0), p['cpu_seconds'])
                if stop and not reason:
                    reason = stop
                    put(args.stage/'STOP.json', {'operation': args.name, 'at': time.time(), 'reasons': reason, 'sample': sample})
                    if child is None:
                        break
                    if args.stop_file:
                        put(args.stop_file, {'guard_stop': reason})
                    else:
                        terminate(known)
                        break
                if child is None:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        p = psutil.Process(child.pid); known[p.pid] = p.create_time()
                    except psutil.NoSuchProcess:
                        pass
                if child.poll() is not None:
                    if child.returncode not in args.allow_exit:
                        raise RuntimeError('Worker exit '+str(child.returncode))
                    break
                time.sleep(5)
    except BaseException as exc:
        failure = repr(exc)
        if not (args.stage/'STOP.json').exists():
            put(args.stage/'STOP.json', {'operation': args.name, 'at': time.time(), 'failure': failure})
        raise
    finally:
        terminate(known)
        if child is not None:
            child.wait(timeout=10)
        for s,h in handlers.items(): signal.signal(s,h)
        put(operation/'receipt.json', {'status': 'stopped' if reason or failure else 'complete',
             'failure': failure, 'reasons': reason, 'returncode': child.returncode if child else None,
             'started': started, 'finished': time.time(), 'wall_seconds': time.time()-started,
             'peak_family_rss_bytes': peak, 'samples': count, 'cpu_seconds_by_identity': cpu_peaks})
    if reason:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
