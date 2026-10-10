"""One-use M1 operation guard for the owner-authorized K50 scoring attempt."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
from time import sleep, time

import psutil
from scripts.research_process_family import owned_processes

GIB = 1024**3


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def host(out):
    def command(*args):
        return subprocess.check_output(args, text=True, timeout=10)
    power = command('pmset', '-g', 'batt')
    swap = command('sysctl', 'vm.swapusage')
    free = command('memory_pressure', '-Q')
    return {'at': time(), 'ac': 'AC Power' in power,
            'power_raw': power, 'swap_raw': swap,
            'swap_bytes': int(float(re.search(r'used = ([\d.]+)M', swap)[1])*1024**2),
            'free_percent': int(re.search(r'free percentage:\s*(\d+)%', free)[1]),
            'pressure_level': int(command('sysctl', '-n', 'kern.memorystatus_vm_pressure_level')),
            'disk_free_bytes': shutil.disk_usage(out).free}


def violation(sample, rss):
    if rss >= 7*GIB:
        return '7 GiB whole-family ceiling'
    if sample['pressure_level'] != 1 or sample['free_percent'] < 15:
        return 'system pressure/headroom'
    if sample['swap_bytes'] > 3_000_000_000:
        return '3 GB total system swap ceiling'
    if sample['disk_free_bytes'] <= 16*GIB:
        return '16 GiB disk floor'
    if not sample['ac']:
        return 'AC power'
    return None


def stop_owned(known):
    """Never signal a reused PID or any process outside this supervisor's family."""
    owned_processes(known)
    live = []
    for pid, created in known.items():
        try:
            process = psutil.Process(pid)
            if process.create_time() == created:
                process.terminate()
                live.append(process)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
    _, live = psutil.wait_procs(live, timeout=5)
    for process in live:
        try:
            if process.create_time() == known[process.pid]:
                process.kill()
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
    _, live = psutil.wait_procs(live, timeout=5)
    if live:
        raise RuntimeError('Owned descendants survived cleanup')


def family_rss(known):
    total = 0
    for process in owned_processes(known):
        try:
            total += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--scoring', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ['--']:
        args.command = args.command[1:]
    identity = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string', 'hw.memsize', 'hw.ncpu'], text=True).splitlines()
    if identity != ['Apple M1', str(16*GIB), '8']:
        raise ValueError('Only the authorized 8-core/16 GiB M1 may run this guard')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'campaign-failure.json').exists():
        raise ValueError('This attempt is stopped; no readmission or retry')
    folder = args.out/'operations'/args.name
    folder.mkdir(parents=True, exist_ok=False)
    known = {}
    child = None
    failure = None
    cleanup_error = None
    peak = 0
    count = 0
    started = time()
    def interrupt(signum, frame):
        raise RuntimeError('Supervisor interrupted')
    original = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
    with (Path.home()/'Local/.hu20-m1-research.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            sample = host(args.out)
            rss = family_rss(known)
            write(folder/'admission.json', dict(sample, family_rss_bytes=rss, identity=identity))
            problem = violation(sample, rss)
            if args.scoring and sample['free_percent']*16*GIB/100 < 9*GIB:
                problem = problem or '9 GiB admission headroom'
            if problem:
                raise RuntimeError('Admission refused: '+problem)
            write(folder/'intent.json', {'command': args.command, 'source': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), 'started': started})
            with (folder/'worker.log').open('x') as log, (folder/'resources.jsonl').open('x') as stream:
                child = subprocess.Popen(['/usr/bin/time','-l',*args.command], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    launched = psutil.Process(child.pid)
                    if launched.ppid() != os.getpid():
                        raise RuntimeError('Launched child identity changed')
                    known[child.pid] = launched.create_time()
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    pass
                while True:
                    rss = family_rss(known)
                    sample = host(args.out)
                    peak = max(peak, rss)
                    stream.write(json.dumps(dict(sample, family_rss_bytes=rss))+'\n')
                    stream.flush()
                    count += 1
                    problem = violation(sample, rss)
                    if problem:
                        raise RuntimeError('Guard breach: '+problem)
                    if child.poll() is not None:
                        if child.returncode != 0:
                            raise RuntimeError('Worker exit '+str(child.returncode))
                        break
                    sleep(.2)
        except BaseException as exc:
            failure = repr(exc)
            write(args.out/'campaign-failure.json', {'operation': args.name, 'at': time(), 'failure': failure})
            raise
        finally:
            try:
                stop_owned(known)
                if child is not None:
                    child.wait(timeout=5)
            except BaseException as exc:
                cleanup_error = repr(exc)
                if failure is None:
                    failure = cleanup_error
                    write(args.out/'campaign-failure.json', {'operation': args.name, 'at': time(), 'failure': failure})
            for s, handler in original.items():
                signal.signal(s, handler)
            write(folder/'receipt.json', {'name': args.name, 'started': started, 'finished': time(), 'seconds': time()-started,
                  'samples': count, 'peak_family_rss_bytes': peak, 'failure': failure, 'cleanup_error': cleanup_error,
                  'returncode': child.returncode if child else None, 'status': 'failed' if failure else 'complete'})
            if cleanup_error:
                raise RuntimeError(cleanup_error)


if __name__ == '__main__':
    main()
