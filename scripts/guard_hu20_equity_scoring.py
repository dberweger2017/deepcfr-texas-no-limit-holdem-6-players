"""One-use machine-bound guard for owner-authorized K50 scoring attempts."""

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


def machine_identity(machine, identity):
    expected = {'m1': ['Apple M1', str(16*GIB), '8'],
                'm4': ['Apple M4', str(16*GIB), '10']}[machine]
    if identity != expected:
        raise ValueError(f'Only the authorized {machine.upper()} may run this guard')


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


def violation(sample, rss, *, swap_ceiling_bytes=3_000_000_000):
    if rss >= 7*GIB:
        return '7 GiB whole-family ceiling'
    if sample['pressure_level'] != 1 or sample['free_percent'] < 15:
        return 'system pressure/headroom'
    if sample['swap_bytes'] > swap_ceiling_bytes:
        return f'{swap_ceiling_bytes/1_000_000_000:g} GB total system swap ceiling'
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
    parser.add_argument('--machine', choices=('m1', 'm4'), default='m1',
                        help='M4 requires the separately owner-authorized fresh attempt')
    parser.add_argument('--swap-ceiling-bytes', type=int,
                        choices=(3_000_000_000, 10_000_000_000),
                        default=3_000_000_000,
                        help='10 GB requires the owner-authorized continuation amendment')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ['--']:
        args.command = args.command[1:]
    identity = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string', 'hw.memsize', 'hw.ncpu'], text=True).splitlines()
    machine_identity(args.machine, identity)
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
    with (Path.home()/f'Local/.hu20-{args.machine}-research.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            sample = host(args.out)
            rss = family_rss(known)
            write(folder/'admission.json', dict(sample, family_rss_bytes=rss, identity=identity,
                                               swap_ceiling_bytes=args.swap_ceiling_bytes))
            problem = violation(sample, rss, swap_ceiling_bytes=args.swap_ceiling_bytes)
            if args.scoring and sample['free_percent']*16*GIB/100 < 9*GIB:
                problem = problem or '9 GiB admission headroom'
            if problem:
                raise RuntimeError('Admission refused: '+problem)
            write(folder/'intent.json', {'command': args.command, 'source': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), 'started': started,
                                        'swap_ceiling_bytes': args.swap_ceiling_bytes})
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
                    problem = violation(sample, rss, swap_ceiling_bytes=args.swap_ceiling_bytes)
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
                  'returncode': child.returncode if child else None, 'status': 'failed' if failure else 'complete',
                  'swap_ceiling_bytes': args.swap_ceiling_bytes})
            if cleanup_error:
                raise RuntimeError(cleanup_error)


if __name__ == '__main__':
    main()
