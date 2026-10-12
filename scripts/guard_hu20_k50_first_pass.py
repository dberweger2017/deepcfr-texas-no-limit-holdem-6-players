"""Guard the owner-authorized, sequential M4 K50 first pass; never restart a stage."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
from time import sleep, time

import psutil
from scripts.guard_hu20_equity_scoring import host, machine_identity, stop_owned, write
from scripts.research_process_family import owned_processes

GIB = 1024**3


def violation(sample, rss, baseline, now):
    if now >= baseline['deadline']:
        return 'six-hour aggregate wall-clock ceiling'
    if rss >= 7 * GIB:
        return 'whole-family RSS must remain below 7 GiB'
    if sample['pressure_level'] != 1 or sample['free_percent'] < 15:
        return 'normal memory pressure / 15 percent free memory'
    if sample['swap_bytes'] - baseline['swap_bytes'] > 512 * 1024**2:
        return '512 MiB fixed-baseline swap growth'
    if sample['disk_free_bytes'] < 16 * GIB:
        return '16 GiB free disk floor'
    if not sample['ac']:
        return 'AC power required'
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.command[:1] == ['--']:
        a.command = a.command[1:]
    if not a.command:
        p.error('worker command required')
    identity = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string', 'hw.memsize', 'hw.ncpu'], text=True).splitlines()
    machine_identity('m4', identity)
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / 'campaign-failure.json').exists():
        raise ValueError('Terminal campaign failure; no retry')
    folder = a.out / 'operations' / a.name
    folder.mkdir(parents=True, exist_ok=False)
    known = {}; child = None; peak = 0; samples = 0; failure = None; cleanup = None
    started = time()
    def interrupt(signum, frame):
        raise RuntimeError('Guard interrupted')
    previous = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
    with (Path.home() / 'Local/.hu20-m4-research.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            initial = host(a.out)
            baseline_file = a.out / 'budget.json'
            if not baseline_file.exists():
                write(baseline_file, {'started': started, 'deadline': started + 21600,
                                     'swap_bytes': initial['swap_bytes'], 'initial_host': initial})
            baseline = json.loads(baseline_file.read_text())
            write(folder / 'intent.json', {'command': a.command, 'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                                         'started': started, 'baseline': baseline, 'identity': identity})
            if problem := violation(initial, psutil.Process().memory_info().rss, baseline, time()):
                raise RuntimeError('Admission refused: ' + problem)
            with (folder / 'worker.log').open('x') as log, (folder / 'resources.jsonl').open('x') as raw:
                child = subprocess.Popen(['/usr/bin/time', '-l', *a.command], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                proc = psutil.Process(child.pid)
                if proc.ppid() != os.getpid():
                    raise RuntimeError('Child identity changed')
                known[child.pid] = proc.create_time()
                while True:
                    # Only known descendants get memory_info(); psutil handles unrelated process metadata.
                    rss = 0
                    for proc in owned_processes(known):
                        try:
                            rss += proc.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.ZombieProcess):
                            pass
                    sample = host(a.out)
                    peak = max(peak, rss); samples += 1
                    row = dict(sample, family_rss_bytes=rss, operation=a.name)
                    raw.write(json.dumps(row) + '\n'); raw.flush()
                    temporary = a.out / 'status.tmp'
                    temporary.write_text(json.dumps(row) + '\n')
                    temporary.replace(a.out / 'status.json')
                    if problem := violation(sample, rss, baseline, time()):
                        raise RuntimeError('Guard breach: ' + problem)
                    if child.poll() is not None:
                        if child.returncode:
                            raise RuntimeError('Worker exit ' + str(child.returncode))
                        break
                    sleep(.2)
        except BaseException as exc:
            failure = repr(exc)
            write(a.out / 'campaign-failure.json', {'operation': a.name, 'at': time(), 'failure': failure})
            raise
        finally:
            try:
                stop_owned(known)
                if child is not None:
                    child.wait(timeout=5)
            except BaseException as exc:
                cleanup = repr(exc)
                if failure is None:
                    failure = cleanup
                    write(a.out / 'campaign-failure.json', {'operation': a.name, 'at': time(), 'failure': failure})
            for s, handler in previous.items():
                signal.signal(s, handler)
            write(folder / 'receipt.json', {'name': a.name, 'status': 'failed' if failure else 'complete', 'started': started,
                'finished': time(), 'seconds': time() - started, 'peak_family_rss_bytes': peak, 'samples': samples,
                'failure': failure, 'cleanup_error': cleanup, 'returncode': child.returncode if child else None})
            if cleanup:
                raise RuntimeError(cleanup)


if __name__ == '__main__':
    main()
