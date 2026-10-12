"""Guard the owner-authorized, sequential M4 K50 first pass; never restart a stage."""
import argparse
import fcntl
import json
import hashlib
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


def file_sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def approved_archive_repair(out, name, command, approval):
    """The owner's exception permits one metadata-only startup repair."""
    if approval is None:
        raise ValueError('Terminal campaign failure; no retry')
    record = json.loads(approval.read_text())
    failed = out / 'campaign-failure.json'
    original = json.loads(failed.read_text())
    receipt = json.loads((out / 'operations/archive-closeout-sidecar/receipt.json').read_text())
    log = (out / 'operations/archive-closeout-sidecar/worker.log').read_text()
    worker = Path.cwd() / 'planning/seal-archive-closeout.py'
    expected_command = [str(Path.cwd() / '.venv/bin/python'), str(worker)]
    if (name not in ('archive-closeout-startup-repair','archive-closeout-corrected-inventory')
            or record.get('owner_statement') != 'Exception permitted'
            or record.get('authorized_operation') != name
            or command != expected_command or record.get('command') != command
            or original.get('operation') != 'archive-closeout-sidecar'
            or original.get('failure') != "RuntimeError('Worker exit 1')"
            or receipt.get('status') != 'failed' or receipt.get('returncode') != 1
            or "ModuleNotFoundError: No module named 'scripts'" not in log
            or file_sha(failed) != record.get('original_failure_sha256')
            or file_sha(out / 'budget.json') != record.get('original_budget_sha256')
            or file_sha(worker) != record.get('corrected_worker_sha256')):
        raise ValueError('Owner exception does not match the exact archival startup failure')
    if name == 'archive-closeout-corrected-inventory':
        prior = out / 'operations/archive-closeout-startup-repair'
        second = json.loads((prior / 'receipt.json').read_text())
        second_log = (prior / 'worker.log').read_text()
        if (second.get('status') != 'failed' or second.get('returncode') != 1
                or "FileNotFoundError: [Errno 2] No such file or directory: 'results/hu20-k50-5b/campaign-failure.json'" not in second_log
                or file_sha(prior / 'receipt.json') != record.get('prior_metadata_failure_receipt_sha256')):
            raise ValueError('Exception cannot cover a different metadata failure')
    return file_sha(approval)


def record_failure(out, folder, name, failure):
    detail = {'operation': name, 'at': time(), 'failure': failure}
    if (out / 'campaign-failure.json').exists():
        # Preserve the original terminal latch byte-for-byte, including during
        # the one owner-approved archival repair.
        write(folder / 'failure.json', detail)
    else:
        write(out / 'campaign-failure.json', detail)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--archive-startup-approval', type=Path)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.command[:1] == ['--']:
        a.command = a.command[1:]
    if not a.command:
        p.error('worker command required')
    identity = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string', 'hw.memsize', 'hw.ncpu'], text=True).splitlines()
    machine_identity('m4', identity)
    a.out.mkdir(parents=True, exist_ok=True)
    approval_sha = None
    if (a.out / 'campaign-failure.json').exists():
        approval_sha = approved_archive_repair(a.out, a.name, a.command, a.archive_startup_approval)
    elif a.archive_startup_approval is not None:
        raise ValueError('Archival exception requires its preserved original failure')
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
                                         'started': started, 'baseline': baseline, 'identity': identity,
                                         'archive_startup_owner_approval_sha256': approval_sha})
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
            record_failure(a.out, folder, a.name, failure)
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
                    record_failure(a.out, folder, a.name, failure)
            for s, handler in previous.items():
                signal.signal(s, handler)
            write(folder / 'receipt.json', {'name': a.name, 'status': 'failed' if failure else 'complete', 'started': started,
                'finished': time(), 'seconds': time() - started, 'peak_family_rss_bytes': peak, 'samples': samples,
                'failure': failure, 'cleanup_error': cleanup, 'returncode': child.returncode if child else None})
            if cleanup:
                raise RuntimeError(cleanup)


if __name__ == '__main__':
    main()
