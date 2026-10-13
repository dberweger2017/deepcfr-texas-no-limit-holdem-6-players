"""Machine protection for the owner-authorized M4 comparison."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
from time import sleep, time

import psutil
from scripts.guard_hu20_equity_scoring import host, machine_identity
from scripts.evaluate_hu100_direct import put

GIB = 1024**3


def sustained(sample, rss, baseline, now, since):
    conditions = {'rss': rss > 13*GIB,
                  'critical_pressure': sample['pressure_level'] == 4,
                  'free_memory': sample['free_percent'] < 8,
                  'swap_growth': sample['swap_bytes']-baseline > 4_000_000_000}
    for key, breached in conditions.items():
        if breached:
            since.setdefault(key, now)
        else:
            since.pop(key, None)
    if sample['disk_free_bytes'] < 16*GIB:
        return 'disk below 16 GiB'
    return next((key for key, start in since.items() if now-start >= 60), None)


def family(known):
    """Enumerate without /bin/ps; unreadable unrelated processes are harmless."""
    # Validate retained identities before traversing their descendants: an old
    # PID may now belong to another task, whose children must never be annexed.
    for pid, created in list(known.items()):
        try:
            if psutil.Process(pid).create_time() != created:
                known.pop(pid)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            known.pop(pid, None)
    children = {}; observed = {}
    for process in psutil.process_iter(['pid', 'ppid', 'create_time']):
        try:
            children.setdefault(process.info['ppid'], []).append(process.pid)
            observed[process.pid] = process.info.get('create_time')
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    # Bind traversal roots to this snapshot as well as their retained identity.
    # Recycling between the first validation and enumeration must not let the
    # replacement's children enter the owned set.
    for pid, created in list(known.items()):
        try:
            if (psutil.Process(pid).create_time()!=created or
                    observed.get(pid) is not None and observed[pid]!=created):
                known.pop(pid)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            known.pop(pid,None)
    pending = [os.getpid(), *known]
    seen = {os.getpid()}
    while pending:
        for pid in children.get(pending.pop(), []):
            if pid not in seen:
                seen.add(pid); pending.append(pid)
    result = [psutil.Process()]
    for pid in sorted((seen | set(known))-{os.getpid()}):
        try:
            process = psutil.Process(pid)
            created = process.create_time()
            if observed.get(pid) is not None and observed[pid] != created:
                raise RuntimeError('Owned identity changed during family discovery')
            if pid in known and known[pid] != created:
                known.pop(pid)
                if pid not in seen:
                    continue
            known[pid] = created
            result.append(process)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            known.pop(pid, None)
    return result


def stop(known):
    family(known)
    live = []
    for pid, created in list(known.items()):
        try:
            process = psutil.Process(pid)
            if process.create_time() == created:
                process.terminate(); live.append(process)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
    _, remaining = psutil.wait_procs(live, timeout=5)
    for process in remaining:
        if process.create_time() == known.get(process.pid):
            process.kill()
    _, remaining = psutil.wait_procs(remaining, timeout=5)
    if remaining:
        raise RuntimeError('Owned workers survived cleanup')


def guard(base: Path, name: str, command: list[str]):
    identity = subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string','hw.memsize','hw.ncpu'], text=True).splitlines()
    machine_identity('m4', identity)
    if (base/'failure.json').exists() and name!='archive':
        raise ValueError('Stopped campaign; preserve partials, no retry')
    folder = base/'guards'/name
    folder.mkdir(parents=True, exist_ok=False)
    known = {}; since = {}; child = None; failure = None; peak = 0; count = 0
    started = time()
    def interrupted(signum, frame):
        raise RuntimeError('Supervisor interrupted')
    handlers = {s:signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    with (Path.home()/'Local/.hu20-m4-research.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            first = host(base)
            if not (base/'baseline.json').exists():
                put(base/'baseline.json', first)
            baseline = json.loads((base/'baseline.json').read_text())['swap_bytes']
            put(folder/'intent.json', {'command':command, 'identity':identity, 'started':started, 'baseline_swap_bytes':baseline})
            # The shared lock coordinates new campaigns; detect older workers
            # that do not use it before loading any large policy.
            for process in psutil.process_iter(['pid','name','cmdline']):
                try:
                    cmd = ' '.join(process.info['cmdline'] or [])
                    python = process.info['name'] in ('Python','python','python3','python3.11')
                    if process.pid != os.getpid() and (process.info['name']=='hu20-trainer' or
                            python and any(token in cmd for token in ('scripts.evaluate_hu','scripts.run_hu','scripts.run_pot_pressure'))):
                        raise RuntimeError('Another research worker is active: '+str(process.pid))
                except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
            with (folder/'worker.log').open('x') as log, (folder/'resources.jsonl').open('x') as raw:
                while True:
                    sample = host(base)
                    rss = 0
                    for process in family(known):
                        try:
                            rss += process.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.ZombieProcess):
                            pass
                    now = time(); reason = sustained(sample,rss,baseline,now,since)
                    peak = max(peak,rss); count += 1
                    raw.write(json.dumps(dict(sample,family_rss_bytes=rss,breach_since=dict(since)))+'\n'); raw.flush()
                    if reason:
                        raise RuntimeError('Guard breach: '+reason)
                    if child is None:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        known[child.pid] = psutil.Process(child.pid).create_time()
                    elif child.poll() is not None:
                        if child.returncode:
                            raise RuntimeError('Worker exit '+str(child.returncode))
                        break
                    sleep(1)
        except BaseException as exc:
            failure = repr(exc)
            if not (base/'failure.json').exists():
                put(base/'failure.json', {'stage':name,'failure':failure,'at':time()})
            raise
        finally:
            stop(known)
            if child is not None:
                child.wait(timeout=5)
            for sig, handler in handlers.items():
                signal.signal(sig,handler)
            put(folder/'receipt.json', {'status':'failed' if failure else 'complete','failure':failure,
                'seconds':time()-started,'peak_family_rss_bytes':peak,'samples':count,'started':started,'finished':time()})
