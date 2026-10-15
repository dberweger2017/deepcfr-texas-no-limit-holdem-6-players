"""M1-only release-work guard; battery pauses request a complete-block checkpoint."""
import fcntl
import json
from pathlib import Path
import re
import signal
import subprocess
from time import sleep, time

import psutil
from scripts.evaluate_hu100_direct import put
from scripts.guard_hu20_equity_scoring import host, machine_identity
from scripts.guard_hu100_k50_v052 import family, stop

GIB = 1024**3


def battery_percent(sample):
    match = re.search(r'(\d+)%;', sample['power_raw'])
    if match is None:
        raise ValueError('Cannot determine M1 battery percentage')
    return int(match[1])


def sustained(sample, rss, baseline, now, since):
    conditions = {'rss': rss > 13*GIB, 'critical_pressure': sample['pressure_level'] == 4,
                  'free_memory': sample['free_percent'] < 8,
                  'swap_growth': sample['swap_bytes']-baseline > 4_000_000_000,
                  'disk': sample['disk_free_bytes'] < 16*GIB}
    for name, breached in conditions.items():
        if breached:
            since.setdefault(name, now)
        else:
            since.pop(name, None)
    return next((name for name, began in since.items() if now-began >= 60), None)


def request_stop(base, reason, *, resumable):
    path = base/'stop-request.json'
    if not path.exists():
        put(path, dict(reason=reason, resumable=resumable, at=time()))


def boundary_stop(base):
    """Called by the evaluator only between complete duplicate blocks."""
    return (Path(base)/'stop-request.json').exists()


def guard(base, name, command, *, active_budget=None):
    base = Path(base); base.mkdir(parents=True, exist_ok=True)
    identity = subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string','hw.memsize','hw.ncpu'], text=True).splitlines()
    machine_identity('m1', identity)
    if (base/'failure.json').exists():
        raise ValueError('Scientific failure latched; no automatic retry')
    if (base/'stop-request.json').exists():
        raise ValueError('Clear only an owner-acknowledged battery pause before resuming')
    folder = base/'guards'/f'{name}-{len(list((base/"guards").glob(name+"-*"))):03d}'
    folder.mkdir(parents=True, exist_ok=False)
    known = {}; since = {}; child = None; failure = None; samples = 0; peak = 0
    started = time(); previous = None; stop_since = None; paused = False
    def interrupted(signum, frame):
        raise RuntimeError('Supervisor interrupted')
    handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (Path.home()/'Local/.hu20-m1-research.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            first = host(base)
            put(folder/'admission.json', dict(first, identity=identity))
            if first['disk_free_bytes'] < 16*GIB:
                raise ValueError('Admission requires at least the 16 GiB disk reserve')
            if battery_percent(first) <= 10:
                paused = True
                request_stop(base, 'battery at or below 10%; plug in before resuming', resumable=True)
                return dict(status='paused', hands_started=False)
            # Warning pressure is allowed on this working laptop. The same
            # declared critical/free-memory thresholds get a full 60 seconds
            # from the first sample, including at admission.
            if not (base/'baseline.json').exists():
                put(base/'baseline.json', first)
            baseline = json.loads((base/'baseline.json').read_text())['swap_bytes']
            put(folder/'intent.json', dict(command=command, identity=identity, started=started,
                baseline_swap_bytes=baseline, active_budget=active_budget))
            with (folder/'worker.log').open('x') as log, (folder/'resources.jsonl').open('x') as raw:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                known[child.pid] = psutil.Process(child.pid).create_time()
                while True:
                    sample = host(base); rss = 0
                    for process in family(known):
                        try:
                            rss += process.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.ZombieProcess):
                            pass
                    now = time(); gap = previous is not None and now-previous > 10
                    if gap:
                        since.clear()
                    previous = now; peak = max(peak, rss); samples += 1
                    reason = sustained(sample, rss, baseline, now, since)
                    if active_budget is not None and now-started >= active_budget:
                        reason = 'declared active-work budget exhausted'
                    if reason:
                        request_stop(base, reason, resumable=False)
                        stop_since = stop_since or now
                        if now-stop_since >= 10:
                            raise RuntimeError('Guard breach: '+reason)
                    if battery_percent(sample) <= 10:
                        request_stop(base, 'battery at or below 10%; plug in before resuming', resumable=True)
                    raw.write(json.dumps(dict(sample, family_rss_bytes=rss, breach_since=dict(since),
                        observation_gap_reset=gap))+'\n'); raw.flush()
                    if child.poll() is not None:
                        if child.returncode:
                            raise RuntimeError('Worker exit '+str(child.returncode))
                        paused = (base/'stop-request.json').exists()
                        if paused and not json.loads((base/'stop-request.json').read_text())['resumable']:
                            raise RuntimeError('Guard stop: '+json.loads((base/'stop-request.json').read_text())['reason'])
                        break
                    sleep(5)
    except BaseException as exc:
        failure = repr(exc)
        if not (base/'failure.json').exists():
            put(base/'failure.json', dict(stage=name, failure=failure, at=time()))
        raise
    finally:
        stop(known)
        if child is not None:
            child.wait(timeout=5)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        put(folder/'receipt.json', dict(status='failed' if failure else 'paused' if paused else 'complete',
            failure=failure, seconds=time()-started, peak_family_rss_bytes=peak,
            samples=samples, started=started, finished=time()))
    return dict(status='paused' if paused else 'complete', seconds=time()-started)
