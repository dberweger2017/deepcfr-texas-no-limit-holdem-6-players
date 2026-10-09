"""Continuous fixed-baseline supervision for one M4 qualification campaign."""
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
from time import monotonic, sleep, time

import psutil

from scripts import run_native_hu100_growth_1b as inherited

GIB = 1024**3
SWAP_GROWTH = 3_000_000_000
CAP_SECONDS = 21_600


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def read(path):
    return json.loads(path.read_text())


def violation(sample, baseline, rss):
    return inherited.limits(sample, baseline, rss, swap_limit=SWAP_GROWTH)


def group_members(pgid):
    members = []
    for process in psutil.process_iter(['pid', 'uids', 'status']):
        try:
            if os.getpgid(process.pid) == pgid and process.info['status'] != psutil.STATUS_ZOMBIE:
                if process.info['uids'].effective != os.geteuid():
                    raise RuntimeError('Owned group contains a different effective UID')
                members.append(process.pid)
        except (ProcessLookupError, psutil.NoSuchProcess):
            pass
    return members


def stop_group(child):
    """Escalate the retained session even after its direct wrapper exits."""
    if group_members(child.pid):
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        # Owned commands have no graceful science-retry behavior. Escalation
        # bounds shutdown and never depends on whether the time wrapper exited.
        if group_members(child.pid):
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    child.wait(timeout=5)
    end = monotonic() + 5
    while group_members(child.pid) and monotonic() < end:
        sleep(.05)
    if group_members(child.pid):
        raise RuntimeError('Owned process group survived SIGKILL')


def family_processes(parent, child):
    family = {p.pid: p for p in [parent, *parent.children(recursive=True)]}
    if child is not None:
        for pid in group_members(child.pid):
            if pid in family:
                continue
            try:
                family[pid] = psutil.Process(pid)
            except psutil.NoSuchProcess:
                pass
    return list(family.values())


class Campaign:
    """Hold ownership and sample idle work, tools and archival work alike."""

    def __init__(self, root, out, source):
        self.root, self.out, self.source = root, out, source
        self.child = None
        self.stop_file = None
        self.stop_at = None
        self.failure = None
        self.done = threading.Event()
        self.peak = 0
        self.count = 0
        self.tool_quote = 0
        self.panel_quote = 0
        self.report_quote = 0
        self.lock = (Path.home() / 'Local/.hu100-m4-research.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # This is also a one-use campaign: releasing the host lock never grants
        # permission to restart its baseline, clock or failed scientific work.
        out.mkdir(parents=True, exist_ok=False)
        inherited.OUT = out
        self.stable = []
        tick = monotonic()
        with (out / 'stable-admission.jsonl').open('x') as stream:
            for _ in range(301):
                sample = inherited.host()
                now = time()
                self.stable.append({'at': now, **sample})
                stream.write(json.dumps(self.stable[-1]) + '\n')
                stream.flush()
                if violation(sample, self.stable[0]['swap_bytes'], 0):
                    raise RuntimeError('Stable admission guard failed')
                if sample['free_percent'] * 16 * GIB / 100 < 8 * GIB:
                    raise RuntimeError('Stable admission needs 8 GiB headroom')
                tick += .2
                sleep(max(0, tick - monotonic()))
        self.swap0 = self.stable[0]['swap_bytes']
        self.started = time()
        self.deadline = self.started + CAP_SECONDS
        self.monotonic_deadline = monotonic() + CAP_SECONDS
        put(out / 'baseline.json', {'source': source, 'host': self.stable[0],
            'stable_samples': len(self.stable), 'started': self.started,
            'deadline': self.deadline, 'cap_seconds': CAP_SECONDS,
            'swap_growth_limit_bytes': SWAP_GROWTH,
            'soft_family_bytes': inherited.FAMILY_SOFT,
            'hard_family_bytes': inherited.FAMILY_HARD,
            'disk_floor_bytes': inherited.DISK_FLOOR})
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()
        self.handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        for s in self.handlers:
            signal.signal(s, self.interrupt)

    @classmethod
    def resume_preparation(cls, root, out, source):
        """Repair only the demonstrated inactive-source gate, with no new clock.

        Existing science or a real resource/correctness breach cannot use this
        path. The original failure, baseline and stream remain immutable history.
        """
        from src.policies.files import file_hash
        failure = read(out / 'campaign-failure.json')
        baseline = read(out / 'baseline.json')
        op = out / 'operations/prepare'
        receipt, intent = read(op / 'receipt.json'), read(op / 'intent.json')
        log = (op / 'worker.log').read_text()
        if (failure['failure'] != "prepare: RuntimeError('prepare exited 1')"
                or failure['source'] != 'dd19995d27950ef6a6b73bef5ec84c970310089f'
                or receipt['source'] != failure['source'] or baseline['source'] != failure['source']
                or receipt['returncode'] != 1 or receipt['child_alive_after_cleanup'] or receipt['cleanup_error']
                or intent['command'][-3:] != ['-m', 'scripts.run_hu100_independent_stages', 'prepare']
                or "ValueError: Native source compatibility differs" not in log
                or sorted(p.name for p in (out / 'operations').iterdir()) != ['prepare']
                or any((out / name).exists() for name in ('training','calibration-32','calibration-512','final','recovery-fixture'))
                or baseline['cap_seconds'] != CAP_SECONDS
                or baseline['swap_growth_limit_bytes'] != SWAP_GROWTH
                or baseline['soft_family_bytes'] != inherited.FAMILY_SOFT
                or baseline['hard_family_bytes'] != inherited.FAMILY_HARD
                or baseline['disk_floor_bytes'] != inherited.DISK_FLOOR):
            raise ValueError('Preparation-only readmission proof differs')
        old = read(root / 'planning/launch.json')
        if psutil.pid_exists(old['controller_pid']):
            raise RuntimeError('Original supervisor still exists')
        self = object.__new__(cls)
        self.root, self.out, self.source = root, out, source
        self.child = self.stop_file = self.stop_at = self.failure = None
        self.done = threading.Event()
        self.tool_quote = self.panel_quote = self.report_quote = 0
        self.lock = (Path.home() / 'Local/.hu100-m4-research.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inherited.OUT = out
        self.stable = [baseline['host']]
        self.swap0 = baseline['host']['swap_bytes']
        self.started, self.deadline = baseline['started'], baseline['deadline']
        self.monotonic_deadline = monotonic() + self.deadline - time()
        self.failure_name = 'readmission-failure.json'
        self.monitor_mode = 'a'
        self.count = 0; self.peak = 0
        last = None
        with (out / 'continuous-resources.jsonl').open() as stream:
            for line in stream:
                last = json.loads(line)
                self.count += 1
                self.peak = max(self.peak, last['family_rss_bytes'])
                if violation(last, self.swap0, last['family_rss_bytes']):
                    raise ValueError('Original stream contains a resource breach')
        tick = monotonic()
        with (out / 'stable-readmission.jsonl').open('x') as stream:
            for _ in range(301):
                sample = inherited.host()
                self.check()
                stream.write(json.dumps({'at':time(), **sample})+'\n'); stream.flush()
                if violation(sample, self.swap0, 0) or sample['free_percent']*16*GIB/100 < 8*GIB:
                    raise RuntimeError('Preparation readmission current resources refused against original baseline')
                tick += .2
                sleep(max(0,tick-monotonic()))
        self.check()
        put(out / 'preparation-readmission.json', {'source':source, 'at':time(),
            'original_source':baseline['source'], 'started':self.started, 'deadline':self.deadline,
            'baseline_sha256':file_hash(out/'baseline.json'), 'original_failure_sha256':file_hash(out/'campaign-failure.json'),
            'original_receipt_sha256':file_hash(op/'receipt.json'), 'current_host':sample,
            'monitoring_gap_seconds':time()-last['at'] if last else None,
            'gap_policy':'no campaign computation while original supervisor stopped; downtime charged to original deadline',
            'science_operations_before_readmission':0, 'no_guard_relaxation_or_baseline_reset':True})
        self.thread = threading.Thread(target=self.monitor,daemon=True)
        self.thread.start()
        self.handlers = {s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
        for sig in self.handlers: signal.signal(sig,self.interrupt)
        return self

    def interrupt(self, signum, frame):
        self.latch('Supervisor interrupted')
        raise RuntimeError(self.failure)

    def latch(self, reason):
        if self.failure is None:
            self.failure = reason
            put(self.out / getattr(self, 'failure_name', 'campaign-failure.json'), {'at': time(), 'failure': reason,
                'source': self.source, 'deadline': self.deadline})
        child = self.child
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def check(self):
        if self.failure:
            raise RuntimeError(self.failure)
        if time() >= self.deadline or monotonic() >= self.monotonic_deadline:
            self.latch('Absolute six-hour deadline exhausted')
            raise TimeoutError(self.failure)

    def remaining(self):
        self.check()
        return min(self.deadline - time(), self.monotonic_deadline - monotonic())

    def monitor(self):
        tick = monotonic()
        parent = psutil.Process(os.getpid())
        try:
            with (self.out / 'continuous-resources.jsonl').open(getattr(self, 'monitor_mode', 'x')) as stream:
                while not self.done.is_set():
                    rss = 0
                    for p in family_processes(parent, self.child):
                        try:
                            rss += p.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    sample = inherited.host()
                    now = time()
                    self.peak = max(self.peak, rss)
                    self.count += 1
                    stream.write(json.dumps({'at': now, 'family_rss_bytes': rss,
                        'swap_growth_bytes': sample['swap_bytes'] - self.swap0, 'source':self.source, 'deadline':self.deadline, **sample}) + '\n')
                    stream.flush()
                    reason = violation(sample, self.swap0, rss)
                    if now >= self.deadline or monotonic() >= self.monotonic_deadline:
                        reason = 'Absolute six-hour deadline exhausted'
                    if reason:
                        self.latch(reason)
                    elif rss >= inherited.FAMILY_SOFT or (self.stop_at is not None and now >= self.stop_at):
                        if self.stop_file is None:
                            self.latch('Tool family exceeds soft ceiling')
                        elif not self.stop_file.exists():
                            put(self.stop_file, {'at': now, 'reason': 'soft RSS or reserved recovery deadline',
                                'family_rss_bytes': rss})
                    tick += .2
                    self.done.wait(max(0, tick - monotonic()))
        except BaseException as exc:
            self.latch('Continuous guard failed: ' + repr(exc))

    def run(self, name, command, *, reserve=1800, quote=0, stop_file=None, accepted=(0,)):
        if name.startswith('final-'):
            reserve += self.report_quote
        if name == 'strict-report':
            quote = self.report_quote
        if not quote and (name.endswith('-export') or name.endswith('-audit')):
            quote = self.tool_quote
        if not quote and (name.endswith('-play') or name.endswith('-reproduce')):
            quote = self.panel_quote
        if self.remaining() < reserve + quote:
            raise CapacityStop('Operation cannot fit with recovery/local archive reserve: ' + name)
        guard = self.out / 'operations' / name
        guard.mkdir(parents=True)
        self.check()
        self.stop_file = stop_file
        self.stop_at = self.deadline - reserve if stop_file is not None else None
        started = time()
        failure = None
        returncode = None
        command = list(map(str, command))
        put(guard / 'intent.json', {'command': command, 'source': self.source,
            'started': started, 'deadline': self.deadline, 'reserve_seconds': reserve,
            'operation_quote_seconds': quote})
        try:
            with (guard / 'worker.log').open('x') as log:
                self.child = subprocess.Popen(['/usr/bin/time', '-l', *command],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=self.root)
                while self.child.poll() is None:
                    self.check()
                    sleep(.1)
                returncode = self.child.returncode
                self.check()
                if returncode not in accepted:
                    raise RuntimeError(f'{name} exited {returncode}')
        except BaseException as exc:
            failure = repr(exc)
            if not isinstance(exc, CapacityStop):
                self.latch(name + ': ' + failure)
            raise
        finally:
            cleanup_error = None
            if self.child is not None:
                try:
                    survived_wrapper = self.child.poll() is not None and bool(group_members(self.child.pid))
                    stop_group(self.child)
                    if survived_wrapper:
                        cleanup_error = 'Owned descendants outlived direct wrapper; killed and campaign stopped'
                except BaseException as exc:
                    cleanup_error = repr(exc)
            alive = self.child is not None and bool(group_members(self.child.pid))
            self.child = None
            self.stop_file = self.stop_at = None
            log = (guard / 'worker.log').read_text()
            high = re.search(r'(\d+)\s+maximum resident set size', log)
            put(guard / 'receipt.json', {'source': self.source, 'started': started,
                'finished': time(), 'seconds': time() - started, 'returncode': returncode,
                'status': 'failed' if failure else 'complete', 'failure': failure,
                'kernel_command_peak_rss_bytes': int(high[1]) if high else None,
                'cleanup_error': cleanup_error, 'child_alive_after_cleanup': alive,
                'deadline': self.deadline})
            if alive or cleanup_error:
                self.latch('Owned child cleanup failed: ' + str(cleanup_error))
        return read(guard / 'receipt.json')

    def finish(self):
        self.check()
        self.done.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError('Guard thread did not finish')
        put(self.out / 'campaign-resources.json', {'source': self.source,
            'started': self.started, 'finished': time(), 'deadline': self.deadline,
            'peak_family_rss_bytes': self.peak, 'samples': self.count,
            'target_sampling_seconds': .2, 'failure': self.failure})
        for s, handler in self.handlers.items():
            signal.signal(s, handler)
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()


class CapacityStop(RuntimeError):
    """A valid incomplete campaign, which may close out while guards pass."""
