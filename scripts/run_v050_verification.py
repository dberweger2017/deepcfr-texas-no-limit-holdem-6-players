"""One-use guarded M4 integration and local-closeout budget, not a research run."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
from time import monotonic, sleep, time
from zipfile import ZIP_DEFLATED, ZipFile

import psutil
from scripts import run_native_hu100_growth_1b as resource
from scripts.hu100_qualification_guard import stop_group, group_members
from src.policies.v050_bundle import ASSET_NAME, MODEL, PROVENANCE, sha


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write('\n')


def archive(out, cloud, source):
    # Canonical model stays in PR207. This ZIP restores evidence/metadata only;
    # explicit exclusions prevent creating a duplicate inference archive.
    paths = [p for p in out.rglob('*') if p.is_file() and p.name != 'access.token'
             and p.suffix not in ('.sqlite-wal', '.sqlite-shm')
             and p.name not in ('resources.jsonl', 'resources-resumed.jsonl', 'resources-closeout.jsonl', 'archive.log')]
    records = [{'path': str(p.relative_to(out)), 'bytes': p.stat().st_size,
                'sha256': sha(p), 'original': str(p.resolve()), 'mtime_ns': p.stat().st_mtime_ns} for p in sorted(paths)]
    manifest = {'source': source, 'kind': 'candidate-integration-evidence-no-model',
                'members': records, 'model_dependency': {'model': MODEL, 'provenance': PROVENANCE}}
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    bound = 256 * 1024**2
    if sum(r['bytes'] for r in records) + len(encoded) > bound:
        raise ValueError('Evidence upper storage bound exceeded')
    target = cloud / 'v050-readiness-M4-20261009.zip'
    cloud.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, 'x', ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted(paths): z.write(p, str(p.relative_to(out)))
        z.writestr('ARCHIVE-MANIFEST.json', encoded)
    if target.stat().st_size > bound: raise ValueError('Evidence ZIP exceeds storage bound')
    with ZipFile(target) as z:
        assert z.read('ARCHIVE-MANIFEST.json') == encoded
        for r in records:
            digest = hashlib.sha256(); size = 0
            with z.open(r['path']) as stream:
                while chunk := stream.read(8 * 1024**2): digest.update(chunk); size += len(chunk)
            assert size == r['bytes'] and digest.hexdigest() == r['sha256']
    return {'status': 'locally-verified-upload-pending', 'path': str(target),
            'bytes': target.stat().st_size, 'sha256': sha(target),
            'manifest_member': 'ARCHIVE-MANIFEST.json',
            'manifest_sha256': hashlib.sha256(encoded).hexdigest(), 'members': len(records),
            'originals_retained': True, 'remote_bytes_downloaded': False}


class Verification:
    def __init__(self, out):
        self.out = out; out.mkdir(parents=True, exist_ok=False)
        self.started = time(); self.tick_start = monotonic()
        self.deadline = self.started + 3600
        self.lock = (Path.home() / 'Local/.hu100-m4-research.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.stop = threading.Event(); self.failure = None; self.child = None
        self.handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        for sig in self.handlers: signal.signal(sig, self.interrupt)
        self.resource_log = out / 'resources.jsonl'
        self.thread = None; self.samples = []; self.peak = 0

    @classmethod
    def resume_preparation(cls, out, old_pid):
        """Resume only the documented offloaded-archive setup, never failed play."""
        old_source = '50f78d53b38cb9d0dd0478ca9cbf3c4181fc6a78'
        failure = json.loads((out / 'failure.json').read_text())
        baseline = json.loads((out / 'baseline.json').read_text())
        summary = json.loads((out / 'resources-summary.json').read_text())
        if (failure != {'at': failure.get('at'), 'error': "RuntimeError('Verification supervisor interrupted')", 'source': old_source}
                or summary['failure'] != 'Verification supervisor interrupted'
                or baseline['cap_seconds'] != 3600 or psutil.pid_exists(old_pid)
                or old_pid != 58545 or not (out / 'retrieval.log').exists()
                or any((out / name).exists() for name in ('package.log', 'pilot', 'main', 'admission.json', 'preparation-readmission.json'))
                or Path('models/v050-retrieved').exists() or Path('models/v050-candidate').exists()):
            raise ValueError('Inactive blocked-retrieval preparation proof differs')
        self = object.__new__(cls)
        self.out = out; self.started = baseline['started']; self.deadline = baseline['deadline']
        self.tick_start = monotonic() - (time() - self.started)
        self.lock = (Path.home() / 'Local/.hu100-m4-research.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.stop = threading.Event(); self.failure = None; self.child = None
        self.handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        for sig in self.handlers: signal.signal(sig, self.interrupt)
        self.resource_log = out / 'resources-resumed.jsonl'
        self.thread = None; self.samples = []; self.peak = summary['peak_family_rss_bytes']
        self.baseline = baseline['host']; resource.OUT = out
        with (out / 'resources.jsonl').open() as stream:
            for row in stream:
                sample = json.loads(row)
                if (resource.limits(sample, self.baseline['swap_bytes'], sample['family_rss_bytes'])
                        or sample['family_rss_bytes'] >= resource.FAMILY_SOFT
                        or sample['swap_bytes'] > 3000000000):
                    raise ValueError('Original stream contains a real guard breach')
        self.check()
        put(out / 'preparation-readmission.json', {'original_source': old_source,
            'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'original_started': self.started, 'original_deadline': self.deadline,
            'baseline_sha256': sha(out / 'baseline.json'), 'failure_sha256': sha(out / 'failure.json'),
            'old_controller_pid': old_pid, 'old_controller_alive': False,
            'monitoring_gap_seconds': time() - summary['finished'],
            'gap_policy': 'no model/runtime work; elapsed repair charged to original deadline',
            'reason': 'accepted canonical ZIP now dataless; use hash-verified indexed nonsynced original',
            'retained_input': '/Users/dberweger/Local/hu100-1b-growth-20261008/results/hu100-1b/training/1000000000/average.gz',
            'no_budget_baseline_or_guard_reset': True, 'prior_completed_hands': 0})
        self.resumed = True
        self.thread = threading.Thread(target=self.monitor, daemon=True); self.thread.start()
        return self

    def admit(self):
        out = self.out
        resource.OUT = out
        self.baseline = resource.host(); self.samples = []; self.peak = 0
        brand = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
        if brand != 'Apple M4': raise ValueError('M4 verification only')
        put(out / 'baseline.json', {'started': self.started, 'deadline': self.deadline,
            'cap_seconds': 3600, 'host': self.baseline, 'brand': brand,
            'guards': {'family_soft_bytes': 6 * 1024**3, 'family_hard_bytes': 8 * 1024**3,
                       'swap_growth_bytes': 512 * 1024**2, 'total_swap_bytes': 3000000000,
                       'disk_floor_bytes': resource.DISK_FLOOR},
            'process_inventory': subprocess.check_output(['ps', '-axo', 'pid,ppid,rss,etime,command'], text=True)})
        self.thread = threading.Thread(target=self.monitor, daemon=True); self.thread.start()
    def interrupt(self, signum, frame):
        self.failure = self.failure or 'Verification supervisor interrupted'
        raise RuntimeError(self.failure)

    def stable_admission(self):
        for _ in range(60):
            self.check(); sleep(1)
            current = self.samples[-1] if self.samples else self.baseline
            if current['free_percent'] * 16 * 1024**3 / 100 < 8 * 1024**3:
                raise RuntimeError('Stable admission lacks 8 GiB headroom')

    def check(self):
        if self.failure: raise RuntimeError(self.failure)
        if time() >= self.deadline or monotonic() - self.tick_start >= 3600:
            raise TimeoutError('Original 60-minute deadline exhausted')

    def monitor(self):
        parent = psutil.Process(os.getpid()); tick = monotonic()
        try:
            with self.resource_log.open('x') as stream:
                while not self.stop.is_set():
                    rss = 0
                    for process in [parent, *parent.children(recursive=True)]:
                        try: rss += process.memory_info().rss
                        except psutil.NoSuchProcess: pass
                    sample = resource.host(); sample.update(at=time(), family_rss_bytes=rss)
                    self.samples.append(sample); self.peak = max(self.peak, rss)
                    stream.write(json.dumps(sample) + '\n'); stream.flush()
                    reason = resource.limits(sample, self.baseline['swap_bytes'], rss)
                    if rss >= resource.FAMILY_SOFT: reason = 'soft whole-family RSS'
                    if sample['swap_bytes'] > 3000000000: reason = 'total swap ceiling'
                    if time() >= self.deadline or monotonic() - self.tick_start >= 3600: reason = '60-minute deadline'
                    if reason:
                        self.failure = self.failure or reason
                        if self.child is not None and self.child.poll() is None:
                            os.killpg(self.child.pid, signal.SIGTERM)
                    tick += .2; self.stop.wait(max(0, tick - monotonic()))
        except BaseException as error:
            self.failure = self.failure or ('Guard monitor failure: ' + repr(error))
            if self.child is not None and self.child.poll() is None:
                os.killpg(self.child.pid, signal.SIGTERM)

    def run(self, label, command):
        self.check(); started = monotonic()
        with (self.out / (label + '.log')).open('x') as stream:
            self.child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while self.child.poll() is None: self.check(); sleep(.1)
                self.check()
                if self.child.returncode != 0: raise RuntimeError(label + ' failed')
            finally:
                # Only our new process group; never touch another agent's jobs.
                stop_group(self.child)
                if group_members(self.child.pid): raise RuntimeError('Owned process group survived closure')
                self.child = None
        result = {'seconds': monotonic() - started, 'command': command, 'status': 'complete', 'owned_process_group_alive': False}
        put(self.out / (label + '-receipt.json'), result)
        return result

    def finish(self):
        self.stop.set()
        if self.thread is not None: self.thread.join(15)
        if self.thread is not None and self.thread.is_alive(): raise RuntimeError('Guard did not close')
        result = {'started': self.started, 'finished': time(), 'deadline': self.deadline,
                  'elapsed_seconds': monotonic() - self.tick_start, 'failure': self.failure,
                  'peak_family_rss_bytes': self.peak, 'samples': len(self.samples),
                  'max_total_swap_bytes': max((s['swap_bytes'] for s in self.samples), default=0),
                  'max_swap_growth_bytes': max((s['swap_bytes'] - self.baseline['swap_bytes'] for s in self.samples), default=0),
                  'min_free_disk_bytes': min((s['disk_free_bytes'] for s in self.samples), default=0),
                  'target_cadence_seconds': .2}
        put(self.out / getattr(self, 'summary_name', 'resources-summary-resumed.json' if getattr(self, 'resumed', False) else 'resources-summary.json'), result)
        for sig, handler in self.handlers.items(): signal.signal(sig, handler)
        fcntl.flock(self.lock, fcntl.LOCK_UN); self.lock.close()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cloud', type=Path, required=True)
    parser.add_argument('--resume-blocked-retrieval', type=int, metavar='OLD_PID')
    args = parser.parse_args()
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    guard = Verification.resume_preparation(args.out, args.resume_blocked_retrieval) if args.resume_blocked_retrieval else Verification(args.out)
    root = Path.cwd(); bundle = root / 'models/v050-candidate'
    common = [sys.executable, '-m', 'scripts.smoke_v050_release']
    try:
        if not args.resume_blocked_retrieval: guard.admit()
        guard.stable_admission()
        put(args.out / ('source-resumed.json' if args.resume_blocked_retrieval else 'source.json'), {'commit': source})
        subprocess.run(['git', 'archive', '-o', str(args.out / ('source-resumed.tar' if args.resume_blocked_retrieval else 'source.tar')), source,
            'AGENTS.md', 'ROADMAP.md', 'readme.md', 'requirements-play.txt', 'requirements-monitoring.txt',
            'src', 'scripts', 'tests', 'configs', 'apps', 'native', '.github',
            'docs/releases/v0.5.0', 'docs/development.md', 'docs/artifact-storage.md',
            'docs/rules.md', 'docs/observations.md'], check=True)
        retrieval = ['--retained', '/Users/dberweger/Local/hu100-1b-growth-20261008/results/hu100-1b/training/1000000000/average.gz'] if args.resume_blocked_retrieval else [
            '--archive', str(Path.home() / 'Local/Research-Cloud/PR-207-hu100-1b' / PROVENANCE['archive_name'])]
        guard.run('retrieval-retained' if args.resume_blocked_retrieval else 'retrieval', [sys.executable,
            '-m', 'scripts.retrieve_v050_candidate', *retrieval, '--out', 'models/v050-retrieved'])
        guard.run('package', [sys.executable, '-m', 'scripts.build_v050_bundle', '--source',
            str(root / 'models/v050-retrieved' / ASSET_NAME), '--out', str(bundle), '--source-sha', source])
        guard.run('standalone', [sys.executable, str(bundle / 'verify_v050_bundle.py'), str(bundle), '--expect-source', source])
        args.out.joinpath('package-metadata').mkdir()
        for p in bundle.iterdir():
            if p.name != ASSET_NAME:
                (args.out / 'package-metadata' / p.name).write_bytes(p.read_bytes())
        guard.run('pilot', [*common, 'pilot', '--bundle', str(bundle), '--out', str(args.out / 'pilot'),
            '--data', str(args.out / 'pilot-data'), '--source', source])
        pilot = json.loads((args.out / 'pilot/summary.json').read_text())
        guard.run('pilot-audit', [*common, 'audit', '--bundle', str(bundle), '--out', str(args.out / 'pilot'),
            '--data', str(args.out / 'pilot-data'), '--source', source])
        audit_cost = json.loads((args.out / 'pilot/independent-audit.json').read_text())
        remaining = guard.deadline - time()
        quote = 2 * max(pilot['startup']['load_seconds'], audit_cost['reader_load_seconds']) * 3 + 3 * (pilot['seconds'] + audit_cost['audit_seconds']) / 3 * 22 + 600 + 300
        disk_required = 256 * 1024**2
        admission = {'status': 'admitted' if remaining >= quote and resource.host()['disk_free_bytes'] - disk_required > resource.DISK_FLOOR else 'refused',
                     'remaining_seconds': remaining, 'quote_seconds': quote, 'pilot_startup_seconds': pilot['startup']['load_seconds'],
                     'pilot_3hand_seconds': pilot['seconds'], 'pilot_audit_seconds': audit_cost['audit_seconds'], 'pilot_reader_load_seconds': audit_cost['reader_load_seconds'], 'loads_remaining': 3,
                     'browser_reserve_seconds': 600, 'closeout_reserve_seconds': 300,
                     'additional_disk_upper_bytes': disk_required, 'source': source}
        put(args.out / 'admission.json', admission)
        if admission['status'] != 'admitted': raise RuntimeError('Measured integration plan admission refused')
        guard.run('main', [*common, 'main', '--bundle', str(bundle), '--out', str(args.out / 'main'),
            '--data', str(args.out / 'main-data'), '--source', source])
        guard.run('main-audit', [*common, 'audit', '--bundle', str(bundle), '--out', str(args.out / 'main'),
            '--data', str(args.out / 'main-data'), '--source', source])
        guard.check()
        # Freeze the stream snapshot; the live final closeout stream and compact
        # terminal receipts remain indexed alongside ZIP, never falsely sealed.
        if args.resume_blocked_retrieval:
            (args.out / 'resources-initial-closed.jsonl').write_bytes((args.out / 'resources.jsonl').read_bytes())
        (args.out / 'resources-prearchive.jsonl').write_bytes(guard.resource_log.read_bytes())
        guard.run('archive', [sys.executable, '-m', 'scripts.archive_v050_evidence', '--out', str(args.out),
                             '--cloud', str(args.cloud), '--source', source])
        guard.check()
    except BaseException as error:
        put(args.out / ('failure-resumed.json' if args.resume_blocked_retrieval else 'failure.json'), {'error': repr(error), 'at': time(), 'source': source})
        raise
    finally:
        print(json.dumps(guard.finish()))


if __name__ == '__main__': main()
