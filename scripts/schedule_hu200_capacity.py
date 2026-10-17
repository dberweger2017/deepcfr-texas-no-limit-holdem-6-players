"""Owner-approved PR301 queue scheduling; qualified trainer and audits stay fixed."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from scripts.guard_hu200_capacity import VrmWindow, motherboard_sensors, stop_files
from scripts.machine_qualification import watchdog as sensors
from scripts.run_hu200_capacity_training import (
    BINARY_SHA, QUALIFIED, PREFIX, SEEDS, Campaign, pin, put, retire)
from src.policies.files import file_hash

BASE = Path('/home/dberweger/hu200-capacity-20261015')
TARGETS = (2_000_000_000, 5_000_000_000, 10_000_000_000, 20_000_000_000)
GIB = 1024**3
FLOOR = 50_000_000_000
BUDGET = 259_000_000_000
GROWTH = 20_000_000_000


def read(path):
    return json.loads(path.read_text())


def checked_audit(folder):
    receipt = read(folder / 'audit.json')
    if (receipt.get('status') != 'verified' or receipt['audit'].get('status') != 'verified'
            or receipt['audit']['all_nodes_verified'] != receipt['native']['diagnostics']['entries']):
        raise ValueError('Incomplete checkpoint audit')
    for item in receipt['files'].values():
        path = Path(item['path'])
        # Retired older saves can still have valid receipts; callers decide which
        # exact recovery must remain present for continuation.
        if path.exists() and (path.stat().st_size, path.stat().st_mtime_ns) != (item['bytes'], item['mtime_ns']):
            raise ValueError('Audited file changed')
    return receipt


def process(pid):
    path = Path('/proc') / str(pid)
    fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, ppid=int(fields[1]), pgid=int(fields[2]), state=fields[0],
                start=fields[19], ticks=int(fields[11]) + int(fields[12]),
                exit_status=int(fields[49]) if len(fields) > 49 else None)


def process_tree():
    result = {}
    for path in Path('/proc').glob('[0-9]*'):
        try:
            result[int(path.name)] = process(int(path.name))
        except (OSError, ValueError):
            continue
    return result


def descendants(tree, pid):
    result = {pid}
    for _ in range(20):
        expanded = result | {i for i, p in tree.items() if p['ppid'] in result}
        if expanded == result:
            return result - {pid}
        result = expanded
    raise RuntimeError('Unexpected process tree depth')


def owned_bytes(root):
    total, seen = 0, set()
    for directory, _, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(directory) / name
            try:
                if path.is_symlink():
                    continue
                stat = path.stat()
            except FileNotFoundError:
                continue
            identity = (stat.st_dev, stat.st_ino)
            if identity not in seen:
                total += stat.st_size
                seen.add(identity)
    return total


def fits(owned, free, reserved, extra):
    return (owned + reserved + extra + GROWTH + FLOOR <= BUDGET
            and free - reserved - extra >= FLOOR)


def allocate(active, kind, seed=None):
    used = {cpu for job in active for cpu in job['cpus']}
    candidates = ([2 * SEEDS.index(seed), 2 * SEEDS.index(seed) + 1]
                  if kind == 'train' else [c for c in (6, 7, 0, 1, 2, 3, 4, 5) if c not in used][:1])
    if not candidates or used.intersection(candidates) or len(used) + len(candidates) > 8:
        return None
    return candidates


def resource_breach(row, rss, jobs, swap_baseline, now, deadline):
    if row['package_c'] >= 85:
        return 'CPU >=85 C'
    if row['available_bytes'] < 6 * GIB:
        return 'Available RAM <6 GiB'
    if row['swap_used_bytes'] - swap_baseline > 256 * 1024**2:
        return 'Swap growth >256 MiB'
    if row['free_disk_bytes'] < FLOOR:
        return 'Disk below 50 GB floor'
    if sum(rss.values()) > 116 * GIB:
        return 'Aggregate family RSS cap'
    if any(rss[j['name']] > j['rss_gib'] * GIB for j in jobs):
        return 'Per-job family RSS cap'
    if now >= deadline:
        return 'Original wall-time cap'
    return None


def sample_counters(active):
    tree = process_tree()
    jobs, phases = {}, {}
    for name, job in active.items():
        rows = []
        for pid in {job['proc'].pid} | descendants(tree, job['proc'].pid):
            if pid not in tree:
                continue
            path = Path('/proc') / str(pid)
            try:
                argv = [a.decode(errors='replace') for a in (path / 'cmdline').read_bytes().split(b'\0') if a]
                io = {k: int(v) for k, v in (line.split(':') for line in (path / 'io').read_text().splitlines())}
                threads = []
                for task in (path / 'task').iterdir():
                    f = (task / 'stat').read_text().rsplit(')', 1)[1].split()
                    threads.append(dict(tid=int(task.name), ticks=int(f[11]) + int(f[12])))
                rows.append(dict(pid=pid, start=tree[pid]['start'], ticks=tree[pid]['ticks'],
                                 argv=argv, io=io, threads=threads))
            except OSError:
                continue
        jobs[name] = rows
        if job['kind'] == 'audit':
            phases[name] = ('native-export' if any('export' in p['argv'] for p in rows)
                            else 'sorted-validation-and-final-hashes' if (job['folder']/'checkpoint.average.gz').exists()
                            else 'checkpoint-verification')
        else:
            phases[name] = job['kind']
    disks = {}
    for line in Path('/proc/diskstats').read_text().splitlines():
        f = line.split()
        if f[2].startswith('nvme') and 'p' not in f[2]:
            disks[f[2]] = dict(read_sectors=int(f[5]), written_sectors=int(f[9]), busy_ms=int(f[12]))
    return dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), monotonic=time.monotonic(),
                hz=os.sysconf('SC_CLK_TCK'), jobs=jobs, phases=phases, disks=disks)


def native_record(base, seed, label):
    folder = base / 'results' / str(seed) / label
    record = read(folder / 'telemetry.jsonl')
    if (record['binary_sha256'] != BINARY_SHA or record['stop_requested']
            or (label != 'twin500m' and record['status'] != 'saved')
            or record['path'] != str(folder / 'checkpoint.gz')):
        raise ValueError('Incomplete or changed native target')
    if label == 'twin500m':
        parent = read(base / 'results' / str(seed) / 'prefix500m/native-record.json')
        if (record['checkpoint_sha256'] != PREFIX[seed]
                or record['completed_nodes'] != parent['completed_nodes']):
            raise ValueError('Twin checkpoint hash/node mismatch')
        path = folder / 'normalize-parent.json'
        if not path.exists():
            put(path, dict(seed=seed, record=parent, large_cap=read(base / 'admission.json')['selected_cap']))
    elif label.isdigit() and record['completed_nodes'] < int(label):
        raise ValueError('Target not reached')
    if not (folder / 'native-record.json').exists():
        put(folder / 'native-record.json', record)
    elif read(folder / 'native-record.json') != record:
        raise ValueError('Native receipt changed')
    return record


def worker(args):
    base, folder = args.base, args.folder
    if args.phase == 'prefix-finalize':
        growing, normalized = {}, {}
        for seed in SEEDS:
            growing[seed] = checked_audit(base / 'results' / str(seed) / 'prefix500m')['native']
            normalized[seed] = checked_audit(base / 'results' / str(seed) / 'twin500m')['native']
            identity = read(base / 'results' / str(seed) / 'twin500m/twin-identity.json')
            if identity['status'] != 'verified' or not identity['all_row_bytes_equal']:
                raise ValueError('Missing exact twin identity')
        if not (base / 'twin-gates.json').exists():
            put(base / 'twin-gates.json', dict(status='all-three-exact', large=growing, normalized=normalized))
        campaign = Campaign.__new__(Campaign)
        campaign.base, campaign.admission = base, read(base / 'admission.json')
        if not (base / 'prefix-memory-admission.json').exists():
            campaign.check_prefix_memory()
        if read(base / 'prefix-memory-admission.json')['status'] != 'passed':
            raise ValueError('Mature prefix memory exceeds admission')
        for seed in SEEDS:
            for label in ('prefix500m', 'twin500m'):
                folder = base / 'results' / str(seed) / label
                checked = checked_audit(folder)
                for kind in ('current', 'average'):
                    item = checked['files'][kind]
                    if Path(item['path']).exists():
                        retire(base, Path(item['path']), item, folder / 'audit.json')
        return
    if args.phase == 'retire':
        checked = checked_audit(folder)
        seed, label = int(folder.parent.name), folder.name
        old_label = 'prefix500m' if label == str(TARGETS[0]) else str(TARGETS[TARGETS.index(int(label)) - 1])
        items = [checked['files']['current']]
        if int(label) not in (TARGETS[1], TARGETS[3]):
            items.append(checked['files']['average'])
        items.append(checked_audit(base / 'results' / str(seed) / old_label)['files']['checkpoint'])
        if int(label) == TARGETS[0]:
            items.append(checked_audit(base / 'results' / str(seed) / 'twin500m')['files']['checkpoint'])
        for item in items:
            retire(base, Path(item['path']), item, folder / 'audit.json')
        put(folder / 'scheduler-retired.json', dict(status='verified', removed=items))
        return
    raise ValueError('Unknown worker phase')


class Queue:
    def __init__(self, base):
        self.base = base
        self.clock = read(base / 'training-clock.json')
        self.deadline = self.clock['deadline_epoch']
        self.admission = read(base / 'admission.json')
        if (self.admission != read(Path('docs/reports/hu200-capacity-cygnus-artifacts/admission.json'))
                or self.clock != read(Path('docs/reports/hu200-capacity-cygnus-artifacts/training-clock.json'))):
            raise ValueError('Original admission or deadline differs from committed pins')
        if (base / 'training-failed.json').exists() or (base / 'training-complete.json').exists():
            raise ValueError('Campaign terminal; never restart')
        handoff = read(base / 'scheduling-handoff.json')
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        if revision != handoff['new_source'] or subprocess.check_output(
                ['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
            raise ValueError('Clean pinned scheduling source required')
        if file_hash(QUALIFIED / 'trainer-first') != BINARY_SHA:
            raise ValueError('Qualified binary changed')
        if pin(base / 'training-clock.json')['sha256'] != handoff['original_clock']['sha256']:
            raise ValueError('Original clock changed')
        self.root = base / 'guards/scheduler'
        self.root.mkdir(exist_ok=False)
        self.active, self.done, self.tasks = {}, set(), {}
        self.audit_limit = 2
        self.batch = []
        self.batch_started = None
        self.batch_decided = False
        self.next_counter = 0
        self.last_counter = None
        self.cpu_deltas = {}
        self.disk_busy_ms = 0
        self.last_disk = None
        self.delivery_pending = False
        self.next_delivery = 0
        self.resource_summary = dict(peak_cpu_c=0, peak_vrm_c=0, peak_helper_rss_bytes=0,
                                     minimum_available_bytes=2**63, minimum_free_disk_bytes=2**63)
        self.build()

    def task(self, name, kind, deps, *, seed=None, label=None):
        self.tasks[name] = dict(name=name, kind=kind, deps=set(deps), seed=seed, label=label)

    def build(self):
        for seed in SEEDS:
            checked_audit(self.base / 'results' / str(seed) / 'prefix500m')
            folder = self.base / 'results' / str(seed) / 'twin500m'
            native_record(self.base, seed, 'twin500m')
            name = f'prefix-audit-{seed}'
            self.task(name, 'audit', [], seed=seed, label='twin500m')
            if (folder / 'audit.json').exists():
                checked_audit(folder)
                self.done.add(name)
        self.task('prefix-finalize', 'prefix-finalize', [f'prefix-audit-{s}' for s in SEEDS])
        if (self.base / 'prefix-memory-admission.json').exists():
            # This handoff is designed for the prefix audit boundary. Refuse a
            # later unreviewed state instead of guessing which old work ran.
            raise ValueError('Handoff passed prefix-finalization boundary')
        self.task('restore', 'restore', ['prefix-finalize'])
        for seed in SEEDS:
            previous = 'prefix-finalize'
            for target in TARGETS:
                label = str(target)
                train, audit, clean = (f'{kind}-{seed}-{label}' for kind in ('train', 'audit', 'retire'))
                deps = [previous] + (['restore'] if target == TARGETS[1] else [])
                self.task(train, 'train', deps, seed=seed, label=label)
                self.task(audit, 'audit', [train], seed=seed, label=label)
                self.task(clean, 'retire', [audit], seed=seed, label=label)
                previous = clean

    def folder(self, task):
        return self.base / 'results' / str(task['seed']) / str(task['label'])

    def reservation(self, job):
        written = max(0, owned_bytes(job['folder']) - job['before_bytes']) if job.get('folder') else 0
        if written > job['reserve_bytes']:
            raise RuntimeError('Writer exceeded its full output reservation')
        return job['reserve_bytes'] - written

    def ready(self, task):
        if not task['deps'].issubset(self.done):
            return False
        if task['kind'] == 'restore':
            index = read(Path('docs/reports/hu200-capacity-cygnus-artifacts/drive-inputs.json'))
            return all((self.base / 'inputs/archives' / i['name']).exists()
                       and (self.base / 'inputs/archives' / i['name']).stat().st_size == i['bytes']
                       for i in index['archives'])
        return True

    def command(self, task):
        kind = task['kind']
        if kind == 'train':
            folder = self.folder(task)
            if folder.exists():
                raise ValueError('Refuse repeated scientific attempt or output directory')
            folder.mkdir()
            seed, target = task['seed'], int(task['label'])
            old = 'prefix500m' if target == TARGETS[0] else str(TARGETS[TARGETS.index(target) - 1])
            parent_folder = folder.parent / old
            parent = checked_audit(parent_folder)['native']
            header = read(parent_folder / 'saved-bytes.json')['header']
            cap = self.admission['selected_cap']
            if header.get('frozen_table') is not None:
                cap = max(cap, header['frozen_table']['entries'])
            recovery = Path(parent['path'])
            expected = checked_audit(parent_folder)['files']['checkpoint']
            if not recovery.exists() or (recovery.stat().st_size, recovery.stat().st_mtime_ns) != (expected['bytes'], expected['mtime_ns']):
                raise ValueError('Verified parent recovery missing or changed')
            return [str(QUALIFIED / 'trainer-first'), 'train', '--stack-bb', '200', '--menu', 'v1',
                    '--seed', str(seed), '--roots-per-seat', '1', '--average-rule', 'opponent-sampled',
                    '--card-buckets', str(QUALIFIED / 'tables'), '--max-entries', str(cap),
                    '--nodes', str(target), '--recovery', '--max-seconds', str(max(1, self.deadline-time.time()-3600)),
                    '--out', str(folder / 'checkpoint.gz'), '--telemetry', str(folder / 'telemetry.jsonl'),
                    '--stop-file', str(folder / 'stop.request'), '--freeze-at-cap',
                    '--resume', parent['path'], '--resume-sha256', parent['checkpoint_sha256']]
        if kind == 'audit':
            return [sys.executable, '-m', 'scripts.run_hu200_capacity_training', 'audit', '--base', str(self.folder(task))]
        if kind == 'restore':
            return [sys.executable, '-m', 'scripts.restore_hu200_capacity_twins', '--base', str(self.base)]
        result = [sys.executable, '-m', 'scripts.schedule_hu200_capacity', kind, '--base', str(self.base)]
        if kind == 'retire':
            result += ['--folder', str(self.folder(task))]
        return result

    def output_reservation(self, task):
        kind = task['kind']
        return (20_000_000_000 if kind == 'train' else 23_000_000_000 if kind == 'audit'
                else 10_000_000_000 if kind == 'restore' else 0)

    def launch(self, task, row, totals):
        kind, name = task['kind'], task['name']
        if kind == 'audit' and sum(j['kind'] == 'audit' for j in self.active.values()) >= self.audit_limit:
            return False
        helpers = [j for j in self.active.values() if j['kind'] != 'train']
        if kind != 'train' and len(helpers) >= 3:
            return False
        cpus = allocate(list(self.active.values()), kind, task['seed'])
        if cpus is None or row['package_c'] >= 68 or row['vrm_c'] >= 83:
            return False
        reserve = self.output_reservation(task)
        if not fits(totals, row['free_disk_bytes'], sum(self.reservation(j) for j in self.active.values()), reserve):
            return False
        folder = self.folder(task) if task['seed'] is not None else None
        command = self.command(task)
        stop = (folder / ('stop.request' if kind == 'train' else kind + '.stop')) if folder else self.root / (name + '.stop')
        if stop.exists():
            raise ValueError('Stop file is not fresh')
        rss = 38 if kind == 'train' else 0.5
        log = (self.root / (name + '.log')).open('x')
        before_bytes = owned_bytes(folder) if folder else 0
        threads = str(len(cpus))
        env = dict(os.environ, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                   OPENBLAS_NUM_THREADS=threads, RAYON_NUM_THREADS=threads,
                   MKL_CBWR='AVX2,STRICT', PYTHONUNBUFFERED='1')
        plan = dict(name=name, kind=kind, cpus=cpus, rss_gib=rss, stop_file=str(stop),
                    command=command, reserve_bytes=reserve, original_deadline=self.deadline)
        put(self.root / (name + '.plan.json'), plan)
        proc = subprocess.Popen(['taskset', '-c', ','.join(map(str, cpus)), *command],
                                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.active[name] = dict(**plan, proc=proc, identity=process(proc.pid)['start'], log=log,
                                 started=time.monotonic(), folder=folder,
                                 before_bytes=before_bytes)
        if kind == 'audit' and not self.batch_decided and len(self.batch) < 2:
            others = [n for n,j in self.active.items() if n != name and j['kind'] == 'audit']
            if others:
                self.batch = [others[0], name]
                self.batch_started = min(self.active[n]['started'] for n in self.batch)
                self.disk_busy_ms, self.last_disk = 0, None
            else:
                self.batch = [name]
                self.batch_started = time.monotonic()
        print(json.dumps(dict(event='started', job=plan, pid=proc.pid)), flush=True)
        return True

    def finish(self, name):
        job = self.active[name]
        code = job['proc'].poll()
        if code != 0:
            raise RuntimeError(f'{name} exit {code}; no retry')
        task = self.tasks[name]
        if task['kind'] == 'train':
            native_record(self.base, task['seed'], task['label'])
        elif task['kind'] == 'audit':
            checked_audit(self.folder(task))
        result = dict(status='passed', exit_codes=[0], elapsed_seconds=time.monotonic()-job['started'],
                      stop_reason=None, hard_stop_reason=None, cpus=job['cpus'])
        put(self.root / (name + '.result.json'), result)
        job['log'].close()
        del self.active[name]
        self.done.add(name)
        print(json.dumps(dict(event='passed', job=name, **result)), flush=True)

    def counters(self, stream):
        sample = sample_counters(self.active)
        stream.write(json.dumps(sample) + '\n')
        # Threads/processes can appear and disappear between samples. Accumulate
        # only identity-matched positive deltas; this is conservative utilization.
        current = {(name, p['pid'], p['start']): p['ticks'] for name, ps in sample['jobs'].items() for p in ps}
        if self.last_counter:
            for key, ticks in current.items():
                if key in self.last_counter:
                    self.cpu_deltas[key[0]] = self.cpu_deltas.get(key[0], 0) + max(0, ticks-self.last_counter[key])/sample['hz']
        self.last_counter = current
        if self.last_disk:
            self.disk_busy_ms += sum(max(0, d['busy_ms']-self.last_disk[k]['busy_ms'])
                                     for k, d in sample['disks'].items() if k in self.last_disk)
        self.last_disk = sample['disks']

    def decide_batch(self, row, totals):
        if self.batch_decided or len(self.batch) < 2 or not set(self.batch).issubset(self.done):
            return
        results = [read(self.root / (name + '.result.json')) for name in self.batch]
        # Normalize by audited rows; normalized-prefix audits also compare both
        # checkpoints, so use the completed original normalized audit if present.
        handoff = read(self.base / 'scheduling-handoff.json')
        baseline_seconds = handoff['completed_guard_elapsed_seconds']
        baseline_rows = read(Path(handoff['completed_audit']['path']))['native']['diagnostics']['entries']
        counts = [checked_audit(self.folder(self.tasks[n]))['native']['diagnostics']['entries'] for n in self.batch]
        wall = time.monotonic()-self.batch_started
        speedup = (sum(counts)/wall)/(baseline_rows/baseline_seconds)
        cpu = sum(self.cpu_deltas.get(n, 0) for n in self.batch)/sum(r['elapsed_seconds'] for r in results)
        busy = self.disk_busy_ms/1000/wall
        admitted = (speedup >= 1.4 and cpu >= 0.8 and busy < 0.5
                    and row['available_bytes'] >= 6*GIB
                    and fits(totals, row['free_disk_bytes'], sum(self.reservation(j) for j in self.active.values()), 23_000_000_000))
        self.audit_limit = 3 if admitted else 2
        receipt = dict(status='measured', first_batch=self.batch, counts=counts, results=results,
                       elapsed_seconds=wall, baseline_seconds=baseline_seconds, baseline_rows=baseline_rows,
                       row_normalized_speedup=speedup, conservative_cpu_fraction=cpu,
                       host_nvme_busy_fraction=busy, admitted_audit_limit=self.audit_limit,
                       resources=self.resource_summary,
                       original_deadline=self.deadline,
                       admission_rule='speedup>=1.4, CPU>=0.8, NVMe busy<0.5, unchanged RAM/disk floors and output reservation')
        put(self.base / 'parallel-audit-batch.json', receipt)
        self.batch_decided = True

    def publish(self):
        for target in TARGETS:
            path = self.base / f'milestone-{target}.json'
            if not path.exists() and all(f'retire-{s}-{target}' in self.done for s in SEEDS):
                put(path, {s: read(self.base / 'results' / str(s) / str(target) / 'native-record.json') for s in SEEDS})
        batch = self.base / 'parallel-audit-batch.json'
        delivered = self.base / 'parallel-audit-batch-delivery.json'
        if (batch.exists() and not delivered.exists() and not self.delivery_pending
                and time.monotonic() >= self.next_delivery):
            self.delivery_pending = True
            self.next_delivery = time.monotonic()+30
            threading.Thread(target=self.deliver_batch, args=(delivered,), daemon=True).start()

    def deliver_batch(self, delivered):
        try:
            payload = dict(event='first_parallel_audit_batch_measured', experiment='PR301',
                           observed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            request = urllib.request.Request((self.base/'campaign-events-url.txt').read_text().strip(),
                data=json.dumps(payload).encode(), headers={'Content-Type':'application/json',
                'User-Agent':'T3Code-HU200Capacity-Observer/1.0'}, method='POST')
            with urllib.request.urlopen(request, timeout=3) as response:
                put(delivered, dict(payload=payload, http_status=response.status))
        except Exception as exc:
            print(json.dumps(dict(event='batch_callback_pending', error_type=type(exc).__name__)), flush=True)
        finally:
            self.delivery_pending = False

    def run(self):
        package = sensors.package_sensor()
        vrm, fan = motherboard_sensors()
        window, requested, reason = VrmWindow(), None, None
        interrupted = []
        previous = {s: signal.signal(s, lambda n, f: interrupted.append(n)) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        started = time.monotonic()
        try:
            with (self.root/'resources.jsonl').open('x', buffering=1) as trace, (self.root/'utilization.jsonl').open('x', buffering=1) as counters:
                while len(self.done) < len(self.tasks):
                    row = sensors.sample(package)
                    row.update(vrm_c=sensors.read_int(vrm)/1000, fan1_rpm=sensors.read_int(fan),
                               free_disk_bytes=shutil.disk_usage(self.base).free)
                    row['fan_rpm'] = {p.stem.removesuffix('_input'):sensors.read_int(p) for p in fan.parent.glob('fan*_input')}
                    if not -20 <= row['vrm_c'] <= 150 or row['fan1_rpm'] < 0:
                        raise RuntimeError('Invalid motherboard reading')
                    rss = {n:sensors.group_rss(j['proc'].pid) for n,j in self.active.items()}
                    row.update(job_rss_bytes=rss, family_rss_bytes=sum(rss.values()),
                               active_jobs={n:dict(kind=j['kind'],cpus=j['cpus']) for n,j in self.active.items()},
                               clean_stop_requested=requested is not None)
                    if hasattr(self, 'resource_summary'):
                        summary = self.resource_summary
                        summary['peak_cpu_c'] = max(summary['peak_cpu_c'], row['package_c'])
                        summary['peak_vrm_c'] = max(summary['peak_vrm_c'], row['vrm_c'])
                        summary['peak_helper_rss_bytes'] = max(summary['peak_helper_rss_bytes'], sum(
                            rss[n] for n,j in self.active.items() if j['kind'] != 'train'))
                        summary['minimum_available_bytes'] = min(summary['minimum_available_bytes'], row['available_bytes'])
                        summary['minimum_free_disk_bytes'] = min(summary['minimum_free_disk_bytes'], row['free_disk_bytes'])
                    hot = window.update(time.monotonic(), row['vrm_c'])
                    row['vrm_hot_seconds'] = 0 if window.since is None else time.monotonic()-window.since
                    trace.write(json.dumps(row)+'\n')
                    breach = resource_breach(row, rss, list(self.active.values()), self.admission['swap_baseline_bytes'], time.time(), self.deadline)
                    if breach or interrupted:
                        raise RuntimeError(breach or f'Supervisor signal {interrupted}')
                    if hot and requested is None:
                        reason, requested = 'VRM >=95 C for 30 continuous observed seconds', time.monotonic()
                        stop_files(list(self.active.values()), reason)
                    if requested is not None and time.monotonic()-requested >= 3600:
                        raise RuntimeError('Clean-save grace exhausted')
                    if time.monotonic() >= self.next_counter:
                        self.counters(counters)
                        self.next_counter = time.monotonic()+5
                    if requested is not None:
                        if all(j['proc'].poll() is not None for j in self.active.values()):
                            raise RuntimeError(reason)
                        time.sleep(1)
                        continue
                    for name in list(self.active):
                        if self.active[name]['proc'].poll() is not None:
                            self.finish(name)
                    totals = owned_bytes(self.base)
                    reservations = sum(self.reservation(j) for j in self.active.values())
                    if not fits(totals, row['free_disk_bytes'], reservations, 0):
                        raise RuntimeError('Existing259GB plan/output reservations exhausted')
                    self.decide_batch(row, totals)
                    # Cleanup first frees verified bytes; then training and ready
                    # audits compete under one disjoint CPU/resource reservation.
                    order = {'retire':0,'prefix-finalize':0,'restore':1,'train':2,'audit':3}
                    for task in sorted(self.tasks.values(), key=lambda t:order[t['kind']]):
                        if task['name'] not in self.done and task['name'] not in self.active and self.ready(task):
                            if self.launch(task,row,totals):
                                totals = owned_bytes(self.base)
                    if (not self.active and row['package_c'] < 68 and row['vrm_c'] < 83
                            and any(t['name'] not in self.done and self.ready(t) for t in self.tasks.values())):
                        raise RuntimeError('Preventive output reservation does not fit the original budget')
                    self.publish()
                    time.sleep(1)
            self.complete()
            put(self.root/'result.json', dict(status='passed', elapsed_seconds=time.monotonic()-started))
        except BaseException as exc:
            stop_files(list(self.active.values()), str(exc))
            self.terminate()
            if not (self.base/'training-failed.json').exists():
                put(self.base/'training-failed.json',dict(status='failed',error=repr(exc),automatic_retry=False,
                    utc=dt.datetime.now(dt.timezone.utc).isoformat(),source='approved scheduling controller'))
            put(self.root/'result.json',dict(status='failed',error=repr(exc),elapsed_seconds=time.monotonic()-started))
            raise
        finally:
            self.terminate()
            for sig, handler in previous.items():
                signal.signal(sig,handler)

    def complete(self):
        milestones = {str(n):read(self.base/f'milestone-{n}.json') for n in TARGETS}
        put(self.base/'training-complete.json', dict(status='all-three-20B-audited', milestones=milestones,
            terminal=milestones[str(TARGETS[-1])], admission=self.admission,
            finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            scheduling_source=read(self.base/'scheduling-handoff.json')['new_source']))

    def terminate(self):
        owned=[]
        for job in self.active.values():
            try:
                if process(job['proc'].pid)['start']==job['identity']:
                    owned.append(job)
            except OSError:
                pass
        for job in owned:
            try:os.killpg(job['proc'].pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for job in owned:
            sensors.terminate_group(job['proc'])
        for job in self.active.values():
            job['log'].close()


def handoff(base, pid):
    """Retire only an idle old controller after a successful audit child exits."""
    original = process(pid)
    argv = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
    if b'scripts.run_hu200_capacity_training' not in argv or b'train' not in argv or str(base).encode() not in argv:
        raise ValueError('Unrecognized old controller')
    source = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    clock = pin(base/'training-clock.json')
    deadline = read(base/'training-clock.json')['deadline_epoch']
    while time.time()<deadline:
        if (base/'training-failed.json').exists():
            raise RuntimeError('Existing scientific failure; no takeover')
        if (base/'prefix-memory-admission.json').exists():
            raise RuntimeError('Prefix-finalization boundary passed; leave old controller running')
        fresh=[p for p in (base/'results').glob('*/twin500m/audit.json') if time.time()-p.stat().st_mtime<3]
        if not fresh:
            # Process-tree polling is needed only during the short successful
            # completion window, not throughout an hour-long streaming audit.
            time.sleep(.1)
            continue
        tree=process_tree()
        if pid not in tree or tree[pid]['start']!=original['start']:
            raise RuntimeError('Old controller exited or changed identity')
        children=descendants(tree,pid)
        # The Python worker must have exited cleanly and written its full audit.
        successful=[]
        for child in children:
            if tree[child]['state']=='Z' and tree[child]['exit_status']==0:
                try:
                    command=(Path('/proc')/str(child)/'cmdline').read_bytes()
                except OSError:
                    command=b''
                successful.append(child)
        # A zombie has no argv; identify the boundary by the most recent verified
        # audit receipt and confirm that no live descendant can be interrupted.
        if successful and fresh and not any(tree[c]['state']!='Z' for c in children):
            os.kill(pid,signal.SIGSTOP)
            stopped=process_tree()
            live=[c for c in descendants(stopped,pid) if stopped[c]['state']!='Z']
            if live:
                os.kill(pid,signal.SIGCONT)
                time.sleep(.02)
                continue
            try:
                checked_audit(fresh[-1].parent)
                guard = base/'guards'/('twin500m-audit-'+fresh[-1].parent.parent.name)
                plan = read(guard/'plan.json')
                if (len(plan['jobs']) != 1 or 'audit' not in plan['jobs'][0]['command']
                        or str(fresh[-1].parent) not in plan['jobs'][0]['command']):
                    raise ValueError('Completed audit differs from active guard plan')
                with (guard/'resources.jsonl').open() as stream:
                    rows = [json.loads(line) for line in stream]
                elapsed = rows[-1]['monotonic']-rows[0]['monotonic']
                if process(pid)['start']!=original['start']:
                    raise RuntimeError('Controller identity changed')
                put(base/'scheduling-handoff.json',dict(status='successful-audit-boundary',
                    old_controller=original,old_completed_children=successful,
                    completed_audit=pin(fresh[-1]),completed_guard_plan=pin(guard/'plan.json'),
                    completed_guard_elapsed_seconds=elapsed,new_source=source,original_clock=clock,
                    original_deadline=deadline,at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    administrative_action='Stop and retire only idle old controller; all scientific children exited0; no retry'))
                # The old supervisor installs signal handlers that would write a
                # false failure or launch further work. Retire its stopped idle
                # process without executing those handlers; no workload is killed.
                os.kill(pid,signal.SIGKILL)
                while (Path('/proc')/str(pid)).exists() and process(pid)['state']!='Z':
                    time.sleep(.01)
                return
            except BaseException:
                try:os.kill(pid,signal.SIGCONT)
                except ProcessLookupError:pass
                raise
        time.sleep(.02)
    raise RuntimeError('Original deadline reached before safe handoff')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('handoff-run','prefix-finalize','retire'))
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--folder',type=Path)
    parser.add_argument('--old-pid',type=int)
    args=parser.parse_args()
    if os.uname().nodename!='cygnus-x1' or args.base.resolve()!=BASE:
        raise ValueError('Only the owned PR301 root on cygnus-x1')
    if args.phase!='handoff-run':
        worker(args)
        return
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Clean source required before administrative handoff')
    handoff(args.base,args.old_pid)
    try:
        with Path('/tmp/cfr-cygnus-exclusive.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            Queue(args.base).run()
    except BaseException as exc:
        if not (args.base/'training-failed.json').exists():
            put(args.base/'training-failed.json',dict(status='failed',error=repr(exc),
                automatic_retry=False,source='scheduler initialization after successful handoff',
                utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        raise


if __name__=='__main__':
    main()
