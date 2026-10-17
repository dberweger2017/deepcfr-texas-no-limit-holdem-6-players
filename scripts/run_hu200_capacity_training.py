"""One-shot three-seed capacity campaign, with serialized every-save audits."""
import argparse
import datetime as dt
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.guard_hu200_capacity import supervise, motherboard_sensors
from scripts.machine_qualification import watchdog as sensors
from scripts.hu200_capacity_preflight import BINARY_SHA, QUALIFIED, verify_rows
from src.blueprint.abstraction import HU200_EQUITY_SCHEMA
from scripts.audit_hu200_capacity_sorted import audit_sorted as audit
from src.policies.files import file_hash

SEEDS = (2026101302, 2026101421, 2026101422)
PREFIX = {
    2026101302: '2c4c1484b7e4ebefca2bb364223d6cd12e886f98b20784b51ff6ac6678f4312d',
    2026101421: '251ab69b5f69785722d2442e628d65e41d578ab20751164d482561cca16f954f',
    2026101422: '45144a01a799b81d185c20e15772fe70793518a3d35b03fa84eb6db0bfa46966'}


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def pin(path):
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=file_hash(path))


def verified_save(path, record):
    saved = pin(path)
    if (saved['bytes'], saved['sha256']) != (record['checkpoint_bytes'], record['checkpoint_sha256']):
        raise ValueError('Independent saved byte/hash check failed')
    keys, rows, count, previous = hashlib.sha256(), hashlib.sha256(), 0, b''
    with gzip.open(path, 'rb') as stream:
        header = json.loads(next(stream))
        for line in stream:
            key = line[2:34]
            if len(key) != 32 or key <= previous or line[:2] != b'["' or line[34:36] != b'",':
                raise ValueError('Noncanonical/nonunique serialized key order')
            previous = key
            keys.update(key)
            rows.update(line)
            count += 1
    if count != record['diagnostics']['entries'] or header['iteration'] != record['iteration']:
        raise ValueError('Save cardinality or iteration differs')
    if header['native_state']['completed_nodes'] != record['completed_nodes']:
        raise ValueError('Save node count differs')
    return dict(file=saved, header=header, entries=count, sorted_keys_sha256=keys.hexdigest(),
                serialized_rows_sha256=rows.hexdigest(), full_gzip_eof_crc_read=True)


def audit_worker(folder):
    if file_hash(QUALIFIED / 'trainer-first') != BINARY_SHA:
        raise ValueError('Qualified executable changed; no export admitted')
    record = json.loads((folder / 'native-record.json').read_text())
    if (folder / 'normalize-parent.json').exists():
        parent = json.loads((folder / 'normalize-parent.json').read_text())
        seed = parent['seed']
        if record['completed_nodes'] != parent['record']['completed_nodes'] or record['checkpoint_sha256'] != PREFIX[seed]:
            raise ValueError('Twin prefix identity failed; stop')
        counts = verify_rows(Path(parent['record']['path']), Path(record['path']),
            expected_entries=record['diagnostics']['entries'], old_cap=parent['large_cap'], new_cap=85_373_862)
        put(folder / 'twin-identity.json', dict(status='verified', seed=seed, counts=counts,
            only_header_difference='config.max_entries', full_twin_sha256=PREFIX[seed], all_row_bytes_equal=True))
    checkpoint = Path(record['path'])
    checked = verified_save(checkpoint, record)
    put(folder / 'saved-bytes.json', checked)
    current = folder / 'checkpoint.current.gz'
    average = folder / 'checkpoint.average.gz'
    subprocess.run([str(QUALIFIED / 'trainer-first'), 'export', str(checkpoint),
                    '--current', str(current), '--average', str(average), '--zero-mass', 'uniform'], check=True)
    spec = dict(name='capacity', seed=checked['header']['config']['seed'], iteration=record['iteration'],
                checkpoint_sha256=checked['file']['sha256'], sha256=file_hash(current))
    result = audit(checkpoint, current, average, spec, file_hash(average), expected_schema=HU200_EQUITY_SCHEMA)
    if result['all_nodes_verified'] != checked['entries']:
        raise ValueError('Every-row audit incomplete')
    put(folder / 'audit.json', dict(status='verified', native=record, audit=result,
        files=dict(checkpoint=checked['file'], current=pin(current), average=pin(average))))


def retire(base, path, expected, replacement):
    """Only this campaign's verified redundant files; never another PR or sync root."""
    resolved = path.resolve()
    if not resolved.is_relative_to((base / 'results').resolve()) or 'Research-Cloud' in resolved.parts:
        raise ValueError('Retirement outside owned nonsynced results')
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (expected['bytes'], expected['mtime_ns']) or file_hash(path) != expected['sha256']:
        raise ValueError('Retirement file changed; retain and stop')
    if not replacement.exists():
        raise ValueError('Missing verified replacement receipt')
    successor = json.loads(replacement.read_text())
    if successor.get('status') != 'verified' or successor.get('audit', {}).get('status') != 'verified':
        raise ValueError('Replacement audit is not verified')
    if successor['audit']['all_nodes_verified'] != successor['native']['diagnostics']['entries']:
        raise ValueError('Replacement audit is incomplete')
    recovery = successor['files']['checkpoint']
    recovery_stat = Path(recovery['path']).stat()
    if (recovery_stat.st_size, recovery_stat.st_mtime_ns) != (recovery['bytes'], recovery['mtime_ns']):
        raise ValueError('Replacement recovery changed; retain original')
    receipt = dict(removed=expected, replacement_receipt=str(replacement),
        reason='Owner requested retaining evaluation and recovery needs; audited redundant campaign file',
        utc=dt.datetime.now(dt.timezone.utc).isoformat())
    with (base / 'retirements.jsonl').open('a') as stream:
        stream.write(json.dumps(receipt) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    path.unlink()


class Campaign:
    def __init__(self, base, admission):
        self.base, self.admission = base, admission
        self.cap = admission['selected_cap']
        self.deadline = time.time() + 72 * 3600
        self.binary = QUALIFIED / 'trainer-first'
        if file_hash(self.binary) != BINARY_SHA or admission['status'] != 'admitted':
            raise ValueError('Missing admission or changed qualified binary')
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        if revision != admission['source'] or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
            raise ValueError('Clean admitted source required')
        if self.cap <= 85_373_928 or self.cap > 301_153_954:
            raise ValueError('Capacity outside the owner-authorized range')
        put(base / 'training-clock.json', dict(start_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
             deadline_epoch=self.deadline, max_seconds=72 * 3600, source=admission['source']))

    def remaining(self):
        seconds = self.deadline - time.time()
        if seconds <= 0:
            raise RuntimeError('Whole campaign time cap reached')
        return seconds

    def cool_before_stage(self, label):
        # Waiting is admission, not a failed scientific attempt or a retry.
        package = sensors.package_sensor()
        vrm, fan = motherboard_sensors()
        path = self.base / 'guards' / 'admissions' / (label + '.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with path.open('x', buffering=1) as readings:
            while True:
                row = sensors.sample(package)
                row.update(vrm_c=sensors.read_int(vrm) / 1000, fan1_rpm=sensors.read_int(fan),
                           free_disk_bytes=shutil.disk_usage(self.base).free)
                readings.write(json.dumps(row) + '\n')
                if (row['package_c'] >= 85 or row['available_bytes'] < 6 * 1024**3
                        or row['swap_used_bytes'] - self.admission['swap_baseline_bytes'] > 256 * 1024**2
                        or row['free_disk_bytes'] < 50_000_000_000 or row['fan1_rpm'] < 0
                        or not -20 <= row['vrm_c'] <= 150):
                    raise RuntimeError('Resource breach during idle admission')
                if row['package_c'] < 68 and -20 <= row['vrm_c'] < 83:
                    return
                if time.monotonic() - started > min(300, self.remaining()):
                    raise RuntimeError('Cooling admission timed out; no workload launched')
                time.sleep(1)

    def guarded(self, label, jobs):
        self.cool_before_stage(label)
        plan = dict(max_seconds=self.remaining(), aggregate_rss_gib=116,
            swap_baseline_bytes=self.admission['swap_baseline_bytes'], clean_save_grace_seconds=3600, jobs=jobs)
        result = supervise(plan, self.base / 'guards' / label)
        if result['status'] != 'passed':
            raise RuntimeError(f'{label} failed; no automatic retry')

    def stage(self, label, nodes, parents=None, normalize=False, rolling=False):
        folders, jobs = {}, []
        for i, seed in enumerate(SEEDS):
            folder = self.base / 'results' / str(seed) / label
            folder.mkdir(parents=True, exist_ok=False)
            folders[seed] = folder
            load_cap = 85_373_862 if normalize else self.cap
            if parents and not normalize:
                header = json.loads((Path(parents[seed]['path']).parent / 'saved-bytes.json').read_text())['header']
                if header.get('frozen_table') is not None:
                    # Complete-iteration overshoot must be admitted by the loader;
                    # the persisted frozen state still forbids every new key.
                    load_cap = max(load_cap, header['frozen_table']['entries'])
            command = [str(self.binary), 'train', '--stack-bb', '200', '--menu', 'v1', '--seed', str(seed),
                '--roots-per-seat', '1', '--average-rule', 'opponent-sampled',
                '--card-buckets', str(QUALIFIED / 'tables'), '--max-entries', str(load_cap),
                '--nodes', str(nodes[seed] if isinstance(nodes, dict) else nodes), '--recovery',
                '--max-seconds', str(max(1, self.remaining() - 3600)), '--out', str(folder / 'checkpoint.gz'),
                '--telemetry', str(folder / 'telemetry.jsonl'), '--stop-file', str(folder / 'stop.request')]
            if not normalize:
                command += ['--freeze-at-cap']
            if parents:
                parent = parents[seed]
                command += ['--resume', parent['path'], '--resume-sha256', parent['checkpoint_sha256']]
                if normalize:
                    command += ['--iterations', str(parent['iteration'])]
            jobs.append(dict(name=str(seed), cpus=[2 * i, 2 * i + 1], rss_gib=38,
                             stop_file=str(folder / 'stop.request'), command=command))
        # Reserve three simultaneous checkpoint outputs before launching any writer.
        if shutil.disk_usage(self.base).free < 50_000_000_000 + 60_000_000_000:
            raise RuntimeError('Preventive three-save disk admission')
        self.guarded(label + '-train', jobs)
        records = {}
        for seed in SEEDS:
            folder = folders[seed]
            record = json.loads((folder / 'telemetry.jsonl').read_text())
            if record['binary_sha256'] != BINARY_SHA or record['stop_requested']:
                raise ValueError('Native binary changed or stop requested')
            if normalize:
                if record['completed_nodes'] != parents[seed]['completed_nodes']:
                    raise ValueError('No-step normalization trained')
                if record['checkpoint_sha256'] != PREFIX[seed]:
                    raise ValueError('Twin 500M identity failure: stop all dependent training')
                put(folder / 'normalize-parent.json', dict(seed=seed, record=parents[seed], large_cap=self.cap))
            elif record['status'] != 'saved':
                raise ValueError('Incomplete native target; retain and stop')
            put(folder / 'native-record.json', record)
            # Sorted full audits need only their two output files, not a SQLite ledger.
            if shutil.disk_usage(self.base).free < 50_000_000_000 + 23_000_000_000:
                raise RuntimeError('Preventive full-audit disk admission')
            self.guarded(label + '-audit-' + str(seed), [dict(name='audit', cpus=[6, 7], rss_gib=4,
                stop_file=str(folder / 'audit.stop'), command=[sys.executable, '-m',
                    'scripts.run_hu200_capacity_training', 'audit', '--base', str(folder)])])
            records[seed] = record
            if rolling:
                checked = json.loads((folder / 'audit.json').read_text())
                items = [dict(path=checked['files']['current']['path'], expected=checked['files']['current'],
                              replacement=str(folder / 'audit.json'))]
                if label not in ('5000000000', '20000000000'):
                    value = checked['files']['average']
                    items.append(dict(path=value['path'], expected=value, replacement=str(folder / 'audit.json')))
                old = Path(parents[seed]['path']).parent
                value = json.loads((old / 'audit.json').read_text())['files']['checkpoint']
                items.append(dict(path=value['path'], expected=value, replacement=str(folder / 'audit.json')))
                self.retire_files(label + '-' + str(seed), items)
        return records

    def retire_files(self, label, items):
        plan = self.base / 'cleanup' / (label + '.json')
        put(plan, dict(base=str(self.base), items=items))
        self.guarded('cleanup-' + label, [dict(name='cleanup', cpus=[6, 7], rss_gib=1,
            stop_file=str(plan.with_suffix('.stop')), command=[sys.executable, '-m',
                'scripts.run_hu200_capacity_training', 'clean', '--base', str(plan)])])

    def trim_exports(self, label, records, keep_average):
        items = []
        for seed in SEEDS:
            folder = Path(records[seed]['path']).parent
            checked = json.loads((folder / 'audit.json').read_text())
            for name in ['current'] + ([] if keep_average else ['average']):
                value = checked['files'][name]
                items.append(dict(path=value['path'], expected=value, replacement=str(folder / 'audit.json')))
        self.retire_files('exports-' + label, items)

    def replace_recovery(self, older, newer):
        items = []
        for seed in SEEDS:
            old = Path(older[seed]['path']).parent
            new = Path(newer[seed]['path']).parent / 'audit.json'
            checked = json.loads((old / 'audit.json').read_text())
            value = checked['files']['checkpoint']
            items.append(dict(path=value['path'], expected=value, replacement=str(new)))
        self.retire_files('recovery-' + Path(older[SEEDS[0]]['path']).parent.name + '-to-' + Path(newer[SEEDS[0]]['path']).parent.name, items)

    def check_prefix_memory(self):
        """Charge every regret-free action for eventual regret allocation."""
        measured = []
        samples = [json.loads(line) for line in
            (self.base / 'guards/prefix500m-train/resources.jsonl').read_text().splitlines()]
        for seed in SEEDS:
            counts = json.loads((self.base / 'results' / str(seed) / 'twin500m/twin-identity.json').read_text())['counts']
            peak = max(row['job_rss_bytes'][str(seed)] for row in samples)
            fixed = self.admission['fixed_card_table_bytes_per_run']
            aging = 8 * counts['regret_free_slots']
            mature_bytes_per_entry = (peak - fixed + aging) / counts['entries']
            measured.append(dict(seed=seed, peak_rss_bytes=peak, entries=counts['entries'],
                action_slots=counts['action_slots'], regret_free_slots=counts['regret_free_slots'],
                mature_bytes_per_entry=mature_bytes_per_entry))
        put(self.base / 'prefix-memory-admission.json', dict(measurements=measured,
            budget_bytes_per_entry=130, status='passed' if all(
                m['mature_bytes_per_entry'] <= 130 for m in measured) else 'failed'))
        if any(m['mature_bytes_per_entry'] > 130 for m in measured):
            raise RuntimeError('Measured mature prefix exceeds declared memory budget; stop')

    def run(self):
        fresh = self.stage('gate-fresh20m', 20_000_000)
        prefix = self.stage('gate-prefix10m', 10_000_000)
        resumed = self.stage('gate-resumed20m', 20_000_000, prefix)
        for seed in SEEDS:
            if fresh[seed]['checkpoint_sha256'] != resumed[seed]['checkpoint_sha256']:
                raise ValueError('Exact resume gate failed; stop')
        put(self.base / 'resume-gates.json', dict(status='byte-identical', fresh=fresh, resumed=resumed))
        for label, records in [('fresh', fresh), ('prefix', prefix), ('resumed', resumed)]:
            self.trim_exports(label, records, False)
        growing = self.stage('prefix500m', 500_000_000)
        normalized = self.stage('twin500m', {s: growing[s]['completed_nodes'] + 1 for s in SEEDS}, growing, True)
        put(self.base / 'twin-gates.json', dict(status='all-three-exact', large=growing, normalized=normalized))
        self.check_prefix_memory()
        self.trim_exports('prefix500m', growing, False)
        self.trim_exports('twin500m', normalized, False)
        previous = growing
        milestones = {}
        for nodes in (2_000_000_000, 5_000_000_000, 10_000_000_000, 20_000_000_000):
            if nodes == 5_000_000_000 and not (self.base / 'twins-restored.json').exists():
                self.guarded('restore-twins', [dict(name='restore', cpus=[6, 7], rss_gib=2,
                    stop_file=str(self.base / 'results/restore.stop'), command=[sys.executable, '-m',
                    'scripts.run_hu200_capacity_training', 'restore-wait', '--base', str(self.base)])])
            records = self.stage(str(nodes), nodes, previous, rolling=True)
            if nodes == 2_000_000_000:
                self.replace_recovery(normalized, records)
            milestones[str(nodes)] = records
            put(self.base / ('milestone-' + str(nodes) + '.json'), records)
            previous = records
        put(self.base / 'training-complete.json', dict(status='all-three-20B-audited', milestones=milestones,
            terminal=previous, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(), admission=self.admission))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'audit', 'clean', 'restore-wait'))
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--admission', type=Path)
    args = parser.parse_args()
    if args.phase == 'restore-wait':
        from scripts.restore_hu200_capacity_twins import restore
        inputs = json.loads(Path('docs/reports/hu200-capacity-cygnus-artifacts/drive-inputs.json').read_text())
        while True:
            if all((args.base / 'inputs/archives' / i['name']).exists() and (args.base / 'inputs/archives' / i['name']).stat().st_size == i['bytes'] for i in inputs['archives']):
                restore(args.base)
                return
            time.sleep(10)
    if args.phase == 'clean':
        plan = json.loads(args.base.read_text())
        for item in plan['items']:
            retire(Path(plan['base']), Path(item['path']), item['expected'], Path(item['replacement']))
        return
    if args.phase == 'audit':
        audit_worker(args.base)
        return
    with Path('/tmp/cfr-cygnus-exclusive.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            Campaign(args.base, json.loads(args.admission.read_text())).run()
        except BaseException as exc:
            put(args.base / 'training-failed.json', dict(status='failed', error=repr(exc),
                utc=dt.datetime.now(dt.timezone.utc).isoformat(), automatic_retry=False))
            raise


if __name__ == '__main__':
    main()
