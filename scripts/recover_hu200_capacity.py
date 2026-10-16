"""One owner-approved PR301 common5B rollback; reuse the qualified global guard."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts import schedule_hu200_capacity as q
from scripts.run_hu200_capacity_training import BINARY_SHA, QUALIFIED, SEEDS, pin, put
from src.policies.files import file_hash

BASE = q.BASE
OUTPUT = 'results-recovery-20261016'
GUARD = 'recovery-20261016'
LEDGER = 'training-deadline-amendment-20261016.json'
ADMISSION = 'recovery-admission-20261016.json'
PINS = Path('docs/reports/hu200-capacity-cygnus-artifacts')
CLOCK_SHA = '0b0141baba6c6041617463ee6f50efe0f262432caca5be7d169d990c0874fe7a'
TARGETS = (10_000_000_000, 20_000_000_000)
REFERENCE = {
    2026101421: '63abc0a8098b7ac9b8c8ce2a6b121640fd681d38a9cf3a48437a8502dbed30b1',
    2026101422: 'b6dd7cb280736a5c93af1590f66f68e5655abc1b5aa83de78c14aa0fa6142fa0',
}


def source_pin():
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Clean recovery source required')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def approved(base):
    receipt = q.read(base/'recovery-approval.json')
    if receipt != q.read(PINS/'recovery-approval.json') or receipt['status'] != 'owner-approved':
        raise ValueError('Exact owner approval required')
    if (receipt['extension_seconds'] != 3600 or receipt['seeds'] != list(SEEDS)
            or receipt['common_parent_nodes'] != 5_000_000_000
            or receipt['recovery_targets'] != list(TARGETS)
            or receipt['later_automatic_retries']):
        raise ValueError('Unapproved retry scope')
    return receipt


def deadline_from(clock, ledger):
    if (clock['deadline_epoch'] != 1792376957.996109 or clock['max_seconds'] != 259200
            or ledger['original_clock_sha256'] != CLOCK_SHA
            or ledger['original_deadline_epoch'] != clock['deadline_epoch']
            or ledger['extension_seconds'] != 3600
            or ledger['amended_deadline_epoch'] != clock['deadline_epoch']+3600
            or ledger['status'] != 'owner-approved-single-extension'):
        raise ValueError('Only one fixed3600s extension is authorized')
    return ledger['amended_deadline_epoch']


def effective_deadline(base):
    approval = approved(base)
    if file_hash(base/'training-clock.json') != CLOCK_SHA:
        raise ValueError('Original clock changed')
    ledger = q.read(base/LEDGER)
    if ledger['approval_sha256'] != file_hash(base/'recovery-approval.json'):
        raise ValueError('Amendment approval pin changed')
    result = deadline_from(q.read(base/'training-clock.json'), ledger)
    if result != approval['amended_deadline_epoch']:
        raise ValueError('Deadline differs from approval')
    return result


def parent_folder(base, seed, target):
    if seed not in SEEDS or target not in TARGETS:
        raise ValueError('Unapproved recovery seed/target')
    if target == TARGETS[0]:
        return base/'results'/str(seed)/'5000000000'
    return base/OUTPUT/str(seed)/str(TARGETS[0])


def intact(item, expected_path):
    path = Path(item['path'])
    if path != expected_path or path.is_symlink() or path.resolve() != path:
        raise ValueError('Unrecognized recovery input path')
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (item['bytes'], item['mtime_ns']):
        raise ValueError('Verified recovery input changed')


def adopt(base, rehash=False):
    readback = q.read(base/'planning/power-loss-byte-readback-20261016.json')
    if readback != q.read(PINS/'power-off-byte-readback.json'):
        raise ValueError('Post-boot readback differs from committed receipt')
    if len(readback['files']) != 11 or not all(x['verified'] for x in readback['files']):
        raise ValueError('Post-boot bytes not verified')
    by_path = {x['path']: x for x in readback['files']}
    adopted = {}
    for seed in SEEDS:
        folder = parent_folder(base, seed, TARGETS[0])
        audit = q.checked_audit(folder)
        if audit != q.read(PINS/f'save-5b/{seed}/audit.json'):
            raise ValueError('Common5B audit differs from committed receipt')
        for kind, name in [('checkpoint', 'checkpoint.gz'), ('average', 'checkpoint.average.gz')]:
            item = audit['files'][kind]
            intact(item, folder/name)
            old = by_path[item['path']]
            if old['sha256'] != item['sha256'] or old['bytes'] != item['bytes']:
                raise ValueError('Recovery parent does not match post-boot readback')
            if kind == 'checkpoint' and rehash and file_hash(folder/name) != item['sha256']:
                raise ValueError('Fresh5B recovery hash mismatch')
        adopted[str(seed)] = dict(audit_sha256=file_hash(folder/'audit.json'),
                                 checkpoint=audit['files']['checkpoint'],
                                 average=audit['files']['average'])
    for seed in SEEDS:
        path = base/'inputs'/f'seed-{seed}'/'final.average.gz'
        old = by_path[str(path)]
        if (path.stat().st_size, path.stat().st_mtime_ns) != (old['bytes'], old['mtime_ns']):
            raise ValueError('Verified twin changed')
    gates = {}
    for name in ['resume-gates.json', 'twin-gates.json', 'prefix-memory-admission.json', 'twins-restored.json']:
        if file_hash(base/name) != file_hash(PINS/name):
            raise ValueError('Previously verified gate changed')
        gates[name] = file_hash(base/name)
    if q.read(base/'twin-gates.json')['status'] != 'all-three-exact' or q.read(base/'prefix-memory-admission.json')['status'] != 'passed':
        raise ValueError('Original global gates incomplete')
    return adopted, gates


def no_attempt(base):
    for name in ['training-failed.json', 'training-complete.json', OUTPUT, 'guards/'+GUARD]:
        if (base/name).exists():
            raise ValueError('Recovery already attempted or campaign terminal; no retry')


def require_admission(row, admission, owned):
    if (row['package_c'] >= 68 or row['vrm_c'] >= 83
            or row['available_bytes'] < admission['memory_required_including_2GiB_helper_and_6GiB_available_floor']
            or row['swap_used_bytes']-admission['swap_baseline_bytes'] > 256*1024**2):
        raise ValueError('Recovery RAM/temperature/swap admission failed')
    if not q.fits(owned, row['free_disk_bytes'], 0, 60_000_000_000):
        raise ValueError('Three trainer reservations exceed original259GB plan')


def prepare(base):
    no_attempt(base)
    if (base/ADMISSION).exists():
        raise ValueError('Recovery admission already sealed')
    approval = approved(base)
    source = source_pin()
    clock = q.read(base/'training-clock.json')
    if file_hash(base/'training-clock.json') != CLOCK_SHA:
        raise ValueError('Original clock changed')
    ledger = dict(status='owner-approved-single-extension', original_clock_sha256=CLOCK_SHA,
                  original_deadline_epoch=clock['deadline_epoch'], extension_seconds=3600,
                  amended_deadline_epoch=approval['amended_deadline_epoch'],
                  approval_sha256=file_hash(base/'recovery-approval.json'))
    deadline_from(clock, ledger)
    if (base/LEDGER).exists():
        if q.read(base/LEDGER) != ledger:
            raise ValueError('Existing amendment differs; never add another hour')
    else:
        put(base/LEDGER, ledger)
    if time.time() >= ledger['amended_deadline_epoch']:
        raise ValueError('Amended campaign deadline already expired')
    admission = q.read(base/'admission.json')
    if admission != q.read(PINS/'admission.json') or file_hash(QUALIFIED/'trainer-first') != BINARY_SHA:
        raise ValueError('Original admission or binary changed')
    for street, expected in admission['card_table_hashes'].items():
        candidates = list((QUALIFIED/'tables').glob(street+'*'))
        candidates = [p for p in candidates if p.is_file()]
        if len(candidates) != 1 or file_hash(candidates[0]) != expected:
            raise ValueError('Qualified K50 table changed')
    parents, gates = adopt(base, rehash=True)
    package = q.sensors.package_sensor()
    vrm, fan = q.motherboard_sensors()
    row = q.sensors.sample(package)
    row.update(vrm_c=q.sensors.read_int(vrm)/1000, fan1_rpm=q.sensors.read_int(fan),
               free_disk_bytes=shutil.disk_usage(base).free)
    owned = q.owned_bytes(base)
    require_admission(row, admission, owned)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    put(base/ADMISSION, dict(status='admitted', source=source, boot_id=boot,
        admitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        original_clock_sha256=CLOCK_SHA, amendment_sha256=file_hash(base/LEDGER),
        approval_sha256=file_hash(base/'recovery-approval.json'),
        deadline_epoch=effective_deadline(base), adopted_parents=parents, adopted_gates=gates,
        owned_bytes=owned, initial_output_reservations=60_000_000_000, resources=row,
        extra_automatic_retries=False))


def validate_native(folder, seed, target):
    record = q.read(folder/'telemetry.jsonl')
    if (record['binary_sha256'] != BINARY_SHA or record['stop_requested']
            or record['status'] != 'saved' or record['path'] != str(folder/'checkpoint.gz')
            or record['requested_nodes'] != target or record['completed_nodes'] < target):
        raise ValueError('Recovery native target incomplete')
    if target == TARGETS[0]:
        if seed in REFERENCE and record['checkpoint_sha256'] != REFERENCE[seed]:
            raise ValueError('Rebuilt10B differs from pre-shutdown compressed bytes')
        if seed == SEEDS[0] and (record['completed_nodes'], record['iteration']) != (10000000240, 7864201):
            raise ValueError('Seed1302 rebuilt10B header differs')
    path = folder/'native-record.json'
    if path.exists():
        if q.read(path) != record:
            raise ValueError('Recovery native receipt changed')
    else:
        put(path, record)
    return record


def retire_worker(base, folder):
    seed, target = int(folder.parent.name), int(folder.name)
    if folder != base/OUTPUT/str(seed)/str(target) or seed not in SEEDS or target not in TARGETS:
        raise ValueError('Retirement outside approved recovery attempt')
    checked = q.checked_audit(folder)
    parent = q.checked_audit(parent_folder(base, seed, target))
    items = [checked['files']['current'], parent['files']['checkpoint']]
    expected_paths = [folder/'checkpoint.current.gz', parent_folder(base, seed, target)/'checkpoint.gz']
    if target == TARGETS[0]:
        items.append(checked['files']['average'])
        expected_paths.append(folder/'checkpoint.average.gz')
    intact(checked['files']['checkpoint'], folder/'checkpoint.gz')
    for item, path in zip(items, expected_paths):
        intact(item, path)
        if file_hash(path) != item['sha256']:
            raise ValueError('Retirement bytes changed; retain and stop')
        receipt = dict(removed=item, replacement_receipt=str(folder/'audit.json'),
            reason='Owner-approved common5B recovery; audited redundant file',
            utc=dt.datetime.now(dt.timezone.utc).isoformat())
        with (base/'retirements.jsonl').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps(receipt)+'\n');stream.flush();os.fsync(stream.fileno())
            path.unlink()
    put(folder/'scheduler-retired.json', dict(status='verified', removed=items))


class RecoveryQueue(q.Queue):
    def __init__(self, base):
        self.base = base
        no_attempt(base)
        self.source = source_pin()
        self.recovery = q.read(base/ADMISSION)
        if self.recovery['status'] != 'admitted' or self.recovery['source'] != self.source:
            raise ValueError('Clean source differs from recovery admission')
        if self.recovery['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
            raise ValueError('Another reboot; no automatic retry')
        self.deadline = effective_deadline(base)
        if self.recovery['deadline_epoch'] != self.deadline or self.recovery['amendment_sha256'] != file_hash(base/LEDGER):
            raise ValueError('Recovery clock changed after admission')
        parents, gates = adopt(base)
        if parents != self.recovery['adopted_parents'] or gates != self.recovery['adopted_gates']:
            raise ValueError('Adopted parents/gates changed')
        self.clock, self.admission = q.read(base/'training-clock.json'), q.read(base/'admission.json')
        if file_hash(QUALIFIED/'trainer-first') != BINARY_SHA:
            raise ValueError('Qualified binary changed')
        vrm, _ = q.motherboard_sensors()
        row = q.sensors.sample(q.sensors.package_sensor())
        row.update(vrm_c=q.sensors.read_int(vrm)/1000,
                   free_disk_bytes=shutil.disk_usage(base).free)
        require_admission(row, self.admission, q.owned_bytes(base))
        self.root = base/'guards'/GUARD
        self.root.mkdir(exist_ok=False)
        (base/OUTPUT).mkdir(exist_ok=False)
        self.active, self.done, self.tasks = {}, set(), {}
        self.audit_limit, self.batch_decided = 3, True
        self.batch, self.batch_started = [], None
        self.next_counter, self.last_counter = 0, None
        self.cpu_deltas, self.disk_busy_ms, self.last_disk = {}, 0, None
        self.resource_summary = dict(peak_cpu_c=0, peak_vrm_c=0, peak_helper_rss_bytes=0,
            minimum_available_bytes=2**63, minimum_free_disk_bytes=2**63)
        self.build()

    def build(self):
        for seed in SEEDS:
            previous = []
            for target in TARGETS:
                label = str(target)
                train, audit, clean = (f'{kind}-{seed}-{label}' for kind in ('train', 'audit', 'retire'))
                self.task(train, 'train', previous, seed=seed, label=label)
                self.task(audit, 'audit', [train], seed=seed, label=label)
                self.task(clean, 'retire', [audit], seed=seed, label=label)
                previous = [clean]

    def folder(self, task):
        return self.base/OUTPUT/str(task['seed'])/task['label']

    def command(self, task):
        kind, seed, target = task['kind'], task['seed'], int(task['label'])
        folder = self.folder(task)
        if kind == 'audit':
            return [sys.executable, '-m', 'scripts.run_hu200_capacity_training', 'audit', '--base', str(folder)]
        if kind == 'retire':
            return [sys.executable, '-m', 'scripts.recover_hu200_capacity', 'retire',
                    '--base', str(self.base), '--folder', str(folder)]
        if kind != 'train' or folder.exists():
            raise ValueError('Refuse repeated or unrecognized scientific attempt')
        parent_path = parent_folder(self.base, seed, target)
        parent = q.checked_audit(parent_path)
        recovery = parent['files']['checkpoint']
        intact(recovery, parent_path/'checkpoint.gz')
        cap = self.admission['selected_cap']
        header = q.read(parent_path/'saved-bytes.json')['header']
        if header.get('frozen_table') is not None:
            cap = max(cap, header['frozen_table']['entries'])
        folder.mkdir(parents=True)
        return [str(QUALIFIED/'trainer-first'), 'train', '--stack-bb', '200', '--menu', 'v1',
            '--seed', str(seed), '--roots-per-seat', '1', '--average-rule', 'opponent-sampled',
            '--card-buckets', str(QUALIFIED/'tables'), '--max-entries', str(cap),
            '--nodes', str(target), '--recovery', '--max-seconds', str(max(1, self.deadline-time.time()-3600)),
            '--out', str(folder/'checkpoint.gz'), '--telemetry', str(folder/'telemetry.jsonl'),
            '--stop-file', str(folder/'stop.request'), '--freeze-at-cap',
            '--resume', recovery['path'], '--resume-sha256', recovery['sha256']]

    def finish(self, name):
        job, task = self.active[name], self.tasks[name]
        if job['proc'].poll() != 0:
            raise RuntimeError(f'{name} exit {job["proc"].poll()}; no retry')
        if task['kind'] == 'train':
            validate_native(self.folder(task), task['seed'], int(task['label']))
        elif task['kind'] == 'audit':
            checked = q.checked_audit(self.folder(task))
            if int(task['label']) == TARGETS[0] and task['seed'] in REFERENCE:
                if checked['files']['checkpoint']['sha256'] != REFERENCE[task['seed']]:
                    raise ValueError('Independent rebuilt10B hash differs')
        elif q.read(self.folder(task)/'scheduler-retired.json')['status'] != 'verified':
            raise ValueError('Recovery retirement receipt missing')
        result = dict(status='passed', exit_codes=[0], elapsed_seconds=time.monotonic()-job['started'],
                      stop_reason=None, hard_stop_reason=None, cpus=job['cpus'])
        put(self.root/(name+'.result.json'), result)
        job['log'].close();del self.active[name];self.done.add(name)
        print(json.dumps(dict(event='passed', job=name, **result)), flush=True)

    def decide_batch(self, row, totals):
        return  # The original measured owner-approved audit admission remains valid.

    def publish(self):
        for target in TARGETS:
            path = self.base/f'milestone-{target}.json'
            if not path.exists() and all(f'retire-{s}-{target}' in self.done for s in SEEDS):
                put(path, {s:q.read(self.base/OUTPUT/str(s)/str(target)/'native-record.json') for s in SEEDS})

    def complete(self):
        milestones = {str(n):q.read(self.base/f'milestone-{n}.json') for n in q.TARGETS}
        put(self.base/'training-complete.json', dict(status='all-three-20B-audited',
            milestones=milestones, terminal=milestones[str(TARGETS[-1])], admission=self.admission,
            finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(), scheduling_source=self.source,
            approved_recovery=pin(self.base/ADMISSION), deadline_amendment=q.read(self.base/LEDGER)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'run', 'retire'])
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--folder', type=Path)
    args = parser.parse_args()
    if os.uname().nodename != 'cygnus-x1' or args.base.resolve() != BASE:
        raise ValueError('Only the owned PR301 root on cygnus-x1')
    if args.phase == 'prepare':
        prepare(args.base)
    elif args.phase == 'retire':
        retire_worker(args.base, args.folder)
    else:
        with Path('/tmp/cfr-cygnus-exclusive.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                RecoveryQueue(args.base).run()
            except BaseException as exc:
                if not (args.base/'training-failed.json').exists():
                    put(args.base/'training-failed.json', dict(status='failed', error=repr(exc),
                        automatic_retry=False, source='owner-approved common5B recovery',
                        utc=dt.datetime.now(dt.timezone.utc).isoformat()))
                raise


if __name__ == '__main__':
    main()
