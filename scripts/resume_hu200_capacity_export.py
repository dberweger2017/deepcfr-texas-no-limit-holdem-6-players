"""One owner-approved export recovery; reuse the qualified exporter and guards."""
import argparse
import datetime as dt
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts import continue_hu200_capacity as prior
from scripts import schedule_hu200_capacity as q
from scripts.run_hu200_capacity_training import BINARY_SHA, QUALIFIED, SEEDS, pin, put
from src.policies.files import file_hash

PHASE = 'export-recovery-20261017'
APPROVAL = 'export-recovery-approval-20261017.json'
FAILURE_SHA = 'fe8b88f140d523772d3b43f31b44a979e202716159816294bcfb744896cfe50b'
PRIOR_LEDGER_SHA = 'c9973591bb59a28919b488a1ab3179eff91708cd1a49e97283e04c09821fe120'
PRIOR_DEADLINE = 1792385089.2587068
STOPPED = dt.datetime.fromisoformat('2026-10-17T14:25:09.233853+00:00').timestamp()
STORAGE = 'storage-export-recovery-20261017'
OUTPUT = 'results-export-recovery-20261017'
COLD = Path('/mnt/data2tb/research/PR-301-hu200-capacity-cygnus/after-export-recovery-20261017')
EXTRA_PHYSICAL_MARGIN = 20_000_000_000


def phase(base):
    return base / PHASE


def output(base):
    return base / OUTPUT / str(SEEDS[0]) / '20000000000'


def approval(base):
    value = q.read(prior.PINS / APPROVAL)
    if value['status'] != 'owner-approved' or value['future_automatic_retries']:
        raise ValueError('A new bounded owner approval is required')
    if file_hash(prior.phase(base) / 'training-failed.json') != FAILURE_SHA:
        raise ValueError('Current failure differs from the approved recovery')
    if (file_hash(prior.phase(base) / 'deadline.json') != PRIOR_LEDGER_SHA
            or prior.checked_ledger(base)['deadline_epoch'] != PRIOR_DEADLINE):
        raise ValueError('Prior immutable deadline changed')
    return value


def adjusted_deadline(stopped, resumed):
    if stopped != STOPPED or not STOPPED <= resumed < PRIOR_DEADLINE:
        raise ValueError('Only the recorded closed interval may be credited')
    return PRIOR_DEADLINE + (resumed - STOPPED)


def checked_ledger(base):
    approval(base)
    value = q.read(phase(base) / 'deadline.json')
    if (value['prior_ledger_sha256'] != PRIOR_LEDGER_SHA
            or value['approval_sha256'] != file_hash(prior.PINS / APPROVAL)
            or value['stopped_epoch'] != STOPPED
            or value['extension_seconds'] != value['resumed_epoch'] - STOPPED
            or value['deadline_epoch'] != adjusted_deadline(STOPPED, value['resumed_epoch'])):
        raise ValueError('Sealed downtime ledger changed')
    return value


def validate(base, rehash=False):
    approval(base)
    storage = q.read(base / STORAGE / 'receipt.json')
    if storage['status'] != 'verified' or storage['relocated_bytes'] < 65_000_000_000:
        raise ValueError('Broad inactive-file relocation is not complete')
    prior.mount_checked()
    for row in storage['files']:
        expected = row['source']
        destination = Path(row['destination'])
        if (Path(expected['path']).exists() or destination.is_symlink()
                or (destination.stat().st_size, destination.stat().st_mtime_ns)
                != (expected['bytes'], expected['mtime_ns'])):
            raise ValueError('Relocated storage metadata differs')
    if file_hash(QUALIFIED / 'trainer-first') != BINARY_SHA:
        raise ValueError('Qualified executable changed')
    for street, sha in q.read(base / 'admission.json')['card_table_hashes'].items():
        files = list((QUALIFIED / 'tables').glob(street + '*'))
        if len(files) != 1 or file_hash(files[0]) != sha:
            raise ValueError('K50 card bytes changed')
    for seed in SEEDS[1:]:
        folder = prior.model_folder(base, seed)
        if q.checked_audit(folder) != q.read(prior.PINS / f'progress-20261017-1408/results-recovery-20261016/{seed}/20000000000/audit.json'):
            raise ValueError('Completed final audit changed')
    folder = prior.model_folder(base, SEEDS[0])
    saved = q.read(folder / 'saved-bytes.json')
    expected = q.read(prior.PINS / 'progress-20261017-1408/results-continuation-20261017/2026101302/20000000000/saved-bytes.json')
    if saved != expected:
        raise ValueError('Retained 20B save verification changed')
    checkpoint = prior.fresh_stat(saved['file'])
    if rehash and file_hash(checkpoint) != saved['file']['sha256']:
        raise ValueError('Retained 20B checkpoint hash changed')
    return saved


def require_resources(base, row):
    # Protect the entire 23GB output even though the checkpoint already exists.
    if (row['package_c'] >= 68 or row['vrm_c'] >= 83
            or row['available_bytes'] < 8 * q.GIB
            or row['swap_used_bytes'] - 1048576 > 256 * 1024**2
            or row['free_disk_bytes'] < q.FLOOR + 23_000_000_000 + EXTRA_PHYSICAL_MARGIN
            or not q.fits(q.owned_bytes(base), row['free_disk_bytes'], 0, 23_000_000_000)):
        raise ValueError('Fresh export admission including physical growth buffer failed')


def prepare(base):
    if phase(base).exists() or (base / OUTPUT).exists() or (base / 'guards' / PHASE).exists():
        raise ValueError('Export recovery already attempted; no automatic retry')
    saved = validate(base, rehash=True)
    row = prior.sample(base)
    require_resources(base, row)
    put(phase(base) / 'admission.json', dict(status='admitted', source=prior.old.source_pin(),
        resources=row, checkpoint=saved['file'], storage_receipt=pin(base / STORAGE / 'receipt.json'),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner_approval_sha256=file_hash(prior.PINS / APPROVAL), utc=prior.now()))


def finish_storage(base):
    audit = q.checked_audit(output(base))
    folder = phase(base) / 'offload'
    rows = []
    for item in [audit['files']['checkpoint'], audit['files']['current']]:
        source = Path(item['path'])
        destination = COLD / source.relative_to(base)
        for parent in [destination, *destination.parents]:
            if parent == prior.HDD.parent:
                break
            if parent.is_symlink():
                raise ValueError('Symlink storage destination')
        rows.append(prior.relocate_one(source, destination, item,
            folder / 'retire.stop', folder / 'journal.jsonl'))
    put(folder / 'receipt.json', dict(status='verified', files=rows, utc=prior.now()))


class ExportRecovery(q.Queue):
    def __init__(self, base):
        self.base = base
        self.source = prior.old.source_pin()
        admitted = q.read(phase(base) / 'admission.json')
        if (admitted['source'] != self.source or admitted['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                or admitted['owner_approval_sha256'] != file_hash(prior.PINS / APPROVAL)
                or admitted['storage_receipt'] != pin(base / STORAGE / 'receipt.json')
                or (phase(base) / 'deadline.json').exists() or output(base).exists()):
            raise ValueError('Fresh source/boot/attempt required')
        validate(base)
        require_resources(base, prior.sample(base))
        resumed = time.time()
        self.deadline = adjusted_deadline(STOPPED, resumed)
        put(phase(base) / 'deadline.json', dict(status='owner-approved-export-stop-credit',
            stopped_epoch=STOPPED, resumed_epoch=resumed, extension_seconds=resumed - STOPPED,
            deadline_epoch=self.deadline, prior_ledger_sha256=PRIOR_LEDGER_SHA,
            approval_sha256=file_hash(prior.PINS / APPROVAL), utc=prior.now()))
        self.root = base / 'guards' / PHASE
        self.root.mkdir(exist_ok=False)
        output(base).mkdir(parents=True, exist_ok=False)
        # Keep the old checkpoint path; only the new export/audit output is fresh.
        (output(base) / 'native-record.json').write_bytes((prior.model_folder(base, SEEDS[0]) / 'native-record.json').read_bytes())
        self.clock, self.admission = q.read(base / 'training-clock.json'), q.read(base / 'admission.json')
        self.active, self.done, self.tasks = {}, set(), {}
        self.audit_limit, self.batch_decided = 1, True
        self.next_counter, self.last_counter = 0, None
        self.cpu_deltas, self.disk_busy_ms, self.last_disk = {}, 0, None
        self.build()

    def build(self):
        self.task('export-audit-1302', 'audit', [], seed=SEEDS[0], label='20000000000')
        self.task('offload-1302', 'retire', ['export-audit-1302'], seed=SEEDS[0], label='20000000000')

    def folder(self, task):
        return output(self.base) if task['kind'] == 'audit' else phase(self.base) / 'offload'

    def command(self, task):
        if task['kind'] == 'audit':
            return [sys.executable, '-m', 'scripts.run_hu200_capacity_training', 'audit', '--base', str(output(self.base))]
        self.folder(task).mkdir(exist_ok=False)
        return [sys.executable, '-m', 'scripts.resume_hu200_capacity_export', 'offload', '--base', str(self.base)]

    def output_reservation(self, task):
        return 23_000_000_000 if task['kind'] == 'audit' else prior.METADATA_RESERVE

    def decide_batch(self, row, totals):
        pass

    def publish(self):
        pass

    def finish(self, name):
        if self.active[name]['proc'].poll() != 0:
            raise RuntimeError('Export/storage failed; no automatic retry')
        if self.tasks[name]['kind'] == 'retire' and q.read(self.folder(self.tasks[name]) / 'receipt.json')['status'] != 'verified':
            raise ValueError('Missing final storage verification')
        super().finish(name)

    def complete(self):
        models = {str(seed):q.read((output(self.base) if seed == SEEDS[0] else prior.model_folder(self.base, seed)) / 'audit.json') for seed in SEEDS}
        terminal = {seed:row['native'] for seed, row in models.items()}
        result = dict(status='all-three-20B-audited', terminal=terminal, audits=models,
            source=self.source, deadline=checked_ledger(self.base), finished_utc=prior.now(),
            storage_receipts=[pin(self.base / STORAGE / 'receipt.json'), pin(phase(self.base) / 'offload/receipt.json')],
            superseded_failures=[pin(self.base / 'training-failed.json'), pin(prior.phase(self.base) / 'training-failed.json')])
        put(self.base / 'milestone-20000000000.json', terminal)
        put(phase(self.base) / 'training-complete.json', result)
        put(self.base / 'training-complete.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'run', 'offload', 'observe'])
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    base = args.base
    if os.uname().nodename != 'cygnus-x1' or base.resolve() != q.BASE:
        raise ValueError('Only the owned bb128 campaign root is authorized')
    if args.phase == 'prepare':
        prepare(base)
    elif args.phase == 'offload':
        finish_storage(base)
    elif args.phase == 'observe':
        observe(base)
    else:
        with Path('/tmp/cfr-cygnus-exclusive.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                ExportRecovery(base).run()
            except BaseException as exc:
                path = phase(base) / 'training-failed.json'
                if not path.exists():
                    put(path, dict(status='failed', attempt=PHASE, error=repr(exc), automatic_retry=False, utc=prior.now()))
                raise


def observe(base):
    ledger = checked_ledger(base)
    sent = phase(base) / 'event-delivery.json'
    url = (base / 'campaign-events-url.txt').read_text().strip()
    target = prior.urllib.parse.urlsplit(url)
    if target.scheme != 'https' or target.hostname != 'relay.t3.codes' or not target.path.startswith('/v1/hooks/'):
        raise ValueError('Unexpected existing event endpoint')
    while time.time() < ledger['deadline_epoch'] + 3600:
        event = ('training_failed' if (phase(base) / 'training-failed.json').exists()
            else 'training_complete' if (phase(base) / 'training-complete.json').exists() else None)
        if event:
            if sent.exists():
                return
            payload = dict(event=event, experiment='PR301', attempt=PHASE, observed_utc=prior.now())
            request = prior.urllib.request.Request(url, data=q.json.dumps(payload).encode(),
                headers={'Content-Type':'application/json', 'User-Agent':'T3Code-HU200Capacity-Observer/1.0'}, method='POST')
            try:
                with prior.urllib.request.urlopen(request, timeout=10) as response:
                    if 200 <= response.status < 300:
                        put(sent, dict(payload=payload, http_status=response.status))
                        return
            except Exception as exc:
                print(q.json.dumps(dict(event='delivery_pending', error_type=type(exc).__name__)), flush=True)
        time.sleep(30)


if __name__ == '__main__':
    main()
