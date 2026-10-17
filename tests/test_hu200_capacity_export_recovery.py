"""Export-only recovery preserves scope, output headroom and closed clock credit."""
import pytest

from scripts import resume_hu200_capacity_export as r
from scripts import schedule_hu200_capacity as q
from scripts.run_hu200_capacity_training import put
from src.policies.files import file_hash


def test_no_training_or_completed_seed_audit_is_repeated(tmp_path):
    queue = r.ExportRecovery.__new__(r.ExportRecovery)
    queue.base, queue.tasks, queue.done = tmp_path, {}, set()
    queue.build()
    assert set(queue.tasks) == {'export-audit-1302', 'offload-1302'}
    assert [t['kind'] for t in queue.tasks.values()] == ['audit', 'retire']
    audit = queue.tasks['export-audit-1302']
    assert queue.command(audit) == [r.sys.executable, '-m',
        'scripts.run_hu200_capacity_training', 'audit', '--base', str(r.output(tmp_path))]
    assert queue.ready(audit) and not queue.ready(queue.tasks['offload-1302'])
    assert queue.output_reservation(audit) == 23_000_000_000
    assert queue.output_reservation(queue.tasks['offload-1302']) == 64 * 1024**2
    queue.done.add(audit['name'])
    assert queue.ready(queue.tasks['offload-1302'])


def test_only_exact_stop_interval_is_credited_once():
    resumed = r.STOPPED + 4567.25
    expected = r.PRIOR_DEADLINE + 4567.25
    assert r.adjusted_deadline(r.STOPPED, resumed) == expected
    assert r.adjusted_deadline(r.STOPPED, resumed) == expected
    for stop, end in [(r.STOPPED-1, resumed), (r.STOPPED, r.STOPPED-1), (r.STOPPED, r.PRIOR_DEADLINE)]:
        with pytest.raises(ValueError):
            r.adjusted_deadline(stop, end)


def test_sealed_deadline_rejects_extra_credit(monkeypatch, tmp_path):
    monkeypatch.setattr(r, 'approval', lambda base:None)
    monkeypatch.setattr(r.prior, 'PINS', tmp_path / 'pins')
    put(r.prior.PINS / r.APPROVAL, {'status':'owner-approved'})
    resumed = r.STOPPED + 1234.5
    value = dict(prior_ledger_sha256=r.PRIOR_LEDGER_SHA,
        approval_sha256=file_hash(r.prior.PINS / r.APPROVAL), stopped_epoch=r.STOPPED,
        resumed_epoch=resumed, extension_seconds=1234.5,
        deadline_epoch=r.adjusted_deadline(r.STOPPED, resumed))
    path = r.phase(tmp_path) / 'deadline.json'
    put(path, value)
    assert r.checked_ledger(tmp_path) == value
    value['extension_seconds'] += 1
    path.write_text(q.json.dumps(value))
    with pytest.raises(ValueError):
        r.checked_ledger(tmp_path)


def test_physical_buffer_is_additional_to_full_output_and_floor(monkeypatch, tmp_path):
    monkeypatch.setattr(q, 'owned_bytes', lambda base:20_000_000_000)
    row = dict(package_c=50, vrm_c=60, available_bytes=100*q.GIB,
        swap_used_bytes=1048576, free_disk_bytes=93_000_000_000)
    r.require_resources(tmp_path, row)
    row['free_disk_bytes'] -= 1
    with pytest.raises(ValueError, match='physical growth buffer'):
        r.require_resources(tmp_path, row)
    # The original guard alone permits this; recovery requires an extra20GB.
    assert q.fits(20_000_000_000, row['free_disk_bytes'], 0, 23_000_000_000)


def test_final_storage_moves_checkpoint_current_only_after_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(r, 'COLD', tmp_path / 'hdd')
    monkeypatch.setattr(r.prior, 'HDD', tmp_path / 'hdd')
    files = {kind:{'path':str(tmp_path / (kind+'.gz'))} for kind in ['checkpoint','current','average']}
    monkeypatch.setattr(q, 'checked_audit', lambda folder:dict(status='verified', files=files))
    seen = []
    def move(source, destination, expected, stop, journal):
        seen.append(expected)
        return dict(source=expected, destination=str(destination))
    monkeypatch.setattr(r.prior, 'relocate_one', move)
    r.finish_storage(tmp_path)
    assert seen == [files['checkpoint'], files['current']]
    assert q.read(r.phase(tmp_path) / 'offload/receipt.json')['status'] == 'verified'
