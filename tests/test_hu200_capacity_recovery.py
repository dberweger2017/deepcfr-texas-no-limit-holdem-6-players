"""Bounded rollback adoption, exact identities, retirement and approved clock."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts import recover_hu200_capacity as r


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n')


def clock_and_ledger():
    clock = dict(deadline_epoch=1792376957.996109, max_seconds=259200)
    ledger = dict(status='owner-approved-single-extension', original_clock_sha256=r.CLOCK_SHA,
        original_deadline_epoch=clock['deadline_epoch'], extension_seconds=3600,
        amended_deadline_epoch=clock['deadline_epoch']+3600)
    return clock, ledger


def test_extension_is_fixed_once_not_a_new_relative_clock():
    clock, ledger = clock_and_ledger()
    assert r.deadline_from(clock, ledger) == 1792380557.996109
    assert clock['deadline_epoch'] == 1792376957.996109
    # Re-reading returns the same absolute deadline without another credit.
    assert r.deadline_from(clock, ledger) == r.deadline_from(clock, ledger)


@pytest.mark.parametrize('field,value', [
    ('extension_seconds', 7200), ('amended_deadline_epoch', 1792384157.996109),
    ('original_clock_sha256', 'different'), ('original_deadline_epoch', 0),
    ('status', 'pending'),
])
def test_unapproved_clock_change_is_refused(field, value):
    clock, ledger = clock_and_ledger()
    ledger[field] = value
    with pytest.raises(ValueError, match='Only one'):
        r.deadline_from(clock, ledger)


def test_recovery_dag_adopts_5b_and_never_repeats_identity_gates(tmp_path):
    queue = r.RecoveryQueue.__new__(r.RecoveryQueue)
    queue.base, queue.tasks, queue.done = tmp_path, {}, set()
    queue.build()
    assert len(queue.tasks) == 18
    assert all(task['label'] in {'10000000000', '20000000000'}
               and 'prefix' not in name for name, task in queue.tasks.items())
    for seed in r.SEEDS:
        first = queue.tasks[f'train-{seed}-10000000000']
        second = queue.tasks[f'train-{seed}-20000000000']
        assert queue.ready(first)
        queue.done.add(f'audit-{seed}-10000000000')
        assert not queue.ready(second)
        queue.done.add(f'retire-{seed}-10000000000')
        assert queue.ready(second)
        assert r.parent_folder(tmp_path, seed, r.TARGETS[0]) == tmp_path/'results'/str(seed)/'5000000000'
        assert r.parent_folder(tmp_path, seed, r.TARGETS[1]) == tmp_path/r.OUTPUT/str(seed)/'10000000000'


@pytest.mark.parametrize('existing', [r.OUTPUT, 'guards/'+r.GUARD, 'training-failed.json', 'training-complete.json'])
def test_second_attempt_or_terminal_campaign_is_never_restarted(tmp_path, existing):
    path = tmp_path/existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('existing')
    with pytest.raises(ValueError, match='no retry'):
        r.no_attempt(tmp_path)


def test_native_reproduction_requires_both_old10b_sha_and_1302_header(tmp_path):
    for seed in r.SEEDS:
        folder = tmp_path/str(seed)
        folder.mkdir()
        record = dict(binary_sha256=r.BINARY_SHA, stop_requested=False, status='saved',
            path=str(folder/'checkpoint.gz'), requested_nodes=r.TARGETS[0],
            completed_nodes=10000000240 if seed == r.SEEDS[0] else 10000000001,
            iteration=7864201, checkpoint_sha256=r.REFERENCE.get(seed, 'header-only-reference'))
        write(folder/'telemetry.jsonl', record)
        assert r.validate_native(folder, seed, r.TARGETS[0]) == record
        (folder/'native-record.json').unlink()
        if seed in r.REFERENCE:
            record['checkpoint_sha256'] = 'mismatch'
        else:
            record['iteration'] += 1
        write(folder/'telemetry.jsonl', record)
        with pytest.raises(ValueError, match='differs'):
            r.validate_native(folder, seed, r.TARGETS[0])
        assert not (folder/'native-record.json').exists()


def audit_fixture(folder):
    folder.mkdir(parents=True)
    files = {}
    for kind in ['checkpoint', 'current', 'average']:
        path = folder/('checkpoint.gz' if kind == 'checkpoint' else 'checkpoint.'+kind+'.gz')
        path.write_bytes(kind.encode())
        files[kind] = r.pin(path)
    receipt = dict(status='verified', audit=dict(status='verified', all_nodes_verified=1),
        native=dict(diagnostics=dict(entries=1)), files=files)
    write(folder/'audit.json', receipt)
    return receipt


def test_retirement_preserves_5b_average_and_interrupted_attempt(tmp_path):
    seed = r.SEEDS[0]
    parent = r.parent_folder(tmp_path, seed, r.TARGETS[0])
    old = audit_fixture(parent)
    interrupted = tmp_path/'results'/str(seed)/'10000000000'/'checkpoint.tmp'
    interrupted.parent.mkdir(parents=True)
    interrupted.write_bytes(b'preserve interruption')
    folder = tmp_path/r.OUTPUT/str(seed)/str(r.TARGETS[0])
    audit_fixture(folder)
    r.retire_worker(tmp_path, folder)
    assert not Path(old['files']['checkpoint']['path']).exists()
    assert Path(old['files']['average']['path']).read_bytes() == b'average'
    assert interrupted.read_bytes() == b'preserve interruption'
    assert (folder/'checkpoint.gz').exists()
    assert not (folder/'checkpoint.current.gz').exists()
    assert not (folder/'checkpoint.average.gz').exists()
    receipt = r.q.read(folder/'scheduler-retired.json')
    assert receipt['status'] == 'verified' and len(receipt['removed']) == 3


def test_changed_recovery_prevents_any_retirement(tmp_path):
    seed = r.SEEDS[0]
    parent = r.parent_folder(tmp_path, seed, r.TARGETS[0])
    audit_fixture(parent)
    folder = tmp_path/r.OUTPUT/str(seed)/str(r.TARGETS[0])
    audit_fixture(folder)
    (folder/'checkpoint.gz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        r.retire_worker(tmp_path, folder)
    assert (parent/'checkpoint.gz').exists()
    assert (folder/'checkpoint.current.gz').exists()


def test_training_command_binds_original5b_parent_and_new_paths(monkeypatch, tmp_path):
    seed = r.SEEDS[0]
    parent = r.parent_folder(tmp_path, seed, r.TARGETS[0])
    audit_fixture(parent)
    write(parent/'saved-bytes.json', dict(header={}))
    queue = r.RecoveryQueue.__new__(r.RecoveryQueue)
    queue.base, queue.deadline = tmp_path, 9999999999
    queue.admission = dict(selected_cap=301153954)
    task = dict(kind='train', seed=seed, label=str(r.TARGETS[0]))
    cmd = queue.command(task)
    assert cmd[cmd.index('--resume')+1] == str(parent/'checkpoint.gz')
    assert cmd[cmd.index('--resume-sha256')+1] == r.file_hash(parent/'checkpoint.gz')
    assert cmd[cmd.index('--max-entries')+1] == '301153954'
    assert cmd[cmd.index('--nodes')+1] == '10000000000'
    assert cmd[cmd.index('--roots-per-seat')+1] == '1'
    assert str(tmp_path/r.OUTPUT) in cmd[cmd.index('--out')+1]
    with pytest.raises(ValueError, match='Refuse repeated'):
        queue.command(task)


def test_adoption_checks_all_three_original_parent_and_readback_pins(monkeypatch, tmp_path):
    base, pins = tmp_path/'owned', tmp_path/'pins'
    monkeypatch.setattr(r, 'PINS', pins)
    entries = []
    for seed in r.SEEDS:
        folder = r.parent_folder(base, seed, r.TARGETS[0])
        audit = audit_fixture(folder)
        write(pins/f'save-5b/{seed}/audit.json', audit)
        for kind in ['checkpoint', 'average']:
            item = audit['files'][kind]
            entries.append(dict(**item, verified=True))
        twin = base/'inputs'/f'seed-{seed}'/'final.average.gz'
        twin.parent.mkdir(parents=True)
        twin.write_bytes(b'twin')
        entries.append(dict(**r.pin(twin), verified=True))
    # The real readback also has the two interrupted10B files.
    entries += [dict(path='reference1', verified=True), dict(path='reference2', verified=True)]
    record = dict(files=entries)
    write(base/'planning/power-loss-byte-readback-20261016.json', record)
    write(pins/'power-off-byte-readback.json', record)
    for name, value in [('resume-gates.json', {}), ('twin-gates.json', {'status':'all-three-exact'}),
                        ('prefix-memory-admission.json', {'status':'passed'}), ('twins-restored.json', {})]:
        write(base/name, value);write(pins/name, value)
    adopted, gates = r.adopt(base, rehash=True)
    assert len(adopted) == 3 and len(gates) == 4
    changed = r.parent_folder(base, r.SEEDS[2], r.TARGETS[0])/'checkpoint.gz'
    changed.write_bytes(b'broken')
    with pytest.raises(ValueError, match='changed'):
        r.adopt(base)


def test_initial_recovery_admission_rechecks_full_three_table_ram_and_disk():
    admission = dict(memory_required_including_2GiB_helper_and_6GiB_available_floor=130192069748,
                     swap_baseline_bytes=1048576)
    row = dict(package_c=50, vrm_c=60, available_bytes=132_000_000_000,
               swap_used_bytes=0, free_disk_bytes=220_000_000_000)
    r.require_admission(row, admission, 95_000_000_000)
    for field, value in [('available_bytes', 129_000_000_000), ('package_c', 68),
                         ('vrm_c', 83), ('swap_used_bytes', 300*1024**2),
                         ('free_disk_bytes', 100_000_000_000)]:
        changed = dict(row, **{field:value})
        with pytest.raises(ValueError):
            r.require_admission(changed, admission, 95_000_000_000)
