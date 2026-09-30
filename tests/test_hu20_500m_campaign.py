from dataclasses import replace
from hashlib import sha256
import json

import pytest

from scripts.hu20_500m_control import budget_action, estimated_cost
from scripts.hu20_500m_worker import acknowledged, next_boundary, rotate, scientific_config
from scripts.hu20_platform_pilot import write
from src.blueprint.solver import PilotConfig


def test_spending_includes_retired_attempts_and_current_storage_allowance():
    records = [dict(created_epoch=100, terminated=3700, upper_rate=.18),
               dict(created_epoch=100, upper_rate=.18), dict(status='unavailable')]
    assert estimated_cost(records, 7300) == pytest.approx(.54)
    assert budget_action(12.99, 15, 2) == 'continue'
    assert budget_action(13, 15, 2) == 'stop'


def test_fixed_total_milestones_do_not_accumulate_overshoot_or_reset_resume():
    assert next_boundary(100000280, [110000000, 120000000]) == 110000000
    assert next_boundary(110000131, [110000000, 120000000]) == 120000000
    assert next_boundary(500000001, [110000000, 500000000]) is None


def fixture_row(root, ident, permanent=False):
    p = root / (ident+'.gz')
    p.write_bytes(ident.encode())
    return dict(id=ident, permanent=permanent,
                files=[dict(name=p.name, sha256=sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)])


def test_rotation_requires_destination_hash_ack_and_preserves_two_verified(tmp_path):
    (tmp_path/'ack').mkdir()
    rows = [fixture_row(tmp_path, str(i)) for i in range(5)]
    for row in rows[:3]:
        write(tmp_path/'ack'/(row['id']+'.json'), dict(id=row['id'], files=row['files']))
    rotate(rows, tmp_path)
    assert not (tmp_path/'0.gz').exists()
    assert all((tmp_path/(str(i)+'.gz')).exists() for i in range(1, 5))


def test_mismatched_ack_never_authorizes_rotation(tmp_path):
    (tmp_path/'ack').mkdir()
    row = fixture_row(tmp_path, '1')
    write(tmp_path/'ack/1.json', dict(id='1', files=[dict(row['files'][0], sha256='wrong')]))
    assert not acknowledged(row, tmp_path)
    rotate([row], tmp_path)
    assert (tmp_path/'1.gz').exists()


def test_permanent_backups_survive_rotation(tmp_path):
    (tmp_path/'ack').mkdir()
    rows = [fixture_row(tmp_path, str(i), permanent=i==0) for i in range(4)]
    for row in rows:
        write(tmp_path/'ack'/(row['id']+'.json'), dict(id=row['id'], files=row['files']))
    rotate(rows, tmp_path)
    assert (tmp_path/'0.gz').exists()
    assert not (tmp_path/'1.gz').exists()
    assert (tmp_path/'2.gz').exists() and (tmp_path/'3.gz').exists()


def test_corrupt_verified_source_cannot_be_silently_deleted(tmp_path):
    (tmp_path/'ack').mkdir()
    rows = [fixture_row(tmp_path, str(i)) for i in range(3)]
    for row in rows:
        write(tmp_path/'ack'/(row['id']+'.json'), dict(id=row['id'], files=row['files']))
    (tmp_path/'0.gz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        rotate(rows, tmp_path)
    assert (tmp_path/'0.gz').exists()


def test_engineering_entry_expansion_changes_no_scientific_field():
    config = PilotConfig(seed=2026093001)
    assert scientific_config(config) == scientific_config(replace(config, max_entries=8000000))
    assert scientific_config(config) != scientific_config(replace(config, roots_per_seat=2))


def test_frozen_schedules_all_seeds_counts_and_fresh_confirmation():
    from pathlib import Path
    plan = json.loads((Path(__file__).parents[1]/'configs/blueprint/hu20-500m-campaign.json').read_text())
    assert [p['seed'] for p in plan['parents']] == [2026093001, 2026093002, 2026093003]
    assert plan['recovery_totals'] == list(range(110000000, 500000001, 10000000))
    assert set(plan['export_totals']) == set(range(120000000, 500000001, 20000000)) | set(range(150000000, 500000001, 50000000))
    roots = [p['root'] for family in ('light_panels', 'broad_panels', 'heldout_panels') for p in plan[family]]
    assert len(roots) == len(set(roots))
    assert plan['chance_samples'] == 4 and plan['lbr_seconds'] == 5
    assert plan['expected_hands'] == 907776
    for baseline, confirmation in zip(plan['broad_panels'], plan['heldout_panels'], strict=True):
        assert baseline['blocks'] == confirmation['blocks']
        assert baseline['root'] != confirmation['root']


def test_export_exception_is_only_the_documented_os_byte(tmp_path):
    import gzip
    from scripts.hu20_500m_worker import export_fingerprint
    payload = b'{"fixed":"policy"}'*1000
    a = tmp_path/'mac.gz'; b = tmp_path/'linux.gz'
    data = gzip.compress(payload, mtime=0)
    a.write_bytes(data[:9]+bytes([19])+data[10:])
    b.write_bytes(data[:9]+bytes([3])+data[10:])
    assert export_fingerprint(a)['sha256'] != export_fingerprint(b)['sha256']
    assert export_fingerprint(a)['os_normalized_sha256'] == export_fingerprint(b)['os_normalized_sha256']
    alternate = gzip.compress(payload, compresslevel=1, mtime=0)
    b.write_bytes(alternate[:9]+bytes([3])+alternate[10:])
    assert export_fingerprint(a)['uncompressed_sha256'] == export_fingerprint(b)['uncompressed_sha256']
    assert export_fingerprint(a)['os_normalized_sha256'] != export_fingerprint(b)['os_normalized_sha256']


def test_actual_continuation_driver_preserves_state_work_chain_and_recovery(tmp_path):
    import time
    from scripts.hu20_500m_worker import run
    from src.blueprint.artifact import save_training
    from src.blueprint.solver import BlueprintTrainer, HU20_UNCAPPED_GAME
    from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
    from src.game.hand import Table
    trainer = BlueprintTrainer(Table(('player-0','player-1'), (2000,2000)),
        PilotConfig(seed=2026093001, raise_cap=None, roots_per_seat=1, max_nodes=250000,
                    max_entries=8000000, max_seconds=300, postflop_replicates=1,
                    abstraction=HU20_UNCAPPED_SCHEMA, game=HU20_UNCAPPED_GAME))
    cp = tmp_path/'parent.gz'
    parent = dict(seed=2026093001, completed_nodes=0, iteration=0, entries=0,
                  checkpoint_path=str(cp), checkpoint_sha256=save_training(trainer,cp))
    plan = dict(target_total_nodes=20000, recovery_totals=[10000,20000],
                export_totals=[10000,20000], permanent_totals=[20000],
                engineering=dict(initial_max_entries=8000000))
    control = tmp_path/'control.json'
    write(control, dict(lease_until=time.time()+300, stop=None))
    direct = run(plan, parent, tmp_path/'direct', control)
    records = [json.loads(line) for line in (tmp_path/'direct/saved.jsonl').read_text().splitlines()]
    middle = next(row for row in records if row['requested_total_nodes']==10000)
    resume = tmp_path/'direct'/(middle['id']+'.record.json')
    resumed = run(plan, parent, tmp_path/'resumed', control, resume)
    assert direct['status'] == resumed['status'] == 'complete'
    assert direct['work_chain_sha256'] == resumed['work_chain_sha256']
    assert direct['completed_nodes'] == resumed['completed_nodes']
    assert direct['cumulative_work'] == resumed['cumulative_work']
    final_a = records[-1]
    final_b = json.loads((tmp_path/'resumed/saved.jsonl').read_text().splitlines()[-1])
    assert final_a['checkpoint_sha256'] == final_b['checkpoint_sha256']
    assert final_a['files'][1]['sha256'] == final_b['files'][1]['sha256']
    write(control, dict(lease_until=time.time()-1, stop='fixture stop'))
    partial = run(plan, parent, tmp_path/'stopped', control)
    assert partial['status'] == 'interrupted' and partial['completed_nodes']==0
    assert partial['discarded_nodes']==0
    assert (tmp_path/'stopped/checkpoint-partial-0.json.gz').exists()


def test_export_backup_failure_cannot_acknowledge_checkpoint_only(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import scripts.hu20_500m_control as control
    ack = []
    calls = []
    monkeypatch.setattr(control.shutil, 'disk_usage', lambda p: SimpleNamespace(free=100*2**30))
    monkeypatch.setattr(control, 'connections', lambda row, root: ([], [], 'fixture'))
    monkeypatch.setattr(control, 'remote_write', lambda *args: ack.append(args))
    def copy(row, item, destination, a):
        calls.append(item['name'])
        if len(calls)==2:
            raise ValueError('policy transport hash mismatch')
    monkeypatch.setattr(control, 'retrieve_file', copy)
    saved = dict(id='120M', files=[dict(name='cp.gz', bytes=1, sha256='a'),
                                 dict(name='policy.gz', bytes=1, sha256='b')])
    with pytest.raises(ValueError, match='hash mismatch'):
        control.backup(dict(seed=1, attempt='1'), saved, SimpleNamespace(root=tmp_path))
    assert calls == ['cp.gz', 'policy.gz'] and ack == []
