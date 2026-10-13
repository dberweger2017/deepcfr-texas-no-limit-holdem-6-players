"""Frozen tables retain every stored update and recover exactly at complete iterations."""
import gzip
import json
from pathlib import Path
import subprocess

import pytest

from src.policies.files import file_hash

BINARY = Path('native/hu20-trainer/target/release/hu20-trainer')
pytestmark = pytest.mark.skipif(not BINARY.exists(), reason='build the native trainer')


def train(path, *args, nodes=300_000, bb=20, cap=2000):
    return subprocess.run([str(BINARY), 'train', '--seed', '5', '--stack-bb', str(bb),
        '--average-rule', 'opponent-sampled', '--recovery', '--nodes', str(nodes),
        '--max-entries', str(cap), '--out', str(path), *map(str, args)], capture_output=True, text=True)


def read(path):
    with gzip.open(path, 'rt') as stream:
        return json.loads(next(stream)), [json.loads(line) for line in stream]


def resume(path):
    return ['--resume', path, '--resume-sha256', file_hash(path)]


@pytest.mark.parametrize('bb', [20, 100, 200])
def test_flag_without_freeze_preserves_checkpoint_bytes(tmp_path, bb):
    a, b = tmp_path/'plain.gz', tmp_path/'enabled.gz'
    assert train(a, bb=bb, cap=1_000_000).returncode == 0
    assert train(b, '--freeze-at-cap', bb=bb, cap=1_000_000).returncode == 0
    assert a.read_bytes() == b.read_bytes()
    assert 'frozen_table' not in read(b)[0]


@pytest.mark.parametrize('roots', [1, 4])
def test_cap_stop_resume_and_interrupted_freeze_are_byte_identical(tmp_path, roots):
    full, stopped, half, continued, resumed = [tmp_path/f'{n}.gz' for n in ('full','stopped','half','continued','resumed')]
    options = ['--roots-per-seat', roots]
    assert train(full, *options, '--freeze-at-cap').returncode == 0
    result = train(stopped, *options)
    assert result.returncode == 3 and 'resource limit' in result.stderr
    h, rows = read(stopped)
    # Complete-iteration overshoot is real; both caller and saved header must admit all rows.
    assert len(rows) > 2000 and h['config']['max_entries'] == len(rows)
    bad = train(tmp_path/'bad-cap.gz', *options, *resume(stopped), '--freeze-at-cap')
    assert bad.returncode == 2 and 'resume checkpoint' in bad.stderr
    assert train(continued, *options, *resume(stopped), '--freeze-at-cap', cap=len(rows)).returncode == 0
    assert full.read_bytes() == continued.read_bytes()
    assert train(half, *options, '--freeze-at-cap', nodes=150_000).returncode == 0
    assert train(resumed, *options, *resume(half), '--freeze-at-cap', cap=len(read(half)[1])).returncode == 0
    assert full.read_bytes() == resumed.read_bytes()
    final_h, final_rows = read(full)
    assert [r[0] for r in rows] == [r[0] for r in final_rows]
    initial = {r[0]: r for r in rows}
    assert any(r[2] != initial[r[0]][2] for r in final_rows)
    assert any(r[3] != initial[r[0]][3] for r in final_rows)
    assert any(initial[r[0]][4] == r[4] == 0 and r[3] != initial[r[0]][3] for r in final_rows)
    state = final_h['frozen_table']
    assert state['since_iteration'] == h['iteration'] and state['entries'] == len(rows)
    assert state['discarded_traverser_visits'] > 0 and state['discarded_updates'] > 0
    coverage = final_h['native_state']
    assert coverage['coverage_start'][2] + sum(coverage['traverser_visits_by_street']) == sum(r[4] for r in final_rows) + state['discarded_traverser_visits']


def test_recovery_refuses_missing_flag_and_tampered_counts(tmp_path):
    frozen = tmp_path/'frozen.gz'
    assert train(frozen, '--freeze-at-cap').returncode == 0
    result = train(tmp_path/'no-flag.gz', *resume(frozen), cap=10000, nodes=400_000)
    assert result.returncode == 2 and 'requires --freeze-at-cap' in result.stderr
    assert not (tmp_path/'no-flag.gz').exists()
    for field in ('discarded_traverser_visits', 'entries', 'since_iteration', 'version'):
        h, rows = read(frozen)
        h['frozen_table'][field] += 1 if field != 'since_iteration' else h['iteration']
        tampered = tmp_path/f'{field}.gz'
        with gzip.open(tampered, 'wt') as stream:
            stream.write(json.dumps(h)+'\n')
            for row in rows: stream.write(json.dumps(row)+'\n')
        result = train(tmp_path/f'{field}-output.gz', *resume(tampered), '--freeze-at-cap', cap=10000, nodes=400_000)
        assert result.returncode == 2 and 'resume checkpoint' in result.stderr


def test_save_telemetry_has_interval_misses_discards_and_rss(tmp_path):
    checkpoint, telemetry = tmp_path/'{nodes}.gz', tmp_path/'receipts.jsonl'
    assert train(checkpoint, '--freeze-at-cap', '--milestones', '100000,200000', '--telemetry', telemetry).returncode == 0
    receipts = [json.loads(line) for line in telemetry.read_text().splitlines()]
    previous_discards = [0, 0]
    missing = [0]*4
    opponent_missing = [0]*4
    for record in receipts:
        data = record['frozen_table']
        state = data['state']
        assert record['process_rss_bytes_after_save'] > 0
        assert state['entries'] == record['diagnostics']['entries']
        fields = ('discarded_traverser_visits', 'discarded_updates')
        for i, field in enumerate(fields):
            assert data[field+'_since_previous_save'] == state[field] - previous_discards[i]
            previous_discards[i] = state[field]
        for actor in ('traverser', 'opponent'):
            for misses, decisions, share in zip(data[f'missing_{actor}_by_street'], data[f'{actor}_decisions_by_street'], data[f'missing_{actor}_share_by_street']):
                assert misses <= decisions
                assert share == (misses/decisions if decisions else None)
        missing = [a+b for a,b in zip(missing, data['missing_traverser_by_street'])]
        opponent_missing = [a+b for a,b in zip(opponent_missing, data['missing_opponent_by_street'])]
    assert all(n > 0 for n in missing + opponent_missing)
    assert receipts[-1]['frozen_table']['interval_start_iteration'] == receipts[-2]['iteration']
    # The resumed interval starts at admission; totals do not get reported a second time.
    first = tmp_path/'100000.gz'
    later = tmp_path/'resume-receipts.jsonl'
    assert train(tmp_path/'resumed.gz', *resume(first), '--freeze-at-cap', '--telemetry', later, cap=10000).returncode == 0
    data = json.loads(later.read_text())['frozen_table']
    initial = read(first)[0]
    assert data['interval_start_iteration'] == initial['iteration']
    assert sum(data['missing_traverser_by_street']) == data['discarded_traverser_visits_since_previous_save']
    assert data['discarded_traverser_visits_since_previous_save'] == data['state']['discarded_traverser_visits'] - initial['frozen_table']['discarded_traverser_visits']


def training_tables(directory, iterations=80):
    """Cover both seats' native root holdings on all streets, for a fixed iteration gate."""
    from src.blueprint.solver import _seed
    from tests.native_equity_tables import write_tables
    seeds = [_seed(5, t, seat, 0, 'deal') for t in range(1, iterations+1) for seat in range(2)]
    result = subprocess.run([str(BINARY), 'engine-deal', *map(str, seeds)], check=True, capture_output=True, text=True)
    situations = []
    for line in result.stdout.splitlines():
        cards = line.split()
        for hole in (tuple(cards[:2]), tuple(cards[2:4])):
            for size in (3, 4, 5): situations.append((hole, tuple(cards[4:4+size])))
    return write_tables(directory, situations)


@pytest.mark.parametrize('bb', [20, 200])
@pytest.mark.parametrize('equity', [False, True])
def test_frozen_exports_load_extract_and_audit_in_python(tmp_path, monkeypatch, bb, equity):
    from src.blueprint import equity_buckets as buckets
    from src.blueprint.artifact import load_training
    from src.blueprint.average import AveragePolicy
    from src.diagnostics.cfr_average import audit, extract
    cards, tables = [], None
    if equity:
        tables = training_tables(tmp_path)
        monkeypatch.setattr(buckets, 'EQUITY_K50_SHA256', buckets.EquityCards(tables, pinned=False).sha256)
        cards = ['--card-buckets', tables, '--card-tables-unpinned']
    checkpoint = tmp_path/'frozen.gz'
    assert train(checkpoint, *cards, '--freeze-at-cap', '--iterations', '80', nodes=10**9, bb=bb, cap=2000).returncode == 0
    h, rows = read(checkpoint)
    assert h['frozen_table']['discarded_traverser_visits'] > 0
    current, average, python_average = [tmp_path/f'{n}.gz' for n in ('current','average','python-average')]
    subprocess.run([str(BINARY), 'export', str(checkpoint), '--current', str(current), '--average', str(average)], check=True, capture_output=True)
    spec = {'seed':5, 'iteration':h['iteration'], 'checkpoint_sha256':file_hash(checkpoint), 'sha256':file_hash(current)}
    extract(checkpoint, spec, python_average, expected_schema=h['abstraction'])
    assert read(average) == read(python_average)
    audited = audit(checkpoint, current, average, spec, file_hash(average), expected_schema=h['abstraction'])
    assert audited['all_nodes_verified'] == len(rows)
    model = AveragePolicy(average, file_hash(average), expected_schema=h['abstraction'], card_buckets=tables)
    assert model.description['iteration'] == h['iteration']
    trainer = load_training(checkpoint, card_buckets=tables)
    assert trainer.evaluation_only and len(trainer.nodes) == len(rows)
    with pytest.raises(ValueError, match='evaluation only'):
        trainer.step()
