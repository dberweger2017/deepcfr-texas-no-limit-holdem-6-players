import io
import json
from pathlib import Path

import pytest

from scripts import guard_hu20_equity_scoring as guard
from scripts import restore_hu20_equity_scoring as restore
from scripts import run_fresh_hu20_equity_scoring as fresh


def setup_run(tmp_path, monkeypatch):
    jobs = [{'job': f'root-{i:02d}'} for i in range(40)]
    (tmp_path/'work').mkdir()
    monkeypatch.setattr(fresh, 'checked_inputs', lambda base: jobs)
    monkeypatch.setattr(fresh.scoring.bench, 'OUT', tmp_path/'work')
    return jobs


def clear_guard(tmp_path, phase, **changes):
    folder = tmp_path/'operations'/phase
    folder.mkdir(parents=True)
    value = {'status': 'complete', 'failure': None, 'cleanup_error': None, 'returncode': 0}
    value.update(changes)
    (folder/'receipt.json').write_text(json.dumps(value))
    (folder/'admission.json').write_text(json.dumps({'identity': ['Apple M4', str(16*1024**3), '10']}))


def posted_quote(tmp_path):
    quote = tmp_path/'full-scoring-quote.json'
    quote.write_text('{}')
    (tmp_path/'quote-posted.json').write_text(json.dumps({
        'quote_sha256': fresh.file_hash(quote),
        'comment_url': 'https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/232#issuecomment-123'}))


def test_machine_selection_refuses_cross_host():
    guard.machine_identity('m4', ['Apple M4', str(16*1024**3), '10'])
    guard.machine_identity('m1', ['Apple M1', str(16*1024**3), '8'])
    with pytest.raises(ValueError):
        guard.machine_identity('m4', ['Apple M1', str(16*1024**3), '8'])
    with pytest.raises(ValueError):
        guard.machine_identity('m1', ['Apple M4', str(16*1024**3), '10'])


def test_member_mismatch_and_overwrite_are_refused(tmp_path):
    with pytest.raises(ValueError, match='Exactness mismatch'):
        restore.copy_member(io.BytesIO(b'different'), tmp_path/'input',
                            {'bytes': 1, 'sha256': 'bad'})
    with pytest.raises(FileExistsError):
        restore.copy_member(io.BytesIO(b'x'), tmp_path/'input', {'bytes': 1, 'sha256': 'bad'})


def test_fresh_evaluation_scores_all40_once(tmp_path, monkeypatch):
    jobs = setup_run(tmp_path, monkeypatch)
    clear_guard(tmp_path, 'pilot')
    posted_quote(tmp_path)
    calls = []
    def score():
        calls.append('all40')
        (tmp_path/'work/eval').mkdir()
    monkeypatch.setattr(fresh.scoring.bench, 'evaluate', score)
    fresh.evaluate(tmp_path)
    assert json.loads((tmp_path/'evaluation-complete.json').read_text()) == {
        'jobs': [j['job'] for j in jobs], 'new_roots': 40}
    with pytest.raises(ValueError, match='Fresh empty'):
        fresh.evaluate(tmp_path)
    assert calls == ['all40']


def test_failed_evaluation_claim_cannot_retry(tmp_path, monkeypatch):
    setup_run(tmp_path, monkeypatch)
    clear_guard(tmp_path, 'pilot')
    posted_quote(tmp_path)
    def fail():
        raise RuntimeError('native exactness failure')
    monkeypatch.setattr(fresh.scoring.bench, 'evaluate', fail)
    with pytest.raises(RuntimeError):
        fresh.evaluate(tmp_path)
    with pytest.raises(FileExistsError):
        fresh.evaluate(tmp_path)
    assert not (tmp_path/'evaluation-complete.json').exists()


def test_no_retained_results_or_unposted_quote(tmp_path, monkeypatch):
    setup_run(tmp_path, monkeypatch)
    clear_guard(tmp_path, 'pilot')
    with pytest.raises(FileNotFoundError):
        fresh.evaluate(tmp_path)
    posted_quote(tmp_path)
    (tmp_path/'continuation.json').write_text('{}')
    with pytest.raises(ValueError, match='Fresh empty'):
        fresh.evaluate(tmp_path)
    assert not (tmp_path/'evaluation-claimed.json').exists()


def test_report_needs_full40_and_clear_guard_not_child_success(tmp_path, monkeypatch):
    jobs = setup_run(tmp_path, monkeypatch)
    clear_guard(tmp_path, 'evaluate', status='failed', failure='AC power')
    (tmp_path/'evaluation-complete.json').write_text(json.dumps({
        'jobs': [j['job'] for j in jobs], 'new_roots': 40}))
    for job in jobs:
        folder = tmp_path/'work/eval'/job['job']
        folder.mkdir(parents=True)
        (folder/'result.json').write_bytes(b'opaque')
    with pytest.raises(ValueError, match='Guard-clear'):
        fresh.report(tmp_path)
    receipt = tmp_path/'operations/evaluate/receipt.json'
    receipt.write_text(json.dumps({'status': 'complete', 'failure': None,
                                  'cleanup_error': None, 'returncode': 0}))
    (tmp_path/'work/eval'/jobs[-1]['job']/'result.json').unlink()
    with pytest.raises(ValueError, match='All 40'):
        fresh.report(tmp_path)
    (tmp_path/'work/eval'/jobs[-1]['job']/'result.json').write_bytes(b'opaque')
    calls = []
    monkeypatch.setattr(fresh.scoring.bench, 'report', lambda: calls.append('full40'))
    fresh.report(tmp_path)
    with pytest.raises(FileExistsError):
        fresh.report(tmp_path)
    assert calls == ['full40']
