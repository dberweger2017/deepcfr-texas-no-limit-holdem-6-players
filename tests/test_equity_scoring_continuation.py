import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts import continue_restored_hu20_equity_bench as continuation
from scripts import guard_hu20_equity_scoring as guard


def fixture_archive(tmp_path, monkeypatch):
    jobs = [{'job': f'job-{i:02d}'} for i in range(40)]
    payloads = {continuation.PRIOR_PREFIX+'/work/eval/'+j['job']+'/result.json':
                b'opaque result bytes, deliberately not JSON' for j in jobs[:5]}
    # Native runtime receipts share the filename but are not complete root results.
    payloads.update({continuation.PRIOR_PREFIX+'/work/eval/'+j['job']+'/solver/result.json':
                     b'opaque nested native runtime receipt' for j in jobs[:5]})
    summary = {'completed_job_ids': [j['job'] for j in jobs[:5]],
               'completed_roots': 5, 'required_roots': 40,
               'executed_scoring_source': continuation.PRIOR_SOURCE,
               'scores_inspected': False, 'failure_receipt': {'status': 'failed'}}
    payloads[continuation.SUMMARY_MEMBER] = json.dumps(summary).encode()
    manifest = {'members': [{'path': k, 'bytes': len(v),
                             'sha256': hashlib.sha256(v).hexdigest()}
                            for k, v in payloads.items()]}
    manifest_bytes = json.dumps(manifest).encode()
    archive_path = tmp_path/'prior.zip'
    with zipfile.ZipFile(archive_path, 'x') as archive:
        for k, v in payloads.items():
            archive.writestr(k, v)
        archive.writestr('ARCHIVE-MANIFEST.json', manifest_bytes)
    monkeypatch.setattr(continuation, 'ARCHIVE_SHA256',
                        continuation.file_hash(archive_path))
    monkeypatch.setattr(continuation, 'MANIFEST_SHA256',
                        hashlib.sha256(manifest_bytes).hexdigest())
    work = tmp_path/'work'
    work.mkdir()
    monkeypatch.setattr(continuation.scoring, 'configure',
                        lambda base: (None, None, jobs))
    monkeypatch.setattr(continuation.scoring.bench, 'OUT', work)
    return archive_path, jobs, work


def test_opaque_seed_then_only_remaining_35_and_no_reexecution(tmp_path, monkeypatch):
    archive, jobs, work = fixture_archive(tmp_path, monkeypatch)
    continuation.seed(tmp_path, archive)
    calls = []
    original = lambda: ('plan', 'records', 'folds', jobs)
    monkeypatch.setattr(continuation.scoring.bench, 'inputs', original)
    monkeypatch.setattr(continuation.scoring.bench, 'evaluate',
                        lambda: calls.append(continuation.scoring.bench.inputs()))
    continuation.evaluate(tmp_path)
    assert calls == [('plan', 'records', 'folds', jobs[5:])]
    assert continuation.scoring.bench.inputs is original
    with pytest.raises(FileExistsError):
        continuation.evaluate(tmp_path)
    assert len(calls) == 1


def test_mutated_retained_root_blocks_new_execution(tmp_path, monkeypatch):
    archive, jobs, work = fixture_archive(tmp_path, monkeypatch)
    continuation.seed(tmp_path, archive)
    (work/'eval'/jobs[0]['job']/'result.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Retained completed root differs'):
        continuation.evaluate(tmp_path)
    assert not (tmp_path/'continuation-evaluation-claimed.json').exists()


def test_prior_archive_hash_and_frozen_job_order_are_required(tmp_path, monkeypatch):
    archive, jobs, _ = fixture_archive(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='provenance differs'):
        continuation.prior_results(archive, list(reversed(jobs)))
    with archive.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='Prior archive differs'):
        continuation.prior_results(archive, jobs)


def test_report_refuses_failed_guard_even_with_completion_marker(tmp_path, monkeypatch):
    archive, jobs, work = fixture_archive(tmp_path, monkeypatch)
    continuation.seed(tmp_path, archive)
    for job in jobs[5:]:
        folder = work/'eval'/job['job']
        folder.mkdir()
        (folder/'result.json').write_bytes(b'opaque')
    continuation.exclusive_json(tmp_path/'continuation-evaluation-complete.json',
                               {'new_roots': 35, 'retained_roots': 5, 'total_roots': 40})
    operation = tmp_path/'operations/evaluate'
    operation.mkdir(parents=True)
    continuation.exclusive_json(operation/'receipt.json',
                               {'status': 'failed', 'failure': 'pressure',
                                'cleanup_error': None})
    with pytest.raises(ValueError, match='Successful full 40-root'):
        continuation.report(tmp_path)


def test_swap_amendment_preserves_other_guards_and_old_default():
    sample = {'pressure_level': 1, 'free_percent': 60,
              'swap_bytes': 6_000_000_000, 'disk_free_bytes': 30*1024**3, 'ac': True}
    assert guard.violation(sample, 0) == '3 GB total system swap ceiling'
    assert guard.violation(sample, 0, swap_ceiling_bytes=10_000_000_000) is None
    assert guard.violation(dict(sample, swap_bytes=10_000_000_001), 0,
                           swap_ceiling_bytes=10_000_000_000) == '10 GB total system swap ceiling'
    assert guard.violation(dict(sample, pressure_level=2), 0,
                           swap_ceiling_bytes=10_000_000_000) == 'system pressure/headroom'
    assert guard.violation(sample, 7*1024**3,
                           swap_ceiling_bytes=10_000_000_000) == '7 GiB whole-family ceiling'
    assert guard.violation(dict(sample, ac=False), 0,
                           swap_ceiling_bytes=10_000_000_000) == 'AC power'
