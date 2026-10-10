"""Continue the frozen bench using five outcome-blind, archived complete roots."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from scripts import score_restored_hu20_equity_bench as scoring
from src.policies.files import file_hash

ARCHIVE_SHA256 = 'f78c1327d413d519a0da57d99f9cc341f92fea074c3a2465317e85d64f4cd935'
MANIFEST_SHA256 = 'fb6f4bc66d3e16a7b75fb8c890af8f1e16e4b7954b3f6d71eebd8bde27417422'
PRIOR_SOURCE = '9310cb7574b196788fb0e508390337dcb4954d3c'
PRIOR_PREFIX = 'results/equity-bench-scoring-admitted-20261010'
SUMMARY_MEMBER = PRIOR_PREFIX+'/final-pressure-stopped-summary.json'
ARCHIVE_ID = '1KKCV4dLBJplxi_NVScGTSXgkcHahXhLH'


def exclusive_json(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def checked_member(archive: zipfile.ZipFile, pins: dict, name: str) -> bytes:
    payload = archive.read(name)
    pin = pins[name]
    if len(payload) != pin['bytes'] or hashlib.sha256(payload).hexdigest() != pin['sha256']:
        raise ValueError('Prior archive member differs: '+name)
    return payload


def prior_results(archive_path: Path, jobs: list) -> list:
    """Verify opaque result bytes; decode only the manifest and stop metadata."""
    if file_hash(archive_path) != ARCHIVE_SHA256:
        raise ValueError('Prior archive differs')
    with zipfile.ZipFile(archive_path) as archive:
        manifest_bytes = archive.read('ARCHIVE-MANIFEST.json')
        if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
            raise ValueError('Prior manifest differs')
        manifest = json.loads(manifest_bytes)
        pins = {pin['path']: pin for pin in manifest['members']}
        if len(pins) != len(manifest['members']):
            raise ValueError('Duplicate archive member')
        summary = json.loads(checked_member(archive, pins, SUMMARY_MEMBER))
        expected = [job['job'] for job in jobs[:5]]
        if (len(jobs) != 40 or len({job['job'] for job in jobs}) != 40
                or summary['completed_job_ids'] != expected
                or summary['completed_roots'] != 5 or summary['required_roots'] != 40
                or summary['executed_scoring_source'] != PRIOR_SOURCE
                or summary['scores_inspected'] is not False
                or summary['failure_receipt']['status'] != 'failed'):
            raise ValueError('Prior completed-root provenance differs')
        result_names = {name for name in pins
                        if name.startswith(PRIOR_PREFIX+'/work/eval/')
                        and name.endswith('/result.json')}
        expected_names = {PRIOR_PREFIX+'/work/eval/'+job+'/result.json' for job in expected}
        if result_names != expected_names:
            raise ValueError('Prior complete-root set differs')
        return [(job, pins[name], checked_member(archive, pins, name))
                for job in expected
                for name in [PRIOR_PREFIX+'/work/eval/'+job+'/result.json']]


def seed(base: Path, archive_path: Path) -> None:
    _, _, jobs = scoring.configure(base)
    verified = prior_results(archive_path, jobs)
    # Complete verification precedes mutation. A partial seed is never resumed.
    destination = scoring.bench.OUT/'eval'
    destination.mkdir(exist_ok=False)
    members = []
    for job, pin, payload in verified:
        folder = destination/job
        folder.mkdir()
        with (folder/'result.json').open('xb') as stream:
            stream.write(payload)
        members.append({'job': job, 'prior_member': pin['path'],
                        'bytes': pin['bytes'], 'sha256': pin['sha256']})
    exclusive_json(base/'continuation.json', {
        'status': 'seeded-five-complete-roots', 'prior_pr': 225,
        'prior_archive_id': ARCHIVE_ID, 'prior_archive_path': str(archive_path.resolve()),
        'prior_archive_sha256': ARCHIVE_SHA256, 'prior_manifest_sha256': MANIFEST_SHA256,
        'prior_executed_source': PRIOR_SOURCE, 'scores_inspected': False,
        'members': members, 'remaining_jobs': [job['job'] for job in jobs[5:]],
    })


def verify_seed(base: Path, jobs: list) -> dict:
    receipt = json.loads((base/'continuation.json').read_text())
    verified = prior_results(Path(receipt['prior_archive_path']), jobs)
    expected = [{'job': job, 'prior_member': pin['path'],
                 'bytes': pin['bytes'], 'sha256': pin['sha256']}
                for job, pin, _ in verified]
    if (receipt['members'] != expected or receipt['scores_inspected'] is not False
            or receipt['remaining_jobs'] != [job['job'] for job in jobs[5:]]
            or receipt['prior_archive_sha256'] != ARCHIVE_SHA256
            or receipt['prior_manifest_sha256'] != MANIFEST_SHA256
            or receipt['prior_executed_source'] != PRIOR_SOURCE):
        raise ValueError('Continuation receipt differs')
    for pin in expected:
        result = scoring.bench.OUT/'eval'/pin['job']/'result.json'
        if result.stat().st_size != pin['bytes'] or file_hash(result) != pin['sha256']:
            raise ValueError('Retained completed root differs: '+pin['job'])
    return receipt


def evaluate(base: Path) -> None:
    _, _, jobs = scoring.configure(base)
    verify_seed(base, jobs)
    directory = scoring.bench.OUT/'eval'
    if {path.name for path in directory.iterdir()} != {job['job'] for job in jobs[:5]}:
        raise ValueError('Continuation already started or unexpected root')
    exclusive_json(base/'continuation-evaluation-claimed.json',
                   {'remaining_jobs': [job['job'] for job in jobs[5:]]})
    original_inputs = scoring.bench.inputs

    def remaining_inputs():
        plan, records, folds, _ = original_inputs()
        return plan, records, folds, jobs[5:]

    scoring.bench.inputs = remaining_inputs
    try:
        scoring.bench.evaluate()
    finally:
        scoring.bench.inputs = original_inputs
    exclusive_json(base/'continuation-evaluation-complete.json',
                   {'new_roots': 35, 'retained_roots': 5, 'total_roots': 40})


def report(base: Path) -> None:
    _, _, jobs = scoring.configure(base)
    verify_seed(base, jobs)
    guard = json.loads((base/'operations/evaluate/receipt.json').read_text())
    complete = json.loads((base/'continuation-evaluation-complete.json').read_text())
    if (guard['status'] != 'complete' or guard['failure'] is not None
            or guard['cleanup_error'] is not None
            or complete != {'new_roots': 35, 'retained_roots': 5, 'total_roots': 40}
            or {path.parent.name for path in (scoring.bench.OUT/'eval').glob('*/result.json')}
            != {job['job'] for job in jobs}):
        raise ValueError('Successful full 40-root continuation required')
    scoring.bench.report()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('seed', 'evaluate', 'report'))
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--prior-archive', type=Path)
    args = parser.parse_args()
    base = args.base.resolve()
    if (base/'campaign-failure.json').exists():
        raise ValueError('Stopped continuation; no retry')
    if args.command == 'seed':
        if args.prior_archive is None:
            parser.error('seed requires --prior-archive')
        seed(base, args.prior_archive)
    elif args.command == 'evaluate':
        evaluate(base)
    else:
        report(base)


if __name__ == '__main__':
    main()
