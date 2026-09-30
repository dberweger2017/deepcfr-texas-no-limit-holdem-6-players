"""Verify a closed Linux archive against the sealed mature M4 reference."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import time


def digest(path, payload=False):
    h = sha256()
    with (gzip.open(path, 'rb') if payload else path.open('rb')) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def export_transport_difference(a, b):
    """Only the previously established export OS byte is exempted."""
    with a.open('rb') as left, b.open('rb') as right:
        x, y = left.read(10), right.read(10)
        if len(x) != 10 or len(y) != 10 or x[:9] != y[:9] or (x[9], y[9]) != (19, 3):
            return False
        while True:
            x, y = left.read(1024 * 1024), right.read(1024 * 1024)
            if x != y:
                return False
            if not x:
                return True


def verify(root, reference, out):
    manifest = json.loads((root / 'work-manifest.json').read_text())
    work = root / 'work'
    actual = {str(p.relative_to(work)) for p in work.rglob('*') if p.is_file()}
    checks = {'exact_archive_membership': actual == set(manifest)}
    for name, info in manifest.items():
        path = work / name
        checks['hash:' + name] = path.stat().st_size == info['bytes'] and digest(path) == info['sha256']
    state = json.loads((work / 'worker.json').read_text())
    checks['all_worker_attempts'] = (state['status'] == 'complete'
        and len(state['attempts']) == 3
        and all(a['status'] == 'complete' and a['guard_failure'] is None for a in state['attempts']))
    a = json.loads((reference / 'direct/result.json').read_text())
    b = json.loads((work / 'direct/result.json').read_text())
    fields = ('added_nodes', 'overshoot_nodes', 'iteration', 'entries', 'new_entries',
              'config', 'origin_entries', 'origin_iteration', 'next_nodes', 'next_streams',
              'work_sha256', 'parent_fingerprint')
    for field in fields:
        checks[field] = a[field] == b[field]
    checks['environment:source'] = a['environment']['source'] == b['environment']['source']
    checks['environment:engine_origin'] = json.loads(a['environment']['engine_origin']) == json.loads(b['environment']['engine_origin'])
    checks['python_version'] = b['environment']['python'].startswith('3.11.14 ')
    transports = {}
    for name in ('final', 'current', 'next'):
        left = reference / 'direct' / (name + '.json.gz')
        right = work / 'direct' / (name + '.json.gz')
        checks['full_payload:' + name] = digest(left, True) == digest(right, True)
        identical = digest(left) == digest(right)
        os_only = name == 'current' and export_transport_difference(left, right)
        checks['explained_transport:' + name] = identical or os_only
        transports[name] = 'byte_identical' if identical else 'gzip_os_byte_9_only' if os_only else 'unexplained'
        for phase in ('direct', 'resumed'):
            path = work / phase / (name + '.json.gz')
            record = json.loads((work / phase / 'result.json').read_text())[name]
            checks[f'rehashed:{phase}:{name}'] = (digest(path) == record['sha256']
                and digest(path, True) == record['uncompressed_sha256'])
        checks['resume_bytes:' + name] = digest(work / 'direct' / (name + '.json.gz')) == digest(work / 'resumed' / (name + '.json.gz'))
    direct_rows = [json.loads(line) for line in (work / 'direct/iterations.jsonl').read_text().splitlines()]
    reference_rows = [json.loads(line) for line in (reference / 'direct/iterations.jsonl').read_text().splitlines()]
    resumed = json.loads((work / 'resumed/result.json').read_text())
    suffix = [json.loads(line) for line in (work / 'resumed/iterations.jsonl').read_text().splitlines()]
    checks['all_cross_platform_iterations'] = reference_rows == direct_rows
    checks['all_resumed_iterations'] = suffix == [r for r in direct_rows if r['added_nodes'] > resumed['resume_added_nodes']]
    checks['reconciled_work'] = (sum(r['nodes'] for r in direct_rows) == b['added_nodes']
        and sum(r['new_entries'] for r in direct_rows) == b['new_entries']
        and all(r['nodes'] == r['attempted_work']['nodes'] for r in direct_rows))
    checks['next_rng_resume'] = resumed['next_streams'] == b['next_streams']
    resources = [json.loads(line) for line in (work / 'resources.jsonl').read_text().splitlines()]
    cap = min(10.5 * 2**30, state['memory_limit_bytes'] * .8)
    checks['resource_guards'] = (bool(resources) and all(r['owned_rss_bytes'] < cap
        and r['swap_growth_bytes'] <= .5 * 2**30 and r['free_disk_bytes'] >= 8 * 2**30 for r in resources))
    result = {'passed': all(checks.values()), 'checks': checks, 'transport': transports,
              'verified_files': len(manifest), 'completed_iterations': len(direct_rows),
              'resumed_iterations': len(suffix), 'finished': time.time(),
              'verification_source_sha256': digest(Path(__file__))}
    out.write_text(json.dumps(result, sort_keys=True) + '\n')
    if not result['passed']:
        raise ValueError('Parity/resource verification failed: ' + str([k for k, v in checks.items() if not v]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'reference', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    verify(args.root, args.reference, args.out)
