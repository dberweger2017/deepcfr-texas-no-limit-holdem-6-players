"""Independently verify common native requests, raw loss arithmetic and paired bootstrap."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def average(values):
    return math.fsum(values) / len(values)


def audit(out, candidate_sha256='a5e9d0fc6f4a448640f52f508187f779e43a0a8fd41158207f03de82adc219e1'):
    manifest = json.loads((out / 'manifest.json').read_text())
    published = json.loads((out / 'summary.json').read_text())
    assert len(manifest['jobs']) == 40
    assert manifest['binary_sha256'] == 'fb32974d9d211fa66001d1efb330ec4af2d825005d24b287dc6cf3c37fa8812f'
    assert sha(manifest['binary']) == manifest['binary_sha256']
    assert manifest['policies'][0]['sha256'] == '4534e7db2f69bedd54098b7eaa3c9bd82450838405ae162270a3b7684db9bedf'
    assert len(candidate_sha256) == 64 and all(c in '0123456789abcdef' for c in candidate_sha256)
    assert manifest['policies'][1]['sha256'] == candidate_sha256
    rows, files = [], []
    for job in manifest['jobs']:
        request = json.loads(Path(job['request']).read_text())
        assert sha(job['request']) == job['request_sha256']
        assert sha(request['compact_path']) == job['compact_sha256']
        assert request['pooling_phase'] == 'lock-only' and request['max_iterations'] == 0
        assert request['policy']['seed'] == 2026093001
        assert {r['metric'] for r in request['pooling_measurements']} == {'v040', 'cfrplus'}
        for measurement in request['pooling_measurements']:
            assert measurement['projection_metric'] == 'v1' and measurement['allow_missing']
        for policy in job['policies']:
            assert sha(policy['path']) == policy['sha256']
        leaf = out / 'evaluated' / job['job']
        result = json.loads((leaf / 'result.json').read_text())
        runtime = result['runtime']
        assert result['job'] == job and runtime['status'] == 'completed' and runtime['failure'] is None
        assert runtime['binary_sha256'] == manifest['binary_sha256']
        assert runtime['request_sha256'] == job['request_sha256']
        response = leaf / 'solver/response.jsonl'
        assert sha(response) == runtime['response_sha256']
        metrics = [r for r in map(json.loads, response.read_text().splitlines()) if r['event'] == 'pooling_metric']
        assert metrics == result['metrics']
        assert {(r['metric'], r['target_solver_seat']) for r in metrics} == {(m, s) for m in ('v040', 'cfrplus') for s in (0, 1)}
        for metric in metrics:
            reference = request['reference_equilibrium_ev_chips'][metric['target_solver_seat'] ^ 1]
            assert metric['reference_responder_value_chips'] == reference
            # The native solver subtracts two f32 values before converting gain to f64.
            gain = struct.unpack('f', struct.pack('f', metric['responder_br_chips'] - reference))[0] / 100
            assert math.isclose(gain, metric['gain_bb'], abs_tol=1e-8)
        prepared = out / 'prepared' / job['job']
        refs = {}
        for record in job['references']:
            assert sha(record['path']) == record['sha256']
        for phase in ('collect', 'relock'):
            refs[phase] = json.loads((prepared / f'reference-{phase}-result.json').read_text())['metrics']
        reference_rows = [json.loads(line) for line in (prepared / 'reference-collect-response.jsonl').read_text().splitlines()]
        reference_complete = [r for r in reference_rows if r['event'] == 'completion'][-1]
        assert reference_complete['current_ev_chips'] == request['reference_equilibrium_ev_chips']
        def seat_mean(metrics, label):
            selected = [r for r in metrics if r['metric'] == label]
            assert {r['target_solver_seat'] for r in selected} == {0, 1} and len(selected) == 2
            return average([r['gain_bb'] for r in selected])
        rows.append({'spot': job['spot'], 'fold': job['fold'], 'B': seat_mean(refs['collect'], 'e_bp'),
                     'P': seat_mean(refs['relock'], 'e_cross_v1'),
                     **{m: seat_mean(metrics, m) for m in ('v040', 'cfrplus')}})
        files.append({'response': str(response), 'sha256': sha(response),
                      'request_sha256': job['request_sha256'], 'compact_sha256': job['compact_sha256'],
                      'runtime': runtime})
    assert len({r['spot'] for r in rows}) == 40 and rows == published['rows']
    rng = random.Random(202610050002)
    draws = [[rng.randrange(40) for _ in range(40)] for _ in range(2000)]
    def summarize(stat):
        values = sorted(stat([rows[i] for i in draw]) for draw in draws)
        return {'mean': stat(rows), 'ci95': [values[49], values[1949]]}
    means = {m: summarize(lambda rs, m=m: average([r[m] for r in rs])) for m in ('B', 'P', 'v040', 'cfrplus')}
    placements = {m: summarize(lambda rs, m=m: (average([r[m] - r['P'] for r in rs])) /
                              average([r['B'] - r['P'] for r in rs])) for m in ('v040', 'cfrplus')}
    delta = summarize(lambda rs: average([r['cfrplus'] - r['v040'] for r in rs]))
    def close(left, right):
        assert math.isclose(left['mean'], right['mean'], abs_tol=1e-12)
        assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(left['ci95'], right['ci95'], strict=True))
    for name in means:
        close(means[name], published['E_bb'][name])
    for name in placements:
        close(placements[name], published['Q'][name])
    close(delta, published['paired_delta_E_bb'])
    result = {'status': 'verified', 'boards': 40, 'raw_metrics_verified': 160, 'independent_bootstrap_matches': True,
              'binary_sha256': manifest['binary_sha256'], 'E_bb': means, 'Q': placements,
              'paired_delta_E_bb': delta, 'files': files, 'bootstrap_seed': 202610050002,
              'bootstrap_draws': 2000, 'scope': manifest['scope']}
    (out / 'audit.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--candidate-sha256', default='a5e9d0fc6f4a448640f52f508187f779e43a0a8fd41158207f03de82adc219e1')
    a = p.parse_args()
    result = audit(a.out, a.candidate_sha256)
    print(json.dumps({key: result[key] for key in ('status', 'boards', 'raw_metrics_verified', 'independent_bootstrap_matches')}))


if __name__ == '__main__':
    main()
