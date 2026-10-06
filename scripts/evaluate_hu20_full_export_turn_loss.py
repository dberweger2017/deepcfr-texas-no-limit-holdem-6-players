"""Apply #149's identical lock-only pipeline to two full-game exports on common turn roots.

Ranges, tree, board folds and reference equilibrium are pinned #149 inputs. Only the locked
policy probabilities change. Q retains #162's common B/P anchors, not policy-dependent ranges.
"""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import random
import shutil
from statistics import mean
from time import time

from scripts.evaluate_hu20_v041_arena import load
from src.diagnostics.board_pooling_results import completion
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.pooling_runtime import run_owned_tool
from src.diagnostics.saved_hu20 import file_hash


def groups_for_board(source, compact, evaluation_lineage):
    groups = {}
    counts = Counter()
    for template, codes in compact['pool_keys']['v1'].items():
        names = compact['tables'][template]['names']
        for key in codes.values():
            if key in groups:
                if groups[key]['names'] != names:
                    raise ValueError('One v1 key changed its native menu')
                continue
            entry = source.entries.get(key)
            status = 'missing' if entry is None else 'zero_mass' if key in getattr(source, 'zero_mass', ()) else 'stored'
            if entry is not None and tuple(entry[0]) != tuple(names):
                raise ValueError('Full-game policy menu differs from the frozen tree')
            p = list(entry[1]) if entry is not None else [1 / len(names)] * len(names)
            # Native load_pool uses lineage as the common request's range identity. Actual training
            # identity stays in source metadata; zero mass retains native uniform fallback telemetry.
            groups[key] = {'lineage': evaluation_lineage, 'metric': 'v1', 'key': key, 'names': names,
                           'probabilities': p, 'mass': float(status == 'stored'), 'roots': 0}
            counts[status] += 1
    return list(groups.values()), dict(counts)


def prepare(arena_plan, policies, prepared, references, out, binary, reference_inventory):
    out.mkdir(parents=True, exist_ok=False)
    old = json.loads((prepared / 'manifest.json').read_text())
    expected = {r['member']: r for r in json.loads(reference_inventory.read_text())['files']}
    jobs = sorted((j for j in old['jobs'] if j['policy_index'] == 0), key=lambda j: j['spot'])
    if len(jobs) != 40:
        raise ValueError('Expected all 40 frozen #149 roots')
    specs = [next(s for s in arena_plan['models'] if s['name'] == name)
             for name in ('R-2026093001', 'O-2026100601')]
    sources = [load(spec, policies) for spec in specs]
    records = []
    for job in jobs:
        name = job['job']
        src = prepared / 'jobs' / name
        if file_hash(src / 'request.json') != job['request_sha256'] or file_hash(src / 'compact.json') != job['compact_sha256']:
            raise ValueError('Frozen preparation input differs')
        leaf = out / 'prepared' / name
        leaf.mkdir(parents=True)
        shutil.copy2(src / 'compact.json', leaf / 'compact.json')
        request = json.loads((src / 'request.json').read_text())
        compact = json.loads((leaf / 'compact.json').read_text())
        measures = []
        policy_records = []
        for label, spec, source in zip(('v040', 'cfrplus'), specs, sources, strict=True):
            groups, coverage = groups_for_board(source, compact, request['policy']['seed'])
            path = leaf / f'{label}.policy.json'
            atomic_json(path, {'format': 'hu20-board-pooling-policy-v1', 'groups': groups,
                              'source': spec, 'evaluation_range_lineage': request['policy']['seed']})
            measures.append({'metric': label, 'projection_metric': 'v1', 'policy_path': str(path.resolve()),
                             'allow_missing': True})
            policy_records.append({'name': label, 'path': str(path), 'sha256': file_hash(path), 'key_coverage': coverage})
        ref = references / 'collect' / name / 'solver' / 'response.jsonl'
        retained = []
        for phase, rel in (('collect', 'solver/response.jsonl'), ('collect', 'result.json'), ('relock', 'result.json')):
            path = references / phase / name / rel
            member = f'inputs/main-06/{phase}/{name}/{rel}'
            if file_hash(path) != expected[member]['sha256']:
                raise ValueError('Archived #149 reference differs')
            dst = leaf / f"reference-{phase}-{Path(rel).name}"
            shutil.copy2(path, dst)
            retained.append({'path': str(dst), 'sha256': file_hash(dst), 'archive_member': member})
        ref_rows = [json.loads(line) for line in ref.read_text().splitlines()]
        request.update(pooling_phase='lock-only', max_iterations=0, compact_path=str((leaf / 'compact.json').resolve()),
                       reference_equilibrium_ev_chips=completion(ref_rows)['current_ev_chips'],
                       reference_response_sha256=file_hash(ref), pooling_measurements=measures)
        atomic_json(leaf / 'request.json', request)
        records.append({'job': name, 'spot': job['spot'], 'fold': job['evaluation_fold'],
                        'request': str(leaf / 'request.json'), 'request_sha256': file_hash(leaf / 'request.json'),
                        'compact_sha256': file_hash(leaf / 'compact.json'), 'policies': policy_records,
                        'references': retained})
        del compact, ref_rows
        gc.collect()
    manifest = {'jobs': records, 'policies': specs, 'binary': str(binary.resolve()), 'binary_sha256': file_hash(binary),
                'reference_inventory_sha256': file_hash(reference_inventory), 'prepared_manifest_sha256': file_hash(prepared / 'manifest.json'),
                'scope': 'one lineage per policy; identical #149 common ranges/tree/reference, all 40 boards; lock-only, no new solving'}
    atomic_json(out / 'manifest.json', manifest)
    return manifest


def evaluate(out, limit=None, max_seconds=7200, job_memory_bytes=7 * 1024**3):
    deadline = time() + max_seconds
    manifest = json.loads((out / 'manifest.json').read_text())
    if file_hash(manifest['binary']) != manifest['binary_sha256']:
        raise ValueError('Qualified lock-only binary changed')
    completed = 0
    for job in manifest['jobs'][:limit]:
        leaf = out / 'evaluated' / job['job']
        if (leaf / 'result.json').exists():
            completed += 1
            continue
        if leaf.exists():
            raise FileExistsError('Preserve the partial native attempt')
        if time() >= deadline:
            raise RuntimeError('Turn-loss aggregate wall deadline')
        if shutil.disk_usage(out).free < 15 * 1024**3:
            raise RuntimeError('Turn-loss free-disk guard')
        if file_hash(job['request']) != job['request_sha256']:
            raise ValueError('Frozen measurement request differs')
        request = json.loads(Path(job['request']).read_text())
        if file_hash(request['compact_path']) != job['compact_sha256']:
            raise ValueError('Frozen compact tree differs')
        for policy in job['policies']:
            if file_hash(policy['path']) != policy['sha256']:
                raise ValueError('Frozen full-game probability export differs')
        leaf.mkdir(parents=True)
        runtime = run_owned_tool(Path(manifest['binary']), Path(job['request']), leaf / 'solver',
                                 memory_bytes=request['memory_budget_bytes'], threads=6,
                                 seconds=min(request['seconds'] + 300, deadline - time()),
                                 job_memory_bytes=job_memory_bytes)
        if runtime['status'] != 'completed':
            raise RuntimeError(runtime['failure'])
        metrics = [row for row in map(json.loads, (leaf / 'solver/response.jsonl').read_text().splitlines())
                   if row['event'] == 'pooling_metric']
        if {(m['metric'], m['target_solver_seat']) for m in metrics} != {(label, seat) for label in ('v040', 'cfrplus') for seat in (0, 1)}:
            raise ValueError('Missing full-game lock-only measurement')
        atomic_json(leaf / 'result.json', {'job': job, 'runtime': runtime, 'metrics': metrics})
        completed += 1
        # Pilot stdout contains cost only; gain values remain unread in retained files until reporting.
        print(json.dumps({'completed': completed, 'job': job['job'], 'runtime': runtime}), flush=True)


def report(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    rows = []
    for job in manifest['jobs']:
        leaf = out / 'prepared' / job['job']
        result = json.loads((out / 'evaluated' / job['job'] / 'result.json').read_text())
        refs = {phase: json.loads((leaf / f'reference-{phase}-result.json').read_text())['metrics']
                for phase in ('collect', 'relock')}
        def seat_mean(metrics, label):
            values = [m['gain_bb'] for m in metrics if m['metric'] == label]
            if len(values) != 2: raise ValueError('Expected both seats')
            return mean(values)
        rows.append({'spot': job['spot'], 'fold': job['fold'], 'B': seat_mean(refs['collect'], 'e_bp'),
                     'P': seat_mean(refs['relock'], 'e_cross_v1'),
                     **{label: seat_mean(result['metrics'], label) for label in ('v040', 'cfrplus')}})
    rng = random.Random(202610050002)
    draws = [[rng.randrange(len(rows)) for _ in rows] for _ in range(2000)]
    def summary(stat):
        values = sorted(stat([rows[i] for i in draw]) for draw in draws)
        return {'mean': stat(rows), 'ci95': [values[49], values[1949]]}
    means = {label: summary(lambda rs, label=label: mean(r[label] for r in rs)) for label in ('B', 'P', 'v040', 'cfrplus')}
    placements = {label: summary(lambda rs, label=label: (mean(r[label] for r in rs) - mean(r['P'] for r in rs))
                                     / (mean(r['B'] for r in rs) - mean(r['P'] for r in rs))) for label in ('v040', 'cfrplus')}
    delta = summary(lambda rs: mean(r['cfrplus'] - r['v040'] for r in rs))
    result = {'boards': len(rows), 'rows': rows, 'E_bb': means, 'Q': placements, 'paired_delta_E_bb': delta,
              'bootstrap_seed': 202610050002, 'bootstrap_draws': 2000, 'percentile_indices': [49, 1949],
              'scope': manifest['scope'], 'Q_definition': '(E-P)/(B-P); common original #149 B500M average and held-out witness anchors'}
    atomic_json(out / 'summary.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'evaluate', 'report'))
    p.add_argument('--out', type=Path, required=True)
    for name in ('arena-plan', 'policies', 'prepared', 'references', 'binary', 'reference-inventory'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--limit', type=int)
    p.add_argument('--max-seconds', type=float, default=7200)
    p.add_argument('--job-memory-gib', type=float, default=7)
    a = p.parse_args()
    if a.command == 'prepare':
        result = prepare(json.loads(a.arena_plan.read_text()), a.policies, a.prepared, a.references,
                         a.out, a.binary, a.reference_inventory)
        print(json.dumps({'jobs': len(result['jobs']), 'binary_sha256': result['binary_sha256']}))
    elif a.command == 'evaluate':
        if a.max_seconds <= 0 or a.job_memory_gib <= 0:
            p.error('--max-seconds and --job-memory-gib must be positive')
        evaluate(a.out, a.limit, a.max_seconds, int(a.job_memory_gib * 1024**3))
    else:
        result = report(a.out)
        print(json.dumps({k: result[k] for k in ('boards', 'E_bb', 'Q', 'paired_delta_E_bb')}))


if __name__ == '__main__':
    main()
