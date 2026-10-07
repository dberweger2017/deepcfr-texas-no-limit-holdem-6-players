"""Predeclared attempt-2 primary/sensitivity report from verified retrieved hands.

Uses the frozen paired estimator and Student-t intervals. No models or new
hands run; incomplete pairs never enter arithmetic. Shared deal block IDs,
rather than list positions, determine the three-lineage averages.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import shutil
import tarfile
from time import time

from scripts import audit_hu20_turn_search as native
from scripts.evaluate_hu20_turn_search import compact_row, summarize_phase
from scripts.finish_hu20_mixed_arena import ArchivedFile
from scripts.hu20_search_arena_control import durable_json
from scripts.hu20_search_evidence import file_hash, verify_archives
from scripts.closeout_hu20_autonomous_arena import fleet
from src.arena.report import estimate


def defect_affected(row):
    return any(v and (':fallback:' in k or ':defect:' in k or
                     k.startswith('range:turn_conditioning_fallback:'))
               for k, v in row['search_counts'].items()) or any(
                   r['status'] in ('fallback', 'defect') for r in row['search_records'])


def paired_analysis(plan, rows, *, exclude_defects=False):
    groups = defaultdict(dict)
    expected = {(m['seed'], p['name'], b) for m in plan['models']
                for p in plan['panels'] for b in range(p['blocks'])}
    for row in rows:
        key = row['seed'], row['panel'], row['block']
        if key not in expected or row['arm'] not in ('base', 'search') or row['rotation'] not in (0, 1):
            raise ValueError('Unexpected paired coordinate')
        position = row['arm'], row['rotation']
        if position in groups[key]:
            raise ValueError('Duplicate hand coordinate')
        groups[key][position] = row
    needed = {('base', 0), ('base', 1), ('search', 0), ('search', 1)}
    selected = []
    counts = defaultdict(Counter)
    per_host = defaultdict(Counter)
    for key in sorted(expected):
        items = groups[key]
        complete = set(items) == needed
        flagged = any(r.get('defect_affected', False) for r in items.values())
        keep = complete and not (exclude_defects and flagged)
        metrics = {'planned': 1, 'complete': int(complete), 'incomplete': int(not complete),
                   'defect_affected_complete': int(complete and flagged), 'included': int(keep),
                   'excluded_defect': int(complete and flagged and exclude_defects)}
        counts[key[1]].update(metrics)
        hosts = {r['pod_id'] for r in items.values()}
        if len(hosts) > 1:
            raise ValueError('Paired block crosses static host partition')
        for host in hosts:
            per_host[host].update(metrics)
        if keep:
            selected.extend(items.values())
    result = summarize_phase(selected, 'arena')
    # summarize_phase's per-lineage arithmetic is unchanged. Its legacy zip
    # assumes full coverage; intersect actual block IDs for the aggregate.
    series = defaultdict(dict)
    for change in result['changes']:
        seed, panel = change['seed'], change['panel']
        block_ids = next(p['blocks'] for p in result['panels']
                         if p['seed'] == seed and p['panel'] == panel and p['arm'] == 'base')
        series[panel][seed] = dict(zip(block_ids, change['paired_delta_chips']))
    result['three_lineage_changes'] = []
    seeds = {m['seed'] for m in plan['models']}
    for panel in plan['panels']:
        by_seed = series[panel['name']]
        common = sorted(set.intersection(*(set(by_seed.get(seed, {})) for seed in seeds)))
        values = [sum(by_seed[seed][block] for seed in sorted(seeds)) / 3 for block in common]
        result['three_lineage_changes'].append({
            'panel': panel['name'], 'search_minus_base': estimate(values), 'blocks': common,
            'scope': 'paired fresh blocks; conditional on three original lineages; exploratory 95%',
            'joint_blocks_excluded': panel['blocks'] - len(common)})
    return {**result, 'paired_blocks_per_panel': {k: dict(v) for k, v in counts.items()},
            'paired_blocks_per_host': {k: dict(v) for k, v in per_host.items()},
            'included_hands': len(selected)}


def materialize(root, ledger):
    from scripts.finish_hu20_mixed_arena import ArchivedDirectory
    directories = []
    controllers = {}
    defects = []
    resumes = []
    plans = []
    for pod in fleet(ledger):
        folder = root / 'retrieved' / pod['id']
        proof = json.loads((folder / 'retrieval-verified.json').read_text())
        if not proof.get('verified') or proof['archive_manifest_sha256'] != file_hash(folder / 'manifest.json'):
            raise ValueError('Retrieval verification missing or changed')
        verify_archives(folder)
        manifest = json.loads((folder / 'manifest.json').read_text())
        local = root / 'audit-attempt-2' / pod['id']
        for chunk in manifest['archives']:
            with tarfile.open(folder / chunk['path'], 'r|gz') as archive:
                for member in archive:
                    parts = Path(member.name).parts
                    wanted = (member.name in ('control-pod.json', 'control-pod.events.jsonl') or
                              len(parts) == 2 and parts[0] == 'arena' and parts[1].endswith('.defects.jsonl') or
                              len(parts) == 3 and parts[0] == 'arena' and parts[1].startswith('worker-') and
                              (parts[2] in ('manifest.json', 'summary.json', 'resume-log.jsonl') or
                               parts[2].endswith('.hands.jsonl.gz')))
                    if not wanted:
                        continue
                    destination = local / member.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as stream, destination.open('wb') as target:
                        shutil.copyfileobj(stream, target, 1024**2)
                    spec = manifest['members'][member.name]
                    if destination.stat().st_size != spec['bytes'] or file_hash(destination) != spec['sha256']:
                        raise ValueError('Materialized file differs')
        controllers[pod['id']] = {'journal': json.loads((local / 'control-pod.json').read_text()),
            'events': [json.loads(line) for line in (local / 'control-pod.events.jsonl').read_text().splitlines()]}
        for path in sorted((local / 'arena').glob('worker-*.defects.jsonl')):
            defects.extend(dict(json.loads(line), pod_id=pod['id']) for line in path.read_text().splitlines())
        for worker in pod['workers']:
            prefix = f'arena/worker-{worker}'
            directory = local / prefix
            plans.append(json.loads((directory / 'summary.json').read_text())['plan'])
            directories.append(ArchivedDirectory(directory, prefix, manifest['members']))
            resume = directory / 'resume-log.jsonl'
            if resume.exists():
                resumes.extend(dict(json.loads(line), pod_id=pod['id'], worker=worker)
                               for line in resume.read_text().splitlines())
    if any(plan != plans[0] for plan in plans):
        raise ValueError('Pod plans differ')
    return directories, plans[0], controllers, defects, resumes


def report(root):
    if not (root / 'CLOSEOUT_COMPLETE.json').exists():
        raise ValueError('Verified termination must finish before reporting')
    ledger = json.loads((root / 'ledger.json').read_text())
    directories, plan, controllers, defects, resumes = materialize(root, ledger)
    launch = json.loads((root / 'ARENA_LAUNCH.json').read_text())
    if native.digest(plan) != launch['plan_sha256']:
        raise ValueError('Retrieved plan differs from the frozen launch receipt')
    original_hash = native.file_hash
    try:
        native.file_hash = lambda p: p.spec['sha256'] if isinstance(p, ArchivedFile) else original_hash(p)
        audit = native.audit(directories, allow_incomplete=True)
    finally:
        native.file_hash = original_hash
    rows = []
    for directory in directories:
        pod_id = directory.local.parent.parent.name
        for path in sorted(directory.glob('*.hands.jsonl.gz')):
            with gzip.open(path, 'rt') as stream:
                for line in stream:
                    row = json.loads(line)
                    rows.append(dict(compact_row(row, 'arena'), pod_id=pod_id, defect_affected=defect_affected(row)))
    primary = paired_analysis(plan, rows)
    sensitivity = paired_analysis(plan, rows, exclude_defects=True)
    spend = {'pods': [{'id': p['id'], 'hours': (p['terminated_at'] - p['created_at']) / 3600,
                      'usd': (p['terminated_at'] - p['created_at']) / 3600 * p['hourly_usd']} for p in ledger['pods']],
             'basis': 'provisioning wall clock × readback compute and disk rates; estimate, not settled invoice',
             'historical_charge_usd': ledger.get('historical_cap_charge_usd', 0)}
    spend['fleet_usd'] = sum(p['usd'] for p in spend['pods'])
    result = {'at': time(), 'plan_sha256': native.digest(plan), 'native_audit': audit,
              'primary': primary, 'sensitivity': sensitivity, 'controllers': controllers,
              'per_decision_defects': defects, 'resume_logs': resumes, 'spend': spend}
    durable_json(root / 'attempt-2-results.json', result)
    def fmt(value):
        if value['bb_per_100'] is None:
            return 'unavailable (0 blocks)'
        ci = value['ci95']
        interval = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else f"interval unavailable: {value['reason']}"
        return f"{value['bb_per_100']:.2f} {interval}; n={value['blocks']}"
    lines = [f"Attempt 2 closeout: {audit['hands']:,}/{plan['expected_hands']:,} completed hands independently replayed. All three pods' full archives and member hashes were verified on the M1 before termination; get-pod 404 and complete list-pods receipts confirm removal.",
             '\nBase+search minus base, BB/100, exploratory paired 95% Student-t intervals conditional on the three frozen lineages:',
             '\n| Panel | Primary | Sensitivity (exclude defect-affected pairs) | Complete / flagged / incomplete lineage pairs |',
             '| --- | --- | --- | --- |']
    for a, b in zip(primary['three_lineage_changes'], sensitivity['three_lineage_changes']):
        c = primary['paired_blocks_per_panel'][a['panel']]
        lines.append(f"| {a['panel']} | {fmt(a['search_minus_base'])} | {fmt(b['search_minus_base'])} | {c['complete']} / {c['defect_affected_complete']} / {c['incomplete']} |")
    lines += ['\nPer-lineage contrasts retain every eligible completed pair. The three-lineage contrast requires a shared block ID in all three lineages; the JSON report records every omitted joint block. Crash-missing pairs enter neither analysis; recovered complete hands enter both unless defect-affected.',
              f"\nRecorded per-decision defects: {len(defects)}; fallbacks/conditioning gaps including interrupted-hand decisions: " + str(sum(c['journal'].get('fallbacks', 0) for c in controllers.values())) + f". Resume-log entries: {len(resumes)}. Full defect and recovery lists are retained in `attempt-2-results.json`.",
              '\nPer-host decision timing (seconds; descriptive):\n\n| Host | Decisions | p95 | p99 | Max |\n| --- | ---: | ---: | ---: | ---: |']
    for host, stats in audit['decision_latency_by_host'].items():
        lines.append(f"| {host} | {stats['decisions']} | {stats['p95_seconds']} | {stats['p99_seconds']} | {stats['max_seconds']} |")
    lines += ['\nFlagged/excluded pairs by pod:\n\n```json\n' + json.dumps({'primary': primary['paired_blocks_per_host'], 'sensitivity': sensitivity['paired_blocks_per_host']}, indent=2) + '\n```',
              '\nSpend record:\n\n```json\n' + json.dumps(spend, indent=2) + '\n```',
              '\nThese are the frozen descriptive contrasts and predeclared defect sensitivity. No model promotion, release claim, or new experiment follows. Attempt 1 remains archived and contributes no observations.']
    (root / 'pr166-attempt-2-results.md').write_text('\n'.join(lines) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    report(args.root.resolve())


if __name__ == '__main__':
    main()
