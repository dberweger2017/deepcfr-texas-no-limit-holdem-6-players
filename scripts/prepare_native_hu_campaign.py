"""Write a future native HU campaign plan and operator commands; never launch jobs.

Historical HU20 controllers bind experiments, hosts and frozen scientific sources.
This preparation entry point uses the existing resource supervisor but keeps new
stage manifests independent of those frozen campaigns.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import platform
from pathlib import Path
import shlex
import subprocess
import sys

REFERENCE = '571e198266eabc6d8bb9de2d1aa76d9a68be0b2512222ea96874168989c6b74d'
SEED = 2026100601


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    hasher = sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''): hasher.update(block)
    return hasher.hexdigest()


def prepare(stage, out, binary, *, quote=None, approval=None):
    binary = binary.resolve()
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Prepare from committed, clean tracked source')
    if not binary.is_file(): raise ValueError('Build the source-pinned native binary first')
    if stage not in ('hu20', 'pilot', 'extension'): raise ValueError('Unknown stage')
    limits = {'rss_gib': 5.5, 'disk_gib': 15.5, 'swap_gib': .5, 'threads': 1,
              'training_seconds': 900 if stage != 'extension' else 6300,
              'total_seconds': 1800 if stage != 'extension' else 7200,
              'audit_seconds': 900, 'max_entries': 2000000 if stage == 'pilot' else 1000000000}
    nodes = {'hu20': 1000000000, 'pilot': 10000000, 'extension': 1000000000}[stage]
    bb = 20 if stage == 'hu20' else 100
    plan = {'version': 1, 'status': 'prepared-only', 'stage': stage, 'source': source,
            'environment': {'python': sys.version, 'platform': platform.platform(),
                            'cargo_lock_sha256': file_hash(Path('native/hu20-trainer/Cargo.lock')),
                            'requirements_sha256': file_hash(Path('requirements-play.txt'))},
            'binary': str(binary), 'binary_sha256': file_hash(binary), 'seed': SEED,
            'stack_bb': bb, 'target_total_nodes': nodes, 'limits': limits,
            'recipe': {'regrets': 'linear-CFR', 'average': 'opponent-sampled',
                       'zero_mass': 'uniform', 'card_descriptor': 'legacy-postflop-descriptor-v1',
                       'menu': 'min-pot-conditional-jam; native reopening; no free fold'},
            'hu20_reference_sha256': REFERENCE, 'strength_claim': False}
    parent = None
    if stage == 'extension':
        if quote is None or approval is None: raise ValueError('Extension requires measured quote and owner authorization receipts')
        q, a = json.loads(quote.read_text()), json.loads(approval.read_text())
        required = ('source', 'binary_sha256', 'seed', 'target_total_nodes', 'parent_path', 'parent_sha256',
                    'forecast_seconds', 'training_forecast_seconds', 'audit_forecast_seconds', 'forecast_rss_gib', 'forecast_disk_free_gib', 'max_entries', 'pilot_audit_verified')
        if any(field not in q for field in required): raise ValueError('Incomplete measured quote')
        if (q['source'] != source or q['binary_sha256'] != plan['binary_sha256'] or q['seed'] != SEED
            or q['target_total_nodes'] != nodes or q['pilot_audit_verified'] is not True
            or q['forecast_seconds'] <= 0 or q['forecast_seconds'] > limits['total_seconds']
            or not 0 < q['training_forecast_seconds'] <= limits['training_seconds']
            or not 0 < q['audit_forecast_seconds'] <= limits['audit_seconds']
            or q['training_forecast_seconds'] + q['audit_forecast_seconds'] > q['forecast_seconds']
            or q['forecast_rss_gib'] <= 0 or q['forecast_rss_gib'] >= limits['rss_gib']
            or q['forecast_disk_free_gib'] < limits['disk_gib']
            or type(q['max_entries']) is not int or q['max_entries'] < 1):
            raise ValueError('Quote fails admission; revise scope with owner before commands are prepared')
        if (a.get('approved') is not True or a.get('quote_sha256') != file_hash(quote)
            or not a.get('owner_instruction_url') or not a.get('approved_at')):
            raise ValueError('Owner receipt must bind the measured quote')
        parent = Path(q['parent_path']).resolve()
        if file_hash(parent) != q['parent_sha256']: raise ValueError('Pilot parent hash differs')
        from gzip import open as gzip_open
        from src.blueprint.average import checked_header
        from src.blueprint.abstraction import HU100_SCHEMA
        with gzip_open(parent, 'rt') as checkpoint:
            h = json.loads(checkpoint.readline())
        checked_header(h, {'seed': SEED, 'iteration': h['iteration']}, expected_schema=HU100_SCHEMA)
        if (h.get('average_rule') != 'opponent-sampled' or 'training_options' in h
            or not 10000000 <= h.get('native_state', {}).get('completed_nodes', 0) < nodes
            or h['native_state'].get('coverage_start') != [0,0,0]
            or len(h['native_state'].get('traverser_visits_by_street', [])) != 4
            or any(type(v) is not int or v <= 0 for v in h['native_state']['traverser_visits_by_street'])):
            raise ValueError('Extension parent is not a recoverable production HU100 pilot')
        limits['max_entries'] = q['max_entries']
        plan.update(quote=q, quote_sha256=file_hash(quote), owner_approval=a)
    out = out.resolve()
    if out.exists(): raise FileExistsError('Use a fresh attempt root')
    pattern = out/'training'/f'HU{bb}-{SEED}-{{nodes}}.json.gz'
    milestones = {'hu20': '100000000,500000000', 'pilot': '100000,1000000,5000000',
                  'extension': '50000000,100000000,250000000,500000000'}[stage]
    command = [str(binary), 'train', '--stack-bb', str(bb), '--seed', str(SEED), '--roots-per-seat', '1',
               '--average-rule', 'opponent-sampled', '--nodes', str(nodes), '--milestones', milestones,
               '--max-seconds', str(limits['training_seconds']), '--max-entries', str(limits['max_entries']), '--out', str(pattern)]
    if stage == 'pilot': command += ['--recovery']
    if parent: command += ['--resume', str(parent), '--resume-sha256', q['parent_sha256']]
    checkpoint = Path(str(pattern).replace('{nodes}', str(nodes)))
    current, average = out/'current.json.gz', out/'average.jsonl.gz'
    jobs = [{'name': 'train', 'command': command}]
    exports = [{'name': 'export', 'command': [str(binary), 'export', str(checkpoint), '--current', str(current), '--average', str(average), '--zero-mass', 'uniform']},
               {'name': 'audit', 'command': [sys.executable, '-m', 'scripts.audit_native_hu_checkpoint', '--checkpoint', str(checkpoint), '--current', str(current), '--average', str(average), '--stack-bb', str(bb), '--target-nodes', str(nodes), '--out', str(out/'audit.json')]}]
    plan.update(command=command, export_audit_jobs=exports, prepared_at=datetime.now(timezone.utc).isoformat())
    plan['plan_sha256'] = digest(plan)
    out.mkdir(parents=True); (out/'training').mkdir()
    for name, value in [('plan.json', plan), ('training-jobs.json', jobs), ('export-audit-jobs.json', exports)]:
        (out/name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    # Absolute deadlines are chosen by the operator at launch, not by this preparation call.
    deadline_command = shlex.join([sys.executable, '-c', f'import time; print(time.time()+{limits["total_seconds"]})'])
    lines = ['set -eu', '# Prepared commands only; obtain stage authorization and idle-worker admission first.',
             'export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',
             f'CAMPAIGN_DEADLINE=$({deadline_command})']
    for name, seconds in [('training', limits['training_seconds']), ('export-audit', limits['audit_seconds'])]:
        guard = [sys.executable, '-m', 'scripts.hu20_scaling_supervise', '--jobs', str(out/f'{name}-jobs.json'),
                 '--out', str(out/f'{name}-guard'), '--rss-gib', str(limits['rss_gib']), '--disk-gib', str(limits['disk_gib']),
                 '--swap-gib', str(limits['swap_gib']), '--require-ac']
        deadline_code = f'import sys,time; print(min(float(sys.argv[1]),time.time()+{seconds}))'
        phase_command = shlex.join([sys.executable, '-c', deadline_code])
        lines += [f'# {name}: total phase cap {seconds}s; includes startup and checkpoint/export writes.',
                  shlex.join(guard) + f' --deadline "$({phase_command} "$CAMPAIGN_DEADLINE")"']
    (out/'commands.txt').write_text('\n'.join(lines)+'\n')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('hu20', 'pilot', 'extension'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=Path('native/hu20-trainer/target/release/hu20-trainer'))
    parser.add_argument('--quote', type=Path); parser.add_argument('--approval', type=Path)
    args = parser.parse_args()
    plan = prepare(args.stage, args.out, args.binary, quote=args.quote, approval=args.approval)
    print(json.dumps({'status': plan['status'], 'plan_sha256': plan['plan_sha256']}))


if __name__ == '__main__': main()
