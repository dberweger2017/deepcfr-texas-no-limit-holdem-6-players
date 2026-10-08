"""Sequential immutable-input export/audit benchmark; call under an external guard."""

import argparse
from gzip import open as gzip_open
import json
import os
from pathlib import Path
import subprocess
import sys
from time import sleep, time

from scripts.native_hu_followup_limits import family_rss, memory_snapshot, unsafe_memory
from src.blueprint.abstraction import HU100_SCHEMA, HU20_UNCAPPED_SCHEMA
from src.diagnostics.cfr_average import audit, extract
from src.policies.files import file_hash


def operation(mode, checkpoint, current, average, out, bb):
    with gzip_open(checkpoint, 'rt') as source:
        header = json.loads(source.readline())
    spec = {'name': 'memory-benchmark', 'seed': header['config']['seed'], 'iteration': header['iteration'],
            'checkpoint_sha256': file_hash(checkpoint), 'sha256': file_hash(current)}
    schema = HU100_SCHEMA if bb == 100 else HU20_UNCAPPED_SCHEMA
    result = (audit(checkpoint, current, average, spec, file_hash(average), expected_schema=schema)
              if mode == 'audit' else extract(checkpoint, spec, out, expected_schema=schema))
    if mode == 'audit':
        with out.open('x') as target:
            json.dump(result, target, indent=2, sort_keys=True)
    print(json.dumps(result, sort_keys=True))


def measure(command, cwd, directory, name, family_root, deadline):
    """200-ms whole-family RSS plus kernel per-command high water; retain raw logs."""
    if deadline - time() < 5:
        raise ValueError('Benchmark deadline exhausted')
    snapshot = memory_snapshot()
    if unsafe_memory(snapshot, admission=True, rss_gib=10):
        raise ValueError('Fresh system-headroom admission refusal')
    (directory / (name + '.admission.json')).write_text(json.dumps(snapshot, indent=2) + '\n')
    started = time(); peak = 0; samples = 0
    with (directory / (name + '.log')).open('xb') as log, (directory / (name + '.rss.jsonl')).open('x') as raw:
        child = subprocess.Popen(['/usr/bin/time', '-l', *command], cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                if time() >= deadline:
                    raise ValueError('Benchmark deadline exhausted')
                listing = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss='], text=True, timeout=5)
                processes = [tuple(map(int, line.split())) for line in listing.splitlines() if line.strip()]
                sizes = family_rss(processes, {family_root, os.getpid(), child.pid})
                rss = sum(sizes); peak = max(peak, rss); samples += 1
                raw.write(json.dumps({'unix_seconds': time(), 'family_rss_bytes': rss}) + '\n')
                raw.flush()
                sleep(.2)
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, command)
        finally:
            # The external supervisor owns the session and handles any descendants.
            if child.poll() is None:
                child.terminate(); child.wait(timeout=5)
    import re
    highwater = re.search(r'(\d+)\s+maximum resident set size', (directory / (name + '.log')).read_text())
    if highwater is None:
        raise ValueError('Missing macOS per-process kernel peak')
    return {'name': name, 'command': command, 'cwd': str(cwd), 'seconds': time() - started,
            'sampled_peak_family_rss_bytes': peak, 'kernel_command_peak_rss_bytes': int(highwater[1]),
            'samples': samples, 'sampling_target_seconds': .2}


def run(plan, directory, family_root, deadline):
    settings = json.loads(plan.read_text()); rows = []
    python = sys.executable
    # Small pilot, then largest HU100 and HU20 reference before intermediate sizes.
    models = settings['models']
    pilot = models[0]
    order = [pilot, models[-2], models[-1], *models[1:-2]]
    for index, model in enumerate(order):
        if index:
            pilot_cost = sum(r['seconds'] for r in rows if r['model'] == pilot['name'])
            scale = model['entries'] / pilot['entries']
            required = 2 * pilot_cost * max(1, scale) + 60
            if deadline - time() < required:
                raise ValueError(f'Cost-only time-fit refusal for {model["name"]}: {required:.2f}s required')
        checkpoint, current, average = (Path(model[k]) for k in ('checkpoint', 'current', 'average'))
        for phase in ('before', 'after'):
            cwd = Path(settings[phase + '_source'])
            binary = Path(settings[phase + '_binary'])
            prefix = model['name'] + '-' + phase
            exported_current = directory / (prefix + '-current.gz')
            exported_average = directory / (prefix + '-average.gz')
            commands = [
                ('export', [str(binary), 'export', str(checkpoint), '--current', str(exported_current), '--average', str(exported_average)]),
                ('audit', [python, '-m', 'scripts.benchmark_hu_export_audit', '--operation', 'audit', '--checkpoint', str(checkpoint),
                           '--current', str(current), '--average', str(average), '--stack-bb', str(model['bb']),
                           '--out', str(directory / (prefix + '-audit.json'))]),
                ('extract', [python, '-m', 'scripts.benchmark_hu_export_audit', '--operation', 'extract', '--checkpoint', str(checkpoint),
                             '--current', str(current), '--average', str(average), '--stack-bb', str(model['bb']),
                             '--out', str(directory / (prefix + '-python-average.gz'))]),
            ]
            for name, command in commands:
                result = measure(command, cwd, directory, prefix + '-' + name, family_root, deadline)
                result.update(model=model['name'], phase=phase, operation=name, entries=model['entries'])
                rows.append(result)
                (directory / 'measurements.json').write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n')
            if file_hash(exported_current) != file_hash(current) or file_hash(exported_average) != file_hash(average):
                raise ValueError('Native export differs from immutable canonical compressed bytes')
        before = directory / (model['name'] + '-before-python-average.gz')
        after = directory / (model['name'] + '-after-python-average.gz')
        if file_hash(before) != file_hash(after):
            raise ValueError('Python extraction compressed bytes differ')
        before_audit = json.loads((directory / (model['name'] + '-before-audit.json')).read_text())
        after_audit = json.loads((directory / (model['name'] + '-after-audit.json')).read_text())
        if before_audit != after_audit or after_audit['all_nodes_verified'] != model['entries']:
            raise ValueError('Full audit receipts differ')
        with (directory / 'equivalence.jsonl').open('a') as target:
            target.write(json.dumps({'model': model['name'], 'entries': model['entries'],
                'native_current_and_average_compressed_byte_identity': True,
                'python_extraction_compressed_byte_identity': True, 'complete_audit_receipt_identity': True}) + '\n')
        if index == 0:
            quote = {'pilot': pilot['name'], 'seconds': sum(r['seconds'] for r in rows), 'allowance': 2,
                'remaining_seconds': deadline-time(), 'deadline': deadline,
                'projected_remaining_seconds': 2*sum(r['seconds'] for r in rows)*sum(m['entries'] for m in order[1:])/pilot['entries']+60}
            (directory / 'timing-pilot.json').write_text(json.dumps(quote, indent=2) + '\n')
    (directory / 'complete.json').write_text(json.dumps({'status': 'verified', 'finished': time(),
        'deadline': deadline, 'models': len(order), 'measurements': len(rows)}) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operation', choices=('audit', 'extract'))
    for name in ('checkpoint', 'current', 'average', 'out', 'plan'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--stack-bb', type=int, choices=(20, 100))
    p.add_argument('--family-root', type=int); p.add_argument('--deadline', type=float)
    a = p.parse_args()
    if a.operation:
        operation(a.operation, a.checkpoint, a.current, a.average, a.out, a.stack_bb)
    else:
        run(a.plan, a.out, a.family_root, a.deadline)


if __name__ == '__main__':
    main()
