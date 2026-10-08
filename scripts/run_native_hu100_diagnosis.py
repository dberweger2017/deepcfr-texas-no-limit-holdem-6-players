"""One-use M1 admission, timing-only pilot and bounded diagnostic stages."""

import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import time
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.diagnose_native_hu100 import write
from scripts.hu20_scaling_supervise import run
from scripts.native_hu_followup_limits import memory_snapshot
from src.policies.files import file_hash

GIB = 1024 ** 3
BINARY = 'native/hu20-trainer/target/release/hu20-trainer'


def initialize(root):
    root.mkdir(parents=True, exist_ok=False)
    mem = memory_snapshot(); available = mem['physical_bytes'] * mem['free_percent'] / 100
    free = shutil.disk_usage(root).free
    limits = {'rss_gib': min(4, available / GIB - 3), 'disk_gib': max(8, free / GIB - 6), 'swap_gib': .25}
    if (platform.node() != 'dberweger-m1' or mem['pressure_level'] != 1 or mem['free_percent'] < 15
            or limits['rss_gib'] < 1 or free < 12 * GIB):
        raise RuntimeError('M1 resource admission refused; retain root, no retry')
    now = time()
    value = {'started': now, 'deadline': now + 1800, 'limits': limits, 'memory': mem,
             'initial_free_disk_bytes': free,
             'swap_baseline': subprocess.check_output(['sysctl', 'vm.swapusage'], text=True),
             'power': subprocess.check_output(['pmset', '-g', 'batt'], text=True),
             'host': platform.node(), 'hardware': subprocess.check_output(['sysctl', 'hw.model'], text=True),
             'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
             'protocol_sha256': file_hash(Path('docs/native-hu100-diagnosis.md'))}
    if 'AC Power' not in value['power']:
        raise RuntimeError('M1 AC admission refused')
    write(root / 'budget.json', value)


def guard(root, phase, jobs):
    budget = json.loads((root / 'budget.json').read_text())
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != budget['source']:
        raise ValueError('Source differs from original budget')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Commit source before compute')
    result = run(jobs, root / ('guard-' + phase), budget['deadline'],
                 swap_before=budget['swap_baseline'], require_ac=True,
                 system_memory_guard=True, **budget['limits'])
    if result['status'] != 'complete':
        raise RuntimeError('Terminal diagnostic stage stop: ' + phase)


def pilot(root):
    py = sys.executable
    guard(root, 'pilot', [
        {'name': 'retrieve', 'command': [py, '-m', 'scripts.diagnose_native_hu100', 'retrieve', '--out', str(root / 'inputs')]},
        {'name': 'build', 'command': ['cargo', 'build', '--release', '--jobs', '2', '--manifest-path', 'native/hu20-trainer/Cargo.toml']},
        {'name': 'pilot-replay', 'command': [py, '-m', 'scripts.diagnose_native_hu100', 'pilot', '--inputs', str(root / 'inputs'), '--out', str(root / 'pilot')]},
        {'name': 'pilot-native', 'command': [BINARY, 'parity', str(root / 'pilot/native-fixtures.jsonl')]},
        {'name': 'pilot-verify', 'command': [py, '-m', 'scripts.diagnose_native_hu100', 'verify', '--out', str(root / 'pilot')]},
    ])
    s = json.loads((root / 'pilot/summary.json').read_text())
    # Freeze reads timing/completeness fields only; pilot outcomes never drive admission.
    quote = 2 * s['model_scan_seconds'] + 2 * (s['seconds'] - s['model_scan_seconds']) * 128 + 240
    budget = json.loads((root / 'budget.json').read_text())
    value = {'status': 'admitted' if quote < budget['deadline'] - time() else 'no-final-budget',
             'pilot_hands': s['hands'], 'pilot_actions': s['actions'],
             'pilot_model_scan_seconds': s['model_scan_seconds'],
             'pilot_other_seconds': s['seconds'] - s['model_scan_seconds'],
             'conservative_final_quote_seconds': quote,
             'frozen_candidate_hands': 102400, 'blocks_per_opponent_checkpoint': 2048,
             'search_states_per_signature': 5000, 'search_seconds_total': 120,
             'final_sample': 'all candidate arms at all five checkpoints and opponents; no losing-hand filter',
             'at': time(), 'deadline': budget['deadline'],
             'source': budget['source'], 'protocol_sha256': budget['protocol_sha256']}
    write(root / 'frozen-analysis.json', value)
    if value['status'] != 'admitted':
        raise RuntimeError('Timing-only quote does not admit the full sample')
    print(json.dumps(value))


def analyse(root):
    if json.loads((root / 'frozen-analysis.json').read_text())['status'] != 'admitted':
        raise RuntimeError('No frozen full-sample admission')
    py = sys.executable
    guard(root, 'analysis', [
        {'name': 'full-replay', 'command': [py, '-m', 'scripts.diagnose_native_hu100', 'analyse', '--inputs', str(root / 'inputs'), '--out', str(root / 'analysis')]},
        {'name': 'full-native', 'command': [BINARY, 'parity', str(root / 'analysis/native-fixtures.jsonl')]},
        {'name': 'independent-arithmetic', 'command': [py, '-m', 'scripts.diagnose_native_hu100', 'verify', '--out', str(root / 'analysis')]},
        {'name': 'focused-tests', 'command': [py, '-m', 'pytest', '-q', 'tests/test_native_hu100_diagnosis.py',
           'tests/test_native_hu100_baseline.py', 'tests/test_native_hu100_learning_curves.py',
           'tests/test_native_hu100_preparation.py', 'tests/test_native_trainer_parity.py']},
        {'name': 'rust-tests', 'command': ['cargo', 'test', '--jobs', '2', '--manifest-path', 'native/hu20-trainer/Cargo.toml']},
        {'name': 'artifacts', 'command': [py, 'scripts/check_repository_artifacts.py']},
    ])


def archive(root, output):
    """Seal all stage outputs, inputs and failures; read every archive member back."""
    members = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'x', ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file():
                name = str(p.relative_to(root))
                spec = {'path': name, 'bytes': p.stat().st_size, 'sha256': file_hash(p)}
                z.write(p, name); members.append(spec)
        raw = json.dumps({'members': members, 'restore': 'fresh ignored root; verify whole ZIP, manifest and selected members'}, sort_keys=True).encode()
        z.writestr('ARCHIVE-MANIFEST.json', raw)
    with ZipFile(output) as z:
        for m in members:
            import hashlib
            h = hashlib.sha256(); size = 0
            with z.open(m['path']) as f:
                while chunk := f.read(1024 * 1024): h.update(chunk); size += len(chunk)
            if size != m['bytes'] or h.hexdigest() != m['sha256']:
                raise ValueError('Archive readback differs')
    write(root / 'archive-receipt.json', {'status': 'locally-verified', 'path': str(output),
          'bytes': output.stat().st_size, 'sha256': file_hash(output), 'members': len(members),
          'manifest_sha256': __import__('hashlib').sha256(raw).hexdigest(), 'finished': time()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('initialize', 'pilot', 'analyse', 'seal', 'archive-worker'))
    p.add_argument('--root', type=Path, required=True); p.add_argument('--archive', type=Path)
    a = p.parse_args()
    if a.command == 'initialize': initialize(a.root)
    elif a.command == 'pilot': pilot(a.root)
    elif a.command == 'analyse': analyse(a.root)
    elif a.command == 'archive-worker': archive(a.root, a.archive)
    else:
        guard(a.root, 'archive', [{'name': 'seal-readback', 'command': [sys.executable, '-m',
              'scripts.run_native_hu100_diagnosis', 'archive-worker', '--root', str(a.root), '--archive', str(a.archive)]}])


if __name__ == '__main__': main()
