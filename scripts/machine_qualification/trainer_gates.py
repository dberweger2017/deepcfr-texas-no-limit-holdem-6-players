"""Only the fresh/resumed #250 correctness gate, never a campaign launcher."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(os.environ['QUALIFICATION_ROOT']).resolve()
repo = Path(os.environ['QUALIFICATION_REPO']).resolve()
output = root / 'gate-results'
output.mkdir(exist_ok=False)
binary = repo / 'native/hu20-trainer/target/release/hu20-trainer'
tables = root / 'tables'
pins = {'flop': '084e243ded9fe93ad03cd59601f22f16c3210592f51e1e3646a60983b572c111',
        'turn': 'c0bc6a8aa0535553118109d18a32d3b4dc6880e937c263cdc87472b1ae9f168f',
        'river': '70c1cb5ecf4cd292e4c89dc2200f7c8381100be3b0288ceeb0a12a3e83977629'}
def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
verified_tables = {}
for street, expected in pins.items():
    path = tables / f'{street}-k50.bin'
    actual = digest(path)
    if actual != expected:
        raise SystemExit(f'Invalid {street} table; refusing training')
    verified_tables[street] = {'path': str(path), 'bytes': path.stat().st_size,
                              'sha256': actual}
(output / 'verified-tables.json').write_text(json.dumps(verified_tables, indent=2) + '\n')
commands = []
receipts = {}
for name, nodes, parent in [('fresh20m', 20000000, None),
                            ('prefix10m', 10000000, None),
                            ('resumed20m', 20000000, 'prefix10m')]:
    command = [str(binary), 'train', '--stack-bb', '200', '--seed', '2026101302',
               '--roots-per-seat', '1', '--average-rule', 'opponent-sampled',
               '--nodes', str(nodes), '--max-entries', '85373862',
               '--card-buckets', str(tables), '--out', str(output / f'{name}.gz'),
               '--telemetry', str(output / f'{name}.telemetry.jsonl')]
    stop = os.environ.get('NPLAYER_STOP_FILE')
    if stop:
        if Path(stop).exists():
            raise SystemExit('Supervisor requested stop; refusing next regression segment')
        command += ['--stop-file', stop]
    if parent:
        checkpoint = output / f'{parent}.gz'
        command += ['--resume', str(checkpoint), '--resume-sha256', digest(checkpoint)]
    commands.append(command)
    (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    subprocess.run(command, check=True, cwd=repo)
    checkpoint = output / f'{name}.gz'
    receipts[name] = {'bytes': checkpoint.stat().st_size, 'sha256': digest(checkpoint)}
    (output / 'checkpoints.json').write_text(json.dumps(receipts, indent=2) + '\n')
    if name != 'prefix10m' and receipts[name]['sha256'] != '4009bbbce8d95a722e4e63e96dfc6e2b67957095f06c53487a29e4bf3d3acc8b':
        raise SystemExit(f'{name} does not reproduce #250; benchmarks blocked')
    print(f'{name} passed: {receipts[name]}', flush=True)
if receipts['fresh20m'] != receipts['resumed20m']:
    raise SystemExit('Fresh/resumed checkpoint mismatch')
print('Exact fresh and resumed 20M gates passed', flush=True)
