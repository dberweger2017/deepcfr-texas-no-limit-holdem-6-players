"""One matched 500M cost-only measurement after the exact recovery gates."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(os.environ['QUALIFICATION_ROOT'])
repo = root / 'repo-main'
gates = json.loads((root / 'gate-results/checkpoints.json').read_text())
expected = '4009bbbce8d95a722e4e63e96dfc6e2b67957095f06c53487a29e4bf3d3acc8b'
assert gates['fresh20m']['sha256'] == gates['resumed20m']['sha256'] == expected
assert json.loads((root / 'build-comparison.json').read_text())['status'] == 'passed'
out = root / 'benchmark-results'
out.mkdir(exist_ok=False)
checkpoint = out / '500m.gz'
command = [str(repo / 'native/hu20-trainer/target/release/hu20-trainer'), 'train',
           '--stack-bb', '200', '--seed', '2026101302', '--roots-per-seat', '1',
           '--average-rule', 'opponent-sampled', '--nodes', '500000000',
           '--max-entries', '85373862', '--card-buckets', str(root / 'tables'),
           '--out', str(checkpoint), '--telemetry', str(out / '500m.telemetry.jsonl')]
(out / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
start = time.monotonic()
subprocess.run(command, cwd=repo, check=True)
wall = time.monotonic() - start
with checkpoint.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
receipt = {'wall_seconds_including_terminal_save': wall,
           'checkpoint_bytes': checkpoint.stat().st_size, 'checkpoint_sha256': digest}
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
if digest != '2c4c1484b7e4ebefca2bb364223d6cd12e886f98b20784b51ff6ac6678f4312d':
    raise SystemExit('Matched 500M checkpoint differs from #250; investigate')
print(json.dumps(receipt), flush=True)
