"""Require path-independent binaries from two clean checkouts at one source pin."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(os.environ['QUALIFICATION_ROOT'])
first = root / 'repo-main'
second = root / 'repo-alternate'
pin = (root / 'source-main.txt').read_text().strip()
subprocess.run(['git', 'clone', '--no-hardlinks', str(first), str(second)], check=True)
subprocess.run(['git', 'checkout', '--detach', pin], cwd=second, check=True)
subprocess.run([str(first / '.venv/bin/python'), '-m', 'scripts.build_native_trainer',
                '--receipt', str(root / 'build-second.json')], cwd=second, check=True)
paths = [root / 'trainer-first', first / 'native/hu20-trainer/target/release/hu20-trainer',
         second / 'native/hu20-trainer/target/release/hu20-trainer']
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
if len(set(hashes.values())) != 1:
    raise SystemExit(f'Binary mismatch: {hashes}')
for p in paths:
    data = p.read_bytes()
    for prefix in [str(first), str(second), '/tmp/hu20-trainer-build']:
        if prefix.encode() in data:
            raise SystemExit(f'Embedded build path in {p}: {prefix}')
receipt = {'source_main': pin, 'binary_hashes': hashes,
           'checkout_and_staging_paths_absent': True, 'status': 'passed'}
(root / 'build-comparison.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
