"""Verify the retained build identity before cross-platform checkpoint gates.

The owner accepts #243's established path-remapped build method instead of a
second checkout here. #129/#133 establish the exact-output comparison approach;
the current HU200 #250 fresh/resume hashes remain the machine admission gate.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(os.environ['QUALIFICATION_ROOT'])
first = root / 'repo-main'
pin = (root / 'source-main.txt').read_text().strip()
build = json.loads((root / 'build-first.json').read_text())
def git(*args):
    return subprocess.check_output(['git', *args], cwd=first, text=True).strip()
if git('rev-parse', 'HEAD') != pin or git('rev-parse', 'HEAD:native') != build['native_tree']:
    raise SystemExit('Checkout source/native tree changed since build receipt')
if git('status', '--porcelain', '--', 'native'):
    raise SystemExit('Native sources modified after build')
if not build['native_tree_clean'] or set(build['remapped_prefixes']) != {'native', 'cargo-home'}:
    raise SystemExit('Build receipt does not identify the established remapped build')
paths = [root / 'trainer-first', first / 'native/hu20-trainer/target/release/hu20-trainer']
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
if set(hashes.values()) != {build['binary_sha256']}:
    raise SystemExit(f'Binary mismatch: {hashes}')
for p in paths:
    data = p.read_bytes()
    for prefix in [str(first), '/tmp/hu20-trainer-build']:
        if prefix.encode() in data:
            raise SystemExit(f'Embedded build path in {p}: {prefix}')
receipt = {'source_main': pin, 'binary_hashes': hashes,
           'checkout_and_staging_paths_absent': True, 'status': 'passed',
           'scope': 'Retained/current binary identity and path scan; second-location build not repeated',
           'historical_method_pr': 243, 'historical_output_parity_prs': [129, 133],
           'cross_platform_gate': 'Pending fresh20M and 10M-resumed20M #250 exact checkpoint SHA256'}
(root / 'build-comparison.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
