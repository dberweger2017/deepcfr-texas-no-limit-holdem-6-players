"""Verify all restored table bytes before any optional real-table test uses them."""
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ['QUALIFICATION_ROOT'])
pins = json.loads((root / 'table-input-pins.json').read_text())
assert {r['street'] for r in pins['tables']} == {'flop', 'turn', 'river'}
verified = []
for pin in pins['tables']:
    path = root / 'tables' / f'{pin["street"]}-k50.bin'
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != pin['sha256'] or path.stat().st_size != pin['bytes']:
        raise SystemExit(f'Invalid restored table: {path}')
    verified.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': actual,
                     'archive_member': pin['archive_member']})
(root / 'tables-verified.json').write_text(json.dumps(verified, indent=2) + '\n')
print('All three restored table sizes and SHA256s verified before use.', flush=True)
