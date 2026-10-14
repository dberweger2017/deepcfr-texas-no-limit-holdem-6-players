"""Read compact progress from the active qualification root; never starts work."""
import argparse
import json
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
a = p.parse_args()
root = a.root
for label in ('memory', 'memory-locked', 'disk', 'network', 'setup', 'build',
              'rust-tests', 'python-tests', 'build-other-path', 'gates', 'trainer-benchmark'):
    job = root / label
    resources = job / 'resources.jsonl'
    if not resources.exists():
        continue
    rows = [json.loads(s) for s in resources.read_text().splitlines()]
    if not rows:
        continue
    last = rows[-1]
    result = json.loads((job / 'result.json').read_text()) if (job / 'result.json').exists() else None
    print(json.dumps({'job': label, 'status': result or 'running',
        'elapsed_s': round(last['monotonic'] - rows[0]['monotonic'], 1),
        'temperature_c': last['package_c'], 'peak_temperature_c': max(r['package_c'] for r in rows),
        'peak_family_rss_gib': round(max(r['family_rss_bytes'] for r in rows) / 2**30, 3),
        'minimum_available_gib': round(min(r['available_bytes'] for r in rows) / 2**30, 3),
        'swap_growth_bytes': last['swap_used_bytes'] - rows[0]['swap_used_bytes'],
        'maximum_throttle_counter': max(last['throttle_counts'].values(), default=0)}))
for folder in root.glob('memory*instances'):
    for log in sorted(folder.glob('instance-*.log')):
        # The initial failed attempt has huge shrinking-allocation logs. Only
        # read its tail here; originals remain intact for the evidence bundle.
        with log.open('rb') as f:
            f.seek(max(0, log.stat().st_size - 65536))
            raw = f.read().decode(errors='replace')
        clean = re.sub(r'[\x00-\x09\x0b-\x1f]', '', raw)
        print(json.dumps({'instance': str(log.relative_to(root)),
            'locked': 'locked.' in raw, 'observed_FAILURE_count_in_tail': raw.count('FAILURE'),
            'completed_patterns': re.findall(r'([A-Za-z0-9 ]+)\s*:\s*ok', clean),
            'tail': clean[-90:]}))
