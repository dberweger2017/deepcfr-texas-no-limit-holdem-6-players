"""Read compact progress from the active qualification root; never starts work."""
import argparse
import json
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
a = p.parse_args()
root = a.root
for job in sorted(root.iterdir()):
    if not job.is_dir():
        continue
    label = job.name
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
            header = f.read(1024).decode(errors='replace')
            size = log.stat().st_size
            # Real pattern logs are modest; the abandoned shrinking-allocation
            # attempt can be hundreds of megabytes per instance.
            start = 0 if size <= 16 * 1024**2 else max(0, size - 65536)
            f.seek(start)
            raw = f.read().decode(errors='replace')
        clean = re.sub(r'[\x00-\x09\x0b-\x1f]', '', raw)
        print(json.dumps({'instance': str(log.relative_to(root)),
            'initial_allocation_locked': 'locked.' in header,
            'scan_scope': 'full' if start == 0 else 'tail',
            'log_bytes': size, 'observed_FAILURE_count': raw.count('FAILURE'),
            'completed_patterns': [s.strip() for s in re.findall(
                r'^\s*([A-Za-z0-9 ]+?)\s*:[^\n]*\bok\s*$', clean, re.MULTILINE)],
            'tail': clean[-90:]}))
