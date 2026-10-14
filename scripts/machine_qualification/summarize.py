"""Make compact thermal CSV/summary receipts from the full on-host streams."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
summary = []
for resources in sorted(a.root.glob('*/resources.jsonl')):
    rows = [json.loads(s) for s in resources.read_text().splitlines()]
    if not rows:
        continue
    job = resources.parent.name
    baseline = rows[0]
    clocks = [v / 1000 for row in rows for v in row['clock_khz']]
    delta = {k: rows[-1]['throttle_counts'][k] - v for k, v in baseline['throttle_counts'].items()}
    result_path = resources.parent / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {'status': 'incomplete'}
    row = {'job': job, **result, 'samples': len(rows),
        'started_utc': baseline['utc'], 'last_sample_utc': rows[-1]['utc'],
        'max_sample_gap_seconds': max((y['monotonic'] - x['monotonic'] for x, y in zip(rows, rows[1:])), default=0),
        'min_temperature_c': min(r['package_c'] for r in rows),
        'max_temperature_c': max(r['package_c'] for r in rows),
        'median_temperature_c': statistics.median(r['package_c'] for r in rows),
        'clock_mhz_all_logical_cpus_min_median_max': [min(clocks), statistics.median(clocks), max(clocks)],
        'clock_scope': 'All 16 logical CPUs, including idle siblings; scaling_cur_freq is a sampled reading, not effective-clock residency',
        'sampled_peak_family_rss_bytes': max(r['family_rss_bytes'] for r in rows),
        'minimum_available_bytes': min(r['available_bytes'] for r in rows),
        'maximum_swap_growth_bytes': max(r['swap_used_bytes'] - baseline['swap_used_bytes'] for r in rows),
        'maximum_throttle_counter_delta': max(delta.values(), default=0),
        'raw_stream_bytes': resources.stat().st_size,
        'raw_stream_sha256': hashlib.sha256(resources.read_bytes()).hexdigest()}
    summary.append(row)
    with (a.output / f'{job}-temperatures.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['utc','elapsed_s','package_c','clock_min_mhz','clock_median_mhz','clock_max_mhz','family_rss_gib','available_gib','swap_growth_bytes','max_throttle_delta'])
        for r in rows:
            c = [v / 1000 for v in r['clock_khz']]
            writer.writerow([r['utc'],round(r['monotonic']-baseline['monotonic'],3),r['package_c'],min(c),round(statistics.median(c),3),max(c),round(r['family_rss_bytes']/2**30,4),round(r['available_bytes']/2**30,4),r['swap_used_bytes']-baseline['swap_used_bytes'],max((r['throttle_counts'][k]-v for k,v in baseline['throttle_counts'].items()),default=0)])
(a.output / 'resource-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
