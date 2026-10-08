"""Conservative advisory estimates from measured export/audit and retained save receipts.

This does not modify historical campaign guards or authorize a new experiment.
"""
import argparse
import json
from math import isfinite
from pathlib import Path

GIB = 1024**3


def estimate(measurements, training, entries, *, rss_gib=10):
    required = {'export', 'audit', 'extract'}
    if type(entries) is not int or entries <= 0 or not isfinite(rss_gib) or rss_gib <= 0:
        raise ValueError('Positive finite budget and integer entry count required')
    rows = [r for r in measurements if r['phase'] in ('after', 'final') and r['model'].startswith('hu100-')]
    if not rows or {r['operation'] for r in rows} != required:
        raise ValueError('Complete HU100 export, audit and extraction measurements required')
    largest = max(r['entries'] for r in rows)
    if entries > largest * 2:
        raise ValueError('Advisory extrapolation limited to 2x largest measured HU100 table')
    for row in rows:
        if any(not isfinite(row[k]) or row[k] <= 0 for k in
               ('entries', 'seconds', 'sampled_peak_family_rss_bytes', 'kernel_command_peak_rss_bytes')):
            raise ValueError('Invalid measured resource cost')
    if not training:
        raise ValueError('Training/save measurements required')
    for row in training:
        if any(not isfinite(row[k]) or row[k] <= 0 for k in
               ('process_rss_bytes_after_save', 'write_seconds', 'checkpoint_bytes')) or row['diagnostics']['entries'] <= 0:
            raise ValueError('Invalid training/save receipt')
    # No claim of a memory plateau from six finite samples. Scale peak costs
    # conservatively even for the fixed-cache paths, then apply 2x plus 10%.
    scale = max(1, entries / largest)
    peak = max(max(r['sampled_peak_family_rss_bytes'], r['kernel_command_peak_rss_bytes']) for r in rows)
    tool_reserve = 2.2 * peak * scale
    training_per_key = max(r['process_rss_bytes_after_save'] / r['diagnostics']['entries'] for r in training)
    serialization = 32 * entries + 64 * 1024**2
    training_reserve = 2.2 * (training_per_key * entries + serialization) + 256 * 1024**2
    save_seconds = 2 * max(r['write_seconds'] / r['diagnostics']['entries'] for r in training) * entries
    tools_seconds = 2 * sum(max(r['seconds'] / r['entries'] for r in rows if r['operation'] == op)
                            for op in sorted(required)) * entries
    checkpoint_bytes = 2 * max(r['checkpoint_bytes'] / r['diagnostics']['entries'] for r in training) * entries
    # A retained checkpoint/current/average set, atomic checkpoint temporary,
    # duplicate archive and sort/index scratch. Also retain inputs independently.
    disk = 8 * checkpoint_bytes + 1024 * entries
    return {'advisory_only': True, 'entry_ceiling': entries, 'largest_measured_hu100_entries': largest,
            'extrapolation_factor': entries / largest, 'memory_allowance': 2.2,
            'export_audit_extract_reserve_bytes': tool_reserve,
            'training_save_reserve_bytes': training_reserve,
            'rss_ceiling_bytes': rss_gib * GIB,
            'fits_rss_estimate': max(tool_reserve, training_reserve) < rss_gib * GIB,
            'checkpoint_save_reserve_seconds': save_seconds,
            'one_set_export_audit_extract_reserve_seconds': tools_seconds,
            'one_set_disk_reserve_bytes': disk,
            'limitations': 'RSS sampling misses transients and counts shared pages repeatedly; historical training RSS is after save, not a family peak. Training hash-map reallocations, table growth, scratch I/O, filesystem cache and future node-to-entry ratios remain uncertain. Fresh guarded timing/save pilot and source-bound admission required; no 1B/10B forecast.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurements', type=Path, required=True)
    p.add_argument('--training-closeout', type=Path, required=True)
    p.add_argument('--entries', type=int, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    closeout = json.loads(a.training_closeout.read_text())
    training = next(v for v in closeout.values() if isinstance(v, list) and v
                    and isinstance(v[0], dict) and 'process_rss_bytes_after_save' in v[0])
    result = estimate(json.loads(a.measurements.read_text()), training, a.entries)
    with a.out.open('x') as target:
        json.dump(result, target, indent=2, sort_keys=True); target.write('\n')


if __name__ == '__main__':
    main()
