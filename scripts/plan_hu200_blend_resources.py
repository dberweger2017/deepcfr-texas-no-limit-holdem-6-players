"""Project the remaining blend stages after verified input retrieval and union scan."""
import argparse
import json
from math import ceil
from pathlib import Path
import shutil
from time import time

from scripts.evaluate_hu100_direct import put
from src.policies.files import file_hash

GIB = 1024**3


def project(scan, sample, *, trace_bytes=4*GIB):
    union = scan['union_keys']
    rows = scan['output']['rows']
    if not union or not rows or rows > union:
        raise ValueError('A nonempty verified compression sample is required')
    outputs = {name: ceil(pin['bytes']/rows*union*1.5)+1024**2
               for name, pin in sample.items()}
    if set(outputs) != {'checkpoint', 'current', 'average'}:
        raise ValueError('Measure the sample checkpoint and both existing-exporter outputs')
    arrays = 35*union+8*scan['action_slots']+768
    if arrays != scan['cache_array_bytes']:
        raise ValueError('Cache projection differs from union dimensions')
    cache = arrays+16384
    seed_caches = [35*s['entries']+8*s['action_slots']+768+16384 for s in scan['inputs']]
    # This is the existing WITHOUT ROWID ledger's stored TEXT payload plus
    # record/page slack. Counts follow the union's exact menu distribution.
    ledger = 0
    for menu in scan['menus']:
        names = menu['names']
        row = json.dumps([names, [1/len(names)]*len(names)], separators=(',', ':'))
        ledger += menu['keys']*(32+len(row.encode())+16)
    audit_workspace = ceil(1.5*ledger)+128*1024**2
    retained = sum(outputs.values())
    evidence = retained+trace_bytes+256*1024**2
    peaks = dict(merge_and_exports=retained,
                 audit=retained+audit_workspace,
                 play=retained+cache+max(seed_caches)+trace_bytes,
                 # All policy caches are inactive and retired after complete replay/reproduction.
                 archive=evidence*2)
    return dict(projected_outputs=outputs, blend_cache_bytes=cache,
                seed_cache_bytes=seed_caches, audit_workspace_bytes=audit_workspace,
                trace_reserve_bytes=trace_bytes, future_peak_additional_bytes=peaks,
                required_free_bytes=max(peaks.values())+18*GIB,
                disk_floor_bytes=16*GIB, margin_bytes=2*GIB,
                compression_multiplier=1.5, audit_payload_multiplier=1.5,
                checkpoint_and_cache_workspace='one streamed output; no sort or full-table RAM copy',
                archive_workspace='one ZIP alongside retained originals; same-filesystem atomic publication required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--scan', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--sample', type=Path, required=True, help='JSON pins for checkpoint/current/average sample')
    parser.add_argument('--sample-audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads((args.base/'all-inputs-verified.json').read_text())
    scan = json.loads(args.scan.read_text())
    sample = json.loads(args.sample.read_text())
    audited = json.loads(args.sample_audit.read_text())
    config_sha = file_hash(args.config)
    if (not inputs['all_seed_averages_reproduced']
            or inputs['configuration_sha256'] != config_sha
            or scan['configuration_sha256'] != config_sha):
        raise ValueError('All original average hashes must reproduce before admission')
    if (sample['checkpoint']['sha256'] != scan['output']['sha256']
            or audited['status'] != 'verified'
            or audited['all_nodes_verified'] != scan['output']['rows']
            or audited['checkpoint_sha256'] != sample['checkpoint']['sha256']
            or audited['current_sha256'] != sample['current']['sha256']
            or audited['average_sha256'] != sample['average']['sha256']):
        raise ValueError('Existing-exporter compression sample audit differs')
    for pin in sample.values():
        path = Path(pin['path'])
        if path.stat().st_size != pin['bytes'] or file_hash(path) != pin['sha256']:
            raise ValueError('Compression sample changed')
    projection = project(scan, sample)
    free = shutil.disk_usage(args.base).free
    put(args.out, dict(projection, status='admitted' if free >= projection['required_free_bytes'] else 'capacity-stop',
                       at=time(), free_bytes=free, configuration_sha256=config_sha,
                       scan_report=str(args.scan.resolve()), scan_sha256=file_hash(args.scan),
                       all_seed_averages_reproduced=True, scope='first union/export/cache/audit/archive capacity gate'))


if __name__ == '__main__':
    main()
