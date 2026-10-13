"""Compact resource readout from all retained five-second M4 samples."""
import json
from pathlib import Path
from scripts.evaluate_hu100_direct import put


def summarize(stage):
    baseline = json.loads((stage/'baseline.json').read_text())
    rows = []
    operations = {}
    for directory in sorted((stage/'operations').iterdir()):
        samples = [json.loads(line) for line in (directory/'resources.jsonl').read_text().splitlines()] if (directory/'resources.jsonl').exists() else []
        if not samples: continue
        def stats(values):
            return {'samples':len(values),'peak_family_rss_bytes':max(r['family_rss_bytes'] for r in values),
                'minimum_free_percent':min(r['free_percent'] for r in values),
                'maximum_swap_growth_bytes':max(r['swap_bytes']-baseline['swap_bytes'] for r in values),
                'minimum_disk_free_bytes':min(r['disk_free_bytes'] for r in values),
                'pressure_levels':sorted({r['pressure_level'] for r in values}),
                'readings_over_each_line':{k:sum(r['breaches'][k] for r in values) for k in values[0]['breaches']},
                'maximum_observed_held_seconds':{k:max(r.get('held_seconds',{}).get(k,0) for r in values) for k in values[0]['breaches']}}
        operations[directory.name] = stats(samples)
        if (directory/'receipt.json').exists(): operations[directory.name]['receipt'] = json.loads((directory/'receipt.json').read_text())
        rows.extend(samples)
    return {'baseline':baseline,'all_operations':stats(rows),'operations':operations,
            'stopped':(stage/'STOP.json').exists(),'stop_record':json.loads((stage/'STOP.json').read_text()) if (stage/'STOP.json').exists() else None,
            'scope':'sampled whole-family peaks; retained streams contain every reading'}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--stage',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); put(a.out,summarize(a.stage))
