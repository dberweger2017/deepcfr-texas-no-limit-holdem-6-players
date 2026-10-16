"""Compact closed-job resource receipts without reading playing outcomes."""
import argparse
import json
from pathlib import Path
import re


def collect(root):
    guards=[]
    for folder in sorted((root/'guards').iterdir()):
        if not (folder/'result.json').exists():continue
        summary=json.loads((folder/'result.json').read_text())
        first=last=None;count=0;maximum_cpu=maximum_vrm=peak_rss=0
        fan_min=float('inf');fan_max=0;minimum_ram=minimum_disk=float('inf')
        with (folder/'resources.jsonl').open() as f:
            for line in f:
                row=json.loads(line);count+=1
                first=first or row;last=row
                maximum_cpu=max(maximum_cpu,row['package_c']);maximum_vrm=max(maximum_vrm,row['vrm_c'])
                peak_rss=max(peak_rss,row['family_rss_bytes'])
                fan_min=min(fan_min,row['fan1_rpm']);fan_max=max(fan_max,row['fan1_rpm'])
                minimum_ram=min(minimum_ram,row['available_bytes']);minimum_disk=min(minimum_disk,row['free_disk_bytes'])
        if not first:raise ValueError('closed guard has no readings')
        guards.append({'name':folder.name,**summary,'readings':count,'first_utc':first['utc'],
            'last_utc':last['utc'],'cpu_max_c':maximum_cpu,'vrm_max_c':maximum_vrm,
            'fan1_min_rpm':fan_min,'fan1_max_rpm':fan_max,'rss_peak_bytes':peak_rss,
            'minimum_available_bytes':minimum_ram,'minimum_disk_bytes':minimum_disk,
            'swap_growth_bytes':last['swap_used_bytes']-first['swap_used_bytes'],
            'cpu_affinity':'0-3','thread_limit':4})
    native=(root/'validation/0.log').read_text()
    python=(root/'validation/1.log').read_text()
    return {'guards':guards,'native_test_summaries':re.findall(r'test result:.*',native),
            'native_passed':sum(map(int,re.findall(r'test result: ok\. (\d+) passed',native))),
            'python_summary':python.splitlines()[-1],
            'hu_build':json.loads((root/'hu-build.json').read_text()),
            'hu250':json.loads((root/'gate-results/checkpoints.json').read_text()),
            'hu100':json.loads((root/'hu-identities/receipt.json').read_text())['hu100']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    with a.out.open('x') as f:json.dump(collect(a.root),f,indent=2);f.write('\n')
