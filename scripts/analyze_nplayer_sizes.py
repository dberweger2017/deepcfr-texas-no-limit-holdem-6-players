"""Descriptive key-growth projections, with explicit hash/allocator headroom."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def analyze(results):
    output=[]
    for run in results:
        if run['name'].startswith('production-'):
            endpoint=run['records'][-2]
            output.append({'name':run['name'],'endpoint':endpoint,'retained_bytes_per_entry_including_scratch':endpoint['retained_training_requested_bytes']/endpoint['entries']})
            continue
        rows=[r for r in run['records'] if 'table_requested_bytes' in r]
        a,b=rows[-2:]
        n=b['nodes'];k=b['entries'];reached=b['reached_keys']
        alpha=math.log(k/a['entries'])/math.log(n/a['nodes'])
        payload=(b['regret_payload_bytes']+b['average_payload_bytes'])/k
        boxes=1+sum(b['average_entries_by_street'])/k
        # Large std HashMap tables use 7/8 maximum load and 16 tail control bytes.
        # Validate the layout against the measured allocation delta, not RSS.
        bucket_bytes=lambda slots:slots*65+16
        slots=b['hash_capacity']*8//7
        assert bucket_bytes(slots)+b['regret_payload_bytes']+b['average_payload_bytes']==b['table_requested_bytes']
        forecasts=[]
        for target in (200_000_000,1_000_000_000,5_000_000_000):
            linear=math.ceil(k*target/n)
            last_slope=max(0,(k-a['entries'])/(n-a['nodes']))
            marginal=math.ceil(k+last_slope*(target-n))
            power=math.ceil(k*(target/n)**alpha)
            projected_slots=1<<max(2,math.ceil(math.log2(linear*8/7)))
            table=bucket_bytes(projected_slots)+linear*payload
            # Largest hash-growth overlap within the projected capacity, 32B per-box
            # allocator allowance,
            # sorted checkpoint references and measured fixed K50 baseline.
            peak=table+bucket_bytes(projected_slots//2)+32*boxes*linear+16*linear+run['records'][0]['baseline_requested_bytes']
            forecasts.append({'nodes':target,'local_power_fit_entries':power,'last_interval_linear_entries':marginal,'origin_linear_entries':linear,'origin_linear_table_requested_gib':table/2**30,'planning_peak_gib':peak/2**30})
        output.append({'name':run['name'],'endpoint':b,'retained_bytes_per_entry':b['table_requested_bytes']/k,'last_interval_entries_per_node':(k-a['entries'])/(n-a['nodes']),'local_power_exponent':alpha,'reached_keys_per_node':reached/n,'forecasts':forecasts})
    return {'kind':'descriptive single-seed growth; forecasts are scenarios, not bounds or saturation claims','models':output}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.write_text(json.dumps(analyze(json.loads(a.results.read_text())),indent=2)+'\n')
