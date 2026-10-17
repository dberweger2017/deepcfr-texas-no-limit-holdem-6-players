"""Descriptive key-growth projections, with explicit hash/chunk headroom."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

# The compact store's segment chunks (store.rs): entry records and f64 values.
CHUNK_ENTRIES=1<<13
CHUNK_VALUES=1<<15


def shard_index_bytes(entries):
    """hashbrown index of u32 positions: smallest power-of-two buckets at 7/8 load, plus control bytes."""
    if entries==0:return 0
    buckets=4 if entries<4 else 8 if entries<8 else 1<<max(4,math.ceil(math.log2(entries*8/7)))
    return 4*buckets+buckets+16


def table_bytes(entries,shards,entry_record,payload_per_entry):
    per=math.ceil(entries/shards)
    records=math.ceil(per/CHUNK_ENTRIES)*CHUNK_ENTRIES*entry_record
    values=math.ceil(per*payload_per_entry/8/CHUNK_VALUES)*CHUNK_VALUES*8
    return shards*(shard_index_bytes(per)+records+values)


def analyze(results):
    output=[]
    for run in results:
        if run['name'].startswith('production-'):
            endpoint=run['records'][-2]
            output.append({'name':run['name'],'endpoint':endpoint,'retained_bytes_per_entry_including_scratch':endpoint['retained_training_requested_bytes']/endpoint['entries']})
            continue
        header=run['records'][0]
        rows=[r for r in run['records'] if 'table_requested_bytes' in r]
        a,b=rows[-2:]
        n=b['nodes'];k=b['entries'];reached=b['reached_keys']
        alpha=math.log(k/a['entries'])/math.log(n/a['nodes'])
        # The store's own byte count must equal the allocator's measured table delta.
        assert b['index_bytes']+b['entry_bytes']+b['value_bytes']==b['table_requested_bytes']
        shards=header['shards'];record=header['entry_record_bytes']
        payload=(b['regret_payload_bytes']+b['average_payload_bytes'])/k
        forecasts=[]
        for target in (200_000_000,1_000_000_000,5_000_000_000):
            linear=math.ceil(k*target/n)
            last_slope=max(0,(k-a['entries'])/(n-a['nodes']))
            marginal=math.ceil(k+last_slope*(target-n))
            power=math.ceil(k*(target/n)**alpha)
            table=table_bytes(linear,shards,record,payload)
            # One shard's old index alive while it grows, one shard's sort order while saving,
            # and the measured fixed K50 baseline.
            per=math.ceil(linear/shards)
            peak=table+shard_index_bytes(per)//2+4*per+header['baseline_requested_bytes']
            forecasts.append({'nodes':target,'local_power_fit_entries':power,'last_interval_linear_entries':marginal,'origin_linear_entries':linear,'origin_linear_table_requested_gib':table/2**30,'planning_peak_gib':peak/2**30})
        output.append({'name':run['name'],'endpoint':b,'model_table_bytes_at_endpoint':table_bytes(k,shards,record,payload),'retained_bytes_per_entry':b['table_requested_bytes']/k,'last_interval_entries_per_node':(k-a['entries'])/(n-a['nodes']),'local_power_exponent':alpha,'reached_keys_per_node':reached/n,'forecasts':forecasts})
    return {'kind':'descriptive single-seed growth; forecasts are scenarios, not bounds or saturation claims','models':output}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.write_text(json.dumps(analyze(json.loads(a.results.read_text())),indent=2)+'\n')
