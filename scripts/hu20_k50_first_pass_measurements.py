"""Full stored-state audit and outcome-blind density for the K50 first pass."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re

from scripts.hu20_k50_first_pass_assets import write, check
from src.blueprint.abstraction import HU20_EQUITY_SCHEMA, HU20_UNCAPPED_SCHEMA
from src.diagnostics.cfr_average import audit, checked_row, average_rule
from src.policies.files import file_hash


def quantiles(histogram):
    count=sum(histogram.values())
    if not count:
        return {'count':0,'mean':None,'quantiles':None}
    ordered=sorted(histogram.items())
    def select(index):
        cumulative=0
        for value,n in ordered:
            cumulative+=n
            if cumulative>index:return value
        raise ValueError('Quantile histogram accounting differs')
    def percentile(p):
        point=p*(count-1);lower=int(point);upper=min(lower+1,count-1)
        return select(lower)+(point-lower)*(select(upper)-select(lower))
    return {'count':count,'mean':sum(value*n for value,n in ordered)/count,
            'zero_fraction':histogram.get(0,0)/count,
            'quantiles':{name:percentile(p) for name,p in [('min',0),('p10',.1),('p25',.25),('median',.5),
                ('p75',.75),('p90',.9),('p99',.99),('max',1)]}}


def metadata(checkpoint, current, average, telemetry=None, archive_metadata=None):
    with gzip.open(checkpoint,'rt') as source:
        header=json.loads(source.readline())
    actual=None
    if telemetry is not None:
        rows=[json.loads(line) for line in telemetry.read_text().splitlines()]
        matching=[row for row in rows if row['checkpoint_sha256']==file_hash(checkpoint)]
        if len(matching)!=1:raise ValueError('Missing/ambiguous native telemetry')
        actual=matching[0]['completed_nodes']
    elif archive_metadata is not None:
        for path in archive_metadata.rglob('*.txt'):
            for match in re.finditer(r'milestone 5000000000 iterations (\d+) nodes (\d+) entries (\d+)',path.read_text()):
                iteration,nodes,entries=map(int,match.groups())
                if iteration==header['iteration']:
                    if actual is not None and actual!=nodes:raise ValueError('Conflicting archived node count')
                    actual=nodes
    if actual is None or actual<5000000000:
        raise ValueError('No verified actual 5B node count')
    return {'name':checkpoint.stem,'seed':header['config']['seed'],'iteration':header['iteration'],
        'strategy':'average','format':'holdem-hu20-stored-cfr-average-diagnostic-v1',
        'abstraction':header['abstraction'],'path':str(average),'bytes':average.stat().st_size,
        'sha256':file_hash(average),'checkpoint_sha256':file_hash(checkpoint),
        'current_sha256':file_hash(current),'checkpoint_path':str(checkpoint),'current_path':str(current),
        'actual_nodes':actual,'requested_nodes':5000000000}


def inspect(checkpoint,current,average,out,*,telemetry=None,archive_metadata=None,tables=None):
    model=metadata(checkpoint,current,average,telemetry,archive_metadata)
    schema=model['abstraction']
    if schema not in (HU20_UNCAPPED_SCHEMA,HU20_EQUITY_SCHEMA):raise ValueError('Unexpected HU20 schema')
    if schema==HU20_EQUITY_SCHEMA:
        if tables is None:raise ValueError('K50 requires explicit tables')
        model['card_buckets']=str(tables)
    spec={'name':model['name'],'seed':model['seed'],'iteration':model['iteration'],
          'checkpoint_sha256':model['checkpoint_sha256'],'sha256':model['current_sha256']}
    receipt=audit(checkpoint,current,average,spec,model['sha256'],expected_schema=schema)
    counts=Counter()
    with gzip.open(checkpoint,'rt') as source:
        header=json.loads(source.readline())
        for line in source:
            _,_,_,_,_,visits=checked_row(json.loads(line),header['iteration'],average_rule(header)=='traverser-reach')
            counts[visits]+=1
    density=quantiles(counts)
    if density['count']!=receipt['all_nodes_verified']:raise ValueError('Audit/density counts differ')
    write(out,{'status':'verified','model':model,'audit':receipt,'density':density,
        'street_mapping':'Stored keys are irreversible hashes without street metadata. Street visit quantiles require independently classified reached keys; no full-table street allocation is inferred.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','current','average','out'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('telemetry','archive-metadata','tables'):p.add_argument('--'+key,type=Path)
    a=p.parse_args();inspect(a.checkpoint,a.current,a.average,a.out,telemetry=a.telemetry,
        archive_metadata=a.archive_metadata,tables=a.tables)


if __name__=='__main__':main()
