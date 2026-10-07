"""Join checkpoint receipts to continuous samples and explicit audit status, once at closeout."""
import argparse
import json
from pathlib import Path

from scripts.prepare_native_hu_campaign import file_hash


def summarize(telemetry_paths,resource_paths,audit_paths):
    samples=sorted([json.loads(line) for path in resource_paths for line in path.read_text().splitlines()],
                   key=lambda r:r['unix_seconds'])
    audits={}
    for path in audit_paths:
        a=json.loads(path.read_text())
        if a['status']!='verified': continue
        for name,spec in a['files'].items():
            p=Path(name)
            if p.stat().st_size!=spec['bytes'] or file_hash(p)!=spec['sha256']:
                raise ValueError('Audited files changed; cannot label checkpoint verified')
        audits[a['audit']['checkpoint_sha256']]={'path':str(path),'sha256':file_hash(path)}
    rows=[]
    for path in telemetry_paths:
        for line in path.read_text().splitlines():
            r=json.loads(line); cp=Path(r['path'])
            if cp.stat().st_size!=r['checkpoint_bytes'] or file_hash(cp)!=r['checkpoint_sha256']:
                raise ValueError('Saved checkpoint differs from atomic receipt')
            near=[s for s in samples if r['write_started']-6 <= s['unix_seconds'] <= r['write_finished']+6]
            before=[s for s in samples if s['unix_seconds']<=r['write_finished']]
            if not near or not before: raise ValueError('Missing checkpoint resource telemetry')
            s=min(near,key=lambda s:abs(s['unix_seconds']-r['write_finished']))
            rows.append({**r,'audit_status':'verified' if r['checkpoint_sha256'] in audits else 'unaudited',
                'audit':audits.get(r['checkpoint_sha256']), 'resource_sample':s,
                'resource_sample_offset_seconds':s['unix_seconds']-r['write_finished'],
                'sampled_peak_aggregate_rss_bytes_so_far':max(s['aggregate_job_rss_bytes'] for s in before)})
    rows.sort(key=lambda r:r['completed_nodes'])
    previous=None
    for r in rows:
        r['entries_growth_since_previous_checkpoint']=None if previous is None else r['diagnostics']['entries']-previous['diagnostics']['entries']
        previous=r
    return {'status':'summarized','checkpoints':rows,'sampling_interval_seconds':5,
        'limitations':'Nearby family/swap/disk samples can miss brief peaks; native process RSS is sampled after each save. Saved means atomic bytes verified, not policy audit.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--telemetry',type=Path,action='append',required=True)
    p.add_argument('--resources',type=Path,action='append',required=True)
    p.add_argument('--audit',type=Path,action='append',default=[])
    p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    if a.out.exists(): raise FileExistsError('Preserve prior summary')
    result=summarize(a.telemetry,a.resources,a.audit)
    with a.out.open('x') as f: json.dump(result,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')


if __name__=='__main__': main()
