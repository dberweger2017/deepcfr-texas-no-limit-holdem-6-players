"""Summarize completed retention gates before choosing calibration concurrency."""

import argparse
import fcntl
import json
from pathlib import Path
from statistics import mean

from scripts.evaluate_hu100_direct import put
from src.blueprint.river_evidence import file_pin


def report(root, three_summary=None):
    parity=json.loads((root/'parity-01/summary.json').read_text())
    cost=json.loads((root/'cost-01/summary.json').read_text())
    if parity['status']!='exact-parity' or not all(cost[k] for k in
            ('latency_target_met','memory_target_met','telemetry_target_met','all_search_work_completed')):
        raise ValueError('Unqualified sequential gate')
    choices=[]
    times=[json.loads(p.read_text())['complete_seconds'] for p in (root/'cost-01').glob('*/timing.json')]
    if len(times)!=24:raise ValueError('Incomplete sequential costs')
    sequential=mean(times)
    choices.append(dict(workers=1,mean_decision_seconds=sequential,
        decisions_per_second=24/sum(times),summary_sha256=file_pin(root/'cost-01/summary.json')['sha256'],summary_path='cost-01/summary.json'))
    for n in (2,3):
        path=three_summary if n==3 and three_summary is not None else root/f'concurrency-{n}/summary.json'
        if not path.exists():continue
        value=json.loads(path.read_text())
        if not value.get('memory_target_met') or not value.get('outputs_match_parity'):continue
        times=[json.loads(p.read_text())['complete_seconds'] for p in path.parent.glob('*/timing.json')]
        if len(times)!=24:raise ValueError('Incomplete concurrent costs')
        choices.append(dict(workers=n,mean_decision_seconds=mean(times),
            decisions_per_second=value['decisions_per_second'],summary_sha256=file_pin(path)['sha256'],summary_path=path.relative_to(root).as_posix()))
    selected=max(choices,key=lambda row:(row['decisions_per_second'],-row['workers']))
    records=[]
    for path in (root/'cost-01').glob('*/native/*/receipt.json'):
        records.append(json.loads(path.read_text()))
    bytes_now=sum(p.stat().st_size for p in (root/'cost-01').rglob('*') if p.is_file())
    result=dict(status='qualified-engineering',cost=cost,choices=choices,selected_workers=selected['workers'],
        sequential_mean_seconds=sequential,chosen_concurrent_mean_seconds=selected['mean_decision_seconds'],
        cost_evidence_bytes=bytes_now,native_calls=len(records),
        sampled_complete_native_bodies=sum(r['profile_retention']['body_retained'] for r in records),
        request_and_header_compression='lossless; original hashes retained',no_calibration_hands=True)
    put(root/'engineering-readout.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--three-summary',type=Path)
    args=parser.parse_args()
    lock=(Path.home()/'Local/.hu20-m4-research.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    report(args.root,args.three_summary)
