"""Streaming independent A/B arithmetic and resource ledger; no model loading."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path

from scripts.hu20_platform_pilot import write
from src.arena.report import estimate
from src.diagnostics.saved_hu20 import file_hash


def utc(epoch):
    return datetime.fromtimestamp(epoch,timezone.utc).isoformat()


def summarize_records(path):
    values={};positions={};tails=defaultdict(Counter);visits=defaultdict(Counter);coordinates={}
    with gzip.open(path,'rt') as source:
        for line in source:
            row=json.loads(line)
            if row['status']!='complete' or not row.get('native_replay_verified'):
                raise ValueError('Incomplete or unreplayed hand in completed evidence')
            key=(row['version'],row['panel'],row['block'],row['rotation'])
            if key in values:raise ValueError('Duplicate hand coordinate')
            values[key]=row['target_chips'];positions[key]='button' if row['rotation']==row['button'] else 'big_blind'
            common=key[1:];world=(row['deal_seed'],row['root_seed'],row['button'])
            if common in coordinates and coordinates[common]!=world:raise ValueError('Unpaired deal')
            coordinates[common]=world
            tails[key[:2]].update(row['tails']['counts'])
            group=row['tails']['first_large_raise_response']
            tails[key[:2]]['first_large_'+group+'_hands']+=1
            tails[key[:2]]['first_large_'+group+'_whole_hand_chips']+=row['target_chips']
            for action in row['actions']:
                if action['logical_player']!=0:continue
                count=action['observation']['visits'];band='0' if count==0 else '1-9' if count<10 else '10-99' if count<100 else '100+'
                visits[(*key[:2],action['street'])][band]+=1
    panels={}
    for panel in sorted({key[1] for key in values}):
        a={(b,r):v for (version,p,b,r),v in values.items() if version=='v1' and p==panel}
        b={(i,r):v for (version,p,i,r),v in values.items() if version=='v2' and p==panel}
        if set(a)!=set(b):raise ValueError('Missing arm coordinate')
        blocks=sorted({i for i,r in a})
        if blocks!=list(range(len(blocks))) or len(a)!=2*len(blocks):raise ValueError('Missing rotation/block')
        record={'blocks':len(blocks),'hands_per_arm':len(a),'per_arm':{},'positions':{}}
        for version,data in [('v1',a),('v2',b)]:
            record['per_arm'][version]={'bb_per_100':estimate([(data[(i,0)]+data[(i,1)])/2 for i in blocks]),
                                       'tails':dict(tails[(version,panel)]),'visit_bands_by_street':{}}
            for street in ('preflop','flop','turn','river'):
                counts=visits[(version,panel,street)];total=sum(counts.values())
                record['per_arm'][version]['visit_bands_by_street'][street]={'decisions':total,'counts':dict(counts),
                   'share':{k:counts[k]/total if total else None for k in ('0','1-9','10-99','100+')}}
        differences=[(b[(i,0)]+b[(i,1)]-a[(i,0)]-a[(i,1)])/2 for i in blocks]
        record['paired_v2_minus_v1_bb_per_100']=estimate(differences);record['paired_block_changes']=differences
        for position in ('button','big_blind'):
            coords=[(i,i%2 if position=='button' else 1-i%2) for i in blocks]
            record['positions'][position]={'v1':estimate([a[c] for c in coords]),'v2':estimate([b[c] for c in coords]),
                        'paired_v2_minus_v1_bb_per_100':estimate([b[c]-a[c] for c in coords])}
        panels[panel]=record
    return {'panels':panels,'hands':len(values),'raw_sha256':file_hash(path)}


def resources(folder):
    worker=json.loads((folder/'worker.json').read_text());training=json.loads((folder/'training/summary.json').read_text())
    windows=[(m['save_started'],m['save_finished']) for m in training['checkpoint_milestones']]
    if 'export_started' in training:windows.append((training['export_started'],training['export_finished']))
    peaks=Counter();minimum=None;swap=0;maximum=None
    with (folder/'resources.jsonl').open() as stream:
        for line in stream:
            r=json.loads(line);t=r['time'];phase=r['phase']
            if phase=='training':phase='serialization' if any(start<=t<=end for start,end in windows) else 'training-or-stream-hashing'
            peaks[phase]=max(peaks[phase],r['owned_rss_bytes']);swap=max(swap,r['swap_growth_bytes'])
            minimum=r['free_disk_bytes'] if minimum is None else min(minimum,r['free_disk_bytes'])
            maximum=r['owned_rss_bytes'] if maximum is None else max(maximum,r['owned_rss_bytes'])
    previous_nodes=previous_seconds=0;growth=[]
    for m in training['checkpoint_milestones']:
        record={k:m[k] for k in ('path','completed_nodes','entries','training_seconds','nodes_per_second','save_seconds','hash_seconds','checkpoint','entries_by_street','traverser_visits_by_street')}
        record['segment_nodes_per_second']=(m['completed_nodes']-previous_nodes)/(m['training_seconds']-previous_seconds)
        record['visits_per_key_by_street']={s:m['traverser_visits_by_street'][s]/n for s,n in m['entries_by_street'].items()}
        previous_nodes=m['completed_nodes'];previous_seconds=m['training_seconds'];growth.append(record)
    return {'worker_status':worker['status'],'source_sha':worker['source_sha'],'memory_limit_bytes':worker['memory_limit_bytes'],
            'sampled_owned_rss_peaks_bytes':dict(peaks),'sampled_owned_rss_peak_bytes':maximum,'minimum_free_disk_bytes':minimum,
            'maximum_swap_growth_bytes':swap,'sampling_caveat':'1-second samples; process ru_maxrss separately retains brief high-water peaks',
            'training_process_peak_rss_bytes':training['peak_rss_bytes'],'training_summary':training,'growth':growth}


def run(root,out):
    out.mkdir(parents=True,exist_ok=False);lease=json.loads((root/'lease.json').read_text());pods=json.loads((root/'pods.json').read_text())
    report={'source_sha':lease['source_sha'],'per_seed':{},'aggregate':{},'lease':lease,'pods':pods,
            'interval_scope':'paired deal-block Student-t, conditional on three saved lineages; exploratory unadjusted95%; same deals across seeds are averaged within block'}
    for pod in pods:
        folder=root/str(pod['seed'])/'results/work';evaluation=json.loads((folder/'evaluation/summary.json').read_text())
        if pod['status']!='verified' or evaluation['status']!='complete':raise ValueError('Incomplete seed; publish its original failure separately')
        actual=summarize_records(folder/'evaluation/hands.jsonl.gz')
        for p in evaluation['panels']:
            recomputed=actual['panels'][p['panel']]
            if (p['paired_v2_minus_v1_bb_per_100']!=recomputed['paired_v2_minus_v1_bb_per_100'] or p['per_arm']!=
                {v:{k:d[k] for k in ('bb_per_100','tails')} for v,d in recomputed['per_arm'].items()}):
                raise ValueError('Linux and independent raw arithmetic differ')
        report['per_seed'][str(pod['seed'])]={**actual,'resources':resources(folder),'evaluation':evaluation,
             'inputs':{v:json.loads((folder/f'evaluation/{v}-input.json').read_text()) for v in ('v1','v2')}}
    seeds=list(report['per_seed']);panels=set.intersection(*(set(report['per_seed'][s]['panels']) for s in seeds))
    for panel in sorted(panels):
        changes=[report['per_seed'][s]['panels'][panel]['paired_block_changes'] for s in seeds]
        if len({len(c) for c in changes})!=1:raise ValueError('Different paired block counts across seeds')
        averaged=[sum(x)/len(seeds) for x in zip(*changes,strict=True)]
        report['aggregate'][panel]={'paired_v2_minus_v1_bb_per_100':estimate(averaged),
            'per_seed_changes':{s:report['per_seed'][s]['panels'][panel]['paired_v2_minus_v1_bb_per_100']['bb_per_100'] for s in seeds},
            'seeds':seeds,'do_not_pool_rotations_or_same_deal_across_lineages_as_independent':True}
    if (root/'operator-finished.json').exists():report['cost_ledger']=json.loads((root/'operator-finished.json').read_text())
    write(out/'summary.json',report)
    write(out/'manifest.json',{str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in root.rglob('*')
        if p.is_file() and p.suffix in ('.json','.jsonl','.gz','.log','.txt') and 'results' in p.relative_to(root).parts})
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.root,a.out)
