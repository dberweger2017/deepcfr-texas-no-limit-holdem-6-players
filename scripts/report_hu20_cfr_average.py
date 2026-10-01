"""Derive separate-panel current/average tables from sealed run summaries."""
import argparse
from collections import Counter,defaultdict
import csv
import gzip
import json
from pathlib import Path

from src.arena.report import estimate


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)


def display(result):
    return f"{result['bb_per_100']:+.2f} [{result['ci95'][0]:+.2f}, {result['ci95'][1]:+.2f}]"


def generate(directory,out):
    out.mkdir(parents=True,exist_ok=False);s=json.loads((directory/'summary.json').read_text())
    if s['status']!='complete':raise ValueError('Do not report an incomplete campaign as complete')
    panels=[];changes=[];parts=[];coverage=[]
    for p in s['panels']:
        for position,result in [('overall',p['overall']),*p['positions'].items()]:
            panels.append({'seed':p['seed'],'strategy':p['strategy'],'panel':p['panel'],'position':position,
                'bb_per_100':result['bb_per_100'],'ci95_low':result['ci95'][0],'ci95_high':result['ci95'][1],
                'paired_blocks':result['blocks'],**({f'all_positions_{k}':v for k,v in p['counts'].items()} if position=='overall' else {})})
        coverage.append({'seed':p['seed'],'strategy':p['strategy'],'panel':p['panel'],'hands':p['hands'],
            'distinct_target_keys':p['distinct_target_keys'],**{f'distinct_{k}':v for k,v in p['distinct_keys_by_street'].items()},
            **p['coverage'],'limited_lbr_decisions':p['limited_lbr_decisions']})
        for response,values in p['whole_hand_partitions'].items():parts.append({'seed':p['seed'],'strategy':p['strategy'],'panel':p['panel'],'first_large_response':response,**values})
    for c in s['changes']:
        for position,result in [('overall',c['overall']),*c['positions'].items()]:
            changes.append({'seed':c['seed'],'panel':c['panel'],'position':position,'average_minus_current_bb_per_100':result['bb_per_100'],
                            'ci95_low':result['ci95'][0],'ci95_high':result['ci95'][1],'paired_blocks':result['blocks']})
    write_csv(out/'panels.csv',panels);write_csv(out/'paired-changes.csv',changes)
    write_csv(out/'coverage.csv',coverage);write_csv(out/'whole-hand-partitions.csv',parts)
    exposure=defaultdict(Counter)
    for model in s['plan']['models']:
        with gzip.open(directory/(model['name']+'.hands.jsonl.gz'),'rt') as f:
            for line in f:
                r=json.loads(line)
                for a in r['actions']:
                    if a['logical_player']:continue
                    c=exposure[r['seed'],r['strategy'],r['panel'],a['average_mass_status']]
                    c['decisions']+=1;c['large_opportunities']+=a['observation']['large_raise_opportunity']
                    if a['kind']=='raise':
                        selected=next(m for m in a['observation']['menu'] if m['kind']=='raise' and m['raise_to']==a['raise_to'])
                        c['large_actions']+=selected['rival_call_amount']>=800
    write_csv(out/'large-action-coverage.csv',[{'seed':seed,'strategy':strategy,'panel':panel,'mass_status':status,**dict(counts)}
              for (seed,strategy,panel,status),counts in sorted(exposure.items())])
    levels=[]
    for panel in s['plan']['panels']:
        for strategy in ('current','average'):
            group=[p for p in s['panels'] if p['panel']==panel['name'] and p['strategy']==strategy]
            if len(group)!=3 or any(p['blocks']!=group[0]['blocks'] for p in group):raise ValueError('Three-lineage level pairing differs')
            levels.append({'panel':panel['name'],'strategy':strategy,'overall':estimate([sum(v)/3 for v in zip(*(p['paired_block_chips'] for p in group))])})
    (out/'three-lineage-levels.json').write_text(json.dumps(levels,indent=2,sort_keys=True)+'\n')
    lines=['# Stored-average comparison tables','', 'Exploratory paired 95% intervals, conditional on three fixed lineages; each panel separate.',
           'Positive changes favor the diagnostic average. Known zero-mass keys are uniform and counted separately from missing keys.','',
           '## Three-lineage comparisons','', '| Panel | Current BB/100 [95%] | Average BB/100 [95%] | Paired average−current [95%] |',
           '| --- | --- | --- | --- |']
    for panel in s['plan']['panels']:
        name=panel['name'];current=next(r['overall'] for r in levels if r['panel']==name and r['strategy']=='current')
        average=next(r['overall'] for r in levels if r['panel']==name and r['strategy']=='average')
        delta=next(r['average_minus_current'] for r in s['three_lineage_changes'] if r['panel']==name)
        lines.append(f'| {name} | {display(current)} | {display(average)} | {display(delta)} |')
    lines+=['','## Per-lineage paired changes','', '512 hands/policy/panel except LBR 256; position intervals are in paired-changes.csv.',
            '| Panel | Seed | Overall average−current BB/100 [95%] | BB / button change point |',
            '| --- | --- | --- | --- |']
    for c in s['changes']:
        lines.append(f"| {c['panel']} | {c['seed']} | {display(c['overall'])} | {c['positions']['big_blind']['bb_per_100']:+.2f} / {c['positions']['button']['bb_per_100']:+.2f} |")
    lines+=['','## Coverage and adverse tails','', 'Counts are completed hands, target decisions or large actions/opportunities, as labelled.',
            'Missing-key fallback excludes known zero-mass uniform decisions. Whole-hand partitions are not bet EV.','',
            '| Panel | Seed | Extraction | Large actions / opportunities | Rival folds / continues | Stack wins / losses | Missing / decisions | Zero-mass / decisions | Distinct keys |',
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for p in s['panels']:
        c=p['counts'];zero=str(p['coverage'].get('zero_mass',0)) if p['strategy']=='average' else 'n/a'
        lines.append(f"| {p['panel']} | {p['seed']} | {p['strategy']} | {c['large_actions']} / {c['large_opportunities']} | {c['rival_folds']} / {c['rival_continuations']} | {c['full_stack_wins']} / {c['full_stack_losses']} | {c['fallback']} / {c['target_decisions']} | {zero} / {c['target_decisions']} | {p['distinct_target_keys']} |")
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    generate(a.directory,a.out)


if __name__=='__main__':main()
