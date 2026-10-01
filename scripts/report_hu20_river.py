"""Native replay audit and paired tables from closed HU20 river evidence."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
from math import sqrt
from statistics import mean, stdev
from time import perf_counter
from scipy.stats import t

from scripts.evaluate_hu20_cfr_average import summarize
from src.arena.runner import public_events
from src.arena.schedule import digest, stream_seed
from src.diagnostics.saved_hu20 import file_hash
from src.diagnostics.stackoff_tails import snapshot, hand_tails
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind
from src.blueprint.abstraction import Choice
from src.blueprint.hu20_river import peak_rss


def audit(directory):
    summary=json.loads((directory/'summary.json').read_text());plan=summary['plan']
    if summary['status']!='complete' or digest(plan)!=summary['plan_sha256']:
        raise ValueError('Incomplete or altered experiment')
    for name,spec in json.loads((directory/'manifest.json').read_text()).items():
        if (directory/name).stat().st_size!=spec['bytes'] or file_hash(directory/name)!=spec['sha256']:
            raise ValueError('Evidence bytes differ')
    expected={(m['name'],p['name'],s,b,r) for m in plan['models'] for p in plan['panels']
              for s in ('current','average') for b in range(p['blocks']) for r in (0,1)}
    rows=[];seen=set();decisions=0;prefixes=defaultdict(dict)
    for model in plan['models']:
        with gzip.open(directory/(model['name']+'.hands.jsonl.gz'),'rt') as f:
            for line in f:
                row=json.loads(line);coordinate=(row['policy'],row['panel'],row['strategy'],row['block'],row['rotation'])
                if coordinate not in expected or coordinate in seen or row['seed']!=model['seed'] or row['status']!='complete':
                    raise ValueError('Unexpected or duplicate hand coordinate')
                if row['deal_seed']!=stream_seed(plan['root'],'test','deal',2,row['block']) or row['button']!=row['block']%2:
                    raise ValueError('Paired deal or button differs')
                hand=Hand.start(Table(('seat0','seat1'),(2000,2000),button=row['button']),hand_id=row['hand_id'],seed=row['deal_seed'])
                coverage=Counter()
                for index,a in enumerate(row['actions']):
                    if hand.actor!=a['seat'] or a['index']!=index or int(hand.actor!=row['rotation'])!=a['logical_player']:
                        raise ValueError('Recorded actor differs')
                    view=hand.observe(hand.actor);observed=a['observation']
                    menu=tuple(Choice(c['name'],Action(ActionKind(c['kind']),c['raise_to'])) for c in observed['menu'])
                    for c in menu:view.legal_actions.validate(c.action)
                    actual=snapshot(view,menu,observed['probabilities'],observed['trained'],None);actual['logical_player']=a['logical_player']
                    if observed!=json.loads(json.dumps(actual)):raise ValueError('Acting-seat observation differs')
                    if not a['logical_player']:
                        search=row['strategy']=='average' and a['street']=='river'
                        if a['river_search']!=search:raise ValueError('Wrong river action source')
                        status='river_search' if search else 'current' if observed['trained'] else 'missing'
                        coverage[status]+=1;coverage[a['street']+':'+status]+=1;decisions+=1
                    hand=hand.apply(Action(ActionKind(a['kind']),a['raise_to']))
                chips=[p.stack-2000 for p in hand.observe(0).players]
                if not hand.finished or sum(chips) or chips!=row['net_chips_by_seat'] or chips[row['rotation']]!=row['target_chips']:
                    raise ValueError('Native payoff differs')
                if digest(public_events(hand.events))!=row['public_events_sha256'] or hand_tails(row)!=row['tails'] or dict(coverage)!=row['coverage']:
                    raise ValueError('Digest or tail/coverage arithmetic differs')
                for record in row['search_records']:
                    if record['status']!='completed' or record['sweeps']!=plan['river_config']['sweeps'] or record['river_delegation']:
                        raise ValueError('Partial or delegated search')
                prefixes[row['policy'],row['panel'],row['block'],row['rotation']][row['strategy']]=[
                    a for a in row['actions'] if a['street']!='river']
                rows.append(row);seen.add(coordinate)
    if seen!=expected or len(rows)!=summary['hands'] or len(rows)!=plan['expected_hands']:
        raise ValueError('Missing frozen hands')
    if summarize(rows)!=summary['comparison']:raise ValueError('Paired arithmetic differs')
    for pair in prefixes.values():
        if pair['current']!=pair['average']:raise ValueError('Earlier-street paired behavior differs')
    return rows,{'status':'verified','hands':len(rows),'target_observations':decisions,
        'identical_earlier_street_pairs':len(prefixes),'no_models_loaded':True}


def write_csv(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def exploratory_interval(values):
    # Preserve the arena's >=30-block gate. These separately labelled intervals
    # describe this small fixed experiment and cannot satisfy a strength gate.
    if len(values)<2 or stdev(values)==0:
        return {'exploratory_t95_low':None,'exploratory_t95_high':None}
    margin=float(t.ppf(.975,len(values)-1))*stdev(values)/sqrt(len(values))
    return {'exploratory_t95_low':mean(values)-margin,'exploratory_t95_high':mean(values)+margin}


def report(directory,out):
    out.mkdir(parents=True,exist_ok=False);began=perf_counter();rows,audited=audit(directory)
    summary=json.loads((directory/'summary.json').read_text());data=summary['comparison']
    panels=[];changes=[];resources=[]
    grouped=defaultdict(list);profit={}
    for row in rows:
        grouped[row['seed'],row['panel'],row['strategy']].append(row)
        position='button' if row['rotation']==row['button'] else 'big_blind'
        profit[row['seed'],row['panel'],row['block'],position,row['strategy']]=row['target_chips']
    for p in data['panels']:
        for position,estimate in [('overall',p['overall']),*p['positions'].items()]:
            hands=grouped[p['seed'],p['panel'],p['strategy']]
            selected=hands if position=='overall' else [h for h in hands if
                ('button' if h['rotation']==h['button'] else 'big_blind')==position]
            values=p['paired_block_chips'] if position=='overall' else [profit[p['seed'],p['panel'],b,position,p['strategy']] for b in p['blocks']]
            panels.append({'seed':p['seed'],'panel':p['panel'],'player':'river_search' if p['strategy']=='average' else 'direct',
                'position':position,'hands':len(selected),**estimate,**exploratory_interval(values),
                'full_stack_wins':sum(h['target_chips']==2000 for h in selected),
                'full_stack_losses':sum(h['target_chips']==-2000 for h in selected)})
    for c in data['changes']:
        for position,estimate in [('overall',c['overall']),*c['positions'].items()]:
            values=c['paired_delta_chips'] if position=='overall' else [
                profit[c['seed'],c['panel'],b,position,'average']-profit[c['seed'],c['panel'],b,position,'current']
                for b in range(len(c['paired_delta_chips']))]
            changes.append({'seed':c['seed'],'panel':c['panel'],'position':position,**estimate,**exploratory_interval(values)})
    for (seed,panel,strategy),hands in sorted(grouped.items()):
        records=[r for h in hands for r in h['search_records']];coverage=Counter()
        for h in hands:coverage.update(h['coverage'])
        own=[a for h in hands for a in h['actions'] if not a['logical_player']]
        tails=Counter()
        for h in hands:tails.update(h['tails']['counts'])
        resources.append({'seed':seed,'panel':panel,'player':'river_search' if strategy=='average' else 'direct',
            'hands':len(hands),'river_intervention_hands':sum(any(a['river_search'] for a in h['actions']) for h in hands),
            'river_actions':coverage['river_search'],'solve_requests':len(records),
            'new_solves':sum(not r['cache_hit'] for r in records),'cache_hits':sum(r['cache_hit'] for r in records),
            'resolves':sum(r['re_solve'] for r in records),'total_completed_sweeps':sum(r['sweeps'] for r in records if not r['cache_hit']),
            'search_seconds':sum(r['seconds'] for r in records),'hand_seconds':sum(h['seconds'] for h in hands),
            'target_decisions':len(own),'blueprint_missing':sum(not a['blueprint_trained'] for a in own),
            'earlier_missing':coverage['missing'],'range_trained':sum(r['range_coverage']['trained'] for r in records),
            'range_missing':sum(r['range_coverage']['missing'] for r in records),
            'range_off_menu_likelihoods':sum(r['range_coverage']['off_menu_raise_likelihoods'] for r in records),
            'min_positive_holdings':min((min(r['range_coverage']['positive_holdings_per_seat']) for r in records),default=None),
            'delegations':sum(r['river_delegation'] for r in records),
            'large_opportunities':tails['large_opportunities'],'large_raises':tails['large_actions'],
            'rival_folds':tails['rival_folds'],'rival_continuations':tails['rival_continuations'],
            'full_stack_wins':tails['full_stack_wins'],'full_stack_losses':tails['full_stack_losses']})
    write_csv(out/'panels.csv',panels);write_csv(out/'paired-changes.csv',changes);write_csv(out/'resources.csv',resources)
    aggregates=[]
    for item in data['three_lineage_changes']:
        selected=[c for c in data['changes'] if c['panel']==item['panel']]
        values=[sum(v)/3 for v in zip(*(c['paired_delta_chips'] for c in selected),strict=True)]
        aggregates.append({**item,**exploratory_interval(values),
            'interpretation':'16 paired blocks; descriptive t interval, arena >=30-block gate unmet'})
    (out/'aggregate.json').write_text(json.dumps(aggregates,indent=2,sort_keys=True)+'\n')
    audited.update(seconds=perf_counter()-began,peak_rss_bytes=peak_rss())
    (out/'audit.json').write_text(json.dumps(audited,indent=2,sort_keys=True)+'\n')
    (out/'manifest.json').write_text(json.dumps({p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in out.iterdir()},indent=2,sort_keys=True)+'\n')
    return audited


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(report(a.directory,a.out)))


if __name__=='__main__':main()
