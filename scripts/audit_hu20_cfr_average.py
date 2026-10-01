"""Recheck closed paired evidence and every native observation/action; no models."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import resource
from time import perf_counter

from scripts.evaluate_hu20_cfr_average import summarize
from src.arena.runner import public_events
from src.arena.schedule import digest,stream_seed
from src.blueprint.abstraction import information_key,HU20_UNCAPPED_SCHEMA,Choice
from src.diagnostics.saved_hu20 import file_hash
from src.diagnostics.stackoff_tails import snapshot,hand_tails
from src.game.hand import Hand,Table
from src.game.types import Action,ActionKind


def audit_run(directory):
    start=perf_counter();summary=json.loads((directory/'summary.json').read_text());plan=summary['plan']
    if summary['status']!='complete' or digest(plan)!=summary['plan_sha256']:raise ValueError('Incomplete or changed plan')
    for name,spec in json.loads((directory/'manifest.json').read_text()).items():
        path=directory/name
        if path.stat().st_size!=spec['bytes'] or file_hash(path)!=spec['sha256']:raise ValueError('Recorded bytes differ')
    expected={(s['name'],p['name'],b,r) for s in plan['models'] for p in plan['panels'] for b in range(p['blocks']) for r in (0,1)}
    seen=set();rows=[];decisions=0
    for spec in plan['models']:
        with gzip.open(directory/(spec['name']+'.hands.jsonl.gz'),'rt') as source:
            for line in source:
                row=json.loads(line);coordinate=row['policy'],row['panel'],row['block'],row['rotation']
                if coordinate not in expected or coordinate in seen or row['seed']!=spec['seed'] or row['strategy']!=spec['strategy']:raise ValueError('Unexpected/duplicate coordinate')
                if row['button']!=row['block']%2 or row['deal_seed']!=stream_seed(plan['root'],'test','deal',2,row['block']):raise ValueError('Paired deal differs')
                hand=Hand.start(Table(('seat0','seat1'),(2000,2000),button=row['button']),hand_id=row['hand_id'],seed=row['deal_seed']);coverage=Counter()
                for index,a in enumerate(row['actions']):
                    if a['index']!=index or hand.actor!=a['seat'] or int(hand.actor!=row['rotation'])!=a['logical_player']:raise ValueError('Action actor/index differs')
                    view=hand.observe(hand.actor);observed=a['observation']
                    menu=tuple(Choice(c['name'],Action(ActionKind(c['kind']),c['raise_to'])) for c in observed['menu'])
                    for c in menu:view.legal_actions.validate(c.action)
                    actual=snapshot(view,menu,observed['probabilities'],observed['trained'],None);actual['logical_player']=a['logical_player']
                    if observed!=json.loads(json.dumps(actual)):raise ValueError('Acting-seat observation differs')
                    if not a['logical_player']:
                        key=information_key(view,menu,schema=HU20_UNCAPPED_SCHEMA)
                        if key!=a['target_key']:raise ValueError('Target key differs')
                        status=a['average_mass_status'];valid={'missing','current'} if spec['strategy']=='current' else {'missing','zero_mass','positive_mass'}
                        if status not in valid or bool(observed['trained'])!=(status!='missing'):raise ValueError('Coverage label differs')
                        coverage[status]+=1;coverage[view.street.value+':'+status]+=1;decisions+=1
                    hand=hand.apply(Action(ActionKind(a['kind']),a['raise_to']))
                chips=[p.stack-2000 for p in hand.observe(0).players]
                if (not hand.finished or sum(chips)!=0 or chips!=row['net_chips_by_seat'] or chips[row['rotation']]!=row['target_chips']
                    or digest(public_events(hand.events))!=row['public_events_sha256']):raise ValueError('Native result differs')
                if dict(coverage)!=row['coverage'] or hand_tails(row)!=row['tails']:raise ValueError('Coverage/tail arithmetic differs')
                seen.add(coordinate);rows.append(row)
    if seen!=expected or len(rows)!=summary['hands']:raise ValueError('Missing planned hand')
    for key,value in summarize(rows).items():
        if summary[key]!=value:raise ValueError('Paired report arithmetic differs')
    return {'status':'verified','actual_hands_replayed':len(rows),'target_decisions_verified':decisions,
            'no_models_loaded':True,'manifest_sha256':file_hash(directory/'manifest.json'),
            'seconds':perf_counter()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('Preserve earlier audit')
    result=audit_run(a.directory);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
