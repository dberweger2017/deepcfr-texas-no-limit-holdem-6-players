"""Verify closed diagnostic files without loading any saved policy."""
import argparse
from collections import Counter
import gzip
import json
from math import isfinite
from pathlib import Path
from random import Random
import resource
from time import perf_counter

from scripts.compare_strong_hu20 import DecisionSample
from src.arena.runner import public_events
from src.arena.schedule import digest, stream_seed
from src.blueprint.search import _sample_world
from src.diagnostics.decision_counterfactual import summarize_values
from src.diagnostics.saved_hu20 import file_hash
from src.diagnostics.stackoff_tails import hand_tails
from src.diagnostics.strong_evaluation import replay_record, summarize_matches
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind


def read_rows(path):
    with gzip.open(path,'rt') as f:
        for line in f:yield json.loads(line)


def root_view(row,index):
    hand=Hand.start(Table(('seat0','seat1'),(2000,2000),button=row['button']),hand_id=row['hand_id'],seed=row['deal_seed'])
    for action in row['actions'][:index]:hand=hand.apply(Action(ActionKind(action['kind']),action['raise_to']))
    return hand.observe(hand.actor)


def audit(directory):
    start=perf_counter();summary=json.loads((directory/'summary.json').read_text());plan=summary['plan']
    if summary['status']!='complete' or summary['plan_sha256']!=digest(plan):raise ValueError('Incomplete or changed plan')
    for name,spec in json.loads((directory/'manifest.json').read_text()).items():
        p=directory/name
        if p.stat().st_size!=spec['bytes'] or file_hash(p)!=spec['sha256']:raise ValueError('Output bytes differ')
    expected={(s['name'],m,b,r) for s in plan['models'] for m in plan['modes'] for b in range(plan['blocks']) for r in (0,1)}
    hands={};rows=[];tails=Counter();samples={};gaps=0;traces=0
    for spec in plan['models']:
        for mode in plan['modes']:samples[spec['name'],mode]=DecisionSample(plan['decision_sample_root'],plan['decisions_per_street_position'])
        for row in read_rows(directory/(spec['name']+'.hands.jsonl.gz')):
            key=row['policy'],row['mode'],row['block'],row['rotation']
            if key not in expected or key in hands:raise ValueError('Unexpected/duplicate hand')
            if row['button']!=row['block']%2 or row['deal_seed']!=stream_seed(plan['deal_root'],'test','deal',2,row['block']):raise ValueError('Deal pairing differs')
            if row['target_chips']!=row['net_chips_by_seat'][row['rotation']] or sum(row['net_chips_by_seat'])!=0:raise ValueError('Target chip identity differs')
            replay_record(row)
            if hand_tails(row)!=row['tails']:raise ValueError('Tail counters differ')
            hands[key]=row;rows.append(row);tails.update(row['tails']['counts'])
            sample=samples[spec['name'],row['mode']]
            for action in row['actions']:
                if action['logical_player']==0:
                    # Rebuild only the currently entitled root observation; sampler
                    # uses the coordinate hash and street/position, never payoff.
                    view=root_view(row,action['index'])
                    sample.consider(view,(),(),action['observation']['trained'],action['index'],row['block'],row['rotation'])
    if set(hands)!=expected or len(rows)!=summary['hands']:raise ValueError('Incomplete actual hands')
    computed=summarize_matches(rows)
    for panel in computed:
        recorded=next(p for p in summary['panels'] if (p['policy'],p['mode'],p['panel'])==(panel['policy'],panel['mode'],panel['panel']))
        if any(recorded[k]!=v for k,v in panel.items()):raise ValueError('Panel arithmetic differs')
    diagnostic_keys=set()
    for spec in plan['models']:
        for mode in plan['modes']:
            selected={(c['block'],c['rotation'],c['index']):v for v,c in samples[spec['name'],mode].selected()}
            actual=set()
            for scored in read_rows(directory/(spec['name']+'.'+mode+'.decisions.jsonl.gz')):
                key=scored['block'],scored['rotation'],scored['index']
                if key not in selected or key in actual:raise ValueError('Decision is not the frozen outcome-blind sample')
                actual.add(key);view=selected[key];row=hands[spec['name'],mode,key[0],key[1]]
                observed=row['actions'][key[2]]['observation'];alternatives=scored['legal_alternatives']
                if scored['root_trained']!=observed['trained']:raise ValueError('Root lookup differs')
                for a in alternatives:
                    view.legal_actions.validate(Action(ActionKind(a['kind']),a['raise_to']))
                    expected_probability=sum(p for c,p in zip(observed['menu'],observed['probabilities']) if (c['kind'],c['raise_to'])==(a['kind'],a['raise_to']))
                    if a['root_probability']!=expected_probability:raise ValueError('Saved-policy root mixture differs')
                fingerprint=digest({'own_cards':view.hole_cards,'board':view.board,'events':[repr(e) for e in view.history]})
                if fingerprint!=scored['observation_sha256']:raise ValueError('Root observation differs')
                values=[w['returns_bb'] for w in scored['world_records']]
                if any(not isfinite(v) or not -20<=v<=20 for r in values for v in r):raise ValueError('Invalid simulated payoff')
                recomputed=summarize_values(values,[a['root_probability'] for a in alternatives],plan['selection_worlds'])
                if any(scored[k]!=v for k,v in recomputed.items()):raise ValueError('Held-out decision arithmetic differs')
                first=scored['world_records'][0]
                world=_sample_world(view,{1-view.seat:scored['range_holdings']},Random(first['world_seed']))
                for i,(a,trace) in enumerate(zip(alternatives,first['traces'],strict=True)):
                    hand=world.apply(Action(ActionKind(a['kind']),a['raise_to']))
                    for action in trace['actions']:
                        observed=hand.observe(hand.actor)
                        if (hand.actor!=action['seat'] or list(observed.board)!=action['board'] or list(observed.hole_cards)!=action['own_cards']):raise ValueError('Continuation observation trace differs')
                        hand=hand.apply(Action(ActionKind(action['kind']),action['raise_to']))
                    if not hand.finished or (hand.observe(view.seat).players[view.seat].stack-2000)/100!=first['returns_bb'][i]:raise ValueError('First-world counterfactual replay differs')
                    traces+=1
                diagnostic_keys.add((spec['name'],mode,*key));gaps+=1
            if actual!=set(selected):raise ValueError('Missing selected decisions')
    recorded_keys={(r['model'],r['mode'],r['block'],r['rotation'],r['index']) for r in summary['decision_diagnostics']}
    if recorded_keys!=diagnostic_keys:raise ValueError('Diagnostic summary identities differ')
    return {'status':'complete','actual_hands_replayed':len(rows),'decision_cases_verified':gaps,
            'first_world_branches_replayed':traces,'seconds':perf_counter()-start,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'input_manifest_sha256':file_hash(directory/'manifest.json'),'no_models_loaded':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Preserve the previous audit output')
    result=audit(a.directory);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
