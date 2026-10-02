"""Outcome-blind retained LBR and fresh self-play turn-root populations."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from random import Random
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.saved_hu20 import file_hash
from src.diagnostics.turn_check import root_record
from src.game.hand import Hand,Table
from src.game.types import Action,ActionKind,Street
from scripts.prepare_flop_check import load_policy
from scripts.select_flop_check_spots import select_strata


def kind(history):
    raises=[e for e in history if hasattr(e,'action') and e.street==Street.PREFLOP and e.action.kind==ActionKind.RAISE]
    return 'limped' if not raises else '3-bet' if len(raises)>1 else 'min-raised' if raises[0].action.raise_to==200 else 'pot-raised'


def stored(plan,hands):
    roots={};decisions=[];counts=Counter();sources=[]
    for relative,wanted in sorted(plan['stored_hands'].items()):
        path=Path(hands)/relative
        if file_hash(path)!=wanted:raise ValueError('Stored turn source hash mismatch')
        sources.append({'path':str(path),'sha256':wanted})
        with gzip.open(path,'rt') as source:
            for number,text in enumerate(source,1):
                row=json.loads(text)
                if row['panel']!='lbr' or row['version']!='v1':continue
                counts['lbr_v1_hands']+=1
                hand=Hand.start(Table(('a','b'),(2000,2000),button=row['button']),hand_id='turn-selection',seed=row['deal_seed'])
                record=None
                for action in row['actions']:
                    view=hand.observe(hand.actor)
                    if view.seat!=action['seat'] or list(view.hole_cards)!=action['observation']['hole_cards']:
                        raise ValueError('Stored turn native replay differs')
                    if record is None and view.street==Street.TURN:
                        record=dict(root_record(hand.events),kind=kind(hand.events),multiplicity=0)
                        counts['hands_with_live_turn_root']+=1
                    if view.street==Street.TURN and action['logical_player']==0 and view.legal_actions.call_amount>0:
                        roots.setdefault(record['spot'],record);roots[record['spot']]['multiplicity']+=1
                        decisions.append({'spot':record['spot'],'source_sha256':wanted,'source_line':number,
                                          'action_index':action['index'],'target_position':(view.seat-view.button)%2})
                    hand=hand.apply(Action(ActionKind(action['kind']),action['raise_to']))
    if len(decisions)!=74:raise ValueError('Pinned turn decision count differs')
    return {'set':'A','roots':list(roots.values()),'decisions':decisions,'counts':dict(counts),'sources':sources,
            'selection_uses_payoffs':False,'root_definition':'start of turn, before any turn action'}


def fresh(source,*,deals=3000,n=None):
    deal_rng=Random(202610020201);action_rng=Random(202610020202);roots={};counts=Counter()
    for index in range(deals):
        hand=Hand.start(Table(('a','b'),(2000,2000),button=index%2),hand_id='turn-selection',seed=deal_rng.getrandbits(64))
        while not hand.finished and hand.observe(hand.actor).street!=Street.TURN:
            menu,p,_=source.distribution(hand.observe(hand.actor));hand=hand.apply(action_rng.choices(menu,weights=p,k=1)[0].action)
        if hand.finished:counts['earlier_terminal']+=1;continue
        record=dict(root_record(hand.events),kind=kind(hand.events),multiplicity=0)
        roots.setdefault(record['spot'],record);roots[record['spot']]['multiplicity']+=1;counts['live_turn_roots']+=1
    population=list(roots.values());selected,strata=select_strata(population,n,202610020203) if n else (population,None)
    return {'set':'B','roots':selected,'corpus_deals':deals,'counts':dict(counts),'corpus_unique_roots':len(population),
        'deal_seed':202610020201,'action_seed':202610020202,'selection_seed':202610020203,
        'source':source.description,'strata':strata,'turn_actions_sampled':0,'selection_uses_payoffs':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--set',choices=('A','B'),required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--inputs',type=Path);p.add_argument('--hands',type=Path,default=Path('docs/reports/hu20-card-v2-artifacts/production'))
    p.add_argument('--n',type=int);a=p.parse_args();plan=json.loads(a.plan.read_text())
    result=stored(plan,a.hands) if a.set=='A' else fresh(load_policy(plan['policies'][0],a.inputs),n=a.n)
    atomic_json(a.out,result)
if __name__=='__main__':main()
