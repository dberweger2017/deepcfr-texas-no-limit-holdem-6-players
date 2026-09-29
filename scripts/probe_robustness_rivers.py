"""Small exploratory conditional native river action values, never hindsight deals.

Adapted from #108's independent oracle approach: replay each declared private
world through native settlement, aggregate opponent reach before comparisons.
Here hero has one known holding and opponent a fixed uniformly weighted 24-hand
range. The root is selected by frozen index, not profit or a realized rival hand.
The target's future actions remain its saved strategy; fixed reactive opponent
responses retain exact private cards. No CFR solver or full-game BR is invoked.
"""
import argparse,gzip,json
from pathlib import Path
from random import Random
from itertools import combinations
from scripts.evaluate_hu20 import write_json
from scripts.evaluate_robustness import load
from scripts.play_robustness import replay_row
from src.blueprint.abstraction import choices
from src.blueprint.search import DECK,_sample_world
from src.diagnostics.robustness import ReactiveAttack
from src.game.hand import Hand,Table
from src.game.types import Action,ActionKind,Street


def probe(view,source,rule,contract,max_nodes=30000):
    all_pairs=list(combinations([c for c in DECK if c not in view.hole_cards+view.board],2))
    pairs=Random(2026136001).sample(all_pairs,24);menus=choices(view,free_fold=False);totals=[0.]*len(menus);nodes=0
    opponent=ReactiveAttack(rule,contract)
    def value(hand):
        nonlocal nodes
        nodes+=1
        if nodes>max_nodes:raise RuntimeError('Conditional probe node cap')
        if hand.finished:return hand.events[-1].stacks[view.seat]-2000
        v=hand.observe(hand.actor)
        if v.seat!=view.seat:return value(hand.apply(opponent.choose_action(v)))
        opts,p,_=source.distribution(v)
        return sum(q*value(hand.apply(c.action)) for c,q in zip(opts,p) if q>0)
    for pair in pairs:
        world=_sample_world(view,{1-view.seat:((pair,1),)},Random(0))
        for i,c in enumerate(menus):totals[i]+=value(world.apply(c.action))/len(pairs)
    menu,p,hit=source.distribution(view)
    actual=sum(q*u for q,u in zip(p,totals))
    return {'hero_cards':view.hole_cards,'board':view.board,'history':[repr(e) for e in view.history],
        'range':pairs,'range_weights':[1/24]*24,'range_rule':'uniform compatible 24 holdings sampled once; not a posterior from prior actions',
        'actions':[{'name':c.name,'kind':c.action.kind.value,'raise_to':c.action.raise_to,'value_chips':u,'target_probability':q} for c,u,q in zip(menu,totals,p)],
        'target_trained':hit,'policy_value_chips':actual,'best_one_action_value_chips':max(totals),
        'conditional_action_gap_chips':max(totals)-actual,'native_nodes':nodes}


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--hands',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());name='2p-2026092801-20M';spec=next(s for s in plan['policies'] if s['name']==name);source=load(spec)
    results=[]
    with gzip.open(a.hands,'rt') as f:
        for line in f:
            row=json.loads(line)
            if row['policy']!=name or row['contract']!='menu' or row['rules'] not in [['pressure'],['passive']] or row['block']>=64:continue
            # Retain first six river target decisions per rule, including neutral ones.
            if sum(r['rule']==row['rules'][0] for r in results)>=6:continue
            hand=replay_row(row)
            n=2;r=row['rotation'];ids=tuple(f'player-{(s-r)%n}' for s in range(n))
            hand=Hand.start(Table(ids,(2000,)*2,button=row['button']),hand_id=f"robustness-{row['phase']}-2-{row['block']}",seed=row['deal_seed'])
            for i,item in enumerate(row['actions']):
                view=hand.observe(hand.actor)
                if view.seat==r and view.street==Street.RIVER:
                    try:record={'status':'complete',**probe(view,source,row['rules'][0],'menu')}
                    except Exception as exc:record={'status':'failed','error':repr(exc)}
                    record.update(policy=name,block=row['block'],rotation=r,decision_index=i,rule=row['rules'][0],deal_seed=row['deal_seed'])
                    results.append(record);break
                hand=hand.apply(Action(ActionKind(item['kind']),item['raise_to']))
    write_json(a.out,{'schema':'exploratory-river-probes-v1','cases':results,'limitations':'conditional 24-holding uniform range; no whole-hand attribution or actual rival cards used'})

if __name__=='__main__':main()
