"""Paired HU200 translation comparison, single-load play and complete reproduction."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from fractions import Fraction
import gzip
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from scripts.evaluate_hu200_diagnosis import OPPONENTS, interval, native_start, rows, seed
from scripts.run_hu200_feasibility import write
from src.arena.policies import make_policy
from src.arena.runner import public_events
from src.arena.schedule import canonical, digest
from src.blueprint.abstraction import HU200_SCHEMA, choices, information_key
from src.blueprint.action_translation import TranslationOptions
from src.blueprint.average import AveragePolicy
from src.game.hand import Hand, Table
from src.game.observation import ActionTaken, replay as observation_replay
from src.game.types import Action, ActionKind
from src.policies.files import file_hash

ARMS=('off','on')
CONTROLS=('check_call','tight_aggressive','loose_aggressive')


def stable(info):
    return {k:v for k,v in info.items() if k!='lookup_seconds'}


def verify_witness(initial, view, menu, info, model):
    """Replay selected public menu history with the independent real game engine."""
    if info['mode']!='translated':return
    witness=initial;real=initial;targets=iter(info['witness_raise_to']);labels={};distance=Fraction(0);changes=0
    for index,event in enumerate(view.history):
        if not isinstance(event,ActionTaken):continue
        if witness.finished or witness.actor!=event.seat:
            raise ValueError('Witness actor/terminal differs')
        wv=witness.observe(witness.actor);rv=real.observe(real.actor)
        if wv.street!=event.street or rv.street!=event.street:raise ValueError('Witness street differs')
        wm=choices(wv,raise_cap=None,free_fold=False);rm=choices(rv,raise_cap=None,free_fold=False)
        action=event.action
        if action.kind==ActionKind.RAISE:
            action=Action(ActionKind.RAISE,next(targets))
            wc=next((c for c in wm if c.action==action),None)
            original=next((c for c in rm if c.action==event.action),None)
            if wc is None or original is not None and wc.name!=original.name:
                raise ValueError('Witness failed preserved menu label')
            wp=action.raise_to-wv.players[event.seat].street_bet
            rp=event.paid
            ratio=Fraction(wp,max(wv.pot,wv.big_blind))
            size=0 if ratio<Fraction(1,2) else 1 if ratio<Fraction(3,2) else 2 if ratio<3 else 3
            labels[index]=f'raise-{size}'+('-all-in' if wp==wv.players[event.seat].stack else '')
            distance+=abs(Fraction(rp,max(rv.pot,rv.big_blind)+rp)-Fraction(wp,max(wv.pot,wv.big_blind)+wp))
            changes+=int((wp==wv.players[event.seat].stack)!=(rp==rv.players[event.seat].stack))
        elif not any(c.action==action for c in wm):raise ValueError('Witness nonraise outside menu')
        wv.legal_actions.validate(action);witness=witness.apply(action);real=real.apply(event.action)
    if next(targets,None) is not None:raise ValueError('Extra witness targets')
    wv=witness.observe(witness.actor);wm=choices(wv,raise_cap=None,free_fold=False)
    if (wv.actor!=view.actor or wv.street!=view.street or wv.hole_cards!=view.hole_cards or wv.board!=view.board
            or tuple((p.folded,p.all_in) for p in wv.players)!=tuple((p.folded,p.all_in) for p in view.players)
            or tuple(c.name for c in wm)!=tuple(c.name for c in menu)
            or information_key(wv,wm,schema=HU200_SCHEMA)!=info['selected_key']
            or information_key(view,menu,schema=HU200_SCHEMA,history_label_overrides=labels)!=info['selected_key']
            or labels!=dict(info['overrides']) or float(distance)!=info['distance'] or changes!=info['all_in_changes']):
        raise ValueError('Witness key/flags/distance/labels differs')
    selected=model.entries.get(info['selected_key'])
    if selected is None or info['selected_key'] in model.zero_mass or selected[0]!=tuple(c.name for c in menu):
        raise ValueError('Witness lacks matching positive mass')


def play_hand(model,root,opponent,block,seat):
    deal=seed(root,'deal',opponent,block);own_seed=seed(root,'candidate',opponent,block,seat)
    rival_seed=seed(root,'rival',opponent,block,seat)
    table=Table(('player-0','player-1'),(20000,20000),button=block%2)
    hand_id=f'HU200-translation/{root}/{opponent}/{block}/{seat}'
    initial=Hand.start(table,hand_id=hand_id,seed=deal);hand=initial
    player=model.policy(own_seed);rival=make_policy(opponent,rival_seed)
    actions=[];features=[];fixtures=[];witness_seconds=0.
    for _ in range(1000):
        if hand.finished:break
        view=hand.observe(hand.actor)
        if observation_replay(view.history,view.seat,view.hole_cards)!=view:raise ValueError('Public replay differs')
        menu=choices(view,raise_cap=None,free_fold=False);key=information_key(view,menu,schema=HU200_SCHEMA)
        fixtures.append(dict(actor=hand.actor,street=view.street.value,pot=view.pot,
            kinds=[k.value for k in view.legal_actions.kinds],call=view.legal_actions.call_amount,
            min_raise_to=view.legal_actions.min_raise_to,max_raise_to=view.legal_actions.max_raise_to,
            menu=[[c.name,c.action.raise_to] for c in menu],key=key))
        action=(player if hand.actor==seat else rival).choose_action(view);view.legal_actions.validate(action)
        if hand.actor==seat:
            actual_menu,p,known,info=player.last_decision
            if actual_menu!=menu or action not in [c.action for c in menu]:raise ValueError('Real candidate menu differs')
            tick=perf_counter();verify_witness(initial,view,menu,info,model);witness_seconds+=perf_counter()-tick
            features.append(dict(decision=len(actions),street=view.street.value,key=key,menu=[asdict(c) for c in menu],
                probabilities=p,known=known,translation=info,legal=asdict(view.legal_actions),action=asdict(action),
                selected_visits=model.visits[info['selected_key']] if info['selected_key'] else None))
        actions.append([hand.actor,action.kind.value,action.raise_to]);hand=hand.apply(action)
    if not hand.finished:raise ValueError('Hand cap')
    final=list(hand.events[-1].stacks)
    if sum(final)!=40000:raise ValueError('Settlement accounting')
    # Full deterministic policy/telemetry reproduction; measured elapsed times alone are excluded.
    tick=perf_counter();again=Hand.start(table,hand_id=hand_id,seed=deal)
    repeat=model.policy(own_seed);other=make_policy(opponent,rival_seed);di=0
    for ai,(actor,kind,target) in enumerate(actions):
        if again.actor!=actor:raise ValueError('Replay actor')
        view=again.observe(actor);action=Action(ActionKind(kind),target);view.legal_actions.validate(action)
        if (repeat if actor==seat else other).choose_action(view)!=action:raise ValueError('Policy reproduction differs')
        if actor==seat:
            menu,p,known,info=repeat.last_decision;d=features[di];di+=1
            if (d['decision']!=ai or menu!=tuple(choices(view,raise_cap=None,free_fold=False))
                    or list(p)!=list(d['probabilities']) or known!=d['known']
                    or stable(info)!=stable(d['translation'])):raise ValueError('Telemetry reproduction differs')
        again=again.apply(action)
    if again.events!=hand.events or list(again.events[-1].stacks)!=final:raise ValueError('Full event replay differs')
    replay_seconds=perf_counter()-tick
    row=dict(opponent=opponent,block=block,seat=seat,deal_seed=deal,candidate_seed=own_seed,rival_seed=rival_seed,
        actions=actions,decisions=features,events=public_events(hand.events),final_stacks=final,
        net_chips=final[seat]-20000,holes=[initial.observe(s).hole_cards for s in (0,1)],button=table.button)
    row['sha256']=digest(row)
    fixture=dict(stack_bb=200,seed=deal,button=table.button,deck=native_start(initial),
        actions=[[k,a] for _,k,a in actions],decisions=fixtures,final_stacks=final)
    return row,fixture,dict(witness_seconds=witness_seconds,replay_seconds=replay_seconds)


def worker(plan,stage,out):
    out.mkdir(parents=True,exist_ok=False);spec=plan['model'];tick=perf_counter()
    model=AveragePolicy(Path(spec['path']),spec['sha256'],expected_schema=HU200_SCHEMA)
    if model.description['entries']!=spec['entries']:raise ValueError('Model entries differ')
    load=perf_counter()-tick;model.record_translation=True
    root=plan['timing_root'] if stage=='timing' else plan['final_root'];blocks=32 if stage=='timing' else plan['blocks']
    costs=[];n=0;actions=0;decisions=0;begun=perf_counter()
    for arm in ARMS:
        model.configure_translation(TranslationOptions(**plan['translation']) if arm=='on' else None)
        folder=out/arm;folder.mkdir()
        with gzip.open(folder/'hands.jsonl.gz','xt',compresslevel=1) as hands,(folder/'native-fixtures.jsonl').open('x') as native:
            for op in OPPONENTS:
                tick=perf_counter();checks=Counter()
                for block in range(blocks):
                    for seat in (0,1):
                        row,fixture,c=play_hand(model,root,op,block,seat);checks.update(c)
                        hands.write(canonical(row)+'\n');hands.flush();native.write(canonical(fixture)+'\n');native.flush()
                        n+=1;actions+=len(row['actions']);decisions+=len(row['decisions'])
                costs.append(dict(arm=arm,opponent=op,seconds=perf_counter()-tick,blocks=blocks,**checks))
        write(folder/'hashes.json',dict(hands_sha256=file_hash(folder/'hands.jsonl.gz'),fixtures_sha256=file_hash(folder/'native-fixtures.jsonl')))
    write(out/'costs.json',dict(status='complete',stage=stage,plan_sha256=digest(plan),model_load_seconds=load,
        play_replay_seconds=perf_counter()-begun,panels=costs,hands=n,actions=actions,decisions=decisions,
        all_events_replayed=True,all_actions_reproduced=True,all_telemetry_reproduced=True,all_selected_witnesses_verified=True))


def verify_rows(path,root,blocks):
    seen=set()
    for row in rows(path):
        coord=(row['opponent'],row['block'],row['seat']);op,b,s=coord
        if row['sha256']!=digest({k:v for k,v in row.items() if k!='sha256'}) or coord in seen:raise ValueError('Changed/duplicate hand')
        if (op not in OPPONENTS or b not in range(blocks) or s not in (0,1)
                or row['deal_seed']!=seed(root,'deal',op,b) or row['candidate_seed']!=seed(root,'candidate',op,b,s)
                or row['rival_seed']!=seed(root,'rival',op,b,s) or sum(row['final_stacks'])!=40000
                or row['net_chips']!=row['final_stacks'][s]-20000 or row['button']!=b%2):raise ValueError('Schedule/accounting differs')
        seen.add(coord);yield row
    if seen!={(op,b,s) for op in OPPONENTS for b in range(blocks) for s in (0,1)}:raise ValueError('Incomplete frozen sample')


def behavior(row):
    # Wall-time telemetry varies; everything that can affect play must remain identical.
    return {**{k:v for k,v in row.items() if k not in ('sha256','decisions')},
            'decisions':[{k:v for k,v in d.items() if k!='translation'} for d in row['decisions']]}


def distribution_summary(values):
    if not values:return dict(count=0)
    return dict(count=len(values),mean=mean(values),p50=float(np.quantile(values,.5)),p95=float(np.quantile(values,.95)),
                p99=float(np.quantile(values,.99)),maximum=max(values))


def report(plan,root):
    folder=root/'final';costs=json.loads((folder/'costs.json').read_text())
    if costs['plan_sha256']!=digest(plan) or not all(costs[k] for k in (
            'all_events_replayed','all_actions_reproduced','all_telemetry_reproduced','all_selected_witnesses_verified')):
        raise ValueError('Unverified worker')
    paired={};counts=Counter();coverage=[];controls={};offs={};absolute={}
    for arm in ARMS:
        hashes=json.loads((folder/arm/'hashes.json').read_text())
        if (hashes['hands_sha256']!=file_hash(folder/arm/'hands.jsonl.gz')
                or hashes['fixtures_sha256']!=file_hash(folder/arm/'native-fixtures.jsonl')):raise ValueError('Worker bytes changed')
        panels=defaultdict(dict);cells=defaultdict(list)
        for row in verify_rows(folder/arm/'hands.jsonl.gz',plan['final_root'],plan['blocks']):
            op=row['opponent'];coord=(op,row['block'],row['seat']);panels[op][coord[1:]]=row['net_chips']
            counts['hands']+=1;counts['actions']+=len(row['actions']);counts['decisions']+=len(row['decisions'])
            if arm=='off':offs[coord]=(digest(behavior(row)),digest(row['holes']),row['deal_seed'])
            else:
                if offs[coord][1:]!=(digest(row['holes']),row['deal_seed']):raise ValueError('Paired deals differ')
                same=offs[coord][0]==digest(behavior(row));controls[op]=controls.get(op,True) and same
                if op in CONTROLS and not same:raise ValueError('On-menu control behavior changed')
            for d in row['decisions']:cells[(op,d['street'])].append(d['translation'])
        for op,v in panels.items():
            values=[(v[b,0]+v[b,1])/2 for b in range(plan['blocks'])]
            paired[arm,op]=values;absolute.setdefault(op,{})[arm]=interval(values)
        for (op,street),data in cells.items():
            c=Counter(d['mode'] for d in data);reason=Counter(d['reason'] for d in data);translated=[d for d in data if d['mode']=='translated']
            distances=Counter()
            for d in translated:
                label=next((name for lim,name in ((.05,'<=.05'),(.1,'.05-.1'),(.25,'.1-.25'),(.5,'.25-.5'),(1,'.5-1')) if d['distance']<=lim),'>1')
                distances[label]+=1
            coverage.append(dict(arm=arm,opponent=op,street=street,decisions=len(data),counts=dict(c),reasons=dict(reason),
                rates={m:c[m]/len(data) for m in ('exact','translated','uniform')},bound_hits=sum(d['bound_reached'] for d in data),
                all_in_changes=sum(d['all_in_changes'] for d in translated),states=distribution_summary([d['states'] for d in data]),
                distances=distribution_summary([d['distance'] for d in translated]),distance_histogram=dict(distances),
                lookup_ms=distribution_summary([1000*d['lookup_seconds'] for d in data]),
                translated_lookup_ms=distribution_summary([1000*d['lookup_seconds'] for d in translated])))
    gains={op:interval([on-off for on,off in zip(paired['on',op],paired['off',op],strict=True)]) for op in OPPONENTS}
    primary=gains['pot_pressure'];random=gains['random']
    decision=dict(primary_pass=primary['interval'][0]>0,random_safeguard_pass=random['interval'][0]>-20,
                  severe_random_regression=random['interval'][1]<-20,on_menu_controls_identical=all(controls[op] for op in CONTROLS))
    write(root/'summary.json',dict(status='verified',blocks=plan['blocks'],plan_sha256=digest(plan),counts=dict(counts),
        absolute=absolute,gains=gains,behavior_identical=controls,decision=decision,
        uncertainty='paired independent deal/private-action blocks; one fixed training lineage; no training-seed or general strength claim'))
    write(root/'coverage.json',coverage)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('worker','report'))
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=('timing','final'))
    a=p.parse_args();plan=json.loads(a.plan.read_text())
    if a.command=='worker':worker(plan,a.stage,a.out)
    else:report(plan,a.out)


if __name__=='__main__':main()
