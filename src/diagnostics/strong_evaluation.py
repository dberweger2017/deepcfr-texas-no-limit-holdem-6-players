"""Small native HU20 harness for heuristic calibration and saved-policy panels."""
from collections import Counter, defaultdict
from random import Random
from statistics import mean, stdev
from math import sqrt

from scipy.stats import t

from src.arena.heuristics import STYLES, StylePolicy, pot_raise
from src.arena.report import estimate
from src.arena.runner import public_events
from src.arena.schedule import digest, stream_seed
from src.blueprint.abstraction import choices
from src.diagnostics.selective_stackoff import SelectiveStackoff
from src.diagnostics.stackoff_tails import hand_tails, snapshot
from src.diagnostics.strong_rollout import StrongRollout, action_menu
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind


class CalibrationControl:
    def __init__(self,name,seed,mode):
        self.name=name;self.random=Random(seed);self.mode=mode
        self.style=StylePolicy(STYLES[name],seed) if name in STYLES else None
        self.selective=SelectiveStackoff(seed) if name=='selective_stackoff' else None

    def choose_action(self,view):
        menu=action_menu(view,self.mode);raises=[c.action for c in menu if c.action.kind==ActionKind.RAISE]
        if self.style:
            action=self.style.choose_action(view)
            if self.mode=='restricted' and action.kind==ActionKind.RAISE:
                # Only this fixed calibration control is projected; saved model
                # actions/queries are never translated or changed.
                return min(raises,key=lambda a:(abs(a.raise_to-action.raise_to),a.raise_to))
            return action
        if self.selective:return self.selective.choose_action(view)
        if self.name=='random':return self.random.choice(menu).action
        if raises and self.name in ('minraise','raise','jam'):
            if self.name=='minraise':return min(raises,key=lambda a:a.raise_to)
            if self.name=='jam' and self.mode=='native':return Action(ActionKind.RAISE,view.legal_actions.max_raise_to)
            if self.name=='raise' and self.mode=='native':return pot_raise(view,1,1)
            if self.name=='raise':return next((c.action for c in menu if c.name=='pot'),raises[0])
            return max(raises,key=lambda a:a.raise_to)
        return Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds else ActionKind.CALL)


def replay_record(row):
    table=Table(('seat0','seat1'),(2000,2000),button=row['button'])
    hand=Hand.start(table,hand_id=row['hand_id'],seed=row['deal_seed'])
    for item in row['actions']:
        if hand.actor!=item['seat']:raise ValueError('Native replay actor differs')
        hand=hand.apply(Action(ActionKind(item['kind']),item['raise_to']))
    if (not hand.finished or digest(public_events(hand.events))!=row['public_events_sha256']
        or [p.stack-2000 for p in hand.observe(0).players]!=row['net_chips_by_seat']):
        raise ValueError('Native replay events/settlement differ')
    return hand


def match(config,mode,root,block,rotation,*,control=None,source=None,name='heuristic',candidate=None):
    seed=stream_seed(root,'test','action','hero',block,rotation)
    rival_seed=stream_seed(root,'test','opponent','rival',block,rotation)
    hero=StrongRollout(seed,config,mode) if source is None else None
    rival=CalibrationControl(control,rival_seed,mode) if control else StrongRollout(rival_seed,config,mode)
    random=Random(seed);deal_seed=stream_seed(root,'test','deal',2,block)
    hand_id=f'strong-v1/{control or name}/{mode}/{block}/{rotation}'
    hand=Hand.start(Table(('seat0','seat1'),(2000,2000),button=block%2),hand_id=hand_id,seed=deal_seed)
    actions=[]
    for index in range(1000):
        if hand.finished:break
        view=hand.observe(hand.actor);logical=int(hand.actor!=rotation)
        if not logical and source is not None:
            menu,probabilities,trained=source.distribution(view)
            action=random.choices(menu,weights=probabilities,k=1)[0].action
            if candidate is not None:candidate(view,menu,probabilities,trained,index,block,rotation)
        else:
            policy=rival if logical else hero;menu=action_menu(view,mode);probabilities=None;trained=None
            action=policy.choose_action(view)
            # Some native calibration controls intentionally use sizes beyond
            # v1's small menu; record that concrete native action explicitly.
            if not any(c.action==action for c in menu):
                from src.blueprint.abstraction import Choice
                menu=(*menu,Choice('control-native',action))
        observed=snapshot(view,menu,probabilities,trained,None)
        observed['logical_player']=logical
        entry={'index':index,'seat':hand.actor,'logical_player':logical,'street':view.street.value,
               'kind':action.kind.value,'raise_to':action.raise_to,'observation':observed}
        policy=rival if logical else hero
        if isinstance(policy,StrongRollout):entry['strong']=policy.telemetry[-1]
        view.legal_actions.validate(action);actions.append(entry);hand=hand.apply(action)
    if not hand.finished:raise RuntimeError('Diagnostic hand exceeded action bound')
    chips=[p.stack-2000 for p in hand.observe(0).players]
    if sum(chips)!=0:raise ValueError('Native chip conservation failed')
    row={'status':'complete','kind':'model' if source else 'calibration','policy':name,'mode':mode,
         'panel':control or mode,'root_seed':root,'block':block,'rotation':rotation,'button':block%2,
         'deal_seed':deal_seed,'hand_id':hand_id,'actions':actions,'target_chips':chips[rotation],
         'net_chips_by_seat':chips,'public_events_sha256':digest(public_events(hand.events))}
    replay_record(row);row['native_replay_verified']=True;row['tails']=hand_tails(row)
    if source is None:
        # Lookup exposure is inapplicable to a heuristic, not a fallback rate.
        row['tails']['counts']={k:v for k,v in row['tails']['counts'].items() if 'trained' not in k and 'fallback' not in k}
    return row


def summarize_matches(rows):
    groups=defaultdict(list)
    for row in rows:
        if not row['native_replay_verified']:raise ValueError('Missing native replay evidence')
        groups[row['policy'],row['mode'],row['panel']].append(row)
    results=[]
    for (name,mode,panel),items in sorted(groups.items()):
        positions={'button':{},'big_blind':{}};counts=Counter()
        for row in items:
            position='button' if row['rotation']==row['button'] else 'big_blind'
            if row['block'] in positions[position]:raise ValueError('Duplicate coordinate')
            positions[position][row['block']]=row['target_chips'];counts.update(row['tails']['counts'])
        if set(positions['button'])!=set(positions['big_blind']):raise ValueError('Incomplete paired block')
        blocks=sorted(positions['button']);series=[mean(positions[p][b] for p in positions) for b in blocks]
        result={'policy':name,'mode':mode,'panel':panel,'overall':estimate(series),
                'positions':{p:estimate([v[b] for b in blocks]) for p,v in positions.items()},
                'counts':dict(counts),'blocks':blocks,'paired_block_chips':series}
        if len(series)>1:
            half=float(t.ppf(1-.05/12,len(series)-1))*stdev(series)/sqrt(len(series))
            result['family95_interval']=[mean(series)-half,mean(series)+half]
        results.append(result)
    return results


def calibration_gates(summary):
    goal=[]
    for mode in ('restricted','native'):
        items=[r for r in summary if r['mode']==mode]
        sanity=[r for r in items if r['panel'] in ('random','call','minraise')]
        others=[r for r in items if r['panel'] not in ('random','call','minraise')]
        goal.append({'mode':mode,'sanity_positive_family95':len(sanity)==3 and all(r.get('family95_interval',[0])[0]>0 for r in sanity),
                     'nonnegative_other_panels':sum(r['overall']['bb_per_100']>=0 for r in others),
                     'other_panel_goal':6,'complete_controls':len(items)==12})
    return {'quality_goal_passed':all(g['sanity_positive_family95'] and g['nonnegative_other_panels']>=6 and g['complete_controls'] for g in goal),'modes':goal}
