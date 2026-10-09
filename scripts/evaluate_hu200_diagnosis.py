"""Frozen HU200 comparison: paired play, exact replay, support and behavior evidence."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import hashlib
import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

from scipy.stats import t

from scripts.diagnose_native_hu100 import signature, support_witness, band
from scripts.run_hu200_feasibility import write
from src.arena.heuristics import hand_score
from src.arena.policies import make_policy
from src.arena.runner import public_events
from src.arena.schedule import canonical, digest
from src.blueprint.abstraction import HU200_SCHEMA, choices, information_key
from src.blueprint.average import AveragePolicy, average_rule, zero_mass_rule
from src.game.hand import Hand, Table, card_name
from src.game.observation import ActionTaken, replay as observation_replay
from src.game.types import Action, ActionKind
from src.policies.files import file_hash

OPPONENTS = ('random', 'check_call', 'tight_aggressive', 'loose_aggressive', 'pot_pressure')
INDEX = Path('docs/reports/hu200-feasibility-artifacts/model-index.json')


def seed(root, *labels):
    return int.from_bytes(hashlib.sha256(canonical([root, *labels]).encode()).digest()[:8], 'big')


def indexed_models(folder):
    index = json.loads(INDEX.read_text())
    result = []
    for m in index['models']:
        if m['target'] not in (20_000_000, 100_000_000):
            continue
        spec = m['files']['average']; path = folder / spec['path']
        if path.stat().st_size != spec['bytes'] or file_hash(path) != spec['sha256']:
            raise ValueError('Indexed model bytes differ')
        with gzip.open(path, 'rt') as f:
            metadata = json.loads(f.readline())
        h = metadata['checkpoint_header']
        if (metadata['source_checkpoint_sha256'] != m['files']['checkpoint']['sha256']
                or h['iteration'] != m['iteration'] or h['config']['seed'] != index['seed']
                or h['config']['game'] != index['game'] or h['abstraction'] != index['schema']
                or h['native_state']['completed_nodes'] != m['actual_nodes']
                or average_rule(h) != 'opponent-sampled' or zero_mass_rule(metadata) != 'uniform'
                or m['audit']['status'] != 'verified' or m['audit']['average_sha256'] != spec['sha256']):
            raise ValueError('Indexed HU200 identity differs')
        result.append({**m, 'path': str(path.resolve()), 'sha256': spec['sha256'], 'header': metadata})
    if [m['target'] for m in result] != [20_000_000, 100_000_000]:
        raise ValueError('Need exactly the two indexed endpoints')
    return result


def interval(values, family=1):
    n = len(values)
    if n < 2:
        return dict(blocks=n, bb_per_100=mean(values) if n else None, interval=None, reason='too-few-blocks')
    center = mean(values); deviation = stdev(values)
    margin = float(t.ppf(1 - .05 / (2 * family), n - 1)) * deviation / sqrt(n)
    return dict(blocks=n, bb_per_100=center, interval=[center-margin, center+margin],
                confidence=1-.05/family, family=family,
                reason='no-observed-variation' if not deviation else None)


def exposure(decisions):
    if not decisions:
        return 'no-decision'
    supports = {d['support'] for d in decisions}
    if 'unsupported-abstract-history-or-menu' in supports:
        return 'ever-unsupported'
    if 'unresolved' in supports:
        return 'unresolved'
    if any(d['lookup'] == 'missing-key' for d in decisions):
        return 'supported-missing'
    if any(d['lookup'] == 'zero-mass' for d in decisions):
        return 'zero-mass-no-missing'
    return 'all-positive'


def native_start(initial):
    p = initial._state.players_state; first = 1-initial.table.button
    return [card_name(p[s].hand[r]) for r in (0, 1) for s in (first, initial.table.button)] + [card_name(c) for c in initial._state.deck]


def play_hand(model, root, opponent, block, seat, support_cache, search):
    deal = seed(root, 'deal', opponent, block)
    own_seed = seed(root, 'candidate', opponent, block, seat)
    rival_seed = seed(root, 'rival', opponent, block, seat)
    table = Table(('player-0', 'player-1'), (20000, 20000), button=block%2)
    hand_id = f'HU200/{root}/{opponent}/{block}/{seat}'
    initial = Hand.start(table, hand_id=hand_id, seed=deal); hand = initial
    player = model.policy(own_seed); rival = make_policy(opponent, rival_seed)
    actions = []; features = []; fixtures = []; prior_off = False
    max_pot = 0; stacked = None
    for _ in range(1000):
        if hand.finished:
            break
        view = hand.observe(hand.actor)
        if observation_replay(view.history, view.seat, view.hole_cards) != view:
            raise ValueError('Observation public replay differs')
        menu = choices(view, raise_cap=None, free_fold=False)
        key = information_key(view, menu, schema=HU200_SCHEMA)
        fixtures.append(dict(actor=hand.actor, street=view.street.value, pot=view.pot,
            kinds=[k.value for k in view.legal_actions.kinds], call=view.legal_actions.call_amount,
            min_raise_to=view.legal_actions.min_raise_to, max_raise_to=view.legal_actions.max_raise_to,
            menu=[[c.name,c.action.raise_to] for c in menu], key=key))
        action = (player if hand.actor == seat else rival).choose_action(view)
        view.legal_actions.validate(action)
        off = action not in [c.action for c in menu]
        if hand.actor == seat:
            actual_menu, probabilities, known, info = model.distribution_with_telemetry(view)
            if menu != actual_menu or off or info['mode'] not in ('exact', 'uniform'):
                raise ValueError('Candidate menu/translation changed')
            support = 'stored-training-witness' if known else 'supported-observed-menu-path'
            proof = None
            if not known and prior_off:
                sig = signature(view, menu)
                if sig not in support_cache:
                    tick = perf_counter()
                    proof = support_witness(initial, sig, deadline=tick+max(0,120-search[0]))
                    search[0] += perf_counter()-tick
                    if proof['support'] == 'supported':
                        witness = initial
                        for a in proof['witness_actions']:
                            wmenu = choices(witness.observe(witness.actor), raise_cap=None, free_fold=False)
                            wa = Action(ActionKind(a['kind']),a['raise_to'])
                            if wa not in [c.action for c in wmenu]:
                                raise ValueError('Support witness action outside menu')
                            witness = witness.apply(wa)
                        wv = witness.observe(witness.actor)
                        if information_key(wv, choices(wv,raise_cap=None,free_fold=False),schema=HU200_SCHEMA) != key:
                            raise ValueError('Support witness full HU200 key differs')
                    support_cache[sig] = proof
                proof = support_cache[sig]; support = proof['support']
            visits = model.visits[key] if known else None
            wager = max(p.street_bet for p in view.players)
            feature = dict(street=view.street.value, position='BTN/SB' if seat==view.button else 'BB',
                key=key, menu=[asdict(c) for c in menu], probabilities=probabilities, lookup=info['reason'],
                visits=visits, visit_band='missing' if visits is None else band(visits,(1,2,10,100),('0','1','2-9','10-99','100+')),
                support=support, proof=proof, prior_off_menu=prior_off, pot_bb=view.pot/100,
                committed_bb=view.players[seat].contributed/100,
                legal=asdict(view.legal_actions), action=asdict(action),
                raise_ratio=(action.raise_to-wager)/(view.pot+view.legal_actions.call_amount) if action.kind==ActionKind.RAISE else None,
                hole_cards=view.hole_cards, board=view.board,
                score=hand_score(view),
                decision=len(actions))
            features.append(feature)
        actions.append([hand.actor, action.kind.value, action.raise_to])
        max_pot=max(max_pot,view.pot)
        before = hand.actor; event_count=len(hand.events); hand=hand.apply(action)
        paid=next(e.paid for e in hand.events[event_count:] if isinstance(e,ActionTaken))
        max_pot=max(max_pot,view.pot+paid)
        if before==seat and action.kind in (ActionKind.CALL,ActionKind.RAISE) and paid==view.players[seat].stack:
            stacked=dict(street=view.street.value,action=action.kind.value,decision=len(actions)-1)
        prior_off |= off
    if not hand.finished:
        raise ValueError('Hand action cap')
    final=list(hand.events[-1].stacks)
    if sum(final)!=40000:
        raise ValueError('Settlement accounting')
    again = Hand.start(table,hand_id=hand_id,seed=deal)
    repeat=model.policy(own_seed); other=make_policy(opponent,rival_seed)
    for actor, kind, target in actions:
        if again.actor != actor:
            raise ValueError('Replay actor')
        view=again.observe(actor); action=Action(ActionKind(kind),target)
        view.legal_actions.validate(action)
        if (repeat if actor==seat else other).choose_action(view) != action:
            raise ValueError('Policy reproduction differs')
        again=again.apply(action)
    if again.events!=hand.events or list(again.events[-1].stacks)!=final:
        raise ValueError('Full event replay differs')
    row=dict(opponent=opponent,block=block,seat=seat,deal_seed=deal,candidate_seed=own_seed,rival_seed=rival_seed,
        actions=actions,decisions=features,events=public_events(hand.events),final_stacks=final,net_chips=final[seat]-20000,
        exposure=exposure(features),max_pot_bb=max_pot/100,stackoff=stacked,final_zero_stack=final[seat]==0,
        holes=[initial.observe(s).hole_cards for s in (0,1)],button=table.button)
    row['sha256']=digest(row)
    fixture=dict(stack_bb=200,seed=deal,button=table.button,deck=native_start(initial),
        actions=[[k,a] for _,k,a in actions],decisions=fixtures,final_stacks=final)
    return row,fixture


def worker(plan, stage, target, out):
    out.mkdir(parents=True,exist_ok=False)
    spec=next(m for m in plan['models'] if m['target']==target)
    tick=perf_counter(); model=AveragePolicy(Path(spec['path']),spec['sha256'],expected_schema=HU200_SCHEMA)
    if model.description['entries'] != spec['entries'] or model.translation is not None:
        raise ValueError('Loaded model identity differs')
    load=perf_counter()-tick; tick=perf_counter()
    cache={};search=[0.];costs={};n=0;actions=0
    root=plan['timing_root'] if stage=='timing' else plan['final_root']
    blocks=32 if stage=='timing' else plan['blocks']
    with gzip.open(out/'hands.jsonl.gz','xt',compresslevel=1) as hands, (out/'native-fixtures.jsonl').open('x') as native:
        for opponent in OPPONENTS:
            start=perf_counter()
            for block in range(blocks):
                for seat in (0,1):
                    row,fixture=play_hand(model,root,opponent,block,seat,cache,search)
                    hands.write(canonical(row)+'\n');hands.flush()
                    native.write(canonical(fixture)+'\n');native.flush()
                    n+=1;actions+=len(row['actions'])
            costs[opponent]=dict(seconds=perf_counter()-start,blocks=blocks)
    write(out/'costs.json',dict(status='complete',plan_sha256=digest(plan),target=target,stage=stage,
        model_load_seconds=load,play_replay_seconds=perf_counter()-tick,opponents=costs,hands=n,actions=actions,
        support_search_seconds=search[0],support_signatures=len(cache),all_events_replayed=True,all_actions_reproduced=True,
        hands_sha256=file_hash(out/'hands.jsonl.gz'),fixtures_sha256=file_hash(out/'native-fixtures.jsonl')))


def rows(path):
    with gzip.open(path,'rt') as f:
        for line in f:
            yield json.loads(line)


def verify_rows(path, root, blocks):
    seen=set(); counts=Counter()
    for row in rows(path):
        spec={k:v for k,v in row.items() if k!='sha256'}
        coord=(row['opponent'],row['block'],row['seat'])
        if row['sha256']!=digest(spec) or coord in seen:
            raise ValueError('Changed or duplicate raw hand')
        op,b,s=coord
        if (row['deal_seed']!=seed(root,'deal',op,b) or row['candidate_seed']!=seed(root,'candidate',op,b,s)
                or row['rival_seed']!=seed(root,'rival',op,b,s) or sum(row['final_stacks'])!=40000
                or row['net_chips']!=row['final_stacks'][s]-20000 or row['exposure']!=exposure(row['decisions'])):
            raise ValueError('Raw schedule/accounting/classification differs')
        seen.add(coord);counts['hands']+=1;counts['actions']+=len(row['actions']);counts['decisions']+=len(row['decisions'])
        yield row
    if seen!={(op,b,s) for op in OPPONENTS for b in range(blocks) for s in (0,1)}:
        raise ValueError('Incomplete frozen sample')


def report(plan, root):
    summaries=[]; cells=[]; representatives={}; paired={};total=Counter()
    for m in plan['models']:
        target=m['target'];out=root/'final'/str(target)
        costs=json.loads((out/'costs.json').read_text())
        if costs['plan_sha256']!=digest(plan) or not costs['all_events_replayed'] or not costs['all_actions_reproduced']:
            raise ValueError('Foreign/unverified worker')
        if file_hash(out/'hands.jsonl.gz')!=costs['hands_sha256'] or file_hash(out/'native-fixtures.jsonl')!=costs['fixtures_sha256']:
            raise ValueError('Worker evidence bytes changed')
        grouped=defaultdict(list); bands=defaultdict(Counter); handcells=defaultdict(lambda:[0,0,0,0]);behavior=defaultdict(Counter)
        for row in verify_rows(out/'hands.jsonl.gz',plan['final_root'],plan['blocks']):
            op=row['opponent'];grouped[op].append(row);total['hands']+=1;total['actions']+=len(row['actions']);total['decisions']+=len(row['decisions'])
            handcells[(op,row['exposure'])][0]+=1;handcells[(op,row['exposure'])][1]+=row['net_chips']
            handcells[(op,row['exposure'])][2]+=row['net_chips']>0;handcells[(op,row['exposure'])][3]+=row['net_chips']<0
            b=behavior[op]
            for label,test in [('large_pot',row['max_pot_bb']>=64),('stackoff',row['stackoff'] is not None),('final_zero_stack',row['final_zero_stack'])]:
                if test:
                    b[label+'_hands']+=1;b[label+'_net_chips']+=row['net_chips'];b[label+'_losses']+=row['net_chips']<0;b[label+'_wins']+=row['net_chips']>0
            if row['stackoff']:
                b['stackoff_'+row['stackoff']['street']+'_'+row['stackoff']['action']]+=1
            for d in row['decisions']:
                c=bands[(op,d['street'])];c['decisions']+=1;c[d['lookup']]+=1;c['visits_'+d['visit_band']]+=1;c['support_'+d['support']]+=1
                c['action_'+d['action']['kind']]+=1
                for kind in (ActionKind.FOLD,ActionKind.CALL,ActionKind.CHECK,ActionKind.RAISE):
                    c['expected_'+kind.value]+=sum(p for item,p in zip(d['menu'],d['probabilities'],strict=True) if item['action']['kind']==kind.value)
                if d['raise_ratio'] is not None:
                    c['raise_ratio_sum']+=d['raise_ratio'];c['raise_count']+=1
                    c['raise_size_'+band(d['raise_ratio'],(.5,1,1.01),('<.5','.5-<1','1','>1'))]+=1
                if d['visits']==0:c['stored_zero_visit_'+d['lookup']]+=1
            sign='win' if row['net_chips']>0 else 'loss' if row['net_chips']<0 else 'tie'
            tags=[row['exposure']+'/'+sign] if sign!='tie' else []
            if row['net_chips']<0 and row['max_pot_bb']>=64:tags.append('large-pot-loss')
            if row['net_chips']<0 and row['stackoff']:tags.append('stackoff-loss')
            rank=digest([op,row['block'],row['seat']])
            for tag in tags:
                key=(target,op,tag)
                if key not in representatives or rank<representatives[key]['rank']:
                    representatives[key]=dict(target=target,tag=tag,rank=rank,hand=row)
        for op,data in grouped.items():
            block_values={b:mean(r['net_chips'] for r in data if r['block']==b) for b in range(plan['blocks'])}
            paired[(target,op)]=block_values
            summaries.append(dict(target=target,opponent=op,hands=len(data),absolute=interval(list(block_values.values())),behavior=dict(behavior[op]),
                exposures={e:dict(hands=v[0],net_bb=v[1]/100,wins=v[2],losses=v[3],contribution_bb_per_100=v[1]/len(data)) for (o,e),v in handcells.items() if o==op}))
        cells.extend(dict(target=target,opponent=op,street=street,**dict(c)) for (op,street),c in bands.items())
    gains={op:interval([paired[(100_000_000,op)][b]-paired[(20_000_000,op)][b] for b in range(plan['blocks'])],5) for op in OPPONENTS}
    write(root/'summary.json',dict(status='verified',plan_sha256=digest(plan),blocks=plan['blocks'],counts=dict(total),panels=summaries,gains=gains))
    write(root/'coverage-behavior.json',cells)
    write(root/'representatives.json',list(representatives.values()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('worker','report'))
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=('timing','final'));p.add_argument('--target',type=int)
    a=p.parse_args();plan=json.loads(a.plan.read_text())
    if a.command=='worker':worker(plan,a.stage,a.target,a.out)
    else:report(plan,a.out)


if __name__=='__main__':main()
