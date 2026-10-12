"""Replay every paired direct hand before numerical readout; preserve street coverage."""
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
from time import perf_counter

from src.arena.runner import public_events
from src.arena.schedule import digest, stream_seed
from src.blueprint.abstraction import choices, information_key, HU20_EQUITY_SCHEMA, HU20_UNCAPPED_SCHEMA
from src.blueprint.equity_buckets import EquityCards
from src.diagnostics.stackoff_tails import public_context
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind
from src.policies.files import file_hash
from scripts.hu20_k50_first_pass_measurements import quantiles


def replay(plan,out,root):
    started=perf_counter()
    cells={};decisions=0;files=[];reached=defaultdict(dict);models={}
    table_paths={pair[role]['card_buckets'] for pair in plan['pairs'] for role in ('candidate','reference')
                 if pair[role].get('abstraction')==HU20_EQUITY_SCHEMA}
    if len(table_paths)>1:raise ValueError('Mixed K50 table dependencies')
    cards=EquityCards(root/next(iter(table_paths))) if table_paths else None
    for lineage,pair in enumerate(plan['pairs'],1):
        for role in ('candidate','reference'):
            spec=pair[role]
            if spec['name'] in models and models[spec['name']]!=spec:raise ValueError('Model alias differs')
            models[spec['name']]=spec
        result=json.loads((out/f'direct-lineage-{lineage}.result.json').read_text())
        if result['status']!='complete' or result['hands']!=2*plan['blocks'] or result['plan_sha256']!=digest(plan):
            raise ValueError('Incomplete or foreign lineage')
        path=out/f'direct-lineage-{lineage}.hands.jsonl.gz'
        files.append({'path':str(path),'bytes':path.stat().st_size,'sha256':file_hash(path)})
        with gzip.open(path,'rt') as source:
            for row in map(json.loads,source):
                coordinate=(lineage,row['block'],row['rotation'])
                if coordinate in cells or row['root_seed']!=plan['root'] or row['lineage']!=lineage:
                    raise ValueError('Duplicate or foreign direct coordinate')
                if row['policy']!=pair['candidate']['name'] or row['reference']!=pair['reference']['name']:
                    raise ValueError('Policy labels differ')
                if row['deal_seed']!=stream_seed(plan['root'],'test','deal',2,row['block']) or row['button']!=row['block']%2:
                    raise ValueError('Foreign deal or button')
                hand=Hand.start(Table(('seat0','seat1'),(2000,2000),button=row['button']),hand_id=row['hand_id'],seed=row['deal_seed'])
                for index,record in enumerate(row['actions']):
                    if hand.actor!=record['seat'] or index!=record['index']:raise ValueError('Actor/order differs')
                    view=hand.observe(hand.actor);obs=record['observation']
                    if obs['hole_cards']!=list(view.hole_cards) or obs['board']!=list(view.board) or obs['pot']!=view.pot or record['street']!=view.street.value:
                        raise ValueError('Recorded observation differs')
                    context=public_context(view)
                    if obs['public_context']!=context or obs['public_context_id']!=digest(context):
                        raise ValueError('Public history or legal bounds differ')
                    role='candidate' if hand.actor==row['rotation'] else 'reference'
                    spec=pair[role];menu=choices(view,raise_cap=None,free_fold=False)
                    key=information_key(view,menu,schema=spec.get('abstraction',HU20_UNCAPPED_SCHEMA),
                                        equity_cards=cards if spec.get('abstraction')==HU20_EQUITY_SCHEMA else None)
                    if role=='candidate':
                        key_recorded=record['target_key'];p=obs['probabilities'];visits=record['training_visits'];known=obs['trained']
                    else:
                        evidence=record['reference_decision']
                        key_recorded=evidence['key'];p=evidence['probabilities'];visits=evidence['training_visits'];known=evidence['trained']
                        if evidence['menu']!=[choice.name for choice in menu]:raise ValueError('Rival menu differs')
                    if key_recorded!=key:raise ValueError('Policy-owned schema key differs')
                    if type(visits) is not int or visits<0 or type(known) is not bool:raise ValueError('Visit/support evidence invalid')
                    if len(p)!=len(menu) or any(not math.isfinite(v) or v<0 for v in p) or not math.isclose(math.fsum(p),1,abs_tol=1e-8):
                        raise ValueError('Nonfinite or unnormalized probabilities')
                    actual=Action(ActionKind(record['kind']),record['raise_to'])
                    view.legal_actions.validate(actual)
                    if not any(choice.action==actual and prob>0 for choice,prob in zip(menu,p,strict=True)):
                        raise ValueError('Action outside its own positive-probability menu')
                    evidence=(record['street'],visits,known)
                    old=reached[spec['name']].get(key)
                    if old is not None and old!=evidence:raise ValueError('Street/visit identity changed')
                    reached[spec['name']][key]=evidence
                    hand=hand.apply(actual);decisions+=1
                if not hand.finished:raise ValueError('Unfinished replay')
                chips=[player.stack-2000 for player in hand.observe(0).players]
                if chips!=row['net_chips_by_seat'] or sum(chips)!=0 or chips[row['rotation']]!=row['target_chips']:
                    raise ValueError('Settlement/accounting differs')
                if digest(public_events(hand.events))!=row['public_events_sha256'] or not row['native_replay_verified']:
                    raise ValueError('Public event digest differs')
                cells[coordinate]=row['target_chips']
    if set(cells)!={(l,b,r) for l in range(1,len(plan['pairs'])+1) for b in range(plan['blocks']) for r in (0,1)}:
        raise ValueError('Incomplete paired deal coverage')
    coverage={}
    for name,keys in reached.items():
        streets={}
        for street in ('preflop','flop','turn','river'):
            known_values=Counter(v for s,v,known in keys.values() if s==street and known)
            missing=sum(s==street and not known for s,_,known in keys.values())
            streets[street]={'stored_reached_keys':quantiles(known_values),'missing_reached_keys':missing}
        covered=sum(known for _,_,known in keys.values())
        coverage[name]={'unique_reached_keys':len(keys),'unique_stored_reached_keys':covered,
                        'street_quantiles_scope':'distinct stored keys reached in this direct contrast, not all table keys',
                        'streets':streets}
    receipt={'status':'verified','hands_replayed':len(cells),'decisions_checked':decisions,'plan_sha256':digest(plan),
        'raw_files':files,'seconds':perf_counter()-started,'policy_owned_key_checks':True,'public_observations_and_legal_bounds_checked':True,
        'coverage':coverage}
    return cells,receipt
