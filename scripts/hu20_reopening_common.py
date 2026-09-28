"""Common public observations and identities for the one-setting HU20 A/B."""

from collections import Counter
from dataclasses import asdict
import gzip
import json
from random import Random

from scripts.tp20_common import density
from src.arena.schedule import digest, stream_seed
from src.blueprint.abstraction import choices, information_key
from src.blueprint.lookup import TableDistribution
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind, Street


def independent_cases(plan):
    """Fixed uniform/passive/repeated-raise trajectories, independent of trained arms."""
    for block in range(plan['independent_blocks']):
        seed=stream_seed(plan['independent_root'],'validation','deal',block)
        for path in ('cap2-uniform','native-uniform','passive','later-repeated-minraise'):
            hand=Hand.start(Table(('player-0','player-1'),(2000,2000),button=block%2),
                            hand_id=f'independent-reopening-{block}-{path}',seed=seed)
            rng=Random(stream_seed(plan['independent_root'],'validation','action',block,path))
            actions=[]
            for index in range(1000):
                if hand.finished:break
                v=hand.observe(hand.actor)
                cap=2 if path=='cap2-uniform' else None
                menu=choices(v,raise_cap=cap,free_fold=False)
                yield {'block':block,'path':path,'seed':seed,'button':block%2,
                       'hand_id':hand.events[0].hand_id,'actions':list(actions),
                       'seat':v.seat,'observation_sha256':digest(asdict(v))}
                if path.endswith('uniform'):
                    action=rng.choice(menu).action
                else:
                    count=sum(e.street==v.street and e.action.kind==ActionKind.RAISE
                              for e in hand.events if hasattr(e,'action') and hasattr(e,'street'))
                    raise_now=path=='later-repeated-minraise' and v.street!=Street.PREFLOP and count<4
                    action=next((c.action for c in menu if raise_now and c.action.kind==ActionKind.RAISE),None)
                    if action is None:
                        action=Action(ActionKind.CHECK if ActionKind.CHECK in v.legal_actions.kinds else ActionKind.CALL)
                actions.append({'kind':action.kind.value,'raise_to':action.raise_to})
                hand=hand.apply(action)
            if not hand.finished:raise RuntimeError('Independent fixture decision guard')


def write_cases(plan,path):
    if path.exists():raise FileExistsError(path)
    with gzip.open(path,'wt') as out:
        count=0
        for row in independent_cases(plan):
            out.write(json.dumps(row,sort_keys=True)+'\n');count+=1
    return count


def case_view(row):
    hand=Hand.start(Table(('player-0','player-1'),(2000,2000),button=row['button']),
                    hand_id=row['hand_id'],seed=row['seed'])
    for a in row['actions']:hand=hand.apply(Action(ActionKind(a['kind']),a['raise_to']))
    v=hand.observe(row['seat'])
    if digest(asdict(v))!=row['observation_sha256']:raise ValueError('Common public observation changed')
    return v


def independent_density(trainer,path):
    source=TableDistribution(trainer);rows=Counter();work=Counter()
    with gzip.open(path,'rt') as saved:
        for line in saved:
            r=json.loads(line);v=case_view(r);menu,p,hit=source.distribution(v)
            key=source.key(v,menu);rows[(v.street.value,key)]+=1
            work[f'{r["path"]}:{v.street.value}:{"trained" if hit else "fallback"}']+=1
    return {'density':density(trainer.nodes,[{'street':s,'key':k,'decisions':n} for (s,k),n in rows.items()]),
            'exposure_by_path_street':dict(work),'observations':sum(rows.values())}
