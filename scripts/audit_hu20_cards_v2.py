"""Model-free refinement audit of retained #139 holdings and public key contexts."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
import subprocess
from time import perf_counter

from src.blueprint.abstraction import HU20_CARD_V2_SCHEMA, HU20_UNCAPPED_SCHEMA, _postflop, choices, information_key
from src.blueprint.cards_v2 import VERSION, postflop_v2
from src.blueprint.hu20_river import peak_rss
from src.diagnostics.saved_hu20 import file_hash
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind, Street

HIGHLIGHTS = (
    ('7h Td Ts 2s 2c', '4d 5s', '7s Kd'),
    ('Td Tc 9s Th 7s', '3h 5s', 'Ad Ks'),
    ('Qh 9s Ad 3d As', '2h 4c', 'Ks Th'),
    ('4h Jh 3s Jd', '2c 7d', '8d Ah'),
    ('8s 8d 8h', '3h 4d', '5h Ac'),
)


def context_templates():
    hand=Hand.start(Table(('player-0','player-1'),(2000,2000)),hand_id='model-free-context',seed=202610010601)
    result={}
    while not hand.finished:
        view=hand.observe(hand.actor)
        result.setdefault(view.street.value,view)
        hand=hand.apply(Action(ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds else ActionKind.CALL))
    return result


def audit(source,out):
    started=perf_counter();out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source/'manifest.json').read_text())
    for name,spec in manifest.items():
        path=source/name
        if path.stat().st_size!=spec['bytes'] or file_hash(path)!=spec['sha256']:
            raise ValueError('Retained #139 evidence differs')
    templates=context_templates();groups=defaultdict(list);buckets=defaultdict(lambda: [set(),set()]);keys=defaultdict(lambda: [set(),set()]);counts=Counter();rows=[]
    with (out/'descriptors.jsonl').open('x') as output:
        for line in (source/'holdings.jsonl').open():
            row=json.loads(line);cards=tuple(row['cards']);board=tuple(row['board']);old=_postflop(cards,board);new=postflop_v2(cards,board)
            if list(old)!=row['bucket'] or new[0]!=old:raise ValueError('v2 is not the declared v1 refinement')
            # Static model-free key contexts: public betting/menu is identical
            # for each schema; only own cards/current board vary. No native
            # simulated deck is used to make these representation queries.
            view=replace(templates[row['street']],hole_cards=cards,board=board)
            menu=choices(view,raise_cap=None,free_fold=False)
            old_key=information_key(view,menu,schema=HU20_UNCAPPED_SCHEMA);new_key=information_key(view,menu,schema=HU20_CARD_V2_SCHEMA)
            buckets[row['street']][0].add(old);buckets[row['street']][1].add(new)
            keys[row['street']][0].add(old_key);keys[row['street']][1].add(new_key)
            groups[row['street'],row['board_index'],old].append((row,new));counts[row['street']]+=1
            data={k:row[k] for k in ('street','board_index','holding_index','cards','board','equity','method')}
            data.update(v1=old,v2=new,v1_key=old_key,v2_key=new_key)
            output.write(json.dumps(data,sort_keys=True)+'\n');rows.append(data)
    collisions=Counter();residual=[]
    for cluster in groups.values():
        for (a,aa),(b,bb) in combinations(cluster,2):
            if a['made_value']==b['made_value']:continue
            street=a['street'];collisions[street+':v1_different_value_pairs']+=1
            if aa!=bb:collisions[street+':separated']+=1
            else:
                collisions[street+':residual']+=1
                residual.append({'street':street,'board':a['board'],'low':a['cards'],'high':b['cards'],
                    'values':[a['made_value'],b['made_value']],'equity_spread':abs(a['equity']-b['equity']),'v2':aa})
    highlights=[]
    for board,lo,hi in HIGHLIGHTS:
        board=tuple(board.split());lo=tuple(lo.split());hi=tuple(hi.split())
        if _postflop(lo,board)!=_postflop(hi,board):raise ValueError('Highlighted v1 collision changed')
        a,b=postflop_v2(lo,board),postflop_v2(hi,board)
        if a==b:raise ValueError('Highlighted #139 collision remains')
        highlights.append({'board':board,'low':lo,'high':hi,'v1':a[0],'v2_low':a,'v2_high':b,'separated':True})
    result={'status':'passed','descriptor_version':VERSION,'schema':HU20_CARD_V2_SCHEMA,
        'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'descriptor_sha256':file_hash('src/blueprint/cards_v2.py'),'source_holdings':manifest['holdings.jsonl'],
        'rows':len(rows),'street_counts':dict(counts),
        'growth':{s:{'v1_buckets':len(v[0]),'v2_buckets':len(v[1]),'bucket_ratio':len(v[1])/len(v[0]),
            'v1_keys':len(keys[s][0]),'v2_keys':len(keys[s][1])} for s,v in buckets.items()},
        'same_board_made_value_pairs':dict(collisions),'highlighted_collisions':highlights,
        'largest_residual_collisions':sorted(residual,key=lambda r:-r['equity_spread'])[:20],
        'scope':'retained uniform-card sample; static identical public key contexts; no policy occupancy/equity recomputation',
        'seconds':perf_counter()-started,'peak_rss_bytes':peak_rss()}
    (out/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (out/'manifest.json').write_text(json.dumps({p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in out.iterdir()},indent=2,sort_keys=True)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('docs/reports/hu20-bucket-coarseness-artifacts'));p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=audit(a.source,a.out);print(json.dumps({k:r[k] for k in ('status','rows','growth','same_board_made_value_pairs','seconds','peak_rss_bytes')}))


if __name__=='__main__':main()
