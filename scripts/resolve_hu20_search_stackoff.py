"""Selected recorded-hand re-solves only; no arena or policy/floor search.

The original requests freeze the scientific inputs. Alternate belief calculations
describe the frozen opponent; they are never supplied to the solver or a policy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile
from collections import Counter
from itertools import combinations
from math import fsum
from time import monotonic

from scripts.diagnose_hu20_search_stackoff import file_hash, write, SEEDS


def scientific_request(request):
    value={k:v for k,v in request.items() if k not in
           ('mode','seconds','dump_path','requested_memory_budget_bytes')}
    if 'requested_memory_budget_bytes' in request:
        value['memory_budget_bytes']=request['requested_memory_budget_bytes']
    return value


def choose_examples(analysis):
    rows=json.loads((analysis/'contrasts.json').read_text())
    selected=[next(r for r in rows if r['seed']==seed and r['delta_bb']<0) for seed in SEEDS]
    contexts={r['observation']['public_context_id'] for r in selected}
    for row in rows:
        if row['delta_bb']<0 and row['observation']['public_context_id'] not in contexts:
            selected.append(row); contexts.add(row['observation']['public_context_id'])
        if len(selected)>=5: break
    return selected


def views(row):
    from src.game.hand import Hand, Table
    from src.game.types import Action, ActionKind
    hand=Hand.start(Table(('seat0','seat1'),(2000,2000),button=row['button']),
                    hand_id=row['hand_id'],seed=row['deal_seed'])
    result=[]
    for a in row['actions']:
        result.append(hand.observe(hand.actor))
        hand=hand.apply(Action(ActionKind(a['kind']),a['raise_to']))
    return result


def extract_verified(z, name, manifest, target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.stat().st_size==manifest[name]['bytes'] and file_hash(target)==manifest[name]['sha256']:
            return
        raise ValueError('Existing extracted asset differs')
    sha=hashlib.sha256(); size=0
    with z.open(name) as source, target.open('xb') as dest:
        for chunk in iter(lambda:source.read(1024**2),b''):
            dest.write(chunk); sha.update(chunk); size+=len(chunk)
    if {'bytes':size,'sha256':sha.hexdigest()}!=manifest[name]:
        raise ValueError('Asset member differs: '+name)


def retrieve(archive, inputs, analysis, out):
    from src.arena.schedule import digest
    from src.blueprint.hu20_river import public_identity
    from src.blueprint.hu20_turn_tree import round_root
    receipt=json.loads((inputs/'restoration.json').read_text())
    stat=archive.stat()
    if (not receipt['whole_zip_verified'] or stat.st_size!=receipt['archive_bytes']
            or {'inode':stat.st_ino,'mtime_ns':stat.st_mtime_ns}!=receipt['archive_stat']):
        raise ValueError('Whole ZIP must be verified first')
    examples=choose_examples(analysis)
    hands=json.loads((analysis/'selective-hands.json').read_text())
    selected=[]; wanted={}; spots=set()
    for example in examples:
        row=next(r for r in hands if r['arm']=='search' and
                 (r['seed'],r['block'],r['rotation'])==(example['seed'],example['block'],example['rotation']))
        selected.append(row)
        for record in row['search_records']:
            if record['status']=='completed': wanted[record['request_identity']]=None
        for view in views(row):
            if view.seat==row['rotation'] and view.street.value in ('turn','river'):
                spots.add(public_identity(round_root(view.history)))
    write(out/'examples.json',examples); write(out/'hands.json',selected)
    provenance={}
    with zipfile.ZipFile(archive) as z:
        raw=z.read('RESEARCH_MEMBER_HASHES.json'); manifest=json.loads(raw)
        if hashlib.sha256(raw).hexdigest()!=receipt['manifest_sha256']:
            raise ValueError('Manifest differs from whole-ZIP restoration')
        for pod in ('gdyfqg9817qme0','ne7ui0na5wd27u','yig5a8bfutpxjg'):
            mname=f'retrieved/{pod}/manifest.json'; aname=f'retrieved/{pod}/evidence-000.tar.gz'
            extract_verified(z,mname,manifest,out/'assets'/mname)
            nested=json.loads((out/'assets'/mname).read_text())
            extract_verified(z,aname,manifest,out/'assets'/aname)
            parents={}; saved={}
            with tarfile.open(out/'assets'/aname,'r|gz') as stream:
                for member in stream:
                    if not member.isfile(): raise ValueError('Unexpected non-file in frozen archive')
                    if member.name.endswith(('request.json','receipt.json','response.jsonl')):
                        data=stream.extractfile(member).read()
                        parent=str(Path(member.name).parent)
                        if member.name.endswith('receipt.json'):
                            record=json.loads(data)
                            if record.get('spot') in spots: parents[parent]=data
                        elif member.name.endswith('request.json') and any(s.encode() in data for s in spots):
                            request=json.loads(data); identity=digest(scientific_request(request))
                            if identity in wanted:
                                if wanted[identity] is not None: continue
                                target=out/'requests'/identity/'request.json'
                                target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
                                saved[parent]=identity; wanted[identity]=str(target)
                                if parent in parents:
                                    rname=parent+'/receipt.json'; rdata=parents[parent]
                                    if {'bytes':len(rdata),'sha256':hashlib.sha256(rdata).hexdigest()}!=nested['members'][rname]:
                                        raise ValueError('Recorded solver receipt differs')
                                    (target.parent/'original-receipt.json').write_bytes(rdata)
                                spec={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
                                if spec!=nested['members'][member.name]: raise ValueError('Request hash differs')
                                provenance[identity]={'zip_member':aname,'nested_member':member.name,**spec}
                        elif member.name.endswith('response.jsonl') and parent in saved:
                            spec={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
                            if spec!=nested['members'][member.name]: raise ValueError('Response hash differs')
                            (out/'requests'/saved[parent]/'original-response.jsonl').write_bytes(data)
            print('scanned',pod,'remaining',sum(v is None for v in wanted.values()),flush=True)
            if all(v is not None for v in wanted.values()): break
        if any(v is None for v in wanted.values()):
            write(out/'missing-requests.json',wanted); raise ValueError('Missing original solve requests')
        name='fixed-work-bundle.tar.gz'
        extract_verified(z,name,manifest,out/'assets'/name)
    write(out/'request-provenance.json',provenance)


def frozen_likelihood(view, action):
    """Integrate v1's single uniform draw, without using its realized deal RNG."""
    from src.diagnostics.selective_stackoff import SelectiveStackoff, TRAP_FREQUENCY
    class Draw:
        def __init__(self,value): self.value=value
        def random(self): return self.value
    rival=SelectiveStackoff(0); result=0
    for draw, mass in ((0,TRAP_FREQUENCY),(.999999999,1-TRAP_FREQUENCY)):
        rival.random=Draw(draw)
        if rival.choose_action(view)==action: result+=mass
    return result


def frozen_range(view):
    from src.blueprint.search import DECK
    from src.game.observation import ActionTaken, replay
    opponent=1-view.seat
    pairs=combinations((c for c in DECK if c not in view.board and c not in view.hole_cards),2)
    result={}
    for hand in pairs:
        weight=1.
        for i,event in enumerate(view.history):
            if isinstance(event,ActionTaken) and event.seat==opponent:
                weight*=frozen_likelihood(replay(view.history[:i],opponent,hand),event.action)
                if not weight: break
        result[tuple(sorted(hand))]=weight
    total=fsum(result.values())
    if not total: raise ValueError('Frozen rule posterior has no support')
    return {h:w/total for h,w in result.items()}


def conditional_range(solution, view):
    """Opponent belief at this node, after root ranges, raises and hero removal."""
    from src.blueprint.hu20_turn_search import observed_likelihood
    from src.game.observation import ActionTaken
    opponent=1-view.seat
    result=dict(solution.ranges[opponent])
    for h in result:
        if set(h).intersection(view.hole_cards): result[h]=0.
    for i,event in enumerate(view.history):
        if i>=len(solution.root) and isinstance(event,ActionTaken) and event.seat==opponent:
            matrix=solution.matrix(view.history[:i])
            for h in result:
                if result[h]: result[h]*=observed_likelihood(matrix.menu,matrix.row(h),event.action)
    total=fsum(result.values())
    if not total: raise ValueError('Search node posterior has no support')
    return {h:w/total for h,w in result.items()}


def describe_range(weights, view):
    from src.game.observation import replay
    from src.diagnostics.selective_stackoff import strength
    from src.diagnostics.stackoff_tails import concrete_category
    from src.game.showdown import hand_value
    opponent=1-view.seat; tiers=Counter(); categories=Counter(); ahead=tie=0.
    hero=hand_value(view.hole_cards+view.board)
    rows=[]
    for h,w in weights.items():
        if not w: continue
        rival=replay(view.history,opponent,h)
        tier=strength(rival); category=concrete_category(rival)
        tiers[tier]+=w; categories[category]+=w
        value=hand_value(tuple(h)+view.board)
        ahead+=w*(hero>value); tie+=w*(hero==value)
        rows.append({'hand':list(h),'weight':w,'tier':tier,'category':category})
    return {'positive_holdings':len(rows),'tiers':dict(tiers),'categories':dict(categories),
            'hero_made_ahead_mass':ahead,'hero_made_tie_mass':tie,
            'top_holdings':sorted(rows,key=lambda r:(-r['weight'],r['hand']))[:12]},rows


def resolve(inputs, out, binary):
    import gc
    from dataclasses import replace
    import numpy as np
    from src.arena.schedule import digest
    from src.blueprint.average import AveragePolicy
    from src.blueprint.hu20_turn_search import HU20TurnSearchPolicy, TurnSearchConfig
    from src.blueprint.hu20_turn_solver import ExternalTurnSolver, parse_profiles
    from src.blueprint.hu20_river import public_identity
    from src.blueprint.hu20_turn_tree import round_root
    plan=json.loads((inputs/'plan.json').read_text())
    harness_source=binary.resolve().parents[2]/'src/main.rs'
    if file_hash(harness_source)!=plan['amendment']['native_harness_source_sha256']:
        raise ValueError('External harness source differs from frozen source')
    specs={s['seed']:s for s in plan['models']}
    # The frozen bundle hash was checked during retrieval; materialize inference files only.
    with tarfile.open(out/'assets/fixed-work-bundle.tar.gz','r|gz') as stream:
        for member in stream:
            spec=next((s for s in specs.values() if Path(member.name).name==s['path']),None)
            if spec:
                path=out/'models'/spec['path']; path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():
                    if path.stat().st_size!=spec['bytes'] or file_hash(path)!=spec['sha256']:
                        raise ValueError('Existing frozen inference model differs')
                    continue
                with stream.extractfile(member) as source,path.open('xb') as dest:
                    for chunk in iter(lambda:source.read(1024**2),b''): dest.write(chunk)
                if path.stat().st_size!=spec['bytes'] or file_hash(path)!=spec['sha256']:
                    raise ValueError('Frozen inference model hash differs')
    requests={p.parent.name:json.loads(p.read_text()) for p in (out/'requests').glob('*/request.json')}
    external=ExternalTurnSolver(binary,out/'solves',allocation_budget=lambda _:3*1024**3)
    reusable={}
    for p in (out/'solves').glob('*/request.json'):
        directory=p.parent; record=json.loads((directory/'receipt.json').read_text())
        if record['status']!='completed': continue
        files=json.loads((directory/'manifest.json').read_text())
        for name in ('request.json','profile.jsonl','receipt.json','response.jsonl'):
            path=directory/name
            if path.stat().st_size!=files[name]['bytes'] or file_hash(path)!=files[name]['sha256']:
                raise ValueError('Prior completed re-solve evidence changed')
        key=digest(scientific_request(json.loads(p.read_text())))
        reusable[key]=(directory,record)
    class RecordedSolver:
        expected_sha256=external.expected_sha256
        @property
        def records(self): return external.records
        def solve(self,request,deadline):
            key=digest(request)
            original=requests.get(key)
            if original is None: raise ValueError('Reconstructed request absent from selected frozen requests: '+key)
            if scientific_request(original)!=request: raise ValueError('Scientific request differs')
            if key in reusable:
                directory,record=reusable[key]
                external.records.append(dict(record,reused_verified_profile=True))
                return parse_profiles(request,directory/'profile.jsonl')
            return external.solve(request,deadline)
    config=TurnSearchConfig(**plan['selected_search_config'])
    examples=json.loads((out/'examples.json').read_text()); hands=json.loads((out/'hands.json').read_text())
    summaries=[]; started=monotonic(); query_checks=0
    for seed in SEEDS:
        source=AveragePolicy(out/'models'/specs[seed]['path'],specs[seed]['sha256'])
        for example,row in [(e,r) for e,r in zip(examples,hands,strict=True) if r['seed']==seed]:
            policy=HU20TurnSearchPolicy(source,RecordedSolver(),config)
            all_views=views(row)
            wagers=[(a['raise_to']-a['observation']['street_bet'] if a['kind']=='raise' else
                     a['observation']['call_amount'] if a['kind']=='call' else 0,a['index'])
                    for a in row['actions'] if a['logical_player']==0 and a['street'] in ('turn','river')]
            largest_index=max(wagers)[1]
            selected_indices={example['decision_index'],largest_index}
            for index,(view,a) in enumerate(zip(all_views,row['actions'],strict=True)):
                if view.seat!=row['rotation'] or view.street.value not in ('turn','river'): continue
                menu,p,_=policy.distribution(view,query_kind='play')
                observed=a['observation']
                if [(c.action.kind.value,c.action.raise_to) for c in menu]!=[(c['kind'],c['raise_to']) for c in observed['menu']]:
                    raise ValueError('Re-solved menu differs from recorded menu')
                error=float(np.max(np.abs(np.array(p)-observed['probabilities'])))
                if error>2e-5: raise ValueError('Re-solved probabilities differ from recorded decision')
                query_checks+=1
                if index not in selected_indices: continue
                solution=policy.live_solutions[view.seat,view.history]
                search=conditional_range(solution,view); truth=frozen_range(view)
                # Hold the recorded turn strategy factors fixed: this is a belief sensitivity,
                # not a second policy solve or a test of the .01 policy's gameplay.
                floor_policy=HU20TurnSearchPolicy(source,RecordedSolver(),replace(config,decision_seconds=120,opponent_likelihood_floor=.01))
                floor_policy.live_turn_models=dict(policy.live_turn_models)
                floor_policy.last_turn_source=dict(policy.last_turn_source)
                floor_policy.played=dict(policy.played)
                root=round_root(view.history)
                floor_ranges,coverage=floor_policy._ranges(root,view.seat,None)
                floored=replace(solution,ranges=floor_ranges)
                from src.game.observation import replay
                root_view=replay(root,view.seat,view.hole_cards)
                root_search=conditional_range(solution,root_view)
                floor=conditional_range(floored,root_view)
                base_menu,base_p,_=source.distribution(view)
                summary={'seed':seed,'block':row['block'],'rotation':row['rotation'],'index':index,
                         'street':view.street.value,'board':list(view.board),'hero':list(view.hole_cards),
                         'pot':view.pot,'call_amount':view.legal_actions.call_amount,'delta_bb':example['delta_bb'],
                         'base_probabilities':[{ 'kind':c.action.kind.value,'raise_to':c.action.raise_to,'p':v} for c,v in zip(base_menu,base_p,strict=True)],
                         'search_probabilities':[{ 'kind':c.action.kind.value,'raise_to':c.action.raise_to,'p':v} for c,v in zip(menu,p,strict=True)],
                         'recorded_probability_max_error':error,'request_identity':digest(solution.request),
                         'floor_coverage':coverage,'search_range_coverage':solution.coverage,
                         'decision_role':'first-divergence' if index==example['decision_index'] else 'largest-late-wager'}
                for label,weights in (('search',search),('frozen_rules',truth),
                                      ('search_at_round_root',root_search),('floor_001_at_round_root_fixed_turn',floor)):
                    desc,raw=describe_range(weights,view); summary[label]=desc
                    name=f'{seed}-{row["block"]}-{row["rotation"]}-{index}-{label}.json'
                    write(out/'range-rows'/name,raw)
                summary['floor_root_total_variation']=sum(abs(root_search.get(h,0)-floor.get(h,0)) for h in root_search)/2
                summary['floor_new_root_support_mass']=sum(w for h,w in floor.items() if not root_search.get(h,0))
                summaries.append(summary);write(out/'resolution-summary.json',{'examples':summaries,'query_checks':query_checks,'seconds':monotonic()-started,
                       'binary_sha256':file_hash(binary),'records':external.records})
                print('resolved',seed,row['block'],row['rotation'],view.street.value,flush=True)
                if index==max(selected_indices): break
            del policy
        del source;gc.collect()
    write(out/'resolution-summary.json',{'examples':summaries,'query_checks':query_checks,'seconds':monotonic()-started,
          'binary_sha256':file_hash(binary),'records':external.records})


def responses(out):
    """Inspect already solved responses to the actual wager, without another solve."""
    from src.arena.schedule import digest
    from src.blueprint.hu20_turn_solver import parse_profiles
    from src.blueprint.hu20_turn_tree import round_root, betting_line, line_key
    from src.blueprint.abstraction import choices
    from src.game.observation import replay
    started=monotonic()
    summary=json.loads((out/'resolution-summary.json').read_text())
    hands=json.loads((out/'hands.json').read_text())
    profile_paths={}
    for p in (out/'solves').glob('*/request.json'):
        req=json.loads(p.read_text()); profile_paths[digest(scientific_request(req))]=p.parent
    for example in summary['examples']:
        row=next(r for r in hands if (r['seed'],r['block'],r['rotation'])==
                 (example['seed'],example['block'],example['rotation']))
        index=example['index']; action=row['actions'][index]
        example['rival_cards_diagnostic_only']=row['actions'][index+1]['observation']['hole_cards'] if index+1<len(row['actions']) else None
        if (action['kind']!='raise' or index+1>=len(row['actions'])
                or row['actions'][index+1]['street']!=action['street']): continue
        before,after=views(row)[index:index+2]
        directory=profile_paths[example['request_identity']]
        manifest=json.loads((directory/'manifest.json').read_text())
        for name in ('request.json','profile.jsonl'):
            path=directory/name
            if path.stat().st_size!=manifest[name]['bytes'] or file_hash(path)!=manifest[name]['sha256']:
                raise ValueError('Completed response profile changed')
        request=json.loads((directory/'request.json').read_text())
        profiles=parse_profiles(request,directory/'profile.jsonl')
        matrix=profiles[line_key(betting_line(round_root(before.history),after.history))]
        prefix=f'{row["seed"]}-{row["block"]}-{row["rotation"]}-{index}'
        search=json.loads((out/'range-rows'/(prefix+'-search.json')).read_text())
        true=json.loads((out/'range-rows'/(prefix+'-frozen_rules.json')).read_text())
        model=Counter(); frozen=Counter()
        for r in search:
            for c,p in zip(matrix.menu,matrix.row(r['hand']),strict=True):
                model[c.action.kind.value]+=r['weight']*p
        for r in true:
            candidate=replay(after.history,after.seat,tuple(r['hand']))
            for c in choices(candidate,raise_cap=None,free_fold=False):
                frozen[c.action.kind.value]+=r['weight']*frozen_likelihood(candidate,c.action)
        if abs(fsum(model.values())-1)>1e-8 or abs(fsum(frozen.values())-1)>1e-8:
            raise ValueError('Response model is not normalized')
        example['solved_response_model']=dict(model)
        example['frozen_rule_response']=dict(frozen)
        example['actual_response']=row['actions'][index+1]['kind']
    summary['response_analysis_seconds']=monotonic()-started
    write(out/'resolution-summary.json',summary)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('retrieve','resolve','responses')); p.add_argument('--archive',type=Path)
    p.add_argument('--inputs',type=Path,required=True); p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--binary',type=Path); a=p.parse_args()
    if a.command=='retrieve': retrieve(a.archive,a.inputs,a.analysis,a.out)
    elif a.command=='resolve': resolve(a.inputs,a.out,a.binary)
    else: responses(a.out)


if __name__=='__main__': main()
