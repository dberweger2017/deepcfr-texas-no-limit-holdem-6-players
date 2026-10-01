"""Frozen heuristic vs verified checkpoints, separate modes and sampled decisions."""
import argparse
from collections import defaultdict
import gc
import gzip
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter

from src.arena.catalog import Checkpoint
from src.arena.report import estimate
from src.arena.schedule import digest, stream_seed
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
from src.blueprint.artifact import FrozenBlueprint, HU20_UNCAPPED_FORMAT
from src.blueprint.solver import HU20_UNCAPPED_GAME
from src.diagnostics.decision_counterfactual import score_decision
from src.diagnostics.saved_hu20 import file_hash
from src.diagnostics.strong_evaluation import match, summarize_matches


class DecisionSample:
    def __init__(self,root,quota):self.root=root;self.quota=quota;self.items=defaultdict(list);self.exposure=defaultdict(int)
    def consider(self,view,menu,probabilities,trained,index,block,rotation):
        position='button' if view.seat==view.button else 'big_blind'
        key=view.street.value,position;self.exposure[key]+=1
        priority=digest([self.root,block,rotation,index])
        group=self.items[key]
        group.append((priority,view,{'index':index,'block':block,'rotation':rotation,'street':key[0],
                     'position':position,'root_trained':trained,'priority':priority}))
        group.sort(key=lambda r:r[0]);del group[self.quota:]
    def selected(self):
        return [(view,coordinate) for _,group in sorted(self.items.items()) for _,view,coordinate in group]
    def counts(self):
        return [{'street':s,'position':p,'eligible_decisions':self.exposure[s,p],
                 'selected':len(self.items[s,p])} for s in ('preflop','flop','turn','river') for p in ('button','big_blind')]


def verify_freeze(path,config):
    freeze=json.loads(path.read_text())
    for name,expected in freeze['files'].items():
        if file_hash(name)!=expected:raise ValueError('Frozen opponent dependency changed')
    if file_hash('configs/diagnostics/strong-rollout-hu20-v1.json')!=freeze['files']['configs/diagnostics/strong-rollout-hu20-v1.json']:
        raise ValueError('Frozen config differs')
    frozen=json.loads(Path('configs/diagnostics/strong-rollout-hu20-v1.json').read_text())
    if config!=frozen:raise ValueError('Configuration differs from frozen v1')
    return freeze


def load_policy(spec,inputs):
    p=inputs/spec['path']
    if p.stat().st_size!=spec['bytes'] or file_hash(p)!=spec['sha256']:raise ValueError('Policy bytes differ before loading')
    source=FrozenBlueprint(Checkpoint(spec['name'],str(p),spec['sha256'],spec['format']),p)
    if (source.players!=2 or source.raise_cap is not None or source.abstraction!=HU20_UNCAPPED_SCHEMA
        or source.game!=HU20_UNCAPPED_GAME or source.description['strategy']!='current'
        or source.description['training_seed']!=spec['seed'] or spec['format']!=HU20_UNCAPPED_FORMAT):
        raise ValueError('Policy game/schema/extraction/lineage differs')
    if 'iteration' in spec and source.description['iteration']!=spec['iteration']:raise ValueError('Policy iteration differs')
    return source


def checkpoint_changes(panels,specs):
    names={s['name']:s for s in specs};changes=[];aggregates=[]
    for mode in ('restricted','native'):
        seed_deltas=[]
        for seed in sorted({s['seed'] for s in specs}):
            rows={names[p['policy']]['milestone']:p for p in panels if p['mode']==mode and names[p['policy']]['seed']==seed}
            if set(rows)!={100000000,500000000}:continue
            old,new=rows[100000000],rows[500000000]
            if old['blocks']!=new['blocks']:raise ValueError('Checkpoint pairing differs')
            delta=[a-b for a,b in zip(new['paired_block_chips'],old['paired_block_chips'])]
            seed_deltas.append(delta);changes.append({'mode':mode,'seed':seed,'500m_minus_100m':estimate(delta)})
        if len(seed_deltas)==3:
            aggregates.append({'mode':mode,'500m_minus_100m':estimate([sum(v)/3 for v in zip(*seed_deltas)]),
                               'scope':'paired deals; conditional on the three original saved lineages, exploratory 95%'})
    return {'seed_changes':changes,'three_lineage_changes':aggregates}


def run(plan,config,freeze,out,inputs):
    out.mkdir(parents=True,exist_ok=False);start=perf_counter();all_rows=[];diagnostics=[];loaded_inputs=[];samples=[];failure=None
    def guard():
        if perf_counter()-start>plan['max_seconds']:raise TimeoutError('Frozen execution window exceeded')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>6*1024**3:raise MemoryError('M1 process RSS limit exceeded')
    try:
        for spec in plan['models']:
            guard();load_start=perf_counter();source=load_policy(spec,inputs)
            loaded_inputs.append({'name':spec['name'],'bytes':spec['bytes'],'sha256':spec['sha256'],
                           'model':source.description,'load_seconds':perf_counter()-load_start})
            guard()
            with gzip.open(out/(spec['name']+'.hands.jsonl.gz'),'wt') as records:
                for mode in plan['modes']:
                    sample=DecisionSample(plan['decision_sample_root'],plan['decisions_per_street_position']);panel_start=perf_counter()
                    for block in range(plan['blocks']):
                        guard()
                        for rotation in (0,1):
                            row=match(config,mode,plan['deal_root'],block,rotation,source=source,name=spec['name'],candidate=sample.consider)
                            all_rows.append(row);records.write(json.dumps(row,sort_keys=True)+'\n');records.flush()
                    samples.append({'model':spec['name'],'mode':mode,'strata':sample.counts()})
                    print(json.dumps({'model':spec['name'],'mode':mode,'stage':'hands','seconds':perf_counter()-panel_start,'hands':2*plan['blocks']}),flush=True)
                    with gzip.open(out/(spec['name']+'.'+mode+'.decisions.jsonl.gz'),'wt') as scored:
                        for view,coordinate in sample.selected():
                            guard();seed=stream_seed(plan['decision_sample_root'],'test','opponent',mode,coordinate['block'],coordinate['rotation'],coordinate['index'])
                            result=score_decision(view,source,config,mode,seed,selection_worlds=plan['selection_worlds'])
                            row={'model':spec['name'],'mode':mode,**coordinate,**result}
                            scored.write(json.dumps(row,sort_keys=True)+'\n');scored.flush();diagnostics.append({k:v for k,v in row.items() if k not in ('range_holdings','world_records')})
                    print(json.dumps({'model':spec['name'],'mode':mode,'stage':'scored','seconds':perf_counter()-panel_start}),flush=True)
                    del sample
            del source;gc.collect()
    except Exception as exc:failure=f'{type(exc).__name__}: {exc}'
    panels=summarize_matches(all_rows) if not failure else []
    result={'status':'incomplete' if failure else 'complete','failure':failure,'plan':plan,'plan_sha256':digest(plan),
        'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'opponent_freeze':freeze,
        'inputs':loaded_inputs,'hands':len(all_rows),'panels':panels,'decision_sampling':samples,'decision_diagnostics':diagnostics,
        'seconds':perf_counter()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if not failure:result.update(checkpoint_changes(panels,plan['models']))
    (out/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (out/'manifest.json').write_text(json.dumps({f.name:{'bytes':f.stat().st_size,'sha256':file_hash(f)} for f in out.iterdir()},indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('status','hands','seconds','peak_rss_bytes','failure')}))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/diagnostics/strong-rollout-hu20-v1.json'))
    p.add_argument('--freeze',type=Path,default=Path('docs/reports/strong-rollout-hu20-v1-artifacts/opponent-freeze.json'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);a=p.parse_args()
    config=json.loads(a.config.read_text());freeze=verify_freeze(a.freeze,config)
    result=run(json.loads(a.plan.read_text()),config,freeze,a.out,a.inputs)
    if result['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
