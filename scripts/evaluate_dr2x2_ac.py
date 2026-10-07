"""Sealed A/C current, paired readout and common-law restricted river evidence."""
import argparse
import gc
from gzip import GzipFile
from io import TextIOWrapper
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter, time

from scripts.evaluate_hu20_cards_v2 import hand
from scripts.evaluate_hu20_river import root_hand
from scripts.evaluate_hu20_stackoff import swap_bytes
from scripts.hu20_platform_pilot import peak_rss, write
from src.arena.schedule import digest
from src.blueprint.river_cfr import profile_quality
from src.blueprint.river_game import river_root_history
from src.blueprint.average import AveragePolicy
from src.diagnostics.history_river import CommonRiverGame, common_ranges, policy_profile, LAW, PROJECTION
from src.diagnostics.saved_hu20 import load_saved, file_hash


def host_swap():
    if sys.platform=='linux':
        memory={line.split(':')[0]:int(line.split()[1])*1024
                for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith(('SwapTotal:','SwapFree:'))}
        return memory['SwapTotal']-memory['SwapFree']
    return swap_bytes()


def science_fields(value):
    """Compare science across hosts, retaining completion/fallback and decisions."""
    if isinstance(value,dict):
        return {k:science_fields(v) for k,v in value.items()
                if not k.endswith('seconds')}
    if isinstance(value,(list,tuple)):
        return [science_fields(v) for v in value]
    return value


def open_average(spec, guard):
    guard()
    path=Path(spec['average_path'])
    source=AveragePolicy(path,spec['average_sha256'],expected_schema=spec['abstraction'])
    if (source.description['source_checkpoint_sha256']!=spec['checkpoint_sha256']
            or source.description['training_seed']!=spec['seed']
            or source.description['iteration']!=spec['iteration']):
        raise ValueError('Average identity differs from frozen input')
    return source,source.visits


def blind_row(row):
    return {**row,'target_chips':None,'net_chips_by_seat':None,'tails':None,
            'validation_payoffs_suppressed':True}


def execute(plan, seed, out, *, admission=False, control=None, inputs=Path('/')):
    if plan['law']!=LAW or plan['projection']!=PROJECTION:
        raise ValueError('Declared river law/projection differs')
    selected=[s for s in plan['models'] if s['seed']==seed]
    if {s['cell'] for s in selected}!={'A','C'} or len(selected)!=2:
        raise ValueError('Exactly one matched A/C pair required')
    if not admission:
        if control is None:raise ValueError('Strength needs explicit admitted controller lease')
        proof=json.loads(control.read_text())
        if proof.get('admitted_plan_sha256')!=digest(plan) or not proof.get('linux_reference_match'):
            raise ValueError('Strength lacks exact-plan Linux admission')
    selected=[{**s,**{k:str(inputs/s[k]) for k in ('path','checkpoint_path','average_path')}} for s in selected]
    out.mkdir(parents=True,exist_ok=False)
    write(out/'plan.json',plan)
    started=perf_counter();swap_before=host_swap();last=-1.;results=[];failure=None;coordinate=None
    def guard():
        nonlocal last
        elapsed=perf_counter()-started
        if admission and elapsed>=plan['limits']['admission_max_seconds']:
            raise TimeoutError('Serialized admission time guard')
        if elapsed-last<1:return
        last=elapsed
        if peak_rss()>=plan['limits']['max_rss_gib']*2**30:
            raise MemoryError('Evaluation RSS guard')
        if shutil.disk_usage(out).free<plan['limits']['min_free_disk_gib']*2**30:
            raise OSError('Evaluation disk guard')
        if host_swap()-swap_before>plan['limits']['max_swap_growth_gib']*2**30:
            raise MemoryError('Evaluation swap guard')
        if control:
            lease=json.loads(control.read_text())
            if lease.get('stop') or lease.get('lease_until',0)<time():
                raise RuntimeError('Controller stop/stale lease; preserve partial evidence')
    try:
        for spec in selected:
            for readout in ('current','average'):
                guard();began=perf_counter()
                source,visits=(load_saved(spec,Path('/'),guard,expected_schema=spec['abstraction'])
                               if readout=='current' else open_average(spec,guard))
                if source.description['iteration']!=spec['iteration']:
                    raise ValueError('Frozen checkpoint iteration differs')
                load_seconds=perf_counter()-began
                stages=['admission'] if admission else (['primary','readout'] if readout=='current' else ['readout'])
                for stage in stages:
                    filename=f"{stage}-{spec['cell']}-{seed}-{readout}.hands.jsonl.gz"
                    timings=[];fingerprints=[];hands=0;begin=perf_counter()
                    with (out/filename).open('xb') as raw:
                        with GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as zipped,TextIOWrapper(zipped,encoding='utf-8') as stream:
                            for panel in plan['panels']:
                                count=(plan['admission_blocks'] if admission else panel['blocks'] if stage=='primary' else panel['readout_blocks'])
                                root=(panel['serialization_root'] if admission else panel['primary_root'] if stage=='primary' else panel['readout_root'])
                                times=[];bytes_written=0
                                for block in range(count):
                                    for rotation in (0,1):
                                        coordinate={'cell':spec['cell'],'seed':seed,'readout':readout,'stage':stage,
                                                    'panel':panel['name'],'root':root,'block':block,'rotation':rotation}
                                        guard();start_hand=perf_counter()
                                        row=hand(source,visits,spec,{**panel,'root':root},block,rotation,plan,resource_only=False,
                                                 failure_sink=lambda failed:write(out/'failed-hand.json',
                                                     {'coordinate':coordinate,'row':blind_row(failed) if admission else failed}))
                                        if row['status']!='complete' or not row['native_replay_verified']:
                                            raise ValueError('Unverified/incomplete hand')
                                        row.update(stage=stage,cell=spec['cell'],readout=readout,
                                                   checkpoint_sha256=spec['checkpoint_sha256'],
                                                   policy_sha256=spec['sha256'] if readout=='current' else spec['average_sha256'])
                                        for action in row['actions']:
                                            observed=action['observation']
                                            if action['logical_player']==0:
                                                observed['readout_mass_status']=('missing' if not observed['trained'] else
                                                    'zero_mass' if observed['key'] in getattr(source,'zero_mass',()) else
                                                    'positive_mass' if readout=='average' else 'current')
                                        if admission:
                                            fingerprints.append(digest(science_fields(row)))
                                            row=blind_row(row)
                                        encoded=json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
                                        stream.write(encoded);stream.flush();hands+=1;bytes_written+=len(encoded.encode())
                                        times.append(perf_counter()-start_hand)
                                    write(out/'progress.json',{'cell':spec['cell'],'readout':readout,'stage':stage,
                                          'panel':panel['name'],'closed_blocks':block+1,'task_hands':hands,
                                          'seconds':perf_counter()-started,'peak_rss_bytes':peak_rss()})
                                timings.append({'panel':panel['name'],'hands':len(times),'seconds':sum(times),
                                                'max_hand_seconds':max(times),'uncompressed_bytes':bytes_written})
                        raw.flush();os.fsync(raw.fileno())
                    receipt={'status':'complete','cell':spec['cell'],'seed':seed,'readout':readout,'stage':stage,
                             'file':filename,'sha256':file_hash(out/filename),'bytes':(out/filename).stat().st_size,
                             'hands':hands,'panels':timings,'seconds':perf_counter()-begin,'load_seconds':load_seconds,
                             'checkpoint_sha256':spec['checkpoint_sha256'],
                             'policy_sha256':spec['sha256'] if readout=='current' else spec['average_sha256']}
                    if admission:receipt.update(scientific_fingerprints=fingerprints,strength_payoffs_suppressed=True)
                    results.append(receipt);write(out/(filename+'.receipt.json'),receipt)
                if not admission:
                    qualities=[]
                    for case in plan['river_roots']:
                        guard()
                        root=river_root_history(root_hand(case['seed'],case['raise_to']).events)
                        game=CommonRiverGame(root,common_ranges(root),raise_cap=plan['river_raise_cap'],max_public_nodes=plan['max_public_nodes'])
                        profile,telemetry=policy_profile(game,source,guard)
                        quality=profile_quality(game,profile)
                        if quality['zero_sum_error_bb']>1e-8:raise ValueError('Restricted quality zero-sum check')
                        qualities.append({'case':case,'range_sha256':digest(common_ranges(root)),'public_nodes':len(game.nodes),
                                          'projection_telemetry':telemetry,'quality':quality})
                        del game,profile;gc.collect()
                    write(out/f"river-{spec['cell']}-{seed}-{readout}.json",{'law':LAW,'projection':PROJECTION,'records':qualities})
                del source,visits;gc.collect()
    except Exception as exc:
        failure=f'{type(exc).__name__}: {exc}'
        write(out/'failure.json',{'failure':failure,'coordinate':coordinate,'no_retry':True})
    summary={'status':'failed-retained' if failure else 'complete','failure':failure,'seed':seed,
             'mode':'serialized-blind-admission' if admission else 'strength', 'plan_sha256':digest(plan),
             'tasks':results,'closed_hands':sum(r['hands'] for r in results),'seconds':perf_counter()-started,
             'peak_rss_bytes':peak_rss(),'swap_growth_bytes':host_swap()-swap_before,
             'candidate_river_quality_computed':not admission and not failure}
    write(out/'summary.json',summary)
    write(out/'manifest.json',{p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in out.iterdir() if p.is_file()})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--admission',action='store_true')
    p.add_argument('--control',type=Path)
    p.add_argument('--inputs',type=Path,default=Path('/'))
    a=p.parse_args();result=execute(json.loads(a.plan.read_text()),a.seed,a.out,admission=a.admission,control=a.control,inputs=a.inputs)
    print(json.dumps({k:result[k] for k in ('status','seed','mode','seconds','closed_hands')}),flush=True)
    raise SystemExit(result['status']!='complete')
