"""Run the frozen, file-boundary turn campaign, one guarded solver at a time."""
import argparse
from datetime import datetime,timedelta,timezone
import fcntl
import json
import math
from pathlib import Path
import signal
import sys
from time import monotonic
from scripts.prepare_flop_check import load_policy
from scripts.preflight_flop_check import prepare_guarded
from scripts.validate_flop_check import response_rows
from src.blueprint.hu20_river import public_ranges
from src.blueprint.river_game import RiverUnsupported
from src.diagnostics.flop_check import atomic_json,compile_tree
from src.diagnostics.flop_check_runtime import append,machine_snapshot,run_tool,swap_usage
from src.diagnostics.saved_hu20 import file_hash
from src.diagnostics.turn_campaign import jobs,secondary_ranges,summarize
from src.diagnostics.turn_check import export,replay_root

GIB=1024**3
HEAVY_FIELDS={'fold_nodes','overfold_groups','selected_lbr_nodes','target_metrics','secondary_metrics'}


def counter_record(row):
    """Keep per-hand evidence on disk, rather than accumulating it in the driver."""
    return {k:v for k,v in row.items() if k not in HEAVY_FIELDS}


def prepare_job(job_path,inputs,out,stored):
    document=json.loads(Path(job_path).read_text());job=document['job'];spec=document['policy']
    out=Path(out);out.mkdir(parents=True,exist_ok=False);root=replay_root(job['root'])
    request,_=compile_tree(root);source=load_policy(spec,inputs)
    try:ranges,coverage=public_ranges(source,root)
    except RiverUnsupported as error:
        if str(error)!='Blueprint likelihood factor has zero support':raise
        atomic_json(out/'excluded.json',{'reason':str(error),'kind':'zero-policy-root-support','source':spec});return
    compact=export(root,spec,inputs,out/'export',source=source)
    secondary,secondary_coverage=secondary_ranges(job['root'],request,stored)
    request.update(mode='solve',spot=job['root']['spot'],policy=spec,range_coverage=coverage,
                   ranges=[[{'hand':list(h),'weight':w} for h,w in ranges[s] if w>0] for s in request['seat_map']],
                   compact_path=str(compact.resolve()),compact_kind='all',secondary_ranges=secondary,
                   secondary_coverage=secondary_coverage,memory_budget_bytes=document['budget_bytes'],compress=True,
                   max_iterations=10000,progress_every=25,target_pct_pot=.2,seconds=900)
    atomic_json(out/'request.json',request)


def status(out,stage,done,total,started,*,job=None,error=None):
    elapsed=monotonic()-started
    atomic_json(out/'status.json',{'stage':stage,'jobs_done':done,'jobs_total':total,'job':job,
        'eta_seconds':elapsed/done*(total-done) if done else None,'elapsed_seconds':elapsed,
        'last_error':error,'updated_utc':datetime.now(timezone.utc).isoformat()})


def run(a):
    config=json.loads(a.protocol.read_text());corpus={g:json.loads((a.repo/config['corpus'][g]['path']).read_text()) for g in ('A','B')}
    if file_hash(a.repo/config['protocol_path'])!=config['protocol_text_sha256']:raise ValueError('Frozen prose protocol differs')
    if config.get('resume_amendment'):
        note=config['resume_amendment']
        if file_hash(a.repo/note['path'])!=note['sha256']:raise ValueError('Resume amendment differs')
    for g in ('A','B'):
        if file_hash(a.repo/config['corpus'][g]['path'])!=config['corpus'][g]['sha256']:raise ValueError('Frozen corpus hash differs')
    if file_hash(a.repo/config['inventory_path'])!=config['inventory_sha256']:raise ValueError('Frozen inventory hash differs')
    inventory=json.loads((a.repo/config['inventory_path']).read_text())
    if file_hash(a.binary)!=config['binary_sha256']:raise ValueError('Frozen solver binary differs')
    for relative,wanted in inventory['repository_source_sha256'].items():
        if file_hash(a.repo/relative)!=wanted:raise ValueError('Frozen repository source differs: '+relative)
    for spec in config['policies']:
        if file_hash(a.inputs/spec['path'])!=spec['sha256']:raise ValueError('Frozen policy bytes differ')
    before=machine_snapshot();budget=int(config['memory_budget_gib']*GIB);baseline=config['swap_baseline_bytes']
    headroom=math.floor(before['reclaimable_bytes']*.8/GIB)*GIB
    if headroom<budget:raise MemoryError('Measured headroom is below the frozen job budget')
    if before['swap_used_bytes']-baseline>GIB:raise MemoryError('Run-baseline swap threshold exceeded before startup')
    schedule=jobs(corpus,config['policies'],config['order_seed']);a.out.mkdir(parents=True,exist_ok=True)
    if len(schedule)!=config['jobs_total']:raise ValueError('Frozen job count differs')
    for relative,wanted in config.get('resume',{}).get('preserved_results',{}).items():
        if file_hash(a.out/relative)!=wanted:raise ValueError('Preserved result differs: '+relative)
    lock=(a.out/'campaign.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    admission_path=a.out/'admission.json'
    if admission_path.exists():
        admission=json.loads(admission_path.read_text())
        if admission['protocol_sha256']!=file_hash(a.protocol):raise ValueError('Resume protocol differs')
        
    else:
        prior=config.get('resume',{}).get('previous_active_seconds',0.)
        if not 0<=prior<config['main_seconds_ceiling']:raise ValueError('Invalid prior campaign time')
        admission={'protocol_sha256':file_hash(a.protocol),'source_commit':a.source_commit,
                   'budget_bytes':budget,'swap_baseline_bytes':baseline,'seconds_ceiling':config['main_seconds_ceiling'],
                   'jobs_total':len(schedule),'machine_before':before,'elapsed_seconds':prior,
                   'resumed_from':config.get('resume'),
                   'started_utc':(datetime.now(timezone.utc)-timedelta(seconds=prior)).isoformat()}
        atomic_json(admission_path,admission)
    started=monotonic()-(datetime.now(timezone.utc)-datetime.fromisoformat(admission['started_utc'])).total_seconds()
    completed=[];outcomes=[];current=None
    for name in ('result.json','failure.json'):
        path=a.out/name
        if path.exists():
            stops=a.out/'retained-stops';stops.mkdir(exist_ok=True)
            number=len(list(stops.glob('*.json')))+1
            path.replace(stops/f'{number:03d}-{name}')
    for job in schedule:
        path=a.out/'spots'/job['job']/'result.json'
        if path.exists():
            row=counter_record(json.loads(path.read_text()));outcomes.append(row)
            if row.get('event')=='spot_complete':completed.append(row)
    done=len(outcomes);deadline=config['main_seconds_ceiling']
    try:
        for job in schedule:
            current=job['job'];base=a.out/'spots'/current
            if (base/'result.json').exists():continue
            # Reserve both bounded worker phases and cleanup before admission.
            if deadline-(monotonic()-started)<2460:
                result={'status':'budget-exhausted','jobs_done':done,'jobs_total':len(schedule),'elapsed_seconds':monotonic()-started,
                        'decision_admitted':False,'note':'Frozen campaign incomplete; no hypothesis decision'}
                atomic_json(a.out/'result.json',result);status(a.out,'budget exhausted',done,len(schedule),started);return
            if file_hash(a.binary)!=config['binary_sha256']:raise ValueError('Solver binary changed during the campaign')
            if swap_usage()-baseline>GIB:raise MemoryError('Run-baseline swap growth exceeded 1 GiB')
            snapshot=machine_snapshot()
            if math.floor(snapshot['reclaimable_bytes']*.8/GIB)*GIB<budget:raise MemoryError('Measured job headroom fell below budget')
            base.mkdir(parents=True,exist_ok=True);attempt=1
            while (base/f'attempt-{attempt:02d}').exists():attempt+=1
            attempt_out=base/f'attempt-{attempt:02d}';attempt_out.mkdir()
            policy=config['policies'][job['policy_index']]
            atomic_json(base/'job.json',{'job':job,'policy':policy,'budget_bytes':budget})
            status(a.out,'preparing native ranges and compact tables',done,len(schedule),started,job=current)
            preparation=attempt_out/'preparation';preparation.mkdir()
            command=[sys.executable,'-m','scripts.run_turn_check','--worker','--job',str(base/'job.json'),
                     '--inputs',str(a.inputs),'--stored',str(a.repo/config['corpus']['A']['path']),
                     '--out',str(attempt_out/'prepared')]
            prepare_guarded(command,preparation,budget,baseline)
            excluded=attempt_out/'prepared/excluded.json'
            if excluded.exists():
                result={'event':'spot_excluded','set':job['set'],'spot':job['root']['spot'],'policy':policy['name'],
                        'lineage':policy['seed'],'strategy':policy['strategy'],**json.loads(excluded.read_text())}
            else:
                path=attempt_out/'prepared/request.json';request=json.loads(path.read_text())
                status(a.out,'solving equilibrium and locked best responses',done,len(schedule),started,job=current)
                remaining=deadline-(monotonic()-started)
                if remaining<1200:
                    atomic_json(a.out/'result.json',{'status':'budget-exhausted','jobs_done':done,'jobs_total':len(schedule),
                        'elapsed_seconds':monotonic()-started,'decision_admitted':False,'note':'Insufficient time for the next bounded job'})
                    status(a.out,'budget exhausted before next job',done,len(schedule),started);return
                runtime=run_tool(a.binary,path,attempt_out/'solver',memory_bytes=budget,threads=2,seconds=1200,initial_swap=baseline)
                if runtime['status']!='completed':raise RuntimeError(runtime['failure'])
                rows=response_rows(attempt_out/'solver/response.jsonl')
                if rows[-1].get('status') in ('oversize','oversize-before-tree-arena'):
                    # Caps cannot be admitted without the declared removed-reach
                    # audit. Never hide a resource failure behind an unaudited tree.
                    raise MemoryError('Native root does not fit; stop before any capped or paid run')
                result=summarize(job,policy,request,rows,runtime,corpus['A'])
                result['request_sha256']=file_hash(path);result['attempt_path']=str(attempt_out)
                completed.append(counter_record(result))
            atomic_json(base/'result.json',result);outcomes.append(counter_record(result));done+=1
            # Heavy per-node tables stay in atomic results, not the monitor stream.
            append(a.out/'progress.jsonl',counter_record(result))
            append(a.out/'progress.jsonl',{'event':'run_counter','jobs_completed':done,'jobs_total':len(schedule),
                'spots_completed_by_set':{g:len({r['spot'] for r in outcomes if r['set']==g}) for g in ('A','B')},
                'jobs_completed_by_set':{g:sum(r['set']==g for r in outcomes) for g in ('A','B')},
                'roots_fully_completed_by_set':{g:sum(sum(r['spot']==root['spot'] for r in outcomes)==6 for root in corpus[g]['roots']) for g in ('A','B')}})
            admission['elapsed_seconds']=monotonic()-started;atomic_json(admission_path,admission)
            status(a.out,'between jobs',done,len(schedule),started)
            # Release the last verbose solver response before the next worker.
            result=None;request=None;rows=None
        atomic_json(a.out/'result.json',{'status':'completed','jobs_done':done,'jobs_total':len(schedule),
            'valid_solved_jobs':len(completed),'elapsed_seconds':monotonic()-started,'machine_after':machine_snapshot()})
        status(a.out,'campaign complete; report pending',done,len(schedule),started)
    except BaseException as error:
        admission['elapsed_seconds']=monotonic()-started;atomic_json(admission_path,admission)
        atomic_json(a.out/'failure.json',{'stage':'main campaign stopped','job':current,'error':repr(error),
                    'jobs_done':done,'jobs_total':len(schedule),'elapsed_seconds':monotonic()-started})
        status(a.out,'stopped',done,len(schedule),started,job=current,error=repr(error));raise
    finally:lock.close()


def main():
    def stop(signum,frame):raise SystemExit('Campaign stopped; owned solver cleanup follows')
    signal.signal(signal.SIGTERM,stop)
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','protocol','inputs','out','job','stored'):p.add_argument('--'+name,type=Path)
    p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--source-commit')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker:
        if not all((a.job,a.inputs,a.out,a.stored)):p.error('Worker requires job, inputs, out and stored corpus')
        return prepare_job(a.job,a.inputs,a.out,json.loads(a.stored.read_text()))
    if not all((a.binary,a.protocol,a.inputs,a.out,a.source_commit)):p.error('Campaign requires frozen protocol, source commit, binary, inputs and out')
    run(a)
if __name__=='__main__':main()
