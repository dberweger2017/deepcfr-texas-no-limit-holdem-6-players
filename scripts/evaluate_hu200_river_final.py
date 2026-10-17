"""Fixed final HU200 river match with bounded, verified cable retirement."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import time

import psutil

from scripts.calibrate_hu200_river_search import schedule as calibration_schedule, CALIBRATION_ROOTS
from scripts.evaluate_hu100_direct import action_seed
from scripts.pilot_hu200_river_search import CONFIG, MODEL_SHA256
from src.arena.river_block_storage import Journal, CableTransfer, canonical, durable, pin, unpack
from src.arena.river_final_schedule import ROOTS, COUNTS, block_at
from src.arena.river_search_calibration import play_hand, block_value
from src.arena.schedule import digest
from src.blueprint.abstraction import HU200_EQUITY_SCHEMA
from src.blueprint.action_translation import TranslationOptions
from src.blueprint.average import AveragePolicy
from src.blueprint.river_ledger import FamilyPeak

RUNTIME=BASE=STOP=None
LAST_CHECK=0.
GIB=1024**3


class FinalStopped(BaseException):pass


class IncompleteHand(RuntimeError):pass


def check():
    global LAST_CHECK
    tick=time.monotonic()
    if tick-LAST_CHECK < .02:return
    LAST_CHECK=tick
    if STOP.is_set():raise FinalStopped('Sibling worker stopped')
    if time.time()>=RUNTIME['deadline']:raise FinalStopped('Fixed final safety deadline')


def initialize(runtime,stop):
    global RUNTIME,BASE,STOP
    RUNTIME,STOP=runtime,stop
    check();started=time.monotonic()
    BASE=AveragePolicy(Path(runtime['model']),MODEL_SHA256,expected_schema=HU200_EQUITY_SCHEMA,
        card_buckets=Path(runtime['tables']),cache=Path(runtime['cache']),cache_read_only=True,
        translation=TranslationOptions(512,128))
    durable(Path(runtime['out'])/'workers'/f'{os.getpid()}-{time.time_ns()}.json',
        dict(load_seconds=time.monotonic()-started,blueprint=BASE.description))


def verify_hand(directory,item,root,rotation,arm):
    checked=json.loads((directory/'verified.json').read_text())
    hand=json.loads((directory/'hand.json').read_text())
    if (not checked['independent_replay'] or not checked['fresh_player_reproduction']
            or checked['native_replay_solves']!=0 or checked['hand_sha256']!=pin(directory/'hand.json')['sha256']
            or hand['deal_seed']!=item.deal_seeds[0] or hand['rotation']!=rotation
            or hand['searched']!=(arm=='search') or sum(hand['net_chips'])
            or hand['target_chips']!=hand['net_chips'][rotation]
            or hand['action_seeds']!=[action_seed(root,item,rotation,r) for r in (0,1)]):
        raise ValueError('Final hand identity or verification differs')
    files=json.loads((directory/'manifest.json').read_text())
    actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    if actual!=set(files)|{'manifest.json','verified.json'}:raise ValueError('Final hand inventory differs')
    for name,wanted in files.items():
        if pin(directory/name)!=wanted:raise ValueError('Final hand checksum differs')
    return hand


def recovered_result(directory,hand):
    rows=[]
    for path in sorted((directory/'telemetry').glob('*.json')):
        if path.name.endswith('-time.json'):continue
        row=json.loads(path.read_text())
        if digest({k:v for k,v in row.items() if k!='sha256'})!=row['sha256']:
            raise ValueError('Recovered telemetry checksum differs')
        timing=json.loads(path.with_name(path.stem+'-time.json').read_text())
        if digest({k:v for k,v in timing.items() if k!='sha256'})!=timing['sha256']:
            raise ValueError('Recovered timing checksum differs')
        rows.append(dict(complete_seconds=timing['complete_seconds'],**row))
    return dict(target_chips=hand['target_chips'],primary_seconds=None,total_seconds=None,
        search_seconds=sum(r['complete_seconds'] for r in rows),search_rows=rows,
        decisions=len(hand['decisions']),verified=True,recovered_without_native=True)


def summarize(value):
    rows=value['search_rows']
    return {**{k:v for k,v in value.items() if k!='search_rows'},'search_rows':[
        {k:r.get(k) for k in ('changed_action','fallback','fallback_cause','model_ev_chips',
            'model_ev_delta_chips','model_ev_unavailable','complete_seconds','native_seconds','tree_nodes','sampled_peak_family_rss_bytes')}
        for r in rows]}


def verify_common_prefix(search,plain):
    """Same deal/RNG cannot diverge before our river action."""
    first=None
    for left,right in zip(search['decisions'],plain['decisions']):
        if any(left[k]!=right[k] for k in ('seat','role','street','observation_sha256')):
            raise ValueError('Paired public common prefix differs before action')
        if left['action']!=right['action']:
            if left['role']!=0 or left['street']!='river':
                raise ValueError('Paired first action difference is outside hero river')
            first=left['index'];break
    if first is None and (len(search['decisions'])!=len(plain['decisions'])
            or search['net_chips']!=plain['net_chips']):
        raise ValueError('Identical paired paths settle differently')
    return first


def verify_block(directory,panel,index,root):
    value=json.loads((directory/'block.json').read_text());item=block_at(panel,root,index)
    if value['panel']!=panel or value['block']!=index or value['root']!=root:
        raise ValueError('Foreign final block')
    returns={'search':[],'plain':[]};hands=0;paired={}
    for rotation in (0,1):
        for arm in (('search',) if panel=='direct' else ('search','plain')):
            hand=verify_hand(directory/f'{rotation}-{arm}',item,root,rotation,arm)
            returns[arm].append(hand['target_chips']);hands+=1;paired[rotation,arm]=hand
    if panel!='direct':
        for rotation in (0,1):verify_common_prefix(paired[rotation,'search'],paired[rotation,'plain'])
    score=block_value(returns['search'])-(block_value(returns['plain']) if panel!='direct' else 0)
    if value['score_bb100']!=score or value['hands']!=hands or not value['all_hands_verified']:
        raise ValueError('Final block score/accounting differs')
    return summarize(value)


def block(job):
    panel,index=job;root=ROOTS[panel];item=block_at(panel,root,index)
    directory=Path(RUNTIME['out'])/'blocks'/panel/str(index)
    try:
        check()
        if (directory/'block.json').exists():return verify_block(directory,panel,index,root)
        directory.mkdir(parents=True,exist_ok=True)
        started=time.monotonic();returns={'search':[],'plain':[]};hands=[]
        for rotation in (0,1):
            seeds=[action_seed(root,item,rotation,r) for r in (0,1)]
            for arm in (('search',) if panel=='direct' else ('search','plain')):
                where=directory/f'{rotation}-{arm}';result_path=directory/'hand-results'/f'{rotation}-{arm}.json'
                if where.exists():
                    if not (where/'verified.json').exists():
                        raise IncompleteHand('Retain partial hand; explicit recorded-state continuation required: '+str(where))
                    hand=verify_hand(where,item,root,rotation,arm)
                    if result_path.exists():
                        saved=json.loads(result_path.read_text());result=saved['result']
                        if saved['sha256']!=digest(result) or result['target_chips']!=hand['target_chips']:
                            raise ValueError('Recovered hand result differs')
                    else:result=recovered_result(where,hand)
                else:
                    result=play_hand(BASE,Path(RUNTIME['binary']),RUNTIME['binary_sha256'],CONFIG,where,
                        deal=item.deal_seeds[0],seeds=seeds,button=item.button,rotation=rotation,
                        hand_id=f'table-0/{index}/{rotation}/0',opponent=None if panel=='direct' else panel,
                        searched=arm=='search',check=check)
                if not result_path.exists():durable(result_path,dict(result=result,sha256=digest(result)))
                returns[arm].append(result['target_chips']);hands.append(result)
        search=block_value(returns['search']);plain=block_value(returns['plain']) if panel!='direct' else None
        result=dict(panel=panel,root=root,block=index,score_bb100=search-(plain or 0),
            search_score_bb100=search,plain_score_bb100=plain,hands=len(hands),
            decisions=sum(h['decisions'] for h in hands),seconds=time.monotonic()-started,
            recovered_timing_unknown=any(h['primary_seconds'] is None for h in hands),
            search_seconds=sum(h['search_seconds'] for h in hands),
            search_rows=[r for h in hands for r in h['search_rows']],all_hands_verified=all(h['verified'] for h in hands))
        durable(directory/'block.json',result)
        return verify_block(directory,panel,index,root)
    except BaseException as exc:
        STOP.set()
        durable(Path(RUNTIME['out'])/'failures'/f'{panel}-{index}-{time.time_ns()}.json',
            dict(panel=panel,block=index,error=repr(exc),partial_hand=isinstance(exc,IncompleteHand)))
        raise


def root_audit(out):
    excluded={b.deal_seeds[0] for p,r in CALIBRATION_ROOTS.items() for b in calibration_schedule(p,r)}
    seen=set();hasher=hashlib.sha256()
    for panel in ROOTS:
        for i in range(COUNTS[panel]):
            seed=block_at(panel,ROOTS[panel],i).deal_seeds[0]
            if seed in excluded or seed in seen:raise ValueError('Final deal collision/calibration overlap')
            seen.add(seed);hasher.update(str(seed).encode()+b'\n')
    durable(out/'deal-disjointness.json',dict(final_blocks=len(seen),calibration_deals=len(excluded),
        final_deal_stream_sha256=hasher.hexdigest(),calibration_overlap=0,internal_duplicates=0))


def barrier(out,journal):
    digest_rows=hashlib.sha256();counts=Counter();hands=0
    for panel,index,phase,raw_manifest,raw_summary,raw_retirement in journal.connection.execute(
            'SELECT panel,number,phase,manifest,summary,retirement FROM blocks ORDER BY panel,number'):
        if panel not in COUNTS or not 0<=index<COUNTS[panel] or phase!='retired':
            raise ValueError('Global barrier has foreign/incomplete block')
        manifest=unpack(raw_manifest);summary=unpack(raw_summary);retirement=unpack(raw_retirement)
        if summary['panel']!=panel or summary['block']!=index or not summary['all_hands_verified']:
            raise ValueError('Global barrier block identity differs')
        if retirement['status']!='verified-owned-block-retired':raise ValueError('Global barrier lacks retirement receipt')
        if (manifest['identity']['declaration_sha256']!=RUNTIME_DECLARATION_SHA
                or manifest['identity']['summary_sha256']!=digest(summary)
                or retirement['manifest_sha256']!=hashlib.sha256((canonical(manifest)+'\n').encode()).hexdigest()):
            raise ValueError('Global barrier declaration differs')
        counts[panel]+=1;hands+=summary['hands'];digest_rows.update(raw_manifest);digest_rows.update(raw_summary)
    if dict(counts)!=COUNTS or hands!=319488:raise ValueError('Incomplete fixed final counts')
    result=dict(status='all-final-hands-replayed-reproduced-transferred-and-verified',blocks=sum(counts.values()),
        counts=dict(counts),primary_hands=hands,native_reproduction_solves=0,receipt_stream_sha256=digest_rows.hexdigest())
    durable(out/'all-final-hands-verified.json',result)
    return result


def execute_pipeline(jobs,pool,storage,produce,ship,capacity,check_now,progress):
    searches={};transfers={};exhausted=False
    try:
        while searches or transfers or not exhausted:
            check_now()
            slots=capacity()
            while not exhausted and len(searches)<slots and len(transfers)<3:
                try:job=next(jobs)
                except StopIteration:exhausted=True;break
                searches[pool.submit(produce,job)]=job
            if not searches and not transfers:
                if exhausted:break
                raise OSError('Fresh dispatch reserve unavailable; originals retained')
            candidates=set(transfers)
            if len(transfers)<3:candidates.update(searches)
            ready,_=wait(candidates,timeout=5,return_when=FIRST_COMPLETED)
            for future in ready & set(transfers):
                job=transfers.pop(future);receipt=future.result();progress(job,receipt)
            for future in list(searches):
                if not future.done():continue
                if future.exception() is not None:future.result()
                if len(transfers)>=3:break
                job=searches.pop(future);summary=future.result()
                transfers[storage.submit(ship,*job,summary)]=job
    except BaseException:
        for future in searches:future.cancel()
        raise


RUNTIME_DECLARATION_SHA=None


def main():
    global RUNTIME_DECLARATION_SHA
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('declaration','model','cache','tables','binary','out','gate-root','root-reservation','qualification'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('host','remote-root','verifier'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--deadline',type=float,required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()!='Apple M4':
        raise ValueError('Finals require authorized M4')
    if not any('scripts.guard_m4_research' in p.cmdline() for p in psutil.Process().parents()):
        raise ValueError('Finals require unchanged guard')
    lock=(Path.home()/'Local/.hu20-m4-research.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads(args.declaration.read_text());source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if (plan['phase']!='fixed-final-strength' or plan['source']!=source or plan['config']!=asdict(CONFIG)
            or plan['counts']!=COUNTS or plan['roots']!=ROOTS or plan['workers']!=3
            or plan['model_sha256']!=MODEL_SHA256 or pin(args.binary)['sha256']!=plan['binary_sha256']
            or args.host!='bb128-lan' or args.remote_root!=plan['remote_root'] or args.verifier!=plan['verifier']):
        raise ValueError('Foreign final declaration')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Clean fixed source required')
    if subprocess.check_output(['git','diff',plan['qualified_search_source'],source,'--',
            'src/blueprint','src/game','src/arena/river_search_calibration.py'],text=True):
        raise ValueError('Search or tested hand player changed after qualification')
    reservation=json.loads(args.root_reservation.read_text())
    if pin(args.root_reservation)['sha256']!=plan['root_reservation_sha256'] or not reservation['fresh'] or not set(ROOTS.values()).issubset(reservation['roots']):
        raise ValueError('Final root reservation differs')
    qualification=json.loads(args.qualification.read_text())
    if pin(args.qualification)['sha256']!=plan['storage_qualification_sha256'] or qualification['status']!='qualified':
        raise ValueError('Storage qualification differs')
    for name,spec in plan['gates'].items():
        path=args.gate_root/spec['path']
        if pin(path)['sha256']!=spec['sha256']:raise ValueError('Gate changed: '+name)
    if not time.time()<args.deadline<=time.time()+plan['safety_seconds']:raise ValueError('Invalid fixed final deadline')
    RUNTIME_DECLARATION_SHA=pin(args.declaration)['sha256']
    if args.resume:
        old=json.loads((args.out/'inputs.json').read_text())
        if old['declaration_sha256']!=RUNTIME_DECLARATION_SHA or old['deadline']!=args.deadline:
            raise ValueError('Restart changes the fixed run or deadline')
    else:
        args.out.mkdir(parents=True,exist_ok=False)
        durable(args.out/'inputs.json',dict(declaration_sha256=RUNTIME_DECLARATION_SHA,plan=plan,deadline=args.deadline,source=source))
        root_audit(args.out)
    journal=Journal(args.out/'receipts.sqlite');transfer=CableTransfer(args.host,args.remote_root,args.verifier,expected_code_sha=plan['storage_verifier_sha256'])
    runtime={k:str(getattr(args,k)) for k in ('model','cache','tables','binary','out')}
    runtime.update(deadline=args.deadline,binary_sha256=plan['binary_sha256'])
    # Recover only closed blocks or already-verified retirements. An unfinished
    # played hand remains intact and stops automatic restart; never replay it.
    for panel,index in journal.connection.execute("SELECT panel,number FROM blocks WHERE phase!='retired'").fetchall():
        row=journal.get(panel,index);transfer.transfer(args.out/'blocks'/panel/str(index),row['manifest']['identity'],row['summary'],journal)
    retired=set(journal.connection.execute("SELECT panel,number FROM blocks WHERE phase='retired'").fetchall())
    jobs=iter((panel,i) for panel in ROOTS for i in range(COUNTS[panel]) if (panel,i) not in retired)
    context=mp.get_context('spawn');started=time.time();done=len(retired);exhausted=False
    with FamilyPeak() as memory,context.Manager() as manager:
        stop=manager.Event()
        # One storage writer overlaps up to three searches. At most three closed
        # blocks wait for transport; disk capacity gates further dispatch.
        def ship(panel,index,summary):
            identity=dict(panel=panel,block=index,root=ROOTS[panel],source=source,
                declaration_sha256=RUNTIME_DECLARATION_SHA,summary_sha256=digest(summary))
            error=None
            for attempt in range(3):
                try:
                    receipt=transfer.transfer(args.out/'blocks'/panel/str(index),identity,summary,journal)
                    if index%128==0:journal.checkpoint()
                    return receipt
                except (OSError,subprocess.SubprocessError) as exc:
                    error=exc;journal.event(panel,index,dict(operation='transport-plumbing-error',attempt=attempt,
                        error=repr(exc),stderr=str(getattr(exc,'stderr',''))[-2000:]))
                    if attempt<2:time.sleep((5,15)[attempt])
            raise error
        with ThreadPoolExecutor(max_workers=1) as storage,ProcessPoolExecutor(
                max_workers=3,mp_context=context,initializer=initialize,initargs=(runtime,stop)) as pool:
            def capacity():
                free=shutil.disk_usage(args.out).free
                return min(3,max(0,(free-16*GIB-256*1024**2)//(512*1024**2)))
            def deadline_check():
                if time.time()>=args.deadline:raise FinalStopped('Fixed final safety deadline')
            def progress(job,receipt):
                nonlocal done
                panel,index=job;done+=1
                print(json.dumps(dict(status='verified-transferred-retired',panel=panel,block=index,
                    completed_blocks=done,total_blocks=sum(COUNTS.values()),elapsed_seconds=time.time()-started)),flush=True)
            try:
                execute_pipeline(jobs,pool,storage,block,ship,capacity,deadline_check,progress)
            except BaseException as exc:
                stop.set()
                durable(args.out/'failures'/f'controller-{time.time_ns()}.json',dict(error=repr(exc),completed_blocks=done,at=time.time()))
                raise
    barrier(args.out,journal)
    from scripts.report_hu200_river_final import report
    report(args.out,journal,plan,memory)
    journal.checkpoint()


if __name__=='__main__':main()
