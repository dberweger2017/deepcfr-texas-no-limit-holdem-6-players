"""Fixed owner-approved calibration; never plays final strength hands."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import shutil
import time

import psutil

from scripts.evaluate_hu100_direct import PANEL_OPPONENTS, action_seed, put
from scripts.pilot_hu200_river_search import CONFIG, MODEL_SHA256
from src.arena.river_search_calibration import block_value, play_hand
from src.arena.schedule import Plan, Scenario, build_schedule, digest
from src.blueprint.abstraction import HU200_EQUITY_SCHEMA
from src.blueprint.action_translation import TranslationOptions
from src.blueprint.average import AveragePolicy
from src.blueprint.river_evidence import file_pin
from src.blueprint.river_ledger import FamilyPeak


class CalibrationStopped(BaseException):
    pass


CALIBRATION_ROOTS = {name:20261017317100+i for i,name in enumerate(("direct",*PANEL_OPPONENTS))}

RUNTIME = BASE = STOP = None
CHECKED_AT = 0.


def check():
    global CHECKED_AT
    now = time.monotonic()
    if now-CHECKED_AT < .02:
        return
    CHECKED_AT = now
    if STOP.is_set():
        raise CalibrationStopped("Sibling worker failed")
    if time.time() >= RUNTIME['deadline']:
        raise CalibrationStopped("Two-hour calibration cap")


def schedule(panel, root):
    count = 128 if panel == 'direct' else 32
    return build_schedule(Plan((Scenario('river-calibration-'+panel,(20000,20000)),),
        candidate='search',baseline='blend',opponents=(panel if panel != 'direct' else 'blend',),
        blocks=count,root_seed=root,split='test'))


def initialize(runtime, stop):
    global RUNTIME, BASE, STOP
    RUNTIME, STOP = runtime, stop
    started = time.monotonic()
    check()
    BASE = AveragePolicy(Path(runtime['model']),MODEL_SHA256,expected_schema=HU200_EQUITY_SCHEMA,
        card_buckets=Path(runtime['tables']),cache=Path(runtime['cache']),cache_read_only=True,
        translation=TranslationOptions(512,128))
    import os
    put(Path(runtime['out'])/'workers'/f'{os.getpid()}.json',dict(
        load_seconds=time.monotonic()-started,blueprint=BASE.description))


def block(job):
    panel, root, index = job
    directory = Path(RUNTIME['out'])/'blocks'/panel/str(index)
    try:
        check()
        started = time.monotonic()
        item = schedule(panel,root)[index]
        returns = {'search':[], 'plain':[]}
        hands = []
        for rotation in (0,1):
            seeds = [action_seed(root,item,rotation,role) for role in (0,1)]
            for arm in (('search',) if panel == 'direct' else ('search','plain')):
                result = play_hand(BASE,Path(RUNTIME['binary']),RUNTIME['binary_sha256'],CONFIG,
                    directory/f'{rotation}-{arm}',deal=item.deal_seeds[0],seeds=seeds,
                    button=item.button,rotation=rotation,
                    hand_id=f'table-0/{index}/{rotation}/0',
                    opponent=None if panel == 'direct' else panel,searched=arm=='search',check=check)
                returns[arm].append(result['target_chips'])
                hands.append(result)
        value = block_value(returns['search'])
        if panel != 'direct':
            value -= block_value(returns['plain'])
        result = dict(panel=panel,root=root,block=index,score_bb100=value,
            hands=len(hands),decisions=sum(h['decisions'] for h in hands),
            seconds=time.monotonic()-started,primary_seconds=sum(h['primary_seconds'] for h in hands),
            search_seconds=sum(h['search_seconds'] for h in hands),
            search_rows=[r for h in hands for r in h['search_rows']],
            all_hands_verified=all(h['verified'] for h in hands))
        put(directory/'block.json',result)
        # Progress contains counts only: nuisance calibration must not leak
        # winning/losing hands or mean effects into engineering decisions.
        print(json.dumps(dict(panel=panel,block=index,status='verified')),flush=True)
        return dict(panel=panel,block=index)
    except BaseException as exc:
        STOP.set()
        put(directory/'block-error.json',dict(panel=panel,block=index,error=repr(exc)))
        raise


def verify_blocks(out, roots):
    results = []
    for panel,root in roots.items():
        for item in schedule(panel,root):
            directory = out/'blocks'/panel/str(item.index)
            value = json.loads((directory/'block.json').read_text())
            expected = 2 if panel == 'direct' else 4
            if (value['panel'] != panel or value['root'] != root or value['block'] != item.index
                    or value['hands'] != expected or not value['all_hands_verified']):
                raise ValueError('Incomplete/foreign calibration block')
            returns = {'search':[], 'plain':[]}
            for rotation in (0,1):
                for arm in (('search',) if panel == 'direct' else ('search','plain')):
                    hand_dir = directory/f'{rotation}-{arm}'
                    checked = json.loads((hand_dir/'verified.json').read_text())
                    hand = json.loads((hand_dir/'hand.json').read_text())
                    if (not checked['independent_replay'] or not checked['fresh_player_reproduction']
                            or checked['native_replay_solves'] != 0
                            or checked['hand_sha256'] != file_pin(hand_dir/'hand.json')['sha256']
                            or hand['deal_seed'] != item.deal_seeds[0]
                            or hand['action_seeds'] != [action_seed(root,item,rotation,r) for r in (0,1)]
                            or hand['rotation'] != rotation or hand['searched'] != (arm=='search')
                            or sum(hand['net_chips']) or hand['target_chips'] != hand['net_chips'][rotation]):
                        raise ValueError('Calibration hand identity/verification differs')
                    pins = json.loads((hand_dir/'manifest.json').read_text())
                    names = {p.relative_to(hand_dir).as_posix() for p in hand_dir.rglob('*') if p.is_file()}
                    if names != set(pins)|{'manifest.json','verified.json'}:
                        raise ValueError('Calibration evidence inventory differs')
                    for name,pin in pins.items():
                        if file_pin(hand_dir/name) != pin:
                            raise ValueError('Calibration evidence bytes differ')
                    returns[arm].append(hand['target_chips'])
            expected_score = block_value(returns['search'])
            if panel != 'direct':expected_score -= block_value(returns['plain'])
            if value['score_bb100'] != expected_score:
                raise ValueError('Calibration block score differs')
            results.append(value)
    put(out/'all-hands-verified.json',dict(status='verified',blocks=len(results),
        primary_hands=sum(r['hands'] for r in results),all_replayed_and_reproduced=True,
        native_reproduction_solves=0))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('declaration','model','cache','tables','binary','out','gate-root','root-reservation'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--deadline',type=float,required=True)
    args = parser.parse_args()
    if subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip() != 'Apple M4':
        raise ValueError('Calibration requires the authorized M4')
    if not any('scripts.guard_m4_research' in p.cmdline() for p in psutil.Process().parents()):
        raise ValueError('Calibration requires the unchanged resource guard')
    lock=(Path.home()/'Local/.hu20-m4-research.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads(args.declaration.read_text())
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if (plan['phase'] != 'fixed-calibration-not-final' or plan['source'] != source
            or plan['config'] != asdict(CONFIG) or plan['budget_seconds'] != 7200
            or plan['direct_blocks'] != 128 or plan['panel_blocks'] != 32
            or not 1 <= plan['workers'] <= 3 or len(plan['roots']) != 8
            or plan['roots'] != CALIBRATION_ROOTS
            or plan['binary_sha256'] != file_pin(args.binary)['sha256']
            or plan['model_sha256'] != MODEL_SHA256):
        raise ValueError('Foreign calibration declaration')
    reservation = json.loads(args.root_reservation.read_text())
    if (file_pin(args.root_reservation)['sha256'] != plan['root_reservation_sha256']
            or not reservation['fresh'] or not set(plan['roots'].values()).issubset(reservation['roots'])):
        raise ValueError('Root reservation differs')
    gates = {}
    for name,pin in plan['gates'].items():
        path = args.gate_root / pin['path']
        if file_pin(path)['sha256'] != pin['sha256']:
            raise ValueError('Qualified gate changed: '+name)
        gates[name] = json.loads(path.read_text())
    if (gates['parity']['status'] != 'exact-parity' or gates['parity']['decisions'] != 24
            or not all(gates['cost'][key] for key in ('latency_target_met','memory_target_met',
                'telemetry_target_met','all_search_work_completed'))
            or (plan['workers'] > 1 and (not gates['concurrency']['memory_target_met']
                or not gates['concurrency']['outputs_match_parity']
                or gates['concurrency']['workers'] != plan['workers']))):
        raise ValueError('Calibration lacks qualified resource/output gates')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Clean calibration source required')
    if subprocess.check_output(['git','diff',plan['qualified_search_source'],source,'--','src/blueprint','src/game'],text=True):
        raise ValueError('Search/game changed after the qualified gates')
    if args.deadline-time.time() > 7200 or args.deadline <= time.time():
        raise ValueError('Invalid calibration clock')
    if (plan['disk_admission']['floor_bytes'] != 16*1024**3
            or shutil.disk_usage(args.out.parent).free < plan['disk_admission']['required_free_bytes']):
        raise ValueError('Fresh calibration temporary/retained disk admission fails')
    args.out.mkdir(parents=True,exist_ok=False)
    put(args.out/'inputs.json',dict(declaration_sha256=file_pin(args.declaration)['sha256'],
        source=source,plan=plan,deadline=args.deadline,calibration_only=True))
    jobs=[(panel,plan['roots'][panel],b.index) for panel in CALIBRATION_ROOTS for b in schedule(panel,plan['roots'][panel])]
    deals=[b.deal_seeds[0] for panel,root in plan['roots'].items() for b in schedule(panel,root)]
    if len(deals) != len(set(deals)):
        raise ValueError('Calibration deal stream collision')
    runtime={k:str(getattr(args,k)) for k in ('model','cache','tables','binary','out')}
    runtime.update(deadline=args.deadline,binary_sha256=plan['binary_sha256'])
    started=time.monotonic()
    context=mp.get_context('spawn')
    with FamilyPeak() as memory:
        with context.Manager() as manager:
            stop=manager.Event()
            with ProcessPoolExecutor(max_workers=plan['workers'],mp_context=context,
                    initializer=initialize,initargs=(runtime,stop)) as pool:
                futures=[pool.submit(block,job) for job in jobs]
                try:
                    for future in as_completed(futures):future.result()
                except BaseException:
                    stop.set()
                    for future in futures:future.cancel()
                    raise
        results=verify_blocks(args.out,plan['roots'])
        from scripts.quote_hu200_river_strength import readout
        summary=readout(results,plan,args.out,time.monotonic()-started,memory)
    if time.time() >= args.deadline:
        raise CalibrationStopped('Calibration verification/readout exceeds two-hour cap')
    summary.update(measured_seconds=time.time()-(args.deadline-7200),
        peak_family_rss_bytes=memory.peak,minimum_host_available_bytes=memory.minimum_available_bytes,
        memory_target_met=memory.peak <= 16*1024**3-3_000_000_000 and memory.minimum_available_bytes >= 3_000_000_000)
    put(args.out/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
