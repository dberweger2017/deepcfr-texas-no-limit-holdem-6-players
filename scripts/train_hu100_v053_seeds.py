"""One guarded M4 campaign: exact #260/#272 recipes for two new seeds.

The controller owns both shared locks through training and subsequent evaluation.
The outer existing M4 guard observes the complete family on one fixed baseline.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import sleep, time

import psutil

from scripts.check_frozen_table import key_digest
from scripts.evaluate_hu100_direct import put
from scripts.run_hu100_k50_v2 import header, pin
from scripts.train_hu100_v2_frozen_5b import is_campaign_process
from src.blueprint.abstraction import HU100_EQUITY_MENU_V2_SCHEMA as SCHEMA
from src.blueprint.equity_buckets import EQUITY_K50_SHA256
from src.diagnostics.cfr_average import audit, checked_header
from src.policies.files import file_hash

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/hu100-v053-seeds'
BINARY = ROOT/'native/hu20-trainer/target/release/hu20-trainer'
TABLES = Path.home()/'Local/hu100-k50-v2/results/hu100-k50-v2/tables'
SEEDS = (2026101501, 2026101502)
CAP = 97_290_497
GIB = 1024**3


def read(path):
    return json.loads(Path(path).read_text())


def revision():
    return subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()


def require_disk(additional=0):
    free = shutil.disk_usage(BASE).free
    if free-additional < 16*GIB:
        raise ValueError(f'Preventive disk admission: {free} free, {additional} additional')
    return free


def command(label, args, allowed=(0,)):
    started = time()
    budget = read(BASE/'budget.json')
    # A scientific subprocess must leave the original closeout reserve intact.
    remaining = budget['deadline']-1800-started
    if remaining <= 0:
        raise ValueError('Scientific budget exhausted; no extension')
    with (BASE/(label+'.log')).open('x') as log:
        done = subprocess.run(list(map(str,args)), stdout=log, stderr=subprocess.STDOUT,
                              timeout=remaining)
    put(BASE/(label+'.operation.json'), dict(command=list(map(str,args)),started=started,
        finished=time(),seconds=time()-started,returncode=done.returncode,source=revision()))
    if done.returncode not in allowed:
        raise ValueError(f'{label}: worker exit {done.returncode}')
    return done.returncode


def train_args(seed, nodes, cap, output, telemetry):
    return [BINARY,'train','--stack-bb',100,'--seed',seed,'--roots-per-seat',1,
        '--average-rule','opponent-sampled','--menu','v2','--card-buckets',TABLES,
        '--nodes',nodes,'--max-entries',cap,'--out',output,'--telemetry',telemetry,
        '--stop-file',BASE/'stop.request']


def verify_lineage(h, seed):
    if (h['config']['seed'] != seed or h['abstraction'] != SCHEMA
            or h['average_rule'] != 'opponent-sampled' or h['config']['roots_per_seat'] != 1
            or h['config']['raise_cap'] is not None or 'training_options' in h
            or h['identity']['stacks'] != [10000,10000]
            or h['identity']['small_blind'] != 50 or h['identity']['big_blind'] != 100
            or h['identity']['card_tables'] != EQUITY_K50_SHA256):
        raise ValueError('Recipe/header differs from authorized lineage')


def verify_frozen(parent, final, initial_keys, final_keys, rows):
    entries = initial_keys['entries']
    if (initial_keys['sorted_key_sha256'] != final_keys['sorted_key_sha256']
            or entries != final_keys['entries'] or any(r['diagnostics']['entries'] != entries for r in rows)
            or final['frozen_table']['entries'] != entries
            or final['frozen_table']['since_iteration'] != parent['iteration']
            or final['native_state']['completed_nodes'] < 5_000_000_000):
        raise ValueError('Frozen membership, boundary or target differs')


def train_seed(seed):
    dest = BASE/str(seed); dest.mkdir(exist_ok=False)
    # Capacity is copied verbatim from #260, independent of this seed's memory.
    require_disk(3_800_000_000)
    telemetry = dest/'stage1.telemetry.jsonl'
    rc = command(f'{seed}-stage1', [*train_args(seed,1_000_000_000,CAP,
        dest/'stage1-requested-{nodes}.checkpoint.gz',telemetry),'--milestones','500000000'], (3,))
    rows = [json.loads(x) for x in telemetry.read_text().splitlines()]
    row = rows[-1]
    if row['diagnostics']['entries'] < CAP or row['stop_requested']:
        raise ValueError('Stage 1 did not terminate at the fixed capacity')
    parent = Path(row['path'])
    parent_pin = pin(parent)
    if parent_pin['sha256'] != row['checkpoint_sha256']:
        raise ValueError('Stage-1 save hash differs from native receipt')
    h = header(parent); verify_lineage(h,seed)
    if 'frozen_table' in h:
        raise ValueError('Stage 1 unexpectedly frozen')
    keys = key_digest(parent)
    if keys['entries'] != row['diagnostics']['entries']:
        raise ValueError('Stage-1 streamed key count differs')
    put(dest/'stage1-verified.json',dict(checkpoint=parent_pin,header=h,keys=keys,
        telemetry=rows,returncode=rc,nominal_cap=CAP,source=revision()))
    require_disk(5_300_000_000)
    final = dest/'stage2-requested-5000000000.checkpoint.gz'
    frozen_log = dest/'stage2.telemetry.jsonl'
    command(f'{seed}-stage2', [*train_args(seed,5_000_000_000,keys['entries'],final,frozen_log),
        '--resume',parent,'--resume-sha256',parent_pin['sha256'],'--freeze-at-cap'])
    frozen_rows = [json.loads(x) for x in frozen_log.read_text().splitlines()]
    final_header = header(final); verify_lineage(final_header,seed)
    final_keys = key_digest(final)
    verify_frozen(h,final_header,keys,final_keys,frozen_rows)
    if file_hash(final) != frozen_rows[-1]['checkpoint_sha256']:
        raise ValueError('Stage-2 checkpoint hash differs')
    put(dest/'training-verified.json',dict(status='verified',header=final_header,
        initial_keys=keys,final_keys=final_keys,telemetry=frozen_rows,source=revision()))
    # Only extra milestones inside this campaign could be superseded. Typical
    # stage 1 reaches capacity before 500M, leaving exactly one terminal save.
    retained = {parent.resolve(),final.resolve()}
    removed = []
    for old in dest.glob('*.checkpoint.gz'):
        if old.resolve() not in retained:
            receipt = pin(old); old.unlink(); removed.append(receipt)
    put(dest/'superseded-checkpoints.json',dict(successor=pin(final),removed=removed,
        outside_own_unsynced_root_deleted=False))
    export_audit(seed,final,dest,frozen_rows[-1])


def export_audit(seed, checkpoint, dest, row):
    require_disk(6_300_000_000)
    current = dest/'5000000000.current.gz'
    average = dest/'5000000000.average.gz'
    command(f'{seed}-export',[BINARY,'export',checkpoint,'--current',current,'--average',average,
        '--zero-mass','uniform'])
    h = header(checkpoint)
    spec = dict(name=f'v2-{seed}',seed=seed,iteration=h['iteration'],
        checkpoint_sha256=file_hash(checkpoint),sha256=file_hash(current))
    checked_header(h,spec,expected_schema=SCHEMA)
    tick=time()
    receipt = audit(checkpoint,current,average,spec,file_hash(average),expected_schema=SCHEMA)
    if receipt['status'] != 'verified' or receipt['all_nodes_verified'] != row['diagnostics']['entries']:
        raise ValueError('Incomplete every-row current/average audit')
    put(dest/'5b.audit.json',dict(receipt,actual_nodes=h['native_state']['completed_nodes'],
        seconds=time()-tick,files={k:pin(p) for k,p in [('checkpoint',checkpoint),('current',current),('average',average)]}))
    metadata = header(average)
    put(dest/'treatment.json',dict(pin(average),name=f'v2-{seed}',abstraction=SCHEMA,
        format=metadata['format'],card_buckets=str(TABLES),entries=row['diagnostics']['entries'],
        iteration=h['iteration'],source_checkpoint_sha256=spec['checkpoint_sha256'],translation=None))


def controller():
    with (Path.home()/'Local/.hu20-m4-research.lock').open('a+') as one, \
         (Path.home()/'Local/.hu100-m4-research.lock').open('a+') as two:
        for stream in (one,two):
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        own={psutil.Process().pid,*[p.pid for p in psutil.Process().parents()]}
        for p in psutil.process_iter(['pid','name','cmdline'],ad_value=None):
            if p.pid not in own and is_campaign_process(p.info['name'],p.info['cmdline']):
                raise ValueError('Other campaign active: '+str(p.pid))
        identity = subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
        if identity != 'Apple M4' or shutil.disk_usage(BASE).free < 30*GIB:
            raise ValueError('M4/30-GiB admission failed')
        tables={street:pin(TABLES/(street+'-k50.bin')) for street in EQUITY_K50_SHA256}
        if any(tables[s]['sha256'] != sha for s,sha in EQUITY_K50_SHA256.items()):
            raise ValueError('Pinned K50 table differs')
        owners=read(BASE/'input-owners.json')
        if any(x['state'] != 'MERGED' for x in owners.values()):
            raise ValueError('Input owners must be merged')
        put(BASE/'admission.json',dict(at=time(),identity=identity,free_bytes=shutil.disk_usage(BASE).free,
            locks=[one.name,two.name],tables=tables,owners=owners,source=revision(),pid=psutil.Process().pid,
            create_time=psutil.Process().create_time(),native_tree=subprocess.check_output(
                ['git','rev-parse','HEAD:native'],text=True).strip()))
        put(BASE/'first-build.json',dict(at=time(),source=revision()))
        command('build',[sys.executable,'-m','scripts.build_native_trainer','--receipt',BASE/'build-receipt.json'])
        for seed in SEEDS:
            train_seed(seed)
        put(BASE/'all-training-verified.json',dict(at=time(),seeds=SEEDS))
        # Evaluation is prepared and pinned while long training runs. This is a
        # finite wait inside the original clock, still covered by the same guard.
        while not (BASE/'evaluation-ready.json').exists():
            if time() >= read(BASE/'budget.json')['deadline']-1800:
                raise ValueError('No evaluation admission before science deadline')
            sleep(5)
        ready=read(BASE/'evaluation-ready.json')
        module=ROOT/'scripts/evaluate_hu100_v053_seeds.py'
        if file_hash(module) != ready['sha256']:
            raise ValueError('Evaluation controller changed after admission')
        command('evaluation',[sys.executable,'-m','scripts.evaluate_hu100_v053_seeds','run'])
        put(BASE/'campaign-complete.json',dict(at=time(),status='complete'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('launch','controller'))
    p.add_argument('--owners',type=Path,help='Recent coordinator-side live PR metadata; avoids M4 credential work')
    a=p.parse_args()
    if a.stage=='launch':
        if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
            raise ValueError('Committed clean checkout required')
        BASE.mkdir(parents=True,exist_ok=False)
        if a.owners is None:
            raise ValueError('Recent live owning-PR metadata required')
        provenance=read(a.owners)
        if not 0 <= time()-provenance['at'] < 600:
            raise ValueError('Owning-PR metadata is stale')
        owners=provenance['owners']
        if set(owners) != {'163','260','272'} or any(x['state'] != 'MERGED' for x in owners.values()):
            raise ValueError('All three input owners must currently be merged')
        put(BASE/'input-owners.json',owners)
        started=time()
        put(BASE/'budget.json',dict(started=started,deadline=started+14*3600,
            closeout_seconds=1800,seed_order=SEEDS,source=revision(),first_build_not_yet_started=True))
        result=subprocess.run([sys.executable,'-m','scripts.guard_m4_research','--stage',str(BASE/'guards'),
            '--name','whole-campaign','--deadline',str(started+14*3600),'--stop-file',str(BASE/'stop.request'),
            '--',sys.executable,'-m','scripts.train_hu100_v053_seeds','controller'])
        put(BASE/'detached-complete.json',dict(at=time(),returncode=result.returncode,locks_released=True))
        raise SystemExit(result.returncode)
    try:
        controller()
    except BaseException as error:
        put(BASE/'controller-failure.json',dict(at=time(),failure=repr(error)))
        raise


if __name__=='__main__':
    main()
