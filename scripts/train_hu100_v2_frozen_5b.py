"""Recover the exact PR260 parent, then train its fixed HU100 table to 5B."""
import argparse
import fcntl
import gzip
import json
from pathlib import Path
import subprocess
import sys
from time import time

import psutil

from scripts.evaluate_hu100_direct import put
from scripts.run_hu100_k50_v2 import pin, header
from src.blueprint.abstraction import HU100_EQUITY_MENU_V2_SCHEMA
from src.blueprint.equity_buckets import EQUITY_K50_SHA256
from src.diagnostics.cfr_average import audit, checked_header
from src.policies.files import file_hash

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/hu100-v2-frozen-5b'
BINARY = ROOT/'native/hu20-trainer/target/release/hu20-trainer'
TABLES = Path.home()/'Local/hu100-k50-v2/results/hu100-k50-v2/tables'
PARENT = Path.home()/'Local/hu100-k50-v2/results/hu100-k50-v2/training/332894788.checkpoint.gz'
SHA = '202058df8aff994612b441d26ffc4e70475ee790cc06a799635e4389e219db2f'
ENTRIES = 97_290_578
SEED = 2026100601


def is_campaign_process(name, cmdline):
    if name == 'hu20-trainer':
        return True
    prefixes=('run_hu','train_hu','evaluate_hu','evaluate_native_hu','run_pot_pressure')
    for argument in (cmdline or [])[1:]:
        if argument.startswith(tuple('scripts.'+p for p in prefixes)):
            return True
        leaf=Path(argument).name
        if leaf=='campaign.py' or leaf.startswith(prefixes):
            return True
    return False


def read(path):
    return json.loads(path.read_text())


def command(args, allowed=(0,)):
    code = subprocess.run(list(map(str, args)), timeout=max(1, read(BASE/'budget.json')['deadline']-time())).returncode
    if code not in allowed:
        raise RuntimeError('Worker exit '+str(code))
    return code


def args(nodes, cap, out, telemetry):
    return [BINARY,'train','--stack-bb',100,'--seed',SEED,'--roots-per-seat',1,
            '--average-rule','opponent-sampled','--menu','v2','--card-buckets',TABLES,
            '--nodes',nodes,'--max-entries',cap,'--out',out,'--telemetry',telemetry,
            '--stop-file',BASE/'stop.request']


def prepare():
    owners = {str(n): json.loads(subprocess.check_output(['gh','pr','view',str(n),
        '--repo','dberweger2017/deepcfr-texas-no-limit-holdem-6-players',
        '--json','state,url,headRefOid'],text=True)) for n in (163,260)}
    if any(o['state'] != 'MERGED' for o in owners.values()):
        raise ValueError('Input owners must be merged')
    tables = {s: pin(TABLES/(s+'-k50.bin')) for s in EQUITY_K50_SHA256}
    if any(p['sha256'] != EQUITY_K50_SHA256[s] for s,p in tables.items()):
        raise ValueError('Pinned tables differ')
    put(BASE/'inputs.json',dict(owners=owners,tables=tables,parent_sha256=SHA,
        original_parent=str(PARENT),source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        main_base='6df0dab9',parent_index=read(ROOT/'docs/reports/hu100-k50-v2-artifacts/model-index.json')))
    command([sys.executable,'-m','scripts.build_native_trainer','--receipt',BASE/'build-receipt.json'])


def recover():
    path = PARENT
    regenerated = not path.exists()
    if regenerated:
        path = BASE/'parent/332894788.checkpoint.gz'
        path.parent.mkdir(exist_ok=False)
        rc = command([*args(1_000_000_000,97_290_497,path,BASE/'parent/telemetry.jsonl'),
                      '--milestones','500000000'],allowed=(3,))
    else:
        rc = None
    actual = pin(path)
    if actual['bytes'] != 3_488_922_486 or actual['sha256'] != SHA:
        raise ValueError('PR260 checkpoint byte identity failed; no continuation')
    h = header(path)
    if (h['native_state']['completed_nodes'] != 332_894_788 or h['iteration'] != 104400
            or h['abstraction'] != HU100_EQUITY_MENU_V2_SCHEMA or 'frozen_table' in h):
        raise ValueError('Wrong parent lineage')
    put(BASE/'parent-verified.json',dict(checkpoint=actual,header=h,regenerated=regenerated,
        returncode=rc,byte_identical_to_pr260=True,checkpoint_downloaded_from_drive=False))


def train():
    parent = read(BASE/'parent-verified.json')['checkpoint']
    if file_hash(Path(parent['path'])) != SHA:
        raise ValueError('Verified parent changed')
    dest = BASE/'training'; dest.mkdir(exist_ok=False)
    command([*args(5_000_000_000,ENTRIES,dest/'5000000000.gz',dest/'telemetry.jsonl'),
             '--resume',parent['path'],'--resume-sha256',SHA,'--freeze-at-cap'])
    rows = [json.loads(x) for x in (dest/'telemetry.jsonl').read_text().splitlines()]
    h = header(dest/'5000000000.gz')
    if (h['native_state']['completed_nodes'] < 5_000_000_000
            or h['frozen_table']['entries'] != ENTRIES
            or h['frozen_table']['since_iteration'] != 104400
            or any(r['diagnostics']['entries'] != ENTRIES for r in rows)):
        raise ValueError('Frozen target, cardinality or boundary differs')
    put(BASE/'training-complete.json',dict(header=h,telemetry=rows,status='verified'))


def exports():
    checkpoint = BASE/'training/5000000000.gz'
    current = BASE/'training/5000000000.current.gz'
    average = BASE/'training/5000000000.average.gz'
    command([BINARY,'export',checkpoint,'--current',current,'--average',average,'--zero-mass','uniform'])
    h = header(checkpoint)
    spec = dict(name='v2-frozen-5b',seed=SEED,iteration=h['iteration'],
                checkpoint_sha256=file_hash(checkpoint),sha256=file_hash(current))
    checked_header(h,spec,expected_schema=HU100_EQUITY_MENU_V2_SCHEMA)
    receipt = audit(checkpoint,current,average,spec,file_hash(average),expected_schema=HU100_EQUITY_MENU_V2_SCHEMA)
    if receipt['status'] != 'verified' or receipt['all_nodes_verified'] != ENTRIES:
        raise ValueError('Every-row audit incomplete')
    put(BASE/'5b.audit.json',dict(receipt,actual_nodes=h['native_state']['completed_nodes'],
        files={k:pin(p) for k,p in [('checkpoint',checkpoint),('current',current),('average',average)]}))
    meta = header(average)
    put(BASE/'treatment.json',dict(pin(average),name='v2-frozen-5b',
        abstraction=HU100_EQUITY_MENU_V2_SCHEMA,format=meta['format'],card_buckets=str(TABLES),
        entries=ENTRIES,iteration=h['iteration'],source_checkpoint_sha256=spec['checkpoint_sha256']))


def run():
    identity = subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
    if identity != 'Apple M4':
        raise ValueError('M4 only')
    BASE.mkdir(parents=True,exist_ok=False)
    with (Path.home()/'Local/.hu20-m4-research.lock').open('a+') as one, (Path.home()/'Local/.hu100-m4-research.lock').open('a+') as two:
        for f in (one,two): fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        own = {psutil.Process().pid,*[p.pid for p in psutil.Process().parents()]}
        for p in psutil.process_iter(['pid','name','cmdline'],ad_value=None):
            if p.pid in own: continue
            if is_campaign_process(p.info['name'],p.info['cmdline']):
                raise ValueError('Other research process active: '+str(p.pid))
        started = time()
        put(BASE/'budget.json',dict(started=started,deadline=started+7*3600,
            locks=[one.name,two.name],identity=identity))
        for stage in ('prepare','recover','train','exports'):
            command([sys.executable,'-m','scripts.guard_m4_research','--stage',BASE/'guards',
                '--name',stage,'--stop-file',BASE/'stop.request','--',sys.executable,
                '-m','scripts.train_hu100_v2_frozen_5b',stage])
    put(BASE/'training-detached-complete.json',dict(at=time(),research_locks_released=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('run','prepare','recover','train','exports'))
    stage = parser.parse_args().stage
    try:
        globals()[stage]()
    except BaseException as exc:
        if BASE.exists(): put(BASE/(stage+'-failure.json'),dict(at=time(),failure=repr(exc)))
        raise


if __name__ == '__main__': main()
