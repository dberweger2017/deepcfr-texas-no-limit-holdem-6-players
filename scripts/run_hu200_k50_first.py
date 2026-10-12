"""Fresh owner-authorized HU200 K50 lineage; no matches or promotion."""
import argparse
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys

from scripts.evaluate_hu100_direct import put
from src.blueprint.abstraction import HU200_EQUITY_SCHEMA
from src.blueprint.equity_buckets import EQUITY_K50_SHA256
from src.diagnostics.cfr_average import audit, checked_header
from src.policies.files import file_hash

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/hu200-k50-first'
TABLES = BASE/'tables'
BINARY = ROOT/'native/hu20-trainer/target/release/hu20-trainer'
SEED = 2026101302
CAP = math.floor((11_000_000_000-1_400_000_000-500_000_000)/(96.9*1.1))
TARGET = 3_000_000_000
REPO = 'dberweger2017/deepcfr-texas-no-limit-holdem-6-players'


def command(args, allowed=(0,)):
    rc = subprocess.run(list(map(str,args)), cwd=ROOT).returncode
    if rc not in allowed:
        raise RuntimeError('Command exit '+str(rc))
    return rc


def pin(path):
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':file_hash(path)}


def prepare():
    owner = json.loads(subprocess.check_output(['gh','pr','view','163','--repo',REPO,'--json','state,mergedAt,url'],text=True))
    if owner['state']!='MERGED':
        raise ValueError('PR163 must be merged before retrieval')
    archive = Path.home()/'Local/Research-Cloud/PR-163-equity-buckets/hu20-equity-buckets-20261005.tar.gz'
    archive_pin = pin(archive)
    if archive_pin['sha256']!='03d4828f030082f213c8ce82e29fa0fa8fd97288bd8035ad0056de85bc0816ca':
        raise ValueError('Indexed PR163 archive differs')
    TABLES.mkdir(parents=True,exist_ok=False)
    args = ['tar','-xzf',archive,'-C',TABLES,'--strip-components','1',
            *[f'hu20-equity-buckets-20261005/{s}-k50.bin' for s in EQUITY_K50_SHA256]]
    command(args)
    tables = {s:pin(TABLES/(s+'-k50.bin')) for s in EQUITY_K50_SHA256}
    if any(tables[s]['sha256']!=sha for s,sha in EQUITY_K50_SHA256.items()):
        raise ValueError('Pinned K50 tables differ')
    put(BASE/'inputs.json', {'status':'verified','owner':owner,'archive':archive_pin,
         'archive_id':'1Gl1JYWN0F7KFJbwoKtcA0zB_rUsWWZ6T',
         'archive_url':'https://drive.google.com/file/d/1Gl1JYWN0F7KFJbwoKtcA0zB_rUsWWZ6T/view',
         'folder_id':'1gwungsUq-b9uey8InGy0yOD2GbaMRhp-',
         'tables':tables,'retrieval_command':list(map(str,args)), 'max_entries':CAP})


def train_command(nodes, out, telemetry, resume=None):
    args = [BINARY,'train','--stack-bb',200,'--seed',SEED,'--roots-per-seat',1,
            '--average-rule','opponent-sampled','--nodes',nodes,'--max-entries',CAP,
            '--card-buckets',TABLES,'--out',out,'--telemetry',telemetry]
    if resume:
        args.extend(['--resume',resume,'--resume-sha256',file_hash(resume)])
    return args


def gates():
    dest = BASE/'gates'; dest.mkdir(exist_ok=False)
    for name,nodes,resume in [('fresh20m',20_000_000,None),('prefix10m',10_000_000,None),
                              ('resumed20m',20_000_000,dest/'prefix10m.gz')]:
        command(train_command(nodes,dest/(name+'.gz'),dest/(name+'.telemetry.jsonl'),resume))
    left,right = pin(dest/'fresh20m.gz'),pin(dest/'resumed20m.gz')
    if (left['bytes'],left['sha256'])!=(right['bytes'],right['sha256']):
        raise ValueError('Fresh/resumed 20M checkpoints differ')
    put(BASE/'gates-complete.json', {'status':'byte-identical','fresh':left,'resumed':right,'binary':pin(BINARY)})


def train():
    gate = json.loads((BASE/'gates-complete.json').read_text())
    if gate['binary']['sha256']!=file_hash(BINARY):
        raise ValueError('Qualified binary changed')
    dest = BASE/'training'; dest.mkdir(exist_ok=False)
    command([*train_command(TARGET,dest/'{nodes}.gz',dest/'telemetry.jsonl'),
             '--milestones','500000000,1000000000,1500000000,2000000000,2500000000',
             '--stop-file',BASE/'native-stop.json'], allowed=(0,3))
    rows = [json.loads(line) for line in (dest/'telemetry.jsonl').read_text().splitlines()]
    for row in rows:
        if row['binary_sha256']!=gate['binary']['sha256']:
            raise ValueError('Telemetry binary differs')
    terminal = rows[-1]
    if terminal['completed_nodes']<TARGET and terminal['diagnostics']['entries']<CAP and not terminal['stop_requested']:
        raise ValueError('Undeclared native training stop')
    put(BASE/'training-complete.json', {'telemetry':rows,'terminal':terminal,
          'max_entries':CAP,'seed':SEED,'schema':HU200_EQUITY_SCHEMA,
          'reason':'guard' if terminal['stop_requested'] else 'entry-cap' if terminal['diagnostics']['entries']>=CAP else 'node-target'})


def exports():
    data = json.loads((BASE/'training-complete.json').read_text())
    rows = data['telemetry']
    selected = [r for r in rows if r['requested_nodes']==1_000_000_000]+[rows[-1]]
    checked = []
    seen = set()
    for record in selected:
        if record['path'] in seen: continue
        seen.add(record['path'])
        checkpoint = Path(record['path'])
        if file_hash(checkpoint)!=record['checkpoint_sha256']:
            raise ValueError('Saved checkpoint hash changed')
        current = checkpoint.with_suffix('.current.gz')
        average = checkpoint.with_suffix('.average.gz')
        command([BINARY,'export',checkpoint,'--current',current,'--average',average,'--zero-mass','uniform'])
        with gzip.open(checkpoint,'rt') as f: header = json.loads(next(f))
        spec = {'name':'hu200-k50','seed':SEED,'iteration':header['iteration'],
                'checkpoint_sha256':file_hash(checkpoint),'sha256':file_hash(current)}
        checked_header(header,spec,expected_schema=HU200_EQUITY_SCHEMA)
        receipt = audit(checkpoint,current,average,spec,file_hash(average),expected_schema=HU200_EQUITY_SCHEMA)
        if receipt['all_nodes_verified']!=record['diagnostics']['entries']:
            raise ValueError('Incomplete full export audit')
        put(checkpoint.with_suffix('.audit.json'),dict(receipt,actual_nodes=record['completed_nodes'],
             schema=HU200_EQUITY_SCHEMA,files={k:pin(p) for k,p in [('checkpoint',checkpoint),('current',current),('average',average)]}))
        checked.append({'actual_nodes':record['completed_nodes'],'requested_nodes':record['requested_nodes'],
                        'average':pin(average),'current':pin(current),'audit':pin(checkpoint.with_suffix('.audit.json'))})
    put(BASE/'exports-complete.json',{'status':'fully-audited','exports':checked})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('prepare','gates','train','exports'))
    a = p.parse_args()
    globals()[a.operation]()


if __name__=='__main__': main()
