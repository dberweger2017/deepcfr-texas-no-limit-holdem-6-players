"""One-use sequential terminal training/export; invoke under the campaign guard."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import time

from scripts.hu20_k50_first_pass_assets import TABLES, CONTROLS, check, restore_controls, write
from src.policies.files import file_hash

BASE=Path('results/hu20-k50-5b')
BINARY=Path('native/hu20-trainer/target/release/hu20-trainer')


def execute(name,command,outputs,*,disk_reserve):
    if any(path.exists() for path in outputs):raise FileExistsError('Preserve prior output: '+name)
    free=shutil.disk_usage(BASE).free
    if free-disk_reserve<16*1024**3:raise ValueError('Stage disk forecast does not fit: '+name)
    write(BASE/'stage-intents'/(name+'.json'),{'command':command,'outputs':[str(p) for p in outputs],
        'started':time(),'disk_free_bytes':free,'additional_disk_reserve_bytes':disk_reserve,
        'binary_sha256':file_hash(BINARY),'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    subprocess.run(command,check=True)


def export_and_audit(seed,*,k50):
    label='k50' if k50 else 'v1'
    checkpoint=BASE/'training'/f'K50-{seed}-5000000000.json.gz' if k50 else BASE/'inputs'/f'O-{seed}-5000000000.json.gz'
    average=BASE/'policies'/f'{label}-{seed}-5b.average.jsonl.gz'
    current=BASE/'policies'/f'{label}-{seed}-5b.current.json.gz'
    average.parent.mkdir(parents=True,exist_ok=True)
    execute(f'{label}-{seed}-export',[str(BINARY),'export',str(checkpoint),'--zero-mass','uniform',
        '--average',str(average),'--current',str(current)],[average,current],disk_reserve=4*1024**3)
    if not k50 and seed==2026100601:
        check(average,'ab8ce0087f6bfefde447ebe48ad754c094db13318803df890488eb1283453193',214796948)
    out=BASE/'audits'/f'{label}-{seed}-5b.json'
    command=[sys.executable,'-m','scripts.hu20_k50_first_pass_measurements','--checkpoint',str(checkpoint),
        '--current',str(current),'--average',str(average),'--out',str(out)]
    if k50:
        command+=['--telemetry',str(BASE/'training'/f'{seed}.telemetry.jsonl'),'--tables',str(BASE/'tables')]
    else:
        command+=['--archive-metadata',str(BASE/'inputs'/f'O-{seed}-5000000000-archive-metadata')]
    execute(f'{label}-{seed}-audit',command,[out],disk_reserve=10*1024**3)


def controls():
    if not json.loads((BASE/'gates-complete.json').read_text())['status']=='verified':raise ValueError('Gates not complete')
    free=shutil.disk_usage(BASE).free
    if free-3*1024**3<16*1024**3:raise ValueError('Control restoration disk forecast does not fit')
    restore_controls(BASE)
    for seed,*_ in CONTROLS:
        export_and_audit(seed,k50=False)
    write(BASE/'controls-complete.json',{'status':'verified','finished':time(),'seeds':[c[0] for c in CONTROLS]})


def treatment():
    if not json.loads((BASE/'controls-complete.json').read_text())['status']=='verified':raise ValueError('Controls not complete')
    gate=json.loads((BASE/'gates-complete.json').read_text())
    if gate['status']!='verified' or gate['binary_sha256']!=file_hash(BINARY):raise ValueError('Gate binary differs')
    for name,(size,sha) in TABLES.items():check(BASE/'tables'/name,sha,size)
    budget=json.loads((BASE/'guards/budget.json').read_text())
    reference=json.loads((BASE/'gates/v1-seed602-fresh-1b.telemetry.jsonl').read_text())
    train_reserve=3*5*1.13*1.25*reference['elapsed_seconds_including_writes']
    downstream_reserve=2*3600
    quote={'at':time(),'remaining_seconds':budget['deadline']-time(),'training_reserve_seconds':train_reserve,
           'downstream_provisional_reserve_seconds':downstream_reserve,
           'basis':'Fresh byte-identical M4 v1 1B train/save time, x5 nodes x3 seeds, 13 percent K50 overhead and 25 percent headroom. Pilot measured play/replay quote still required before final play.'}
    quote['admitted']=train_reserve+downstream_reserve<=quote['remaining_seconds']
    write(BASE/'training-time-admission.json',quote)
    if not quote['admitted']:raise ValueError('Training and downstream time forecast do not fit six hours')
    for seed,*_ in CONTROLS:
        checkpoint=BASE/'training'/f'K50-{seed}-5000000000.json.gz'
        telemetry=BASE/'training'/f'{seed}.telemetry.jsonl'
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        command=[str(BINARY),'train','--nodes','5000000000','--seed',str(seed),
                 '--average-rule','opponent-sampled','--max-entries','1000000000','--card-buckets',str(BASE/'tables'),
                 '--telemetry',str(telemetry),'--out',str(checkpoint)]
        execute(f'k50-{seed}-train',command,[checkpoint,telemetry],disk_reserve=2*1024**3)
        records=[json.loads(line) for line in telemetry.read_text().splitlines()]
        if len(records)!=1 or records[0]['status']!='saved' or records[0]['stop_requested'] or records[0]['completed_nodes']<5000000000:
            raise ValueError('Terminal treatment target incomplete')
        check(checkpoint,records[0]['checkpoint_sha256'],records[0]['checkpoint_bytes'])
        export_and_audit(seed,k50=True)
    write(BASE/'treatment-complete.json',{'status':'verified','finished':time(),'seeds':[c[0] for c in CONTROLS],
        'binary_sha256':file_hash(BINARY)})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('controls','treatment'))
    a=p.parse_args();{'controls':controls,'treatment':treatment}[a.stage]()


if __name__=='__main__':main()
