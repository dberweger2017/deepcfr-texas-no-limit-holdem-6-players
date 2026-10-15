"""Stream closed PR278 seed evidence into member-hashed ZIPs on the M1.

No intermediate payload copy, synced deletion, or M4 original removal occurs.
The M4 seed supervisor must have exited before selection. Complete local ZIP
readback is separate from subsequent native/cloud upload acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from zipfile import ZipFile, ZIP_STORED

from src.policies.files import file_hash

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/Users/dberweger/Local/hu200-v055-confirmation-20261014'
BASE=REMOTE+'/results/hu200-v055-confirmation'
DEST=Path.home()/'Local/Research-Cloud/PR-278-hu200-v055-confirmation'
OUT=ROOT/'results/hu200-v055-archive-receipts'
CHUNK=1<<20


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def select(seed):
    owner=json.loads(subprocess.check_output(['gh','pr','view','278','--json','state,headRefName,headRefOid,url'],text=True))
    if owner['state']!='OPEN' or owner['headRefName']!='feature/hu200-v055-confirmation':
        raise ValueError('Archive owner differs from explicitly authorized open PR278')
    script=f'BASE = {BASE!r}\nREMOTE = {REMOTE!r}\nSEED = {seed!r}\n'+'''import hashlib,json,os
from pathlib import Path
base=Path(BASE)
seed=SEED
receipt=json.loads((base/f'seed-{seed}-supervisor-receipt.json').read_text())
science_path=base/f'seed-{seed}/science-complete.json'
science=json.loads(science_path.read_text()) if science_path.exists() else None
import psutil
claim=json.loads((base/f'seed-{seed}-claim.json').read_text())
try:
 process=psutil.Process(claim['pid'])
 if process.create_time()==claim['create_time'] and process.status()!=psutil.STATUS_ZOMBIE: raise ValueError('Seed supervisor is still live')
except psutil.NoSuchProcess: pass
files=list(sorted((base/f'seed-{seed}').rglob('*')))
files += [base/name for name in ('input-owners.json','readmission.json','baseline.json','build-receipt.json','inputs.json','detached-launch.json',f'seed-{seed}-claim.json',f'seed-{seed}-worker.log',f'seed-{seed}-resources.jsonl',f'seed-{seed}-supervisor-receipt.json')]
files += [Path(REMOTE)/'planning/hu200-v055/campaign.py',Path(REMOTE)/'native/hu20-trainer/target/release/hu20-trainer']
result={}
for path in files:
 if path.is_symlink(): raise ValueError('Unexpected evidence symlink')
 if not path.is_file(): continue
 stat=path.stat();digest=hashlib.sha256()
 with path.open('rb') as stream:
  while chunk:=stream.read(1<<20):digest.update(chunk)
 member='research/'+str(path.relative_to(base)) if path.is_relative_to(base) else 'source/'+str(path.relative_to(Path(REMOTE)))
 result[member]={'path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':digest.hexdigest(),'host':'m4'}
print(json.dumps({'files':result,'science':science,'supervisor':receipt}))
'''
    raw=subprocess.check_output(['ssh','m4','~/Local/hu100-1b-growth-20261008/.venv/bin/python -'],input=script,text=True)
    selected=json.loads(raw)
    metadata=dict(pr=278,owner=owner,seed=seed,source='9c449929e4b41e8c60383e682bea1e0e1e4c01d5',
        remote_seed_supervisor=selected['supervisor'],input_tables='PR163 accepted indexed archive; excluded; see research/inputs.json',
        originals_retained=True,remote_archive_bytes_downloaded=False)
    put(OUT/f'seed-{seed}-selection.json',dict(selected,metadata=metadata))
    return selected,metadata


def seal(files,destination,metadata):
    tick=time.time()
    pins={member:dict(bytes=row['bytes'],sha256=row['sha256'],source_path='m4:'+row['path'],source_mtime_ns=row['mtime_ns']) for member,row in files.items()}
    manifest=(json.dumps(dict(version=1,metadata=metadata,members=pins),indent=2,sort_keys=True)+'\n').encode()
    # Destination folder is created and verified through the Drive connector;
    # requiring its existing local counterpart avoids duplicate sync folders.
    if not destination.parent.is_dir(): raise ValueError('Canonical cloud folder not materialized')
    with destination.open('xb') as raw,ZipFile(raw,'w',compression=ZIP_STORED,allowZip64=True) as archive:
        for member,row in files.items():
            command=['python3','-c','import os,sys;s=os.stat(sys.argv[1]);assert (s.st_size,s.st_mtime_ns)==(int(sys.argv[2]),int(sys.argv[3]))',row['path'],str(row['bytes']),str(row['mtime_ns'])]
            subprocess.run(['ssh','m4',shlex.join(command)],check=True)
            child=subprocess.Popen(['ssh','m4',shlex.join(['cat',row['path']])],stdout=subprocess.PIPE)
            digest=hashlib.sha256();count=0
            try:
                with archive.open(member,'w',force_zip64=True) as target:
                    while chunk:=child.stdout.read(CHUNK):target.write(chunk);digest.update(chunk);count+=len(chunk)
                if child.wait()!=0:raise RuntimeError('Remote evidence read failed')
            finally:
                child.stdout.close()
                if child.poll() is None:child.terminate();child.wait()
            if count!=row['bytes'] or digest.hexdigest()!=row['sha256']:raise ValueError('Streamed member SHA256/size differs')
        archive.writestr('ARCHIVE-MANIFEST.json',manifest)
    with ZipFile(destination) as archive:
        if set(archive.namelist())!=set(pins)|{'ARCHIVE-MANIFEST.json'} or archive.read('ARCHIVE-MANIFEST.json')!=manifest:
            raise ValueError('Archive manifest/inventory differs')
        for member,pin in pins.items():
            digest=hashlib.sha256();count=0
            with archive.open(member) as stream:
                while chunk:=stream.read(CHUNK):digest.update(chunk);count+=len(chunk)
            if count!=pin['bytes'] or digest.hexdigest()!=pin['sha256']:raise ValueError('Full local member readback differs')
    return dict(status='locally-verified',path=str(destination),bytes=destination.stat().st_size,sha256=file_hash(destination),
        manifest_path='ARCHIVE-MANIFEST.json',manifest_sha256=hashlib.sha256(manifest).hexdigest(),members=len(pins),
        all_member_sizes_and_hashes_readback=True,seconds=time.time()-tick,cloud_upload_accepted=False,remote_archive_bytes_downloaded=False)


def main(seed, archive_kind='both'):
    selected,metadata=select(seed); files=selected['files']
    average=selected['science']['exports']['files']['average']['path'] if selected['science'] else None
    avg_files={'averages/seed-'+str(seed)+'/final.average.gz':next(row for row in files.values() if row['path']==average)} if average else {}
    choices=[(label,members) for label,members in (('full',files),('final-average',avg_files))
        if archive_kind in ('both',label)]
    expected=sum(row['bytes'] for _,members in choices for row in members.values())
    if shutil.disk_usage(DEST).free<expected+2*1024**3:raise ValueError('Insufficient M1 archive space; preserve M4 originals')
    for label,members in choices:
        if not members: continue
        path=DEST/f'hu200-v055-seed-{seed}-{label}-20261014.zip'
        try: result=seal(members,path,metadata)
        except BaseException as exc:
            put(OUT/f'seed-{seed}-{label}-failure.json',dict(failure=repr(exc),path=str(path),at=time.time()));raise
        put(OUT/f'seed-{seed}-{label}-receipt.json',result)
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,choices=(2026101421,2026101422),required=True)
    parser.add_argument('--archive-kind',choices=('both','full','final-average'),default='both',
        help='Seal selected ZIPs when the same canonical folder is served by both Macs')
    args=parser.parse_args();main(args.seed,args.archive_kind)
