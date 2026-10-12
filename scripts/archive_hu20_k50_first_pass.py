"""Archive only PR238 evidence; original indexed inputs remain explicit dependencies."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from time import time

from scripts.archive_hu100_4b_seed_ladder import seal
from scripts.hu20_k50_first_pass_assets import check, write
from scripts.run_hu20_k50_first_pass_training import BASE, BINARY
from src.policies.files import file_hash

SOURCE_BASE='045c793a4abd2e371d4d77de663fd3dd2f00316b'


def read(path):
    return json.loads(path.read_text())


def inventory():
    dependencies={}
    table=read(BASE/'tables-restoration.json')
    for item in table['members']:
        path=Path(item['path'])
        dependencies[path.as_posix()]={'local_path':str(path),'bytes':item['bytes'],'sha256':item['sha256'],
            'archive':table['archive'],'drive_id':table['drive_id'],'drive_folder':table['drive_folder'],
            'member':'hu20-equity-buckets-20261005/'+path.name,'retrieval':table['retrieval']}
    ref=read(BASE/'reference-restoration.json')
    dependencies[ref['member']['path']]={**ref['member'],'archive':ref['archive'],
        'drive_folder':ref['drive_folder'],'member':ref['archive_member'],'retrieval':ref['retrieval']}
    for item in read(BASE/'controls-restoration.json'):
        dependencies[item['member']['path']]={**item['member'],'archive':item['archive'],
            'member':item['archive_member'],'manifest_sha256':item['manifest_sha256'],
            'retrieval':item['retrieval'],'index':'RESULTS_INDEX.md PR182/185 milestone archive IDs'}
    release=read(BASE/'release-restoration.json')
    dependencies[release['asset']['path']]={**release['asset'],
        'release_url':'https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/releases/tag/v0.4.2',
        'member':Path(release['asset']['path']).name,'retrieval':release['retrieval']}
    files=[];excluded=[]
    for path in sorted(BASE.rglob('*')):
        if not path.is_file() or path.is_symlink():continue
        rel=path.relative_to(BASE)
        # A live guard stream changes while sealing; its closed receipt is
        # retained in a separate closeout sidecar with independent acceptance.
        if rel.parts[:3]==('guards','operations','archive-final') or rel.as_posix() in (
            'guards/status.json','guards/status.tmp','archive-receipt.json','archive-admission.json'):
            continue
        if path.as_posix() in dependencies:
            pin=dependencies[path.as_posix()]
            check(path,pin['sha256'],pin['bytes']);excluded.append(pin);continue
        files.append((path,'research/'+rel.as_posix()))
    for path in sorted(Path('planning').glob('*')):
        if path.is_file() and not path.is_symlink() and path.suffix in ('.json','.md','.log','.txt'):
            files.append((path,'provenance/planning/'+path.name))
    files.append((BINARY,'research/bin/hu20-trainer'))
    changed=subprocess.check_output(['git','diff','--name-only',SOURCE_BASE,'HEAD'],text=True).splitlines()
    for name in changed:
        path=Path(name)
        if path.is_file() and path.stat().st_size<=1024**2:
            files.append((path,'source/'+name))
    return files,excluded


def pack():
    owner=json.loads(subprocess.check_output(['gh','pr','view','238','--repo',
        'dberweger2017/deepcfr-texas-no-limit-holdem-6-players','--json','state,isDraft,headRefOid'],text=True))
    if owner['state']!='OPEN':raise ValueError('Unexpected campaign PR state')
    if read(BASE/'final-readout.json')['status']!='verified':raise ValueError('Final replayed readout required')
    files,excluded=inventory()
    logical=sum(path.stat().st_size for path,_ in files)
    cloud=Path.home()/'Local/Research-Cloud/PR-238-hu20-k50-5b-first-pass'
    # This is the single native Research-Cloud PR238 folder; preserve any
    # existing accepted files, never delete/evict/overwrite in a synced folder.
    cloud.mkdir(parents=True,exist_ok=True)
    free=shutil.disk_usage(cloud).free
    if free-logical-2*1024**3<16*1024**3:raise ValueError('Archive storage reserve does not fit')
    write(BASE/'archive-admission.json',{'at':time(),'logical_member_bytes':logical,'members':len(files),
        'disk_free_bytes':free,'required_free_bytes':logical+18*1024**3,'owner_pr':owner})
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    receipt=seal(files,cloud/'hu20-k50-5b-first-pass-M4-20261012.zip',{
        'pr':238,'source':source,'source_base':SOURCE_BASE,'binary_sha256':file_hash(BINARY),
        'canonical_plan_sha256':file_hash(BASE/'canonical-final-plan.json'),
        'indexed_input_dependencies':excluded,'originals_retained':True,
        'live_archive_guard_excluded':'research/guards/operations/archive-final',
        'scope':'Owner-authorized equal-node K50 first pass, every final hand independently replayed; no adoption or release'})
    write(BASE/'archive-receipt.json',receipt)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    pack()


if __name__=='__main__':main()
