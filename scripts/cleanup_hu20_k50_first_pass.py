"""Plan and remove only unchanged, accepted PR238 research originals after merge."""
import argparse
import json
import re
from pathlib import Path
import shutil
import subprocess
from time import time
from zipfile import ZipFile
import psutil

ROOT=Path.cwd()
BASE=ROOT/'results/hu20-k50-5b'


def read(path):
    return json.loads(path.read_text())


def owned_candidate(path,pin):
    """A manifest cannot authorize deletion outside this task or through symlinks."""
    absolute=path if path.is_absolute() else ROOT/path
    try:relative=absolute.relative_to(BASE)
    except ValueError:return False,'outside own research root'
    if any(part in ('..','.git') for part in relative.parts):return False,'unsafe relative path'
    if relative.parts[0]=='guards':return False,'preserve guard controls, failure latch and operation history'
    if absolute.suffix not in ('.gz','.bin') and pin['bytes']<=1024**2:
        return False,'preserve compact receipts and provenance'
    if absolute.is_symlink() or any(parent.is_symlink() for parent in absolute.parents if parent!=ROOT):
        return False,'symlinked source'
    if not absolute.is_file():return False,'missing/nonregular'
    stat=absolute.stat()
    if stat.st_size!=pin['bytes'] or stat.st_mtime_ns!=pin['source_mtime_ns']:
        return False,'size or modification history changed'
    return True,None


def candidates():
    selected={};retained=[]
    for name in ('main-archive-acceptance.json','closeout-archive-acceptance.json'):
        acceptance=read(BASE/name)
        if acceptance['status'] not in ('main-archive-accepted','closeout-archive-accepted'):
            raise ValueError('Current archive acceptance required')
        archive=acceptance['local_integrity']
        raw=subprocess.check_output(['fileproviderctl','evaluate',archive['path']],text=True)
        flags={key:int(match[1]) if (match:=re.search(r'\b'+key+r'\s*=\s*(\d+)\s*;',raw)) else None
               for key in ('documentSize','isUploaded','isUploading','hasUnresolvedConflicts','isTrashed','isSyncPaused','isExcludedFromSync')}
        if (flags['documentSize']!=archive['bytes'] or flags['isUploaded']!=1 or flags['isUploading']!=0
                or any(flags[key]!=0 for key in ('hasUnresolvedConflicts','isTrashed','isSyncPaused','isExcludedFromSync'))
                or re.search(r'\buploadingError\s*=',raw)):
            raise ValueError('Current native archive upload acceptance changed')
        with ZipFile(archive['path']) as bundle:
            manifest=json.loads(bundle.read(archive['manifest_path']))
        for member,pin in manifest['members'].items():
            path=Path(pin['source_path'])
            valid,reason=owned_candidate(path,pin)
            absolute=path if path.is_absolute() else ROOT/path
            if not valid:
                if absolute.is_relative_to(BASE):retained.append({'path':str(absolute),'reason':reason})
                continue
            selected[str(absolute)]={'path':str(absolute),'bytes':pin['bytes'],
                'source_mtime_ns':pin['source_mtime_ns'],'sha256':pin['sha256'],
                'archive_id':acceptance['independent_cloud_metadata']['id'],'archive_url':acceptance['independent_cloud_metadata']['url'],
                'archive_sha256':archive['sha256'],'member':member}
    input_receipt=read(ROOT/'planning/indexed-input-cleanup-pins.json')
    if input_receipt['status']!='current-native-and-independent-input-archives-accepted':
        raise ValueError('Indexed input archives have no current acceptance')
    for pin in input_receipt['copies']:
        path=Path(pin['path'])
        valid,reason=owned_candidate(path,pin)
        if valid:selected[str(path)]=pin
        else:retained.append({'path':str(path),'reason':reason})
    return list(selected.values()),retained


def dependencies():
    denied=[];active=[]
    current=psutil.Process()
    own={current.pid,*[p.pid for p in current.parents()]}
    for p in psutil.process_iter():
        try:
            row=p.as_dict(attrs=['pid','ppid','cmdline','create_time'])
            command=' '.join(row['cmdline'] or [])
            if row['pid'] in own:continue
            if any(row[key] is None for key in ('cmdline','ppid','create_time')):
                denied.append(row['pid']);continue
            if ('hu20-trainer' in command and 'train' in command or 'evaluate_hu20' in command
                    or str(BASE) in command or 'K50-202610060' in command):
                active.append(row)
        except psutil.NoSuchProcess:pass
        except psutil.AccessDenied:denied.append(p.pid)
    if denied or active:raise ValueError('Active/uncertain dependencies; retain all originals')
    public=json.loads(subprocess.check_output(['gh','search','prs','"hu20-k50-5b-first-pass"','--repo',
        'dberweger2017/deepcfr-texas-no-limit-holdem-6-players','--state','open','--match','body',
        '--json','number,url,title','--limit','100'],text=True))
    foreign=[r for r in public if r['number']!=238 and 'cleanup' not in r['title'].lower()]
    if foreign:raise ValueError('Other open PR claim; retain all originals')
    return {'checked_at':time(),'process_metadata_active':active,'denied_metadata':denied,
            'public_open_claims':public,'scope':'Own paths/process metadata/public plans only; no other PR work root read'}


def plan():
    files,retained=candidates()
    result={'status':'planned-no-deletion','at':time(),'candidates':files,'retained':retained,
            'candidate_bytes':sum(r['bytes'] for r in files),
            'policy':'After PR238 merge only; accepted archives, exact path/size/mtime and dependency review. No synced deletion/eviction, shared Git, other roots or changed/uncertain files.'}
    path=ROOT/'planning/cleanup-plan.json'
    with path.open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps({'candidate_files':len(files),'candidate_bytes':result['candidate_bytes'],'retained':len(retained)}))


def execute():
    owner=json.loads(subprocess.check_output(['gh','pr','view','238','--repo',
        'dberweger2017/deepcfr-texas-no-limit-holdem-6-players','--json','state,mergedAt,mergeCommit'],text=True))
    if owner['state']!='MERGED':raise ValueError('Do not remove open PR evidence')
    review=dependencies()
    planned=read(ROOT/'planning/cleanup-plan.json')
    now,retained=candidates()
    current={r['path']:r for r in now}
    receipt={'status':'running','started':time(),'owner_pr':owner,'dependencies':review,
             'free_disk_before':shutil.disk_usage(BASE).free,'removed':[],'retained':retained,
             'archives_reaudited':False,'original_policy':'Owner-authorized cleanup after accepted archives and merge'}
    destination=ROOT/'planning/cleanup-receipt.json'
    def persist():
        temporary=destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(receipt,indent=2)+'\n');temporary.replace(destination)
    persist()
    for pin in planned['candidates']:
        path=Path(pin['path'])
        if current.get(pin['path'])!=pin:
            receipt['retained'].append({'path':str(path),'reason':'changed since cleanup plan'});persist();continue
        valid,reason=owned_candidate(path,pin)
        if not valid:
            receipt['retained'].append({'path':str(path),'reason':reason});persist();continue
        path.unlink()
        receipt['removed'].append(pin);persist()
    receipt.update(status='complete',finished=time(),free_disk_after=shutil.disk_usage(BASE).free,
                   removed_bytes=sum(r['bytes'] for r in receipt['removed']))
    persist();print(json.dumps({'removed_files':len(receipt['removed']),'removed_bytes':receipt['removed_bytes'],
                               'free_disk_after':receipt['free_disk_after']}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('plan','execute'))
    args=p.parse_args();{'plan':plan,'execute':execute}[args.stage]()


if __name__=='__main__':main()
