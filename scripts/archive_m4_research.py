"""Seal owner-authorized M4 evidence without altering previous PR roots/uploads."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

from scripts.archive_hu100_4b_seed_ladder import seal
from scripts.evaluate_hu100_direct import put

ROOT = Path(__file__).resolve().parents[1]
REPO = 'dberweger2017/deepcfr-texas-no-limit-holdem-6-players'


def archive(stage, pr, short, base_ref, *, averages=()):
    owner = json.loads(subprocess.check_output(['gh','pr','view',str(pr),'--repo',REPO,
                     '--json','state,mergedAt,url,headRefName,headRefOid'],text=True))
    branch = subprocess.check_output(['git','branch','--show-current'],text=True).strip()
    if owner['state']=='OPEN' and owner['headRefName']!=branch:
        raise ValueError('Only this explicitly authorized stage root may be archived while open')
    dest = Path.home()/f'Local/Research-Cloud/PR-{pr}-{short}'
    dest.mkdir(parents=True,exist_ok=True)
    files = []
    omitted = []
    for path in sorted(stage.rglob('*')):
        if not path.is_file() or path.is_symlink(): continue
        rel = path.relative_to(stage)
        if rel.parts[0]=='tables':
            # Pinned PR163 inputs have a separately accepted archive and retrieval receipt.
            omitted.append(str(rel)); continue
        if rel.parts[:2]==('operations','archive') or path.name in (
                'archive-supervisor.log','guard-held.json','archive-receipt.json','averages-receipt.json'):
            continue
        files.append((path,'research/'+rel.as_posix()))
    binary = ROOT/'native/hu20-trainer/target/release/hu20-trainer'
    if binary.exists(): files.append((binary,'research/bin/hu20-trainer'))
    changed = subprocess.check_output(['git','diff','--name-only',base_ref,'HEAD'],text=True).splitlines()
    for name in changed:
        path = ROOT/name
        if path.is_file() and path.stat().st_size<=1024**2:
            files.append((path,'source/'+name))
    metadata = {'pr':pr,'owner_status':owner,'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'source_base':base_ref,'originals_retained':True,'indexed_table_dependencies':omitted,
                'excluded_live_archive_guard':'research/operations/archive','remote_archive_bytes_downloaded':False}
    receipt = seal(files,dest/(short+'-full-M4-20261013.zip'),metadata)
    put(stage/'archive-receipt.json',receipt)
    if averages:
        selected = [(path,'research/'+path.relative_to(stage).as_posix()) for path in dict.fromkeys(averages)]
        put(stage/'averages-receipt.json',seal(selected,dest/(short+'-averages-M4-20261013.zip'),metadata))


def native_status(path):
    raw = subprocess.check_output(['fileproviderctl','evaluate',str(path)],text=True,timeout=30)
    def number(key):
        found = re.search(r'\b'+key+r'\s*=\s*(\d+)\s*;',raw)
        return int(found[1]) if found else None
    script = '''function run(argv) {
        ObjC.import("Foundation");
        const url = $.NSURL.fileURLWithPath(argv[0]);
        const versions = $.NSFileVersion.unresolvedConflictVersionsOfItemAtURL(url);
        return JSON.stringify({exists: Boolean($.NSFileManager.defaultManager.fileExistsAtPath(argv[0])),
            nil: versions.isNil(), unresolved_count: versions.isNil() ? null : Number(versions.count)});
    }'''
    conflict = json.loads(subprocess.check_output(['/usr/bin/osascript','-l','JavaScript','-e',script,str(path)],text=True,timeout=20))
    uploaded, uploading, size = number('isUploaded'),number('isUploading'),number('documentSize')
    accepted = (uploaded==1 and uploading==0 and size==path.stat().st_size
                and conflict['exists'] and not conflict['nil'] and conflict['unresolved_count']==0
                and not re.search(r'\buploadingError\s*=',raw))
    item_id = subprocess.check_output(['xattr','-p','com.google.drivefs.item-id#S',str(path)],text=True).strip() if accepted else None
    parent_id = subprocess.check_output(['xattr','-p','com.google.drivefs.item-id#S',str(path.parent)],text=True).strip() if accepted else None
    return {'at':time.time(),'path':str(path),'name':path.name,'bytes':size,'is_uploaded':uploaded,
            'is_uploading':uploading,'conflicts':conflict,'native_accepted':accepted,
            'cloud_file_id':item_id,'cloud_folder_id':parent_id,'raw':raw,
            'remote_archive_bytes_downloaded':False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('pack','native'))
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--pr',type=int)
    p.add_argument('--short')
    p.add_argument('--base-ref')
    p.add_argument('--average',type=Path,action='append',default=[])
    p.add_argument('--receipt',type=Path)
    a = p.parse_args()
    if a.operation=='pack':
        if not a.pr or not a.short or not a.base_ref: p.error('pack requires pr, short and base-ref')
        archive(a.stage,a.pr,a.short,a.base_ref,averages=a.average)
    else:
        data = {}
        for name in ('archive-receipt.json','averages-receipt.json'):
            path = a.stage/name
            if path.exists(): data[name] = native_status(Path(json.loads(path.read_text())['path']))
        put(a.receipt,data)


if __name__=='__main__': main()
