"""Seal explicit completed Phase 2 evidence on bb128, retaining every original.

Use the current owning-PR status receipt supplied by the operator. This is the
owner-authorized PR292 archival only; it cannot clean or offload any input.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile


def check_stop():
    stop=os.environ.get('NPLAYER_STOP_FILE')
    if stop and Path(stop).exists(): raise RuntimeError('supervisor stop; partial archive retained')


def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def seal(root, paths, out, owner_status, part_bytes=128*1024**2):
    owner=json.loads(owner_status.read_text())
    if owner['number']!=292 or owner['state'] not in ('OPEN','MERGED'):
        raise ValueError('need current PR292 status receipt')
    if owner['state']=='OPEN' and owner['headRefName']!='feature/nplayer-blueprint-phase2':
        raise ValueError('unexpected owning branch')
    out.mkdir(parents=True,exist_ok=False)
    root=root.resolve()
    files={}
    for relative in paths:
        base=root/relative
        if not base.exists() or base.is_symlink(): raise ValueError('explicit local source required')
        selected=sorted(base.rglob('*')) if base.is_dir() else [base]
        for path in selected:
            if path.is_symlink(): continue
            if path.is_file():
                resolved=path.resolve()
                if not resolved.is_relative_to(root) or resolved.is_relative_to(out.resolve()):
                    raise ValueError('archive source escapes root or contains itself')
                files[str(path.relative_to(root))]=path
    manifest=[]
    for member,path in sorted(files.items()):
        check_stop()
        before=path.stat();sha=digest(path);after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError('source changed during inventory')
        manifest.append({'member':member,'bytes':before.st_size,'sha256':sha,
                         'source_mtime_ns':before.st_mtime_ns})
    metadata={'format':'nplayer-phase2-archive-v1','owner_status':owner,
        'source_root':str(root),'originals_retained':True,'created_unix':time.time(),
        'members':manifest}
    payload=(json.dumps(metadata,indent=2,sort_keys=True)+'\n').encode()
    manifest_sha=hashlib.sha256(payload).hexdigest()
    archive=out/'evidence.zip'
    with zipfile.ZipFile(archive,'x',allowZip64=True) as z:
        z.writestr('MANIFEST.json',payload,compress_type=zipfile.ZIP_DEFLATED)
        for row in manifest:
            check_stop()
            path=files[row['member']]
            stat=path.stat()
            if (stat.st_size,stat.st_mtime_ns)!=(row['bytes'],row['source_mtime_ns']):
                raise ValueError('source changed before sealing')
            compression=zipfile.ZIP_STORED if path.suffix=='.gz' else zipfile.ZIP_DEFLATED
            z.write(path,row['member'],compress_type=compression,compresslevel=3)
            stat=path.stat()
            if (stat.st_size,stat.st_mtime_ns)!=(row['bytes'],row['source_mtime_ns']):
                raise ValueError('source changed while sealing')
    with zipfile.ZipFile(archive) as z:
        if z.read('MANIFEST.json')!=payload: raise ValueError('manifest readback mismatch')
        for row in manifest:
            check_stop()
            count=0;sha=hashlib.sha256()
            with z.open(row['member']) as f:
                while chunk:=f.read(4*1024**2): count+=len(chunk);sha.update(chunk)
            if count!=row['bytes'] or sha.hexdigest()!=row['sha256']:
                raise ValueError('member readback mismatch')
    parts=[]
    # Byte parts are transport units. Restoration concatenates before ZIP extraction.
    with archive.open('rb') as source:
        index=0
        while True:
            check_stop()
            first=source.read(min(part_bytes,4*1024**2))
            if not first: break
            path=out/f'evidence.zip.part{index:04d}'
            sha=hashlib.sha256();count=0
            with path.open('xb') as target:
                chunk=first
                while chunk:
                    target.write(chunk);sha.update(chunk);count+=len(chunk)
                    if count==part_bytes: break
                    chunk=source.read(min(4*1024**2,part_bytes-count))
            parts.append({'member':path.name,'bytes':count,'sha256':sha.hexdigest()});index+=1
    # Verify the part sequence against the sealed ZIP independently of ZIP readback.
    combined=hashlib.sha256()
    for part in parts:
        check_stop()
        with (out/part['member']).open('rb') as f:
            while chunk:=f.read(4*1024**2): combined.update(chunk)
    archive_sha=digest(archive)
    if combined.hexdigest()!=archive_sha: raise ValueError('transport-part readback mismatch')
    receipt={'status':'sealed','archive':str(archive),'bytes':archive.stat().st_size,
        'sha256':archive_sha,'manifest_member':'MANIFEST.json','manifest_sha256':manifest_sha,
        'members':len(manifest),'member_readback':'all sizes/SHA256 and ZIP CRC pass',
        'part_readback':'concatenated SHA256 matches archive','parts':parts,
        'originals_retained':True,'cloud_upload':'pending','remote_bytes_downloaded':False}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--member',action='append',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--owner-status',type=Path,required=True)
    a=p.parse_args()
    if not os.environ.get('NPLAYER_STOP_FILE'): raise RuntimeError('VRM supervisor required')
    seal(a.root,a.member,a.out,a.owner_status)
