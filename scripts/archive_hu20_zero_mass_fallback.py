"""Seal every research output and exact input into a member-hashed ZIP without removal."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import time
import zipfile

ROOT=Path.home()/'Local/hu20-zero-mass-fallback-20261006'
DEST=Path.home()/'Local/Research-Cloud/PR-179-HU20-zero-mass-fallback'
FLOOR=8*1024**3

def guard():
    free=shutil.disk_usage(ROOT).free
    if free<FLOOR:raise OSError(f'Free disk {free} below 8 GiB; stop without deletion')
    return free

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):
    with gzip.open(path,'rt') as f:return json.load(f)
def write(path,value):
    with gzip.open(path,'xt') as f:json.dump(value,f,indent=2,sort_keys=True)

def main():
    guard();DEST.mkdir(parents=True,exist_ok=True);archive=DEST/'hu20-zero-mass-fallback-complete-20261006.zip'
    paths={str(p.relative_to(ROOT)):p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    # Include the exact checkpoint and original policy inputs, rather than relying on another archive.
    for item in read(ROOT/'input-verification.json.gz'):
        p=Path(item['path'] if item['kind']=='checkpoint' else item['spec']['path'])
        expected=item['sha256'] if item['kind']=='checkpoint' else item['spec']['sha256']
        assert sha(p)==expected
        name='exact-inputs/'+('checkpoints/' if item['kind']=='checkpoint' else 'policies/')+p.name
        paths[name]=p
    members=[{'path':name,'bytes':p.stat().st_size,'sha256':sha(p)} for name,p in sorted(paths.items())]
    manifest={'created_at':time.time(),'root':str(ROOT),'source_host':'M1 only','members':members,
        'logical_bytes':sum(m['bytes'] for m in members),'member_count':len(members),'originals_retained':True}
    encoded=json.dumps(manifest,sort_keys=True,indent=2).encode()
    write(ROOT/'archive-manifest.json.gz',manifest)
    # STORE avoids trying to compress gzip members a second time, keeping peak space predictable.
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for m in members:
            guard()
            with paths[m['path']].open('rb') as src,z.open('research/'+m['path'],'w',force_zip64=True) as out:
                while chunk:=src.read(8*1024**2):guard();out.write(chunk)
        z.writestr('ARCHIVE-MANIFEST.json.gz',gzip.compress(encoded,mtime=0))
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())=={'ARCHIVE-MANIFEST.json.gz',*('research/'+m['path'] for m in members)}
        assert gzip.decompress(z.read('ARCHIVE-MANIFEST.json.gz'))==encoded
        for m in members:
            guard();h=hashlib.sha256();size=0
            with z.open('research/'+m['path']) as f:
                while chunk:=f.read(8*1024**2):h.update(chunk);size+=len(chunk)
            assert size==m['bytes'] and h.hexdigest()==m['sha256'],m['path']
    receipt={'status':'verified','archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),
        'members_verified':len(members),'manifest_sha256':hashlib.sha256(encoded).hexdigest(),'at':time.time(),
        'free_disk_bytes':guard(),'originals_retained':True,'deleted_or_evicted':False,
        'cloud_byte_readback':'not claimed; member readback is local ZIP verification'}
    write(ROOT/'archive-receipt.json.gz',receipt);write(DEST/'archive-receipt.json.gz',receipt)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    try:main()
    except Exception as e:
        write(ROOT/('archive-failure-'+str(time.time_ns())+'.json.gz'),{'error':str(e),'free_disk_bytes':shutil.disk_usage(ROOT).free,'at':time.time()})
        raise
