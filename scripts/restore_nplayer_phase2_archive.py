"""Verify indexed transport parts and restore named members into a fresh directory."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile


def restore(receipt,parts,out,members):
    manifest_receipt=json.loads(receipt.read_text())
    out.mkdir(parents=True,exist_ok=False)
    # The assembled ZIP is disposable scratch; downloaded parts remain untouched.
    with tempfile.TemporaryFile(dir=out) as combined:
        whole=hashlib.sha256();total=0
        for part in manifest_receipt['parts']:
            name=part.get('cloud_name',part['member'])
            if Path(name).name!=name:raise ValueError('transport part must be a basename')
            path=parts/name
            digest=hashlib.sha256();count=0
            with path.open('rb') as f:
                while chunk:=f.read(4*1024**2):
                    digest.update(chunk);whole.update(chunk);combined.write(chunk);count+=len(chunk)
            if count!=part['bytes'] or digest.hexdigest()!=part['sha256']:
                raise ValueError(f'transport part mismatch: {name}')
            total+=count
        if total!=manifest_receipt['bytes'] or whole.hexdigest()!=manifest_receipt['sha256']:
            raise ValueError('whole archive mismatch')
        combined.seek(0)
        with zipfile.ZipFile(combined) as z:
            payload=z.read(manifest_receipt['manifest_member'])
            if hashlib.sha256(payload).hexdigest()!=manifest_receipt['manifest_sha256']:
                raise ValueError('manifest mismatch')
            indexed=json.loads(payload)['members']
            rows={row['member']:row for row in indexed}
            if len(rows)!=len(indexed) or len(set(z.namelist()))!=len(z.namelist()):
                raise ValueError('duplicate archive member')
            selected=members or list(rows)
            for member in selected:
                key=PurePosixPath(member)
                if member not in rows or key.is_absolute() or '..' in key.parts:
                    raise ValueError('unindexed or unsafe member')
                destination=out.joinpath(*key.parts)
                destination.parent.mkdir(parents=True,exist_ok=True)
                digest=hashlib.sha256();count=0
                with z.open(member) as source,destination.open('xb') as target:
                    while chunk:=source.read(4*1024**2):
                        digest.update(chunk);target.write(chunk);count+=len(chunk)
                if count!=rows[member]['bytes'] or digest.hexdigest()!=rows[member]['sha256']:
                    raise ValueError('restored member mismatch')
    result={'status':'passed','archive_sha256':manifest_receipt['sha256'],
            'restored_members':selected,'input_parts_retained':True}
    with (out/'RESTORATION.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',required=True,type=Path)
    p.add_argument('--parts',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--member',action='append',default=[])
    a=p.parse_args();print(json.dumps(restore(a.receipt,a.parts,a.out,a.member),indent=2))
