"""Restore selected member identities from a pinned, deduplicated HU100 archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from zipfile import ZipFile

from src.policies.files import file_hash


def safe_member(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not name.startswith('research/'):
        raise ValueError('Unsafe archive member')
    return p


def restore(archive, destination, archive_sha256, manifest_sha256, members):
    if file_hash(archive) != archive_sha256:
        raise ValueError('Archive SHA256 differs')
    destination.mkdir(parents=True, exist_ok=False)
    restored = {}
    with ZipFile(archive) as z:
        raw = z.read('ARCHIVE-MANIFEST.json')
        if hashlib.sha256(raw).hexdigest() != manifest_sha256:
            raise ValueError('Manifest SHA256 differs')
        manifest = json.loads(raw)
        canonical = manifest['members']
        aliases = manifest['hardlink_aliases']
        wanted = set(members) if members else set(canonical) | set(aliases)
        for name in sorted(wanted):
            safe_member(name)
            target_name = aliases[name]['canonical_member'] if name in aliases else name
            safe_member(target_name)
            pin = canonical[target_name]
            if name in aliases and any(aliases[name][k] != pin[k] for k in ('bytes', 'sha256')):
                raise ValueError('Alias identity differs')
            target = destination / target_name
            if target_name not in restored:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(target_name) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst, 1024**2)
                if target.stat().st_size != pin['bytes'] or file_hash(target) != pin['sha256']:
                    raise ValueError('Restored member differs')
                restored[target_name] = {'bytes':pin['bytes'],'sha256':pin['sha256']}
            if name != target_name:
                alias = destination / name
                alias.parent.mkdir(parents=True, exist_ok=True)
                os.link(target, alias)
                if alias.stat().st_size != pin['bytes'] or file_hash(alias) != pin['sha256']:
                    raise ValueError('Restored alias differs')
                restored[name] = {'canonical_member':target_name,'bytes':pin['bytes'],'sha256':pin['sha256']}
    receipt = {'archive':str(archive),'archive_sha256':archive_sha256,'manifest_sha256':manifest_sha256,
        'destination':str(destination),'restored':restored,'automatic_relaunch':False}
    (destination/'RESTORATION-RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--archive-sha256',required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--member',action='append',default=[])
    a=p.parse_args()
    print(json.dumps(restore(a.archive,a.destination,a.archive_sha256,a.manifest_sha256,a.member)))

if __name__=='__main__':main()
