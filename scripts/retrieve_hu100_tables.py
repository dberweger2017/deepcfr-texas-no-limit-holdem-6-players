"""Restore only the three pinned K50 release tables from PR163's indexed archive."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile

from scripts.evaluate_hu100_direct import put
from scripts.restore_hu20_equity_scoring import copy_member
from src.policies.hu100_bundle import CARD_TABLES, regular, sha

ARCHIVE_BYTES = 1991949530
ARCHIVE_SHA256 = '03d4828f030082f213c8ce82e29fa0fa8fd97288bd8035ad0056de85bc0816ca'
MANIFEST_SHA256 = 'a5d64a7882da34840dad53905e4a03284f822487732ce691ecc82cdf1585bdef'
PREFIX = 'hu20-equity-buckets-20261005/'


def retrieve(archive, manifest, destination):
    archive, manifest, destination = map(Path, (archive, manifest, destination))
    regular(archive); regular(manifest)
    if destination.exists() or destination.is_symlink():
        raise ValueError('Preserve existing table destination')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(destination.parent).free < sum(p['bytes'] for p in CARD_TABLES.values())+16*1024**3:
        raise ValueError('Insufficient disk for tables and reserve')
    if archive.stat().st_size!=ARCHIVE_BYTES or sha(archive)!=ARCHIVE_SHA256 or sha(manifest)!=MANIFEST_SHA256:
        raise ValueError('Table archive or member manifest differs')
    members={r['path']:r for r in json.loads(manifest.read_text())['members']}
    if any({k:members.get(name,{}).get(k) for k in ('bytes','sha256')}!=pin for name,pin in CARD_TABLES.items()):
        raise ValueError('Table member pins differ')
    destination.mkdir(); found={}
    with tarfile.open(archive,'r|gz') as bundle:
        for member in bundle:
            name=member.name.removeprefix(PREFIX)
            if member.name==PREFIX+name and name in CARD_TABLES:
                if name in found or not member.isfile() or member.size!=CARD_TABLES[name]['bytes']:
                    raise ValueError('Duplicate or nonregular table member')
                with bundle.extractfile(member) as stream:
                    found[name]=copy_member(stream,destination/name,CARD_TABLES[name])
    if set(found)!=set(CARD_TABLES):
        raise ValueError('Missing pinned table')
    receipt=dict(status='verified',archive=str(archive.resolve()),archive_bytes=ARCHIVE_BYTES,
        archive_sha256=ARCHIVE_SHA256,manifest=str(manifest.resolve()),manifest_sha256=MANIFEST_SHA256,
        drive_archive_id='1Gl1JYWN0F7KFJbwoKtcA0zB_rUsWWZ6T',
        members={PREFIX+name:value for name,value in found.items()},native_drive_archive_hydrated_and_verified=True)
    put(destination/'retrieval.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(retrieve(args.archive,args.manifest,args.out),sort_keys=True))


if __name__=='__main__':main()
