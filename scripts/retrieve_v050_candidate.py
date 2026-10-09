"""Restore only the pinned #207 average from its existing archive; never export."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

from src.policies.v050_bundle import ASSET_NAME, MODEL, PROVENANCE, regular, sha, verify_model


RETAINED_SOURCE = Path('/Users/dberweger/Local/hu100-1b-growth-20261008/results/hu100-1b/training/1000000000/average.gz')

def retrieve(archive, destination):
    archive, destination = Path(archive), Path(destination)
    regular(archive)
    if archive.stat().st_size != PROVENANCE['archive_bytes'] or sha(archive) != PROVENANCE['archive_sha256']:
        raise ValueError('PR207 archive size/hash differs')
    if destination.exists() or destination.is_symlink():
        raise ValueError('Preserve existing destination')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < MODEL['bytes'] + int(15.5 * 1024**3):
        raise ValueError('Insufficient disk with retained 15.5 GiB floor')
    with ZipFile(archive) as z:
        encoded = z.read(PROVENANCE['manifest_member'])
        if hashlib.sha256(encoded).hexdigest() != PROVENANCE['manifest_sha256']:
            raise ValueError('PR207 member manifest hash differs')
        records = [r for r in json.loads(encoded)['members'] if r['path'] == PROVENANCE['member']]
        if len(records) != 1 or any(records[0][k] != MODEL[k] for k in ('bytes', 'sha256')):
            raise ValueError('PR207 model member identity differs')
        if len([i for i in z.infolist() if i.filename == PROVENANCE['member']]) != 1:
            raise ValueError('Duplicate/missing PR207 model member')
        destination.mkdir()
        target = destination / ASSET_NAME
        with z.open(PROVENANCE['member']) as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst, 8 * 1024**2)
    verify_model(target)
    receipt = {'status': 'verified-local-archive-and-model', 'archive': str(archive.resolve()),
               'target': str(target.resolve()), 'model': MODEL, 'provenance': PROVENANCE,
               'remote_cloud_bytes_downloaded_by_this_command': False}
    (destination / 'retrieval.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt



def retrieve_retained(source, destination):
    """Reuse fully allocated indexed original; do not claim a fresh archive audit."""
    source, destination = Path(source), Path(destination)
    expected = RETAINED_SOURCE
    if source != expected:
        raise ValueError('Retained source must be the indexed PR207 original')
    verify_model(source)
    if destination.exists() or destination.is_symlink():
        raise ValueError('Preserve existing destination')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < MODEL['bytes'] + int(15.5 * 1024**3):
        raise ValueError('Insufficient disk with retained floor')
    destination.mkdir()
    target = destination / ASSET_NAME
    with source.open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst, 8 * 1024**2)
    verify_model(target)
    receipt = {'status': 'verified-indexed-retained-model', 'source': str(source),
               'target': str(target.resolve()), 'model': MODEL, 'provenance': PROVENANCE,
               'archive_audit': 'existing accepted PR207 receipt; no fresh whole-ZIP verification',
               'remote_cloud_bytes_downloaded_by_this_command': False,
               'source_bytes': source.stat().st_size, 'source_mtime_ns': source.stat().st_mtime_ns}
    (destination / 'retrieval.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--archive', type=Path)
    selection.add_argument('--retained', type=Path, help='Indexed fully allocated M4 PR207 original')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = retrieve(args.archive, args.out) if args.archive else retrieve_retained(args.retained, args.out)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
