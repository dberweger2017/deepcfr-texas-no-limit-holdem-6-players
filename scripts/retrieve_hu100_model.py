"""Restore one HU100 release's pinned average from its archived research ZIP; never export."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

from src.policies.hu100_bundle import RELEASES, regular, sha, verify_model


def retrieve(release, archive, destination):
    pins = RELEASES[release]
    model, provenance = pins['model'], pins['provenance']
    archive, destination = Path(archive), Path(destination)
    regular(archive)
    if archive.stat().st_size != provenance['archive_bytes'] or sha(archive) != provenance['archive_sha256']:
        raise ValueError(f"PR{provenance['pr']} archive size/hash differs")
    if destination.exists() or destination.is_symlink():
        raise ValueError('Preserve existing destination')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < model['bytes'] + int(15.5 * 1024**3):
        raise ValueError('Insufficient disk with retained 15.5 GiB floor')
    with ZipFile(archive) as z:
        encoded = z.read(provenance['manifest_member'])
        if hashlib.sha256(encoded).hexdigest() != provenance['manifest_sha256']:
            raise ValueError('Archive member manifest hash differs')
        records = [r for r in json.loads(encoded)['members'] if r['path'] == provenance['member']]
        if len(records) != 1 or any(records[0][k] != model[k] for k in ('bytes', 'sha256')):
            raise ValueError('Archived model member identity differs')
        if len([i for i in z.infolist() if i.filename == provenance['member']]) != 1:
            raise ValueError('Duplicate or missing model member')
        destination.mkdir()
        target = destination / model['file']
        with z.open(provenance['member']) as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst, 8 * 1024**2)
    verify_model(target, release)
    receipt = {'status': 'verified-local-archive-and-model', 'release': release,
               'archive': str(archive.resolve()), 'target': str(target.resolve()),
               'model': model, 'provenance': provenance,
               'remote_cloud_bytes_downloaded_by_this_command': False}
    (destination / 'retrieval.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', choices=sorted(RELEASES), required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(retrieve(args.release, args.archive, args.out), sort_keys=True))


if __name__ == '__main__':
    main()
