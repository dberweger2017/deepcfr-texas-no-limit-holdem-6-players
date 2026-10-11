"""Restore hash-pinned #222 exports/references, #190 evaluator and #163 tables."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

from scripts.continue_restored_hu20_equity_bench import exclusive_json
from src.policies.files import file_hash

MANIFEST_SHA256 = 'feec3c2212e7edc7411f4eff1cf1e73745994d8525c749e6aa9b01544dd764a4'
TABLE_ARCHIVE_SHA256 = '03d4828f030082f213c8ce82e29fa0fa8fd97288bd8035ad0056de85bc0816ca'


def copy_member(stream, path, pin):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as target:
        shutil.copyfileobj(stream, target, 1024**2)
    if path.stat().st_size != pin['bytes'] or file_hash(path) != pin['sha256']:
        raise ValueError('Exactness mismatch: '+str(path))
    return dict(pin, restored_path=str(path.resolve()), mtime_ns=path.stat().st_mtime_ns)


def restore(base, cloud):
    index = json.loads(Path('docs/reports/hu20-equity-bench-artifacts/model-input-index.json').read_text())
    destination = base/'restored'
    destination.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'in-progress', 'archives': [], 'members': [], 'aliases': [],
               'science_started': False}
    archive = cloud/'PR-222-hu20-equity-bench/hu20-equity-bench-partial-20261010.zip'
    if file_hash(archive) != index['archive_sha256']:
        raise ValueError('PR222 whole archive differs')
    with zipfile.ZipFile(archive) as bundle:
        manifest = bundle.read('ARCHIVE-MANIFEST.json')
        if hashlib.sha256(manifest).hexdigest() != MANIFEST_SHA256:
            raise ValueError('PR222 manifest differs')
        records = json.loads(manifest)['members']
        pins = {p['path']: p for p in records}
        if len(pins) != len(records):
            raise ValueError('Duplicate archive member')
        selected = {p['path'] for p in index['required_model_and_input_members']}
        for pin in index['required_model_and_input_members']:
            if any(pins[pin['path']][k] != pin[k] for k in ('bytes', 'sha256')):
                raise ValueError('Published member locator differs')
        selected.update(p for p in pins if p.startswith('research/references/'))
        selected.update(('research/input-pins.json', 'research/matched-freeze.json',
                         'research/visits.json', 'research/reference-restoration.json',
                         'research/qualified-final-source.json'))
        selected.update(p['same_bytes_as_member'] for p in index['restoration_aliases'])
        for name in sorted(selected):
            with bundle.open(name) as stream:
                receipt['members'].append(copy_member(stream, destination/name, pins[name]))
        for alias in index['restoration_aliases']:
            target = destination/alias['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.hardlink_to(destination/alias['same_bytes_as_member'])
            if file_hash(target) != alias['sha256'] or target.stat().st_size != alias['bytes']:
                raise ValueError('Alias differs')
            receipt['aliases'].append(dict(alias, restored_path=str(target.resolve()),
                                           mtime_ns=target.stat().st_mtime_ns))
    receipt['archives'].append({'path': str(archive), 'drive_id': index['archive_id'],
                               'sha256': index['archive_sha256'], 'manifest_sha256': MANIFEST_SHA256})
    archive = cloud/'PR-190-HU20-bucket-validation/verified-inputs-prepared-pilots.zip'
    pin = index['lock_evaluator']
    if file_hash(archive) != pin['archive_sha256']:
        raise ValueError('PR190 whole archive differs')
    with zipfile.ZipFile(archive) as bundle:
        records = json.loads(bundle.read('ARCHIVE-MANIFEST.json'))['members']
        record, = [p for p in records if p['path'] == pin['member']]
        if any(record[k] != pin[k] for k in ('bytes', 'sha256')):
            raise ValueError('Evaluator pins differ')
        with bundle.open(pin['member']) as stream:
            receipt['members'].append(copy_member(stream, destination/'lock-evaluator', pin))
    (destination/'lock-evaluator').chmod(0o755)
    receipt['archives'].append({'path': str(archive), 'drive_id': pin['archive_id'],
                               'sha256': pin['archive_sha256']})
    archive = cloud/'PR-163-equity-buckets/hu20-equity-buckets-20261005.tar.gz'
    if file_hash(archive) != TABLE_ARCHIVE_SHA256:
        raise ValueError('PR163 whole archive differs')
    hashes = index['external_inputs']['tables']['sha256']
    sizes = {'flop': 12867944, 'turn': 139600524, 'river': 1231562564}
    with tarfile.open(archive, 'r|gz') as bundle:
        found = set()
        for member in bundle:
            street = member.name.split('/')[-1].removesuffix('-k50.bin')
            if street in hashes and member.name == f'hu20-equity-buckets-20261005/{street}-k50.bin':
                pin = {'path': member.name, 'bytes': sizes[street], 'sha256': hashes[street]}
                with bundle.extractfile(member) as stream:
                    receipt['members'].append(copy_member(stream, destination/'tables'/f'{street}-k50.bin', pin))
                found.add(street)
        if found != set(hashes):
            raise ValueError('Missing table')
    receipt['archives'].append({'path': str(archive), 'drive_id': '1Gl1JYWN0F7KFJbwoKtcA0zB_rUsWWZ6T',
                               'sha256': TABLE_ARCHIVE_SHA256})
    receipt.update(status='verified', restoration_command=
                   'python -m scripts.restore_hu20_equity_scoring --base '+str(base))
    exclusive_json(base/'restoration.json', receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--cloud', type=Path, default=Path.home()/'Local/Research-Cloud')
    args = parser.parse_args()
    if (args.base/'campaign-failure.json').exists():
        raise ValueError('Stopped attempt; no retry')
    restore(args.base.resolve(), args.cloud.resolve())


if __name__ == '__main__':
    main()
