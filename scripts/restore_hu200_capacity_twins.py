"""Verify indexed Drive archives on bb128 and restore the three fixed twins."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

from scripts.hu200_capacity_preflight import QUALIFIED
from scripts.run_hu200_capacity_training import pin, put
from src.blueprint.abstraction import HU200_EQUITY_SCHEMA
from src.blueprint.average import ZERO_MASS_RULES
from src.policies.files import file_hash


def restore(base):
    index = json.loads(Path('docs/reports/hu200-capacity-cygnus-artifacts/drive-inputs.json').read_text())
    restored, removals = [], []
    for item in index['archives']:
        archive = base / 'inputs/archives' / item['name']
        if archive.stat().st_size != item['bytes'] or file_hash(archive) != item['sha256']:
            raise ValueError('Archive size/hash mismatch; preserve and stop')
        archive_stat = archive.stat()
        folder = base / 'inputs' / ('seed-' + str(item['seed']))
        folder.mkdir(exist_ok=False)
        target = folder / 'final.average.gz'
        with zipfile.ZipFile(archive) as zipped:
            raw = zipped.read('ARCHIVE-MANIFEST.json')
            if hashlib.sha256(raw).hexdigest() != item['manifest_sha256']:
                raise ValueError('Manifest hash mismatch')
            manifest = json.loads(raw)
            member = manifest['members'][item['member']]
            if member['sha256'] != item['model_sha256']:
                raise ValueError('Indexed member hash mismatch')
            with zipped.open(item['member']) as source, target.open('xb') as destination:
                shutil.copyfileobj(source, destination, 1 << 20)
        model = pin(target)
        if model['bytes'] != member['bytes'] or model['sha256'] != item['model_sha256']:
            raise ValueError('Restored model mismatch')
        with gzip.open(target, 'rt') as source:
            metadata = json.loads(next(source))
        header = metadata['checkpoint_header']
        if (header['abstraction'] != HU200_EQUITY_SCHEMA or header['config']['seed'] != item['seed']
                or header['native_state']['completed_nodes'] < 20_000_000_000
                or metadata['zero_mass_rule'] != ZERO_MASS_RULES['uniform']):
            raise ValueError('Twin identity mismatch')
        spec = dict(name='twin-' + str(item['seed']), path=str(target), sha256=model['sha256'],
            bytes=model['bytes'], format=metadata['format'], abstraction=HU200_EQUITY_SCHEMA,
            card_buckets=str(QUALIFIED / 'tables'), entries=header['frozen_table']['entries'],
            iteration=header['iteration'], source_checkpoint_sha256=metadata['source_checkpoint_sha256'])
        receipt = dict(status='verified', seed=item['seed'], archive=item, member=member,
            model=model, spec=spec, whole_archive_sha256_verified=True,
            embedded_manifest_sha256_verified=True, selected_member_size_hash_crc_verified=True)
        put(folder / 'restoration.json', receipt)
        restored.append(receipt)
        removals.append(dict(path=str(archive), bytes=archive_stat.st_size, mtime_ns=archive_stat.st_mtime_ns,
            sha256=item['sha256'], drive_url='https://drive.google.com/file/d/' + item['id'] + '/view',
            archive_member=item['member'], retained_model=str(target)))
    # These are verified transport replicas of accepted #265/#278 archives.
    # Every original Drive archive and each restored model remains intact.
    for item in removals:
        path = Path(item['path'])
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (item['bytes'], item['mtime_ns']):
            raise ValueError('Transport replica changed; retain')
        if path.resolve().parent != (base / 'inputs/archives').resolve():
            raise ValueError('Transport path escaped owned archive directory')
    for item in removals:
        with (base / 'transport-copy-retirements.jsonl').open('a') as log:
            log.write(json.dumps(dict(action='remove-verified-transport-replica', file=item, at=time.time())) + '\n')
            log.flush()
            os.fsync(log.fileno())
        Path(item['path']).unlink()
    put(base / 'transport-copy-retirements.json', dict(status='verified-replicas-retired-after-restoration',
        removed=removals, original_drive_files_retained=True, owning_prs=[265, 278], at=time.time()))
    put(base / 'twins-restored.json', dict(status='verified', models=restored,
        source='read-only indexed synced Drive archives; SSH transport; all verification on bb128'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    restore(parser.parse_args().base)


if __name__ == '__main__':
    main()
