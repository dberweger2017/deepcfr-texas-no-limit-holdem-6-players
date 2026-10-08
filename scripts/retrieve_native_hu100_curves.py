"""Restore only the five indexed averages into fresh nonsynced working storage."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

from src.arena.artifacts import write_json
from src.policies.files import file_hash

ARCHIVE = Path.home() / 'Local/Research-Cloud/PR-197-native-recovery-HU100/native-recovery-hu100-followup-20261008.zip'
INDEX = Path('docs/reports/native-recovery-hu100-artifacts/followup-model-index.json')


def retrieve(config, out):
    index = json.loads(INDEX.read_text())
    settings = json.loads(config.read_text())
    if ARCHIVE.stat().st_size != index['archive_bytes'] or file_hash(ARCHIVE) != index['archive_sha256']:
        raise ValueError('Indexed input archive differs')
    restored = []
    with ZipFile(ARCHIVE) as z:
        manifest = z.read(index['manifest_member'])
        if sha256(manifest).hexdigest() != index['manifest_sha256']:
            raise ValueError('Input member manifest differs')
        for spec, indexed in zip(settings['models'], index['models'], strict=True):
            asset = indexed['assets']['average']
            if (spec['sha256'] != asset['sha256'] or spec['bytes'] != asset['bytes']
                    or spec['archive_member'] != asset['path'] or spec['actual_nodes'] != indexed['actual_nodes']):
                raise ValueError('Model configuration differs from index')
            path = Path(spec['path']); path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(asset['path']) as source, path.open('xb') as destination:
                shutil.copyfileobj(source, destination)
            if path.stat().st_size != spec['bytes'] or file_hash(path) != spec['sha256']:
                raise ValueError('Restored model hash differs')
            with gzip.open(path, 'rt') as source:
                header = json.loads(source.readline())
            if (header['checkpoint_header']['native_state']['completed_nodes'] != spec['actual_nodes']
                    or header['checkpoint_header']['iteration'] != spec['iteration']
                    or header['source_checkpoint_sha256'] != spec['source_checkpoint_sha256']):
                raise ValueError('Actual completed nodes/checkpoint identity differs')
            restored.append({'model': spec, 'checkpoint_header': header['checkpoint_header'],
                             'retrieval_command': 'python -m scripts.retrieve_native_hu100_curves --config '
                             + str(config) + ' --out ' + str(out)})
    write_json(out, {'status': 'verified', 'archive': str(ARCHIVE), 'archive_id': index['archive_id'],
                    'archive_sha256': index['archive_sha256'], 'manifest_sha256': index['manifest_sha256'],
                    'local_synced_archive_used': True, 'remote_bytes_downloaded': False, 'models': restored})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    retrieve(a.config, a.out)
