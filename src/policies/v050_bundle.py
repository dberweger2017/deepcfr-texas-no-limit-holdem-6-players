"""Standalone Python 3.11 verifier for the v0.5.0 HU100 bundle and its publication binding."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

ASSET_NAME = 'O1B-HU100-opponent-sampled-average-seed-2026100601.jsonl.gz'
RELEASE = 'v0.5.0'
MODEL = {
    'file': ASSET_NAME, 'bytes': 1173264021,
    'sha256': '47d493c2ca0a750ffec8ba5490bd8fdec0a582e0cf2fe3e4309868f6ae620fa9',
    'checkpoint_sha256': 'cca0b54a609f47c60b29e9fe5a920475a91ec641df2ff0e3ea48fdac615147ec',
    'seed': 2026100601, 'iteration': 885307, 'entries': 41010014,
    'actual_nodes': 1000002065, 'training_budget_nodes': 1000000000,
    'game': 'hu100-native-reopening-100bb-52card-no-ante-rake-v1',
    'schema': 'hu100-native-reopening-ordered-history-card-v1',
    'format': 'holdem-hu100-stored-cfr-average-research-v1',
    'extraction': 'normalize-lifetime-iteration-opponent-sampled-accumulator-v1',
    'players': 2, 'raise_cap': None, 'resumable': False,
}
INFERENCE = {'adapter': 'hu100-public-menu-translation-v1',
             'translation': {'version': 'hu100-public-menu-translation-v1',
                             'max_states': 512, 'max_events': 128}}
PROVENANCE = {
    'pr': 207, 'scientific_source_commit': 'bd0e7a417064f736091dc2b667954b50becb4b69',
    'drive_folder_id': '1qhlOHmphBGSFfiM82S7T4B_KhdabyRUS',
    'drive_archive_id': '1iowJoQBQqB3tLRnU6GD0JcniF0qDIvgj',
    'archive_name': 'hu100-1b-campaign-M4-retry-20261009.zip',
    'archive_bytes': 20517119304,
    'archive_sha256': 'ba3e82d8fa79be32d445c54eb240069c4a717cb75af9713f3eaf7ac86364fddf',
    'manifest_member': 'ARCHIVE-MANIFEST.json',
    'manifest_sha256': '609d4899a363b384aa58b0daaef2ff85f2bcae7b135c2a4e3604813fc4182d58',
    'member': 'research/training/1000000000/average.gz',
}
ASSETS = {ASSET_NAME, 'MODEL_CARD.md', 'RELEASE_NOTES.md', 'INSTALL.md',
          'release-manifest.json', 'verify_v050_bundle.py'}
PUBLICATION_GATE = ('Owner chat authorization; reviewed green-check merge; exact tagged source; '
                    'verified draft download before stable Latest')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def full_sha(value):
    return isinstance(value, str) and len(value) == 40 and all(c in '0123456789abcdef' for c in value)


def regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Bundle assets must be regular files')


def verify_model(path):
    regular(path)
    if path.stat().st_size != MODEL['bytes'] or sha(path) != MODEL['sha256']:
        raise ValueError('Model bytes/SHA256 differ from the fixed PR207 export')
    with gzip.open(path, 'rt') as stream:
        header = json.loads(stream.readline())
    checkpoint = header['checkpoint_header']
    if (header['format'] != MODEL['format'] or header['kind'] != 'diagnostic-inference'
            or header['extraction'] != MODEL['extraction']
            or header['source_checkpoint_sha256'] != MODEL['checkpoint_sha256']
            or checkpoint['config']['seed'] != MODEL['seed']
            or checkpoint['config']['game'] != MODEL['game']
            or checkpoint['config']['raise_cap'] is not None
            or checkpoint['abstraction'] != MODEL['schema']
            or checkpoint['iteration'] != MODEL['iteration']
            or checkpoint['average_rule'] != 'opponent-sampled'
            or checkpoint['identity']['players'] != 2):
        raise ValueError('Model lineage, game or extraction differs')


def manifest(source, approved=False):
    """The bundle's identity. Approval binds the publication to the source it was built from."""
    if not full_sha(source):
        raise ValueError('Full lowercase package-source Git SHA required')
    if type(approved) is not bool:
        raise ValueError('Publication approval must be an explicit boolean')
    return {'release': RELEASE,
            'status': 'owner-approved-publication' if approved else 'unpublished-owner-review',
            'package_source_commit': source,
            'approved_release_source_commit': source if approved else None,
            'owner_publication_approval': approved, 'release_tag': RELEASE if approved else None,
            'publication_gate': PUBLICATION_GATE,
            'model': MODEL, 'inference': INFERENCE, 'provenance': PROVENANCE,
            'table': {'players': 2, 'stack_chips': 10000, 'small_blind': 50,
                      'big_blind': 100, 'chip_unit': '0.01', 'reset_each_hand': True},
            'evidence': {'pr215_overall_recipe_qualified': False,
                         'external_benchmark_established': False,
                         'translated_pot_profitability_established': False},
            'dependency': {'python': '3.11', 'pokers_git_revision':
                           '5db20e3d5d6862b32a7402035c1340b622d3b005'}}


def verify_bundle(directory, expected_source=None, *, require_publication=False):
    directory = Path(directory)
    if directory.is_symlink() or set(p.name for p in directory.iterdir()) != ASSETS | {'SHA256SUMS'}:
        raise ValueError('Bundle members differ from the declared assets')
    for name in ASSETS | {'SHA256SUMS'}:
        regular(directory / name)
    hashes = {}
    for row in (directory / 'SHA256SUMS').read_text().splitlines():
        fields = row.split('  ')
        if len(fields) != 2 or fields[1] not in ASSETS or fields[1] in hashes:
            raise ValueError('Invalid, duplicate or undeclared checksum member')
        digest, name = fields
        if sha(directory / name) != digest:
            raise ValueError('Checksum differs: ' + name)
        hashes[name] = digest
    if set(hashes) != ASSETS:
        raise ValueError('Missing release checksum')
    data = json.loads((directory / 'release-manifest.json').read_text())
    source, approved = data.get('package_source_commit'), data.get('owner_publication_approval')
    # Rebuilding the expected manifest pins every field, so no edit can waive a pin or the hold.
    if (type(approved) is not bool or data != manifest(source, approved)
            or expected_source is not None and source != expected_source):
        raise ValueError('Release identity, source or publication status differs')
    if require_publication and (expected_source is None or not approved):
        raise ValueError('Publication verification needs the expected tagged source and owner approval')
    verify_model(directory / ASSET_NAME)
    return {'status': data['status'], 'release': RELEASE,
            'package_source_commit': source, 'model_sha256': MODEL['sha256'],
            'inference': INFERENCE, 'hashes': hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--expect-source')
    parser.add_argument('--require-publication', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(verify_bundle(args.directory, args.expect_source,
                                       require_publication=args.require_publication), sort_keys=True))
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
