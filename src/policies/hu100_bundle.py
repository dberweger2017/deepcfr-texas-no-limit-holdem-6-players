"""Standalone Python 3.11 verifier for HU100 release bundles and their publication binding.

Each release pins its exact model, translation settings and provenance. A bundle names its
release in `release-manifest.json`; every field must equal the pinned manifest rebuilt from
that release, its package source commit and its approval. v0.5.0 bundles shipped this
verifier as `verify_v050_bundle.py`; later releases ship it as `verify_hu100_bundle.py`.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

_SHARED = {
    'game': 'hu100-native-reopening-100bb-52card-no-ante-rake-v1',
    'schema': 'hu100-native-reopening-ordered-history-card-v1',
    'format': 'holdem-hu100-stored-cfr-average-research-v1',
    'extraction': 'normalize-lifetime-iteration-opponent-sampled-accumulator-v1',
    'players': 2, 'raise_cap': None, 'resumable': False,
}
INFERENCE = {'adapter': 'hu100-public-menu-translation-v1',
             'translation': {'version': 'hu100-public-menu-translation-v1',
                             'max_states': 512, 'max_events': 128}}
TABLE = {'players': 2, 'stack_chips': 10000, 'small_blind': 50,
         'big_blind': 100, 'chip_unit': '0.01', 'reset_each_hand': True}
DEPENDENCY = {'python': '3.11', 'pokers_git_revision': '5db20e3d5d6862b32a7402035c1340b622d3b005'}
PUBLICATION_GATE = ('Owner chat authorization; reviewed green-check merge; exact tagged source; '
                    'verified draft download before stable Latest')
RELEASES = {
    'v0.5.0': {
        'verifier': 'verify_v050_bundle.py',
        'name': 'v0.5.0 · HU100 · O1B · seed 2026100601 · translation 512/128',
        'model': {
            'file': 'O1B-HU100-opponent-sampled-average-seed-2026100601.jsonl.gz', 'bytes': 1173264021,
            'sha256': '47d493c2ca0a750ffec8ba5490bd8fdec0a582e0cf2fe3e4309868f6ae620fa9',
            'checkpoint_sha256': 'cca0b54a609f47c60b29e9fe5a920475a91ec641df2ff0e3ea48fdac615147ec',
            'seed': 2026100601, 'iteration': 885307, 'entries': 41010014,
            'actual_nodes': 1000002065, 'training_budget_nodes': 1000000000, **_SHARED,
        },
        'provenance': {
            'pr': 207, 'scientific_source_commit': 'bd0e7a417064f736091dc2b667954b50becb4b69',
            'drive_folder_id': '1qhlOHmphBGSFfiM82S7T4B_KhdabyRUS',
            'drive_archive_id': '1iowJoQBQqB3tLRnU6GD0JcniF0qDIvgj',
            'archive_name': 'hu100-1b-campaign-M4-retry-20261009.zip',
            'archive_bytes': 20517119304,
            'archive_sha256': 'ba3e82d8fa79be32d445c54eb240069c4a717cb75af9713f3eaf7ac86364fddf',
            'manifest_member': 'ARCHIVE-MANIFEST.json',
            'manifest_sha256': '609d4899a363b384aa58b0daaef2ff85f2bcae7b135c2a4e3604813fc4182d58',
            'member': 'research/training/1000000000/average.gz',
        },
        'evidence': {'pr215_overall_recipe_qualified': False,
                     'external_benchmark_established': False,
                     'translated_pot_profitability_established': False},
    },
    'v0.5.1': {
        'verifier': 'verify_hu100_bundle.py',
        'name': 'v0.5.1 · HU100 · O2B · seed 2026100601 · translation 512/128',
        'model': {
            'file': 'O2B-HU100-opponent-sampled-average-seed-2026100601.jsonl.gz', 'bytes': 1603422956,
            'sha256': 'e6f79ccac39a651352424e05382681f4d1994bb570b025f692556d170f87e9ae',
            'checkpoint_sha256': 'e84039c7a934a966a2c01f237748a23951124a8b665edc2585ef4809e5681676',
            'seed': 2026100601, 'iteration': 1812907, 'entries': 54626283,
            'actual_nodes': 2000000460, 'training_budget_nodes': 2000000000, **_SHARED,
        },
        'provenance': {
            'pr': 223, 'scientific_source_commit': '38c83f2eab09b5247a8c16344ce6d0a3d00c43af',
            'drive_folder_id': '1K9iLLPtHZrduVPQf_btzv3OyV1Oe19jp',
            'drive_archive_id': '15DBpUxBH2MYbyr1bgKLPyAqSvuwtkFGd',
            'archive_name': 'hu100-terminal-2b-ladder-20261010.zip',
            'archive_bytes': 7503720854,
            'archive_sha256': 'e7836f6400e39b589723191c4b6f24ceafdea45cfb8e077843294212fe5e48fb',
            'manifest_member': 'ARCHIVE-MANIFEST.json',
            'manifest_sha256': '6071b334e45c89922dc73e95e874668b7c3d7449b39624ebc3d97f1c3576fcd0',
            'member': 'research/training/2000000000/average.gz',
        },
        'evidence': {'direct_gain_over_v050_established': True,
                     'external_benchmark_established': False,
                     'translated_pot_profitability_established': False},
    },
}


def assets(release):
    pins = RELEASES[release]
    return {pins['model']['file'], 'MODEL_CARD.md', 'RELEASE_NOTES.md', 'INSTALL.md',
            'release-manifest.json', pins['verifier']}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def full_sha(value):
    return isinstance(value, str) and len(value) == 40 and all(c in '0123456789abcdef' for c in value)


def regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Bundle assets must be regular files')


def verify_model(path, release):
    regular(path)
    MODEL = RELEASES[release]['model']
    if path.stat().st_size != MODEL['bytes'] or sha(path) != MODEL['sha256']:
        raise ValueError(f"Model bytes/SHA256 differ from {release}'s fixed export")
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


def manifest(release, source, approved=False):
    """A release's identity. Approval binds the publication to the source it was built from."""
    if release not in RELEASES:
        raise ValueError('Unknown HU100 release')
    if not full_sha(source):
        raise ValueError('Full lowercase package-source Git SHA required')
    if type(approved) is not bool:
        raise ValueError('Publication approval must be an explicit boolean')
    pins = RELEASES[release]
    return {'release': release,
            'status': 'owner-approved-publication' if approved else 'unpublished-owner-review',
            'package_source_commit': source,
            'approved_release_source_commit': source if approved else None,
            'owner_publication_approval': approved, 'release_tag': release if approved else None,
            'publication_gate': PUBLICATION_GATE,
            'model': pins['model'], 'inference': INFERENCE, 'provenance': pins['provenance'],
            'table': TABLE, 'evidence': pins['evidence'], 'dependency': DEPENDENCY}


def bundle_release(directory):
    data = json.loads((Path(directory) / 'release-manifest.json').read_text())
    if data.get('release') not in RELEASES:
        raise ValueError('Unknown HU100 release')
    return data['release']


def verify_bundle(directory, expected_source=None, *, require_publication=False):
    directory = Path(directory)
    if directory.is_symlink() or not (directory / 'release-manifest.json').is_file():
        raise ValueError('A bundle directory with release-manifest.json is required')
    release = bundle_release(directory)
    names = assets(release)
    if set(p.name for p in directory.iterdir()) != names | {'SHA256SUMS'}:
        raise ValueError('Bundle members differ from the declared assets')
    for name in names | {'SHA256SUMS'}:
        regular(directory / name)
    hashes = {}
    for row in (directory / 'SHA256SUMS').read_text().splitlines():
        fields = row.split('  ')
        if len(fields) != 2 or fields[1] not in names or fields[1] in hashes:
            raise ValueError('Invalid, duplicate or undeclared checksum member')
        digest, name = fields
        if sha(directory / name) != digest:
            raise ValueError('Checksum differs: ' + name)
        hashes[name] = digest
    if set(hashes) != names:
        raise ValueError('Missing release checksum')
    data = json.loads((directory / 'release-manifest.json').read_text())
    source, approved = data.get('package_source_commit'), data.get('owner_publication_approval')
    # Rebuilding the expected manifest pins every field, so no edit can waive a pin or the hold.
    if (type(approved) is not bool or data != manifest(release, source, approved)
            or expected_source is not None and source != expected_source):
        raise ValueError('Release identity, source or publication status differs')
    if require_publication and (expected_source is None or not approved):
        raise ValueError('Publication verification needs the expected tagged source and owner approval')
    verify_model(directory / RELEASES[release]['model']['file'], release)
    return {'status': data['status'], 'release': release,
            'package_source_commit': source, 'model_sha256': RELEASES[release]['model']['sha256'],
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
