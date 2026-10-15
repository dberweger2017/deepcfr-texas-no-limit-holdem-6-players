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
import os

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
K50_MENU_V2_INFERENCE = {
    'adapter': 'hu100-public-menu-translation-equity-k50-menu-v2',
    'translation': {'version': 'hu100-public-menu-translation-equity-k50-menu-v2',
                    'max_states': 512, 'max_events': 128}}
CARD_TABLES = {
    'flop-k50.bin': {'bytes': 12867944, 'sha256': '084e243ded9fe93ad03cd59601f22f16c3210592f51e1e3646a60983b572c111'},
    'turn-k50.bin': {'bytes': 139600524, 'sha256': 'c0bc6a8aa0535553118109d18a32d3b4dc6880e937c263cdc87472b1ae9f168f'},
    'river-k50.bin': {'bytes': 1231562564, 'sha256': '70c1cb5ecf4cd292e4c89dc2200f7c8381100be3b0288ceeb0a12a3e83977629'},
}
PART_BYTES = 1024**3  # Every downloadable part stays strictly below GitHub's 2 GiB limit.
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
    'v0.5.2': {
        'verifier': 'verify_hu100_bundle.py',
        'name': 'v0.5.2 · HU100 · O4B · seed 2026100601 · translation 512/128',
        'model': {
            'file': 'O4B-HU100-opponent-sampled-average-seed-2026100601.jsonl.gz', 'bytes': 2121415396,
            'sha256': '0ed7f9512996bb3e9535a7258a5d0e4bc8c00f22734d511c642c0e5b95cbd3c0',
            'checkpoint_sha256': '65fd61d6999d7a06f36fc1c718aea20e7c23856aaa491df7d517e312c1c33368',
            'seed': 2026100601, 'iteration': 3716073, 'entries': 70466972,
            'actual_nodes': 4000000174, 'training_budget_nodes': 4000000000, **_SHARED,
        },
        'provenance': {
            'pr': 226, 'scientific_source_commit': 'd3120c2c195b5eca136b7f0d9969e32ba5159fff',
            'drive_folder_id': '1YP4RR_c6HlPjGyGSUoZQaVI33kgF-TfW',
            'drive_archive_id': '1qfgsSf9CodCtYE6rjH57zI1ovbT0t23E',
            'archive_name': 'hu100-4b-seed-ladder-M4-20261010-recovery-20261011.zip',
            'archive_bytes': 22545560259,
            'archive_sha256': 'ea58899645688f3fb391c12b501dc0941754878e96eec165b835e1847b4105b1',
            'manifest_member': 'ARCHIVE-MANIFEST.json',
            'manifest_sha256': '5dad61abfe120225592fb3b29f11bb812febc6ca438635ccb1d333b9ea4dd664',
            'member': 'research/training/2026100601/4000000000/average.gz',
        },
        'evidence': {'direct_gain_over_v051_established': True,
                     'independent_seed_replication_established': True,
                     'external_benchmark_established': False,
                     'translated_pot_profitability_established': False},
    },
}

RELEASES['v0.5.3'] = {
    'verifier': 'verify_hu100_bundle.py',
    'name': 'v0.5.3 · HU100 · frozen K50 menu v2 O5B · seed 2026100601 · translation 512/128',
    'model': {
        **_SHARED,
        'file': 'O5B-HU100-K50-menu-v2-frozen-average-seed-2026100601.jsonl.gz',
        'bytes': 3196291551,
        'sha256': 'c3a8edc4cc78098f07ac3e4ecbbb3dacae298a266b9fcd5fc788d0e2b6fcfc8e',
        'checkpoint_sha256': '401fe31263c763c5b805b0cff2f6af332d2dc7a3e9b7d8b99f1120b0216f58d6',
        'seed': 2026100601, 'iteration': 1794658, 'entries': 97290578,
        'actual_nodes': 5000006023, 'training_budget_nodes': 5000000000,
        'schema': 'hu100-native-reopening-ordered-history-equity-k50-menu-v2',
    },
    'inference': K50_MENU_V2_INFERENCE,
    'card_tables': CARD_TABLES,
    'part_bytes': PART_BYTES,
    'provenance': {
        'pr': 272, 'scientific_source_commit': 'fd066d26e61cab1ecc2d7f9e131ce2beac296998',
        'drive_folder_id': '1YeDWiUGZjIgiwNsA2BAoc6SLWVb7hsZq',
        'drive_archive_id': '1p7LmF9xRGXroTLDqNiY4eRwPuMC2IRiX',
        'archive_name': 'hu100-v2-frozen-5b-average-M4-20261014.zip',
        'archive_bytes': 3196298276,
        'archive_sha256': 'ea29e019781d390d6fb284c7e4675b73f197587ad28103656247f89304d7df2d',
        'manifest_member': 'ARCHIVE-MANIFEST.json',
        'manifest_sha256': 'bc31a32e532d276fb95e501ff1e014c8fc419219b48c5a1c71c5005b923e6e67',
        'member': 'averages/5000000000.average.gz',
    },
    'evidence': {'candidate_declared_without_seed_selection': True,
                 'three_fixed_seed_evaluation_pr': 296,
                 'translation_safeguard_pr': 298,
                 'translation_safeguard_passed': True,
                 'external_benchmark_established': False},
}


def inference(release):
    return RELEASES[release].get('inference', INFERENCE)


def part_layout(release):
    pins = RELEASES[release]
    if 'part_bytes' not in pins:
        return []
    model = pins['model']
    return [{'file': model['file'] + f'.part-{i:03d}',
             'bytes': min(pins['part_bytes'], model['bytes'] - offset)}
            for i, offset in enumerate(range(0, model['bytes'], pins['part_bytes']), 1)]


def assets(release):
    pins = RELEASES[release]
    payload = {p['file'] for p in part_layout(release)} if 'part_bytes' in pins else {pins['model']['file']}
    return payload | set(pins.get('card_tables', {})) | {
        'MODEL_CARD.md', 'RELEASE_NOTES.md', 'INSTALL.md', 'release-manifest.json', pins['verifier']}


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
    verify_header(header, release)


def verify_header(header, release):
    MODEL = RELEASES[release]['model']
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
    if 'card_tables' in RELEASES[release]:
        tables = {name.removesuffix('-k50.bin'): pin['sha256']
                  for name, pin in RELEASES[release]['card_tables'].items()}
        if (checkpoint['identity'].get('card_tables') != tables
                or checkpoint.get('native_state', {}).get('completed_nodes') != MODEL['actual_nodes']):
            raise ValueError('K50 model table identity or node count differs')


def manifest(release, source, approved=False, *, model_parts=None):
    """A release's identity. Approval binds the publication to the source it was built from."""
    if release not in RELEASES:
        raise ValueError('Unknown HU100 release')
    if not full_sha(source):
        raise ValueError('Full lowercase package-source Git SHA required')
    if type(approved) is not bool:
        raise ValueError('Publication approval must be an explicit boolean')
    pins = RELEASES[release]
    data = {'release': release,
            'status': 'owner-approved-publication' if approved else 'unpublished-owner-review',
            'package_source_commit': source,
            'approved_release_source_commit': source if approved else None,
            'owner_publication_approval': approved, 'release_tag': release if approved else None,
            'publication_gate': PUBLICATION_GATE,
            'model': pins['model'], 'inference': inference(release), 'provenance': pins['provenance'],
            'table': TABLE, 'evidence': pins['evidence'], 'dependency': DEPENDENCY}
    if 'part_bytes' in pins:
        layout = part_layout(release)
        if (not isinstance(model_parts, list) or len(model_parts) != len(layout)
                or any(not isinstance(p, dict) or set(p) != {'file', 'bytes', 'sha256'}
                       or {k: p[k] for k in ('file', 'bytes')} != expected
                       or not isinstance(p['sha256'], str) or len(p['sha256']) != 64
                       or any(c not in '0123456789abcdef' for c in p['sha256'])
                       for p, expected in zip(model_parts, layout))):
            raise ValueError('Model parts differ from the fixed ordered layout')
        data.update(model_parts=model_parts, card_tables=pins['card_tables'])
    elif model_parts is not None:
        raise ValueError('This release has no model parts')
    return data


def verify_parts(directory, release, parts, destination=None):
    """Validate parts and the fixed whole hash, optionally reassemble to a fresh file.

    Part digests detect download damage; the source-code-pinned whole digest is
    the trust anchor. A changed part manifest cannot substitute a different model.
    """
    directory = Path(directory)
    output = Path(destination) if destination is not None else None
    temporary = None
    stream = None
    whole = hashlib.sha256()
    try:
        if output is not None:
            if output.exists() or output.is_symlink():
                raise ValueError('Preserve existing assembled model')
            output.parent.mkdir(parents=True, exist_ok=True)
            candidate = output.with_name(output.name + f'.assembling-{os.getpid()}')
            stream = candidate.open('xb')
            temporary = candidate
        for part in parts:
            path = directory / part['file']
            regular(path)
            if path.stat().st_size != part['bytes']:
                raise ValueError('Model part size differs: ' + part['file'])
            digest = hashlib.sha256()
            with path.open('rb') as src:
                while chunk := src.read(8 * 1024**2):
                    digest.update(chunk); whole.update(chunk)
                    if stream is not None:
                        stream.write(chunk)
            if digest.hexdigest() != part['sha256']:
                raise ValueError('Model part SHA256 differs: ' + part['file'])
        if whole.hexdigest() != RELEASES[release]['model']['sha256']:
            raise ValueError('Reassembled model differs from fixed export')
        # The complete gzip header is contained in the first fixed-size part.
        with gzip.open(directory / parts[0]['file'], 'rt') as src:
            verify_header(json.loads(src.readline()), release)
        if stream is not None:
            stream.close(); stream = None
            # An exclusive hardlink publishes without overwriting a concurrent file.
            os.link(temporary, output)
        return whole.hexdigest()
    finally:
        if stream is not None:
            stream.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def bundle_release(directory):
    data = json.loads((Path(directory) / 'release-manifest.json').read_text())
    if data.get('release') not in RELEASES:
        raise ValueError('Unknown HU100 release')
    return data['release']


def verify_bundle(directory, expected_source=None, *, require_publication=False, assemble=None):
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
    if (type(approved) is not bool or data != manifest(release, source, approved, model_parts=data.get('model_parts'))
            or expected_source is not None and source != expected_source):
        raise ValueError('Release identity, source or publication status differs')
    if require_publication and (expected_source is None or not approved):
        raise ValueError('Publication verification needs the expected tagged source and owner approval')
    pins = RELEASES[release]
    for name, pin in pins.get('card_tables', {}).items():
        if (directory / name).stat().st_size != pin['bytes'] or sha(directory / name) != pin['sha256']:
            raise ValueError('Pinned card table differs: ' + name)
    if 'part_bytes' in pins:
        if assemble is not None and Path(assemble).resolve().is_relative_to(directory.resolve()):
            raise ValueError('Assemble outside the immutable asset directory')
        verify_parts(directory, release, data['model_parts'], assemble)
    else:
        if assemble is not None:
            raise ValueError('This release needs no reassembly')
        verify_model(directory / pins['model']['file'], release)
    return {'status': data['status'], 'release': release,
            'package_source_commit': source, 'model_sha256': RELEASES[release]['model']['sha256'],
            'inference': inference(release), 'hashes': hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--expect-source')
    parser.add_argument('--require-publication', action='store_true')
    parser.add_argument('--assemble', type=Path, help='Reassemble split model to this fresh file outside the asset directory')
    args = parser.parse_args()
    try:
        print(json.dumps(verify_bundle(args.directory, args.expect_source,
                                       require_publication=args.require_publication, assemble=args.assemble), sort_keys=True))
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
