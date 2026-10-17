"""Standalone Python 3.11 stdlib verifier for pinned HU200 release assets.

Checksums detect transfer damage; source-pinned whole-file hashes prevent a
rehashed part manifest from substituting a different model. Publication binds
all assets to one reviewed source commit and an explicit owner approval.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

PART_BYTES = 1024**3
SCHEMA = 'hu200-native-reopening-ordered-history-equity-k50-v1'
GAME = 'hu200-native-reopening-200bb-52card-no-ante-rake-v1'
FORMAT = 'holdem-hu200-stored-cfr-average-research-v1'
EXTRACTION = 'normalize-lifetime-iteration-opponent-sampled-accumulator-v1'
INFERENCE = {'adapter': 'hu200-public-menu-translation-equity-k50-v1',
             'translation': {'version': 'hu200-public-menu-translation-equity-k50-v1',
                             'max_states': 512, 'max_events': 128}}
TABLE = {'players': 2, 'stack_chips': 20000, 'small_blind': 50,
         'big_blind': 100, 'chip_unit': '0.01', 'reset_each_hand': True}
CARD_TABLES = {
    'flop-k50.bin': {'bytes': 12867944, 'sha256': '084e243ded9fe93ad03cd59601f22f16c3210592f51e1e3646a60983b572c111'},
    'turn-k50.bin': {'bytes': 139600524, 'sha256': 'c0bc6a8aa0535553118109d18a32d3b4dc6880e937c263cdc87472b1ae9f168f'},
    'river-k50.bin': {'bytes': 1231562564, 'sha256': '70c1cb5ecf4cd292e4c89dc2200f7c8381100be3b0288ceeb0a12a3e83977629'},
}
MODELS = {
    'blend': {
        'file': 'HU200-v055-three-seed-blend.average.gz', 'bytes': 6282968150,
        'sha256': 'ce3ecc2cb8ebffa1fb3e0a7f11c79d0900b74008b55d85093e1915ff4dde6485',
        'checkpoint_sha256': '59a6dd353fb4ef06d77a86783c7251521d8c9f8b92a785bb7e11a9ae724d4123',
        'seed': 2026101529701, 'iteration': 1, 'entries': 156208331,
        'synthetic_identity': 'hu200-v055-sequence-blend-2026101529701',
    },
    '2026101302': {
        'file': 'HU200-frozen-20B-seed-2026101302.average.gz', 'bytes': 3164563975,
        'sha256': '9128f71c602f18a9cb7219b18cd43ff4e3f2b7ed6a30f9babe7ed033840ac7b6',
        'checkpoint_sha256': 'c5929bda17457911347e47cd77a83dd8b34618a5542731bed96ece1fb4288b6f',
        'seed': 2026101302, 'iteration': 16670844, 'entries': 85373907,
    },
    '2026101421': {
        'file': 'HU200-frozen-20B-seed-2026101421.average.gz', 'bytes': 3162671080,
        'sha256': 'c08d86a9615022df7015d34e5dc3de2705bbb8346ed6976bd72284e71d91f444',
        'checkpoint_sha256': 'c70e8e01b3b27c7fa8b8e0561bdbe5b77b321125de646f04614c82ecacacc676',
        'seed': 2026101421, 'iteration': 16745107, 'entries': 85373916,
    },
    '2026101422': {
        'file': 'HU200-frozen-20B-seed-2026101422.average.gz', 'bytes': 3162064602,
        'sha256': 'b191c5ed8b943d138e6076fb4a1b0117c98964d92dd2d6253e157f696f691f6f',
        'checkpoint_sha256': 'e265c384686f003521e49158c5162e57175d3b53e405140cb494d30a8ec9660c',
        'seed': 2026101422, 'iteration': 16827728, 'entries': 85373928,
    },
}
SEEDS = ('2026101302', '2026101421', '2026101422')
for seed, size, nodes in zip(SEEDS, (4960844586, 4957752665, 4957316628),
                             (20000002032, 20000000322, 20000000096)):
    MODELS[seed + '-checkpoint'] = {
        **MODELS[seed], 'kind': 'checkpoint',
        'file': f'HU200-frozen-20B-seed-{seed}.checkpoint.gz',
        'bytes': size, 'sha256': MODELS[seed]['checkpoint_sha256'], 'actual_nodes': nodes,
    }
RECIPE = {'file': 'blend-config.json', 'bytes': 8498,
          'sha256': '83c72410433ef57904fbf97829a84ecb3dae7a74132e48fb87dad3f0b909a47f'}
RELEASES = {
    'v0.5.5': {'models': ['blend'], 'prerelease': False, 'latest': True,
               'name': 'v0.5.5 · HU200 · three-seed blend · translation 512/128',
               'evidence': {'construction_pr': 306, 'qualification_pr': 318,
                            'protocol_version': 2, 'slumbot_hands': 30000,
                            'raw_bb_per_100': -19.67, 'ci95': [-38.57, -0.76],
                            'qualified': True, 'beats_slumbot': False}},
    'hu200-frozen-20b-seeds': {'models': [*SEEDS, *(seed + '-checkpoint' for seed in SEEDS)],
                            'prerelease': True, 'latest': False,
                            'name': 'HU200 frozen 20B research seeds',
                            'evidence': {'confirmation_pr': 304, 'protocol_version': 1,
                                         'recipe_qualified': False, 'aivat_analysis_pr': 316}},
}
PROVENANCE = {'input_proofs_source_commit': 'f0165a6f7ba7575b5c5849f4f1b887ec41ce47ad',
              'input_proofs': 'docs/reports/hu200-v055-blend-run-artifacts/',
              'blend_scientific_source': '13ccb498482c228776128e70ad9e733fb0f5621a',
              'blend_drive_archive_id': '1Jn57qQY7gliu3F32_B1-WM_xAIO9__hV',
              'blend_archive_sha256': '35075f721cb60de6f2ab1d56c9172c0d6bdfe08cfe94bfd34bb619c54ac54582',
              'blend_member': 'research/hu200-v055-blend/merge/average.gz'}
VERIFIER = 'verify_hu200_bundle.py'
DOCS = {'INSTALL.md', 'MODEL_CARD.md', 'RELEASE_NOTES.md'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Bundle assets must be regular files')


def digest(value, length=64):
    return isinstance(value, str) and len(value) == length and all(c in '0123456789abcdef' for c in value)


def part_layout(model):
    pin = MODELS[model]
    return [{'file': pin['file'] + f'.part-{i:03d}', 'bytes': min(PART_BYTES, pin['bytes'] - offset)}
            for i, offset in enumerate(range(0, pin['bytes'], PART_BYTES), 1)]


def assets(release):
    return (DOCS | {VERIFIER, 'release-manifest.json'} | set(CARD_TABLES)
            | ({RECIPE['file']} if release == 'hu200-frozen-20b-seeds' else set())
            | {p['file'] for model in RELEASES[release]['models'] for p in part_layout(model)})


def verify_header(header, model):
    pin = MODELS[model]
    raw = pin.get('kind') == 'checkpoint'
    checkpoint = header if raw else header['checkpoint_header']
    tables = {name.removesuffix('-k50.bin'): p['sha256'] for name, p in CARD_TABLES.items()}
    if raw:
        if (header.get('format') != 'holdem-hu200-native-reopening-blueprint-v1'
                or header.get('kind') != 'training'
                or header.get('native_state', {}).get('completed_nodes') != pin['actual_nodes']):
            raise ValueError('Raw frozen checkpoint identity differs')
    elif (header.get('format') != FORMAT or header.get('kind') != 'diagnostic-inference'
          or header.get('extraction') != EXTRACTION
          or header.get('source_checkpoint_sha256') != pin['checkpoint_sha256']
          or header.get('zero_mass_rule') != 'uniform in retained menu; reported separately from missing keys'):
        raise ValueError('HU200 average extraction differs')
    if (checkpoint['config']['seed'] != pin['seed'] or checkpoint['config']['game'] != GAME
            or checkpoint['config']['raise_cap'] is not None or checkpoint['abstraction'] != SCHEMA
            or checkpoint['iteration'] != pin['iteration'] or checkpoint['average_rule'] != 'opponent-sampled'
            or checkpoint['identity']['players'] != 2 or checkpoint['identity']['card_tables'] != tables
            or checkpoint['table']['stacks'] != [20000, 20000]
            or checkpoint['table']['small_blind'] != 50 or checkpoint['table']['big_blind'] != 100
            or checkpoint['table']['chip_unit'] != '0.01'):
        raise ValueError('HU200 model lineage, game, tables or extraction differs')
    if model == 'blend':
        blend = checkpoint.get('blend_provenance', {})
        if (blend.get('model_identity') != pin['synthetic_identity']
                or blend.get('kind') != 'synthetic-sequence-form-average'
                or [(s['seed'], s['average_sha256'], s['sha256'], s['iterations']) for s in blend['inputs']]
                != [(MODELS[m]['seed'], MODELS[m]['sha256'], MODELS[m]['checkpoint_sha256'], MODELS[m]['iteration'])
                    for m in SEEDS]):
            raise ValueError('HU200 blend ingredient identity differs')


def verify_model(path, model='blend'):
    path = Path(path); regular(path)
    pin = MODELS[model]
    if path.stat().st_size != pin['bytes'] or sha(path) != pin['sha256']:
        raise ValueError('Model bytes/SHA256 differ from the pinned HU200 export')
    with gzip.open(path, 'rt') as stream:
        verify_header(json.loads(stream.readline()), model)


def manifest(release, source, approved=False, *, model_parts):
    if not digest(source, 40) or type(approved) is not bool:
        raise ValueError('Full source commit and explicit boolean approval required')
    pins = RELEASES[release]
    if not isinstance(model_parts, dict) or set(model_parts) != set(pins['models']):
        raise ValueError('Model set differs from the release')
    for model, parts in model_parts.items():
        layout = part_layout(model)
        if (not isinstance(parts, list) or len(parts) != len(layout)
                or any(not isinstance(p, dict) or set(p) != {'file', 'bytes', 'sha256'}
                       or {k: p[k] for k in ('file', 'bytes')} != expected or not digest(p['sha256'])
                       for p, expected in zip(parts, layout))):
            raise ValueError('Model parts differ from the fixed ordered layout')
    data = {'release': release, 'package_source_commit': source,
            'owner_publication_approval': approved,
            'approved_release_source_commit': source if approved else None,
            'release_tag': release if approved else None,
            'publication_gate': 'Owner-approved packaging merge; green exact-source CI; verified draft and public downloads',
            'prerelease': pins['prerelease'], 'latest': pins['latest'],
            'models': {m: MODELS[m] for m in pins['models']}, 'model_parts': model_parts,
            'schema': SCHEMA, 'game': GAME, 'format': FORMAT, 'resumable': False,
            'inference': INFERENCE, 'table': TABLE, 'card_tables': CARD_TABLES,
            'evidence': pins['evidence'], 'provenance': PROVENANCE,
            'dependency': {'python': '3.11', 'pokers_git_revision': '5db20e3d5d6862b32a7402035c1340b622d3b005'}}
    if release == 'hu200-frozen-20b-seeds':
        data['rebuild'] = {'recipe': RECIPE, 'tool': 'native/hu20-trainer/examples/hu200_blend.rs',
                           'input_lookup': '--input-dir', 'average_sha256': MODELS['blend']['sha256'],
                           'checkpoint_sha256': MODELS['blend']['checkpoint_sha256']}
    return data


def verify_parts(directory, model, parts, destination=None):
    output = Path(destination) if destination is not None else None
    temporary = None; stream = None; whole = hashlib.sha256()
    try:
        if output is not None:
            if output.exists() or output.is_symlink():
                raise ValueError('Preserve existing assembled model')
            output.parent.mkdir(parents=True, exist_ok=True)
            candidate = output.with_name(output.name + f'.assembling-{os.getpid()}')
            stream = candidate.open('xb'); temporary = candidate
        for part in parts:
            path = Path(directory) / part['file']; regular(path)
            if path.stat().st_size != part['bytes']:
                raise ValueError('Model part size differs: ' + part['file'])
            hashed = hashlib.sha256()
            with path.open('rb') as src:
                while chunk := src.read(8 * 1024**2):
                    hashed.update(chunk); whole.update(chunk)
                    if stream is not None: stream.write(chunk)
            if hashed.hexdigest() != part['sha256']:
                raise ValueError('Model part SHA256 differs: ' + part['file'])
        if whole.hexdigest() != MODELS[model]['sha256']:
            raise ValueError('Reassembled model differs from the pinned HU200 export')
        with gzip.open(Path(directory) / parts[0]['file'], 'rt') as src:
            verify_header(json.loads(src.readline()), model)
        if stream is not None:
            stream.close(); stream = None
            os.link(temporary, output)
    finally:
        if stream is not None: stream.close()
        if temporary is not None: temporary.unlink(missing_ok=True)


def verify_bundle(directory, expected_source=None, *, require_publication=False, assemble=None):
    directory = Path(directory)
    if directory.is_symlink(): raise ValueError('Bundle must be a regular directory')
    regular(directory / 'release-manifest.json')
    data = json.loads((directory / 'release-manifest.json').read_text())
    release = data.get('release')
    if release not in RELEASES: raise ValueError('Unknown HU200 release')
    source, approved = data.get('package_source_commit'), data.get('owner_publication_approval')
    if (data != manifest(release, source, approved, model_parts=data.get('model_parts'))
            or expected_source is not None and source != expected_source):
        raise ValueError('Release identity, source or publication status differs')
    if require_publication and (expected_source is None or not approved):
        raise ValueError('Publication verification requires the expected source and approval')
    names = assets(release)
    if set(p.name for p in directory.iterdir()) != names | {'SHA256SUMS'}:
        raise ValueError('Bundle members differ from the declared assets')
    for name in names | {'SHA256SUMS'}: regular(directory / name)
    hashes = {}
    for row in (directory / 'SHA256SUMS').read_text().splitlines():
        fields = row.split('  ')
        if len(fields) != 2 or fields[1] not in names or fields[1] in hashes or not digest(fields[0]):
            raise ValueError('Invalid, duplicate or undeclared checksum member')
        hashed, name = fields
        if sha(directory / name) != hashed: raise ValueError('Checksum differs: ' + name)
        hashes[name] = hashed
    if set(hashes) != names: raise ValueError('Missing release checksum')
    for name, pin in CARD_TABLES.items():
        if (directory / name).stat().st_size != pin['bytes'] or hashes[name] != pin['sha256']:
            raise ValueError('Pinned card table differs: ' + name)
    if release == 'hu200-frozen-20b-seeds':
        recipe = directory / RECIPE['file']
        if recipe.stat().st_size != RECIPE['bytes'] or hashes[RECIPE['file']] != RECIPE['sha256']:
            raise ValueError('Pinned blend recipe differs')
    if assemble is not None:
        if Path(assemble).resolve().is_relative_to(directory.resolve()):
            raise ValueError('Assemble outside the immutable bundle')
        Path(assemble).mkdir(parents=True, exist_ok=True)
    for model, parts in data['model_parts'].items():
        verify_parts(directory, model, parts, Path(assemble) / MODELS[model]['file'] if assemble is not None else None)
    return {'status': 'verified', 'release': release, 'package_source_commit': source,
            'owner_publication_approval': approved, 'hashes': hashes,
            'models': {m: MODELS[m]['sha256'] for m in RELEASES[release]['models']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--expect-source')
    parser.add_argument('--require-publication', action='store_true')
    parser.add_argument('--assemble', type=Path, help='Fresh models in this directory outside the immutable bundle')
    args = parser.parse_args()
    try:
        print(json.dumps(verify_bundle(args.directory, args.expect_source,
                                      require_publication=args.require_publication, assemble=args.assemble), sort_keys=True))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
