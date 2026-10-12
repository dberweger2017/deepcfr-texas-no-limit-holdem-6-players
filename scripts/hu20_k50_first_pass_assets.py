"""Pinned restoration and exact native-state gates for the K50 first pass."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import zipfile
from itertools import zip_longest

from src.policies.files import file_hash
from src.diagnostics.cfr_average import exact_numbers

TABLES = {
    'flop-k50.bin': (12867944, '084e243ded9fe93ad03cd59601f22f16c3210592f51e1e3646a60983b572c111'),
    'turn-k50.bin': (139600524, 'c0bc6a8aa0535553118109d18a32d3b4dc6880e937c263cdc87472b1ae9f168f'),
    'river-k50.bin': (1231562564, '70c1cb5ecf4cd292e4c89dc2200f7c8381100be3b0288ceeb0a12a3e83977629'),
}
REFERENCE = 'a0ef695d6807aa4554ba01b32d27f64c39ccdc72d03be5a21c99cf67e52ad716'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def check(path, sha, size=None):
    actual = file_hash(path)
    if actual != sha or (size is not None and path.stat().st_size != size):
        raise ValueError('Hash/size mismatch: ' + str(path))
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': actual}


def restore_reference(base):
    archive = Path.home() / 'Local/Research-Cloud/PR-185-HU20-O-10B/O-2026100602-1000000000-milestone.zip'
    whole = check(archive, '7c51abc6963d5a415ffa3a8a130245ee63a81d1357db11e92e685280afb9613b', 215176734)
    destination = base / 'inputs/v1-seed602-1b.json.gz'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        manifest = source.read('ARCHIVE-MANIFEST.json')
        if hashlib.sha256(manifest).hexdigest() != 'f4048a0514380046c3189d662fec81259fad8b6b26b8819b631bf20fa648f16f':
            raise ValueError('Reference manifest differs')
        with source.open('O-2026100602-1000000000.json.gz') as incoming, destination.open('xb') as target:
            shutil.copyfileobj(incoming, target, 1024**2)
        # Retain the original metadata/command evidence without extracting unrelated payloads.
        metadata = base / 'inputs/v1-seed602-1b-archive-metadata'
        metadata.mkdir(exist_ok=False)
        for item in source.infolist():
            if item.is_dir() or item.filename.endswith('.json.gz'):
                continue
            relative = Path(item.filename)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe reference metadata path')
            if item.file_size > 1024**2:
                continue
            target = metadata / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read(item))
    member = check(destination, REFERENCE, 215171058)
    with gzip.open(destination, 'rt') as stream:
        header = json.loads(stream.readline())
    if header['config']['seed'] != 2026100602 or header['average_rule'] != 'opponent-sampled':
        raise ValueError('Reference recipe differs')
    write(base / 'reference-restoration.json', {'status': 'verified', 'archive': whole, 'member': member,
          'archive_member': 'O-2026100602-1000000000.json.gz', 'drive_folder': '1uTSTXQKn19JaGWTTx_lbrANfxO01xPtL',
          'header': header, 'retrieval': 'Extract the named member from the whole-hash-verified milestone ZIP; verify the manifest and member SHA256.'})


def restore_tables(base):
    archive = Path.home() / 'Local/Research-Cloud/PR-163-equity-buckets/hu20-equity-buckets-20261005.tar.gz'
    whole = check(archive, '03d4828f030082f213c8ce82e29fa0fa8fd97288bd8035ad0056de85bc0816ca', 1991949530)
    destination = base / 'tables'
    destination.mkdir(exist_ok=False)
    extracted = []
    with tarfile.open(archive, 'r:gz') as source:
        for item in source:
            name = Path(item.name).name
            if name not in TABLES or item.name != 'hu20-equity-buckets-20261005/' + name:
                continue
            if not item.isfile() or item.size != TABLES[name][0]:
                raise ValueError('Unexpected pinned table member')
            with source.extractfile(item) as incoming, (destination / name).open('xb') as target:
                shutil.copyfileobj(incoming, target, 1024**2)
            extracted.append(check(destination / name, TABLES[name][1], TABLES[name][0]))
    if len(extracted) != 3:
        raise ValueError('Missing pinned table')
    write(base / 'tables-restoration.json', {'status': 'verified', 'archive': whole, 'members': extracted,
         'drive_id': '1Gl1JYWN0F7KFJbwoKtcA0zB_rUsWWZ6T', 'drive_folder': '1gwungsUq-b9uey8InGy0yOD2GbaMRhp-',
         'retrieval': 'tar -xzf TABLE_ARCHIVE --strip-components=1 -C NEW_TABLE_DIR hu20-equity-buckets-20261005/{flop,turn,river}-k50.bin; verify all exact member hashes before use'})


def recipe_gate(base):
    reference = base / 'inputs/v1-seed602-1b.json.gz'
    check(reference, REFERENCE)
    with gzip.open(reference, 'rt') as source:
        header = json.loads(source.readline())
    fresh = base / 'gates/v1-seed602-fresh-1b.json.gz'
    check(fresh, REFERENCE, reference.stat().st_size)
    with gzip.open(fresh, 'rt') as source:
        if json.loads(source.readline()) != header:
            raise ValueError('Recipe header differs')
    write(base / 'recipe-gate.json', {'status': 'verified', 'checkpoint_byte_identity': True,
          'seed': 2026100602, 'requested_nodes': 1000000000,
          'reference_sha256': REFERENCE, 'max_entries': header['config']['max_entries'],
          'binary_sha256': file_hash(Path('native/hu20-trainer/target/release/hu20-trainer')),
          'scope': 'Current-main production v1 reproduction. K50 treatment changes only card buckets.'})


def recovery_gate(base):
    fresh = base / 'gates/k50-fresh-4000000.json.gz'
    resumed = base / 'gates/k50-resumed-4m.json.gz'
    rows = 0
    with gzip.open(fresh, 'rt') as left, gzip.open(resumed, 'rt') as right:
        a, b = json.loads(left.readline()), json.loads(right.readline())
        state = b.pop('native_state', None)
        if state is None or not exact_numbers(a, b):
            raise ValueError('Recovery header differs beyond native_state')
        expected = {name.split('-')[0]: sha for name, (_, sha) in TABLES.items()}
        # Exact pinned identity format is checked rather than guessing its field names.
        identity_text = json.dumps(a['identity'])
        if not all(sha in identity_text for sha in expected.values()):
            raise ValueError('Checkpoint table identities missing')
        for original, recovered in zip_longest(left, right):
            if original is None or recovered is None or not exact_numbers(json.loads(original), json.loads(recovered)):
                raise ValueError('Recovery row differs')
            rows += 1
    write(base / 'recovery-gate.json', {'status': 'verified', 'all_rows_identical': rows,
          'fresh_sha256': file_hash(fresh), 'resumed_sha256': file_hash(resumed),
          'identity': a['identity'], 'iteration': a['iteration'], 'resumed_native_state': state,
          'allowed_header_difference': 'Only documented native_state added by resume; no scientific row differences.'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('reference', 'tables', 'recipe-gate', 'recovery-gate'))
    p.add_argument('--base', type=Path, required=True)
    a = p.parse_args()
    {'reference': restore_reference, 'tables': restore_tables, 'recipe-gate': recipe_gate,
     'recovery-gate': recovery_gate}[a.stage](a.base)


if __name__ == '__main__':
    main()
