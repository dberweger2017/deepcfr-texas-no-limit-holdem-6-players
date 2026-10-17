"""Verify indexed archives/members and journal chains, and write the run specification."""
import hashlib
import json
from pathlib import Path
import zipfile

from src.arena.external.journal import verify_journal

ROOT = Path('results/inputs')
PINS = {
    'pr318': ('v055-blend-confirmation-20261017.zip', '9b453e6073200476f91d39c4162a65f3513c61f7525ee4fc26295f7f41d1dfe1', '4b2449e1528604e3d44e8e97ee9cde88f8cd03d5672a409269d9538c5e1cb4e7'),
    'pr304': ('hu200-v055-slumbot-confirmation-20261015.zip', '51f4587dc3fec12a65537be51e27572c3f1eec147952c8ed383f1c7689c57137', '4c3d53ca809f9b03715aa0a3adb0edbfca5b1da32d822efc589baf3f83f35e1a'),
    'pr316': ('slumbot-aivat-real-runs-perhand-20261017.zip', '51a525a617eb90c83d746e6bfd26347fa47f93774b3d379c172a4e091ca298ee', '61a5d592f98015cd4ce264bbc3b45b8d7fd93aabf3abcb5776f8af7edb821943'),
}

def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    receipts = {'archives': {}, 'journals': {}}
    for label, (name, digest, manifest_digest) in PINS.items():
        source = ROOT / name
        assert sha(source.read_bytes()) == digest, name
        with zipfile.ZipFile(source) as archive:
            manifest = archive.read('ARCHIVE-MANIFEST.json')
            assert sha(manifest) == manifest_digest
            members = json.loads(manifest)['members']
            for member in members:
                data = archive.read(member['path'])
                assert len(data) == member['bytes'] and sha(data) == member['sha256'], member['path']
                # Historical archive includes a Git HEAD; it is provenance, not a repository.
                if '.git' not in Path(member['path']).parts:
                    target = ROOT / label / member['path']
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('xb') as stream:
                        stream.write(data)
            (ROOT / label / 'ARCHIVE-MANIFEST.json').write_bytes(manifest)
        receipts['archives'][label] = {'name': name, 'bytes': source.stat().st_size, 'sha256': digest,
                                       'manifest_sha256': manifest_digest, 'verified_members': len(members)}
    for part in ('prefix', 'part2'):
        source = ROOT / f'exploratory-{part}.jsonl'
        target = ROOT / 'exploratory' / part / 'journal.jsonl'
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    prefix_receipt = json.loads((ROOT / 'exploratory/prefix/PREFIX.json').read_text())
    assert sha((ROOT / 'exploratory/prefix/journal.jsonl').read_bytes()) == prefix_receipt['prefix_sha256']
    receipts['exploratory_prefix'] = prefix_receipt
    runs = {}
    for letter in 'abc':
        runs[f'confirmation-{letter}'] = [str(p.parent) for p in sorted((ROOT / 'pr318').glob(f'account-{letter}-*/journal.jsonl'))]
    runs['confirmation-pooled'] = sum((runs[f'confirmation-{c}'] for c in 'abc'), [])
    for seed in ('2026101302', '2026101421', '2026101422'):
        parts = sorted((ROOT / 'pr304').glob(f'**/seed-{seed}*/journal.jsonl'))
        parts.sort(key=lambda p: ('part2' in str(p), str(p)))
        runs[seed] = [str(p.parent) for p in parts]
    runs['exploratory'] = [str(ROOT / 'exploratory' / part) for part in ('prefix', 'part2')]
    for paths in runs.values():
        assert paths
        for path in paths:
            journal = Path(path) / 'journal.jsonl'
            if str(journal) not in receipts['journals']:
                checked = verify_journal(journal)
                receipts['journals'][str(journal)] = {**checked, 'bytes': journal.stat().st_size,
                                                     'sha256': sha(journal.read_bytes())}
                print(path, checked, flush=True)
    source_files = ('src/arena/external/analysis.py', 'src/arena/external/aivat.py',
                    'src/arena/external/style.py', 'src/arena/external/journal.py',
                    'docs/reports/slumbot-aivat-real-runs/dump_aivat.py')
    receipts['analysis_source_sha256'] = {p: sha(Path(p).read_bytes()) for p in source_files}
    assert receipts['analysis_source_sha256'][source_files[-1]] == (
        '21d13c2659788c50376c7fc4e6c325de26084473133a860250cd54e90b8cc43d')
    Path('results/outputs/input-integrity.json').write_text(json.dumps(receipts, indent=2) + '\n')
    Path('results/outputs/runs.json').write_text(json.dumps(runs, indent=2) + '\n')


if __name__ == '__main__':
    main()
