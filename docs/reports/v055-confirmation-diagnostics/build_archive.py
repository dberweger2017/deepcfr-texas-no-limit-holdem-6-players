"""Build a member-hashed, read-back ZIP from the final uncompressed evidence directory.

Run on the research cache: ROOT --name ARCHIVE.zip. Writes the manifest beside the payload
and the archive; prints the compact receipt to stdout so it can be retained outside ROOT.
Never uploads, removes evidence or overwrites an existing archive.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert Path(args.name).name == args.name and args.name.endswith('.zip')
    archive_path = root / args.name
    assert not archive_path.exists()
    members = []
    for path in sorted(root.rglob('*')):
        assert not path.is_symlink(), path
        if not path.is_file() or path in (root / '.drive-receipt.json', root / 'ARCHIVE-MANIFEST.json', root / 'KEEP'):
            continue
        members.append({'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
                        'sha256': digest(path)})
    manifest = {'archive': args.name, 'content': 'PR330 confirmation card-only AIVAT and public-action diagnosis; '
                'inputs, per-hand corrections, summaries, public tables, reproduction sources, logs and failures',
                'members': members}
    manifest_path = root / 'ARCHIVE-MANIFEST.json'
    with manifest_path.open('x') as stream:
        stream.write(json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(manifest_path, manifest_path.name)
        for member in members:
            archive.write(root / member['path'], member['path'])
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == len(members) + 1
        assert archive.read(manifest_path.name) == manifest_path.read_bytes()
        for member in members:
            data = archive.read(member['path'])
            assert len(data) == member['bytes']
            assert hashlib.sha256(data).hexdigest() == member['sha256']
            assert digest(root / member['path']) == member['sha256']
    print(json.dumps({'archive': args.name, 'bytes': archive_path.stat().st_size,
                      'sha256': digest(archive_path), 'manifest': manifest_path.name,
                      'manifest_sha256': digest(manifest_path), 'payload_members': len(members),
                      'all_members_read_back': True, 'all_uncompressed_payload_sha256_match': True}, indent=2))


if __name__ == '__main__':
    main()
