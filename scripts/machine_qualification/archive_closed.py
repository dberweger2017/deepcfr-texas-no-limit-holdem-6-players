"""Seal this PR's closed on-host evidence, retaining every original.

Run after CPU/GPU diagnostics and before extra RAM. Retrievable table inputs,
tool caches, shared Git and the reproducible fio scratch extent are excluded.
The active archive guard is preserved later in the separate closeout bundle.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile


def digest(stream):
    sha, size = hashlib.sha256(), 0
    while block := stream.read(1024 * 1024):
        sha.update(block)
        size += len(block)
    return size, sha.hexdigest()


def main():
    root = Path(os.environ['QUALIFICATION_ROOT']).resolve()
    owner = json.loads((root / 'archive-owner-state.json').read_text())
    if not owner['url'].endswith('/pull/283') or owner['headRefName'] != 'feature/cygnus-x1-qualification':
        raise SystemExit('Archive ownership does not match this authorized PR/root')
    # A live/unfinished load is not an immutable archive input.
    for stream in root.glob('*/resources.jsonl'):
        if stream.parent.name.startswith(('archive-', 'memory-extra')):
            continue
        if not stream.with_name('result.json').is_file():
            raise SystemExit(f'Unclosed guard input: {stream.parent}')
    out = root / 'archive-payloads'
    out.mkdir(exist_ok=False)
    destination = out / 'cygnus-main-20261015.zip'
    selected, excluded = [], []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        top = relative.parts[0]
        if path.is_symlink():
            continue
        if not path.is_file():
            continue
        if (top in ('repo-main', 'repo-alternate', 'tables', 'node', 'archive-payloads')
                or top.startswith(('node-v22.', 'archive-', 'memory-extra'))
                or relative.as_posix() == 'disk-results/fio-scratch.bin'):
            # Avoid retaining an enormous cache inventory in the manifest.
            if top not in ('repo-main', 'repo-alternate') and not top.startswith('node-v22.'):
                excluded.append({'path': relative.as_posix(), 'bytes': path.stat().st_size})
            continue
        selected.append((path, 'research/' + relative.as_posix()))
    # Restore runtime source from the pinned Git revision; also preserve the
    # actual native source and maintained build entry point beside the binary.
    for path in sorted((root / 'repo-main/native').rglob('*')):
        relative = path.relative_to(root / 'repo-main')
        if 'target' not in relative.parts and path.is_file() and path.suffix in ('.rs', '.toml', '.lock'):
            selected.append((path, 'runtime-source/' + relative.as_posix()))
    selected.append((root / 'repo-main/scripts/build_native_trainer.py',
                     'runtime-source/scripts/build_native_trainer.py'))
    started = time.monotonic()
    pins = {}
    for path, member in selected:
        if member in pins:
            raise SystemExit('Duplicate archive member')
        stat = path.stat()
        with path.open('rb') as stream:
            size, sha = digest(stream)
        if size != stat.st_size or path.stat().st_mtime_ns != stat.st_mtime_ns:
            raise SystemExit(f'Archive source changed while hashing: {path}')
        pins[member] = {'bytes': size, 'sha256': sha,
                        'source_path': str(path), 'source_mtime_ns': stat.st_mtime_ns}
    expected = {'research/trainer-first': {
        'bytes': json.loads((root / 'build-first.json').read_text())['binary_bytes'],
        'sha256': json.loads((root / 'build-first.json').read_text())['binary_sha256']}}
    for name, pin in json.loads((root / 'gate-results/checkpoints.json').read_text()).items():
        expected[f'research/gate-results/{name}.gz'] = pin
    benchmark = json.loads((root / 'benchmark-results/receipt.json').read_text())
    expected['research/benchmark-results/500m.gz'] = {
        'bytes': benchmark['checkpoint_bytes'], 'sha256': benchmark['checkpoint_sha256']}
    for member, pin in expected.items():
        if any(pins[member][key] != pin[key] for key in ('bytes', 'sha256')):
            raise SystemExit(f'Previously verified artifact changed: {member}')
    manifest = {'version': 1, 'owning_pr': owner,
        'scope': 'Closed inventory, RAM partial, storage/network, tests/failures, native gates/500M and CPU/GPU diagnostics',
        'source_main': (root / 'source-main.txt').read_text().strip(),
        'originals_retained': True, 'excluded': excluded,
        'excluded_inputs_and_tooling': ['PR163 tables: indexed input-restoration receipt',
            'repo Git/venv/target caches; restore pinned source and build receipt',
            'Node distribution: node-install.json URL and verified hash',
            '8 GiB reproducible fio scratch extent: verification/results retained',
            'Active archive guard: separate final closeout bundle'],
        'members': pins}
    raw = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    with zipfile.ZipFile(destination, 'x', allowZip64=True) as archive:
        for path, member in selected:
            before = path.stat()
            if before.st_size != pins[member]['bytes'] or before.st_mtime_ns != pins[member]['source_mtime_ns']:
                raise SystemExit(f'Archive source changed before packing: {path}')
            archive.write(path, member,
                compress_type=zipfile.ZIP_STORED if path.suffix == '.gz' else zipfile.ZIP_DEFLATED,
                compresslevel=3)
        archive.writestr('ARCHIVE-MANIFEST.json', raw)
    with zipfile.ZipFile(destination) as archive:
        if set(archive.namelist()) != set(pins) | {'ARCHIVE-MANIFEST.json'}:
            raise SystemExit('Archive membership differs')
        if archive.read('ARCHIVE-MANIFEST.json') != raw:
            raise SystemExit('Archive manifest readback differs')
        for member, pin in pins.items():
            with archive.open(member) as stream:
                size, sha = digest(stream)
            if size != pin['bytes'] or sha != pin['sha256']:
                raise SystemExit(f'Archive readback differs: {member}')
    with destination.open('rb') as stream:
        size, sha = digest(stream)
    receipt = {'status': 'locally verified on cygnus; upload not accepted yet',
        'path': str(destination), 'bytes': size, 'sha256': sha,
        'manifest_path': 'ARCHIVE-MANIFEST.json',
        'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'members': len(pins),
        'all_member_sizes_and_hashes_readback': True,
        'seconds': time.monotonic() - started, 'originals_retained': True,
        'cloud_upload_accepted': False, 'remote_archive_bytes_downloaded': False}
    (out / 'archive-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
