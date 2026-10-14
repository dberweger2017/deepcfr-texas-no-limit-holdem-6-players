"""Install a checksum-verified Node 22 binary in the owned test root, as in CI."""
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import urllib.request

root = Path(os.environ['QUALIFICATION_ROOT'])
base = 'https://nodejs.org/dist/latest-v22.x/'
checksums = urllib.request.urlopen(base + 'SHASUMS256.txt', timeout=30).read()
matches = [line.split() for line in checksums.decode().splitlines()
           if re.search(r' node-v22\.\d+\.\d+-linux-x64\.tar\.xz$', line)]
if len(matches) != 1:
    raise SystemExit('Expected exactly one official Node 22 Linux x64 archive')
digest, filename = matches[0]
archive = root / filename
with urllib.request.urlopen(base + filename, timeout=60) as source, archive.open('xb') as target:
    while block := source.read(1024 * 1024):
        target.write(block)
with archive.open('rb') as stream:
    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
if actual != digest:
    raise SystemExit('Node archive checksum mismatch')
with tarfile.open(archive) as source:
    source.extractall(root, filter='data')
(root / 'node').symlink_to(filename.removesuffix('.tar.xz'), target_is_directory=True)
(root / 'node-install.json').write_text(json.dumps({
    'url': base + filename, 'sha256_source': base + 'SHASUMS256.txt',
    'sha256': actual, 'bytes': archive.stat().st_size,
    'checksum_manifest_sha256': hashlib.sha256(checksums).hexdigest(),
}, indent=2) + '\n')
