"""A verified binary cache of a loaded stored average, mapped read-only on later loads.

Parsing a large average's gzipped JSON rows takes minutes (8 for #250's 85M-entry HU200
average on an M1). The compact arrays it builds are written once into a folder named after the
model's SHA256, schema and this format; later loads hash every cache file against its manifest
and map the arrays read-only, in seconds. Mapped pages belong to the files, so the OS can share
and reclaim them. The canonical model stays the source of truth: AveragePolicy still checks its
hash and header before using a cache, and lookups return the same values as a parsed load.

Callers choose the cache folder explicitly; keep it out of synced folders.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from src.blueprint.compact_policy import CompactCounts, CompactEntries, CompactFlags, _Index

FORMAT = 'compact-average-cache-v1'
ARRAYS = ('keys', 'menu', 'offsets', 'probabilities', 'flags', 'counts')
DTYPES = {'keys': 'S16', 'menu': '<u2', 'offsets': '<u8', 'probabilities': '<f8', 'flags': '|u1', 'counts': '<u8'}


def folder(root: Path, source_sha256: str, schema: str) -> Path:
    return Path(root) / f'{source_sha256}-{schema}-{FORMAT}'


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def save(root: Path, source_sha256: str, schema: str, built) -> Path:
    """Write the arrays and manifest, then publish the folder with one rename."""
    entries, zero_mass, visits = built
    arrays = {'keys': entries.index.keys, 'menu': entries.menu, 'offsets': entries.offsets,
              'probabilities': entries.probabilities, 'flags': zero_mass.flags.view(np.uint8),
              'counts': visits.counts}
    target = folder(root, source_sha256, schema)
    temporary = target.with_name(f'{target.name}.tmp-{os.getpid()}')
    temporary.mkdir(parents=True)
    try:
        files = {}
        for name in ARRAYS:
            array = np.ascontiguousarray(arrays[name], dtype=DTYPES[name])
            path = temporary / f'{name}.npy'
            np.save(path, array, allow_pickle=False)
            files[name] = {'bytes': path.stat().st_size, 'sha256': _file_sha256(path), 'length': len(array)}
        manifest = {'format': FORMAT, 'source_sha256': source_sha256, 'schema': schema,
                    'entries': len(entries), 'menus': [list(m) for m in entries.menus], 'files': files}
        (temporary / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        try:
            temporary.rename(target)
        except OSError:
            if not target.is_dir():
                raise
            shutil.rmtree(temporary)        # Another process published the same cache first.
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def load(root: Path, source_sha256: str, schema: str):
    """(entries, zero-mass flags, visit counts) from a verified cache, or None when there is none."""
    target = folder(root, source_sha256, schema)
    if not target.is_dir():
        return None
    manifest = json.loads((target / 'manifest.json').read_text())
    if (manifest.get('format') != FORMAT or manifest.get('source_sha256') != source_sha256
            or manifest.get('schema') != schema or set(manifest.get('files', {})) != set(ARRAYS)):
        raise ValueError('Average cache manifest differs from the requested model')
    arrays = {}
    for name in ARRAYS:
        path = target / f'{name}.npy'
        expected = manifest['files'][name]
        if path.stat().st_size != expected['bytes'] or _file_sha256(path) != expected['sha256']:
            raise ValueError(f'Average cache file differs from its manifest: {name}')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if array.dtype != np.dtype(DTYPES[name]) or array.ndim != 1 or len(array) != expected['length']:
            raise ValueError(f'Average cache array has the wrong shape or type: {name}')
        arrays[name] = array
    count = manifest['entries']
    menus = tuple(tuple(m) for m in manifest['menus'])
    if (any(len(arrays[n]) != count for n in ('keys', 'menu', 'offsets', 'flags', 'counts'))
            or (count and int(arrays['menu'].max()) >= len(menus))):
        raise ValueError('Average cache arrays disagree')
    index = _Index(arrays['keys'])
    return (CompactEntries(index, menus, arrays['menu'], arrays['offsets'], arrays['probabilities']),
            CompactFlags(index, arrays['flags'].view(np.bool_)),
            CompactCounts(index, arrays['counts']))
