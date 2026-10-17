"""Write the existing sorted-average cache format without retaining its arrays in RAM.

Probabilities pass through unchanged. The normal AveragePolicy reader verifies
and maps this cache; this command performs no policy normalization.
"""
import argparse
import gzip
import json
from math import fsum, isfinite
from pathlib import Path
import struct

import numpy as np

from src.blueprint import policy_cache
from src.blueprint.average import (EXTRACTIONS, average_rule, checked_header,
                                   format_for_schema, zero_mass_rule)
from src.policies.files import file_hash, local_cache


def build(path: Path, sha256: str, schema: str, cache: Path, entries: int, slots: int):
    """Counts come from the verified union; every row and both counts are checked."""
    cache = local_cache(cache)
    if entries < 1 or slots < entries or file_hash(path) != sha256:
        raise ValueError('Invalid counts or canonical model SHA256')
    before = path.stat()
    target = policy_cache.folder(cache, sha256, schema)
    if target.exists():
        raise FileExistsError('Preserve existing cache; validate it with policy_cache.load')
    temporary = target.with_name(target.name + '.streaming-partial')
    temporary.mkdir(parents=True, exist_ok=False)
    streams = {}
    menus, menu_ids = [], {}
    lengths = {name: slots if name == 'probabilities' else entries for name in policy_cache.ARRAYS}
    try:
        for name in policy_cache.ARRAYS:
            stream = (temporary / (name + '.npy')).open('xb')
            streams[name] = stream
            np.lib.format.write_array_header_1_0(stream, {
                'descr': np.lib.format.dtype_to_descr(np.dtype(policy_cache.DTYPES[name])),
                'fortran_order': False, 'shape': (lengths[name],)})
        count = offset = 0
        previous = None
        with gzip.open(path, 'rt') as source:
            metadata = json.loads(source.readline())
            header = metadata['checkpoint_header']
            checked_header(header, {'seed': header['config']['seed'], 'iteration': header['iteration']},
                           expected_schema=schema)
            rule = average_rule(header)
            zero = zero_mass_rule(metadata)
            if (metadata.get('format') != format_for_schema(schema)
                    or metadata.get('kind') != 'diagnostic-inference'
                    or metadata.get('extraction') != EXTRACTIONS[rule]
                    or entries > header['config']['max_entries']):
                raise ValueError('Average identity or entry cap differs')
            for line in source:
                key, names, probabilities, total, visits = json.loads(line)
                if (not isinstance(key, str) or len(key) != 32
                        or any(c not in '0123456789abcdef' for c in key)
                        or not names or len(names) != len(set(names))
                        or any(not isinstance(n, str) for n in names)
                        or len(names) != len(probabilities)
                        or any(type(p) not in (int, float) or not isfinite(p) or p < 0 for p in probabilities)
                        or abs(fsum(probabilities) - 1) > 1e-8
                        or type(total) not in (int, float) or not isfinite(total) or total < 0
                        or type(visits) is not int or visits < 0 or visits >= 2**64
                        or (rule == 'traverser-reach' and total > header['iteration'] * visits
                            + 1e-9 * max(1, header['iteration'] * visits))
                        or (not total and zero == 'uniform' and probabilities != [1/len(names)]*len(names))):
                    raise ValueError('Invalid sorted average row')
                digest = bytes.fromhex(key)
                if previous is not None and digest <= previous:
                    raise ValueError('Average keys must strictly increase')
                previous = digest
                names = tuple(names)
                if names not in menu_ids:
                    menu_ids[names] = len(menus)
                    menus.append(names)
                menu = menu_ids[names]
                if menu >= 2**16 or count >= entries or offset + len(names) > slots:
                    raise ValueError('Cache dimensions or menu IDs overflow')
                streams['keys'].write(digest)
                streams['menu'].write(struct.pack('<H', menu))
                streams['offsets'].write(struct.pack('<Q', offset))
                streams['probabilities'].write(struct.pack('<' + 'd'*len(names), *probabilities))
                streams['flags'].write(bytes([not total]))
                streams['counts'].write(struct.pack('<Q', visits))
                count += 1
                offset += len(names)
        if count != entries or offset != slots:
            raise ValueError('Cache row/action counts differ from union')
    finally:
        for stream in streams.values():
            stream.close()
    # An incomplete build stays at its explicit partial path for the failure archive.
    after = path.stat()
    if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
            or file_hash(path) != sha256):
        raise ValueError('Canonical average changed during cache build')
    files = {name: {'bytes': (temporary/(name+'.npy')).stat().st_size,
                    'sha256': file_hash(temporary/(name+'.npy')), 'length': lengths[name]}
             for name in policy_cache.ARRAYS}
    manifest = {'format': policy_cache.FORMAT, 'source_sha256': sha256, 'schema': schema,
                'entries': entries, 'menus': [list(m) for m in menus], 'files': files}
    (temporary/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    temporary.rename(target)
    policy_cache.load(cache, sha256, schema)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--schema', required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--entries', type=int, required=True)
    parser.add_argument('--action-slots', type=int, required=True)
    args = parser.parse_args()
    manifest = build(args.model, args.sha256, args.schema, local_cache(args.cache),
                     args.entries, args.action_slots)
    print(json.dumps({'status': 'verified', 'entries': manifest['entries'],
                      'array_bytes': sum(p['bytes'] for p in manifest['files'].values())}))


if __name__ == '__main__':
    main()
