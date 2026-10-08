"""Build native average specs from full audit receipts bound to exact bytes."""
import gzip
import json
from pathlib import Path
from src.policies.files import file_hash


def audited_average_spec(path: Path, audit: dict, *, checkpoint_sha256: str, actual_nodes: int) -> dict:
    # Native average headers omit entries. The full audit is authoritative, and
    # its hashes must bind the receipt to this average and its recovery parent.
    if (audit.get('status') != 'verified' or audit.get('audit', {}).get('status') != 'verified'
        or audit['audit']['checkpoint_sha256'] != checkpoint_sha256
        or audit['native_state']['completed_nodes'] != actual_nodes
        or audit['audit']['all_nodes_verified'] != audit['entries']
        or audit['audit']['counts']['entries'] != audit['entries']):
        raise ValueError('Verified consistent native audit required')
    expected = [v for p, v in audit['files'].items() if Path(p).name == 'average.gz']
    if len(expected) != 1 or expected[0]['sha256'] != audit['audit']['average_sha256']:
        raise ValueError('Average audit asset identity differs')
    if path.stat().st_size != expected[0]['bytes'] or file_hash(path) != expected[0]['sha256']:
        raise ValueError('Audited average bytes changed')
    with gzip.open(path, 'rt') as f:
        h = json.loads(next(f))
    if (h['source_checkpoint_sha256'] != checkpoint_sha256
        or h['checkpoint_header']['native_state']['completed_nodes'] != actual_nodes
        or h['checkpoint_header']['iteration'] != audit['iteration']
        or ('entries' in h and h['entries'] != audit['entries'])):
        raise ValueError('Average header and audit lineage differ')
    return {'name': 'HU100-average-' + str(actual_nodes), 'path': str(path),
            **expected[0], 'format': h['format'], 'actual_nodes': actual_nodes,
            'entries': audit['entries'], 'iteration': audit['iteration'],
            'source_checkpoint_sha256': checkpoint_sha256}
