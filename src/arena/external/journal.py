"""Append-only, hash-chained public-prefix evidence; transport owns all secrets."""

from hashlib import sha256
import json
from pathlib import Path

from src.arena.external.slumbot import verify_record


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class PrefixJournal:
    def __init__(self, path: Path, identity: dict):
        # A new evaluation never overwrites evidence from a partial attempt.
        self.file = path.open('x', encoding='utf-8')
        self.previous = '0' * 64
        self.sequence = 0
        self._write({'identity': identity, 'scope': 'public decision prefixes; not terminal settlement verification'})

    def _write(self, payload):
        row = {'sequence': self.sequence, 'previous': self.previous, 'payload': payload}
        row['sha256'] = sha256(canonical(row).encode()).hexdigest()
        self.file.write(canonical(row) + '\n'); self.file.flush()
        self.previous = row['sha256']; self.sequence += 1

    def append(self, record):
        verify_record(json.loads(canonical(record)))
        self._write({'decision': record})

    def failure(self, reason: str):
        # Caller supplies a fixed classification, never an exception string or raw
        # transport body that could contain credentials or future hidden data.
        if reason not in ('timeout-ambiguous', 'protocol-error', 'incompatible-game', 'resource-stop'):
            raise ValueError('Use a known failure classification')
        self._write({'failure': reason})

    def close(self):
        self.file.close()


def verify_journal(path: Path, *, expected_tail=None):
    previous = '0' * 64; count = 0; rows = 0
    with path.open(encoding='utf-8') as source:
        for sequence, line in enumerate(source):
            rows += 1
            row = json.loads(line); expected = row.pop('sha256')
            if (row['sequence'] != sequence or row['previous'] != previous
                    or sha256(canonical(row).encode()).hexdigest() != expected):
                raise ValueError('External journal chain differs')
            payload = row['payload']
            if sequence == 0:
                if set(payload) != {'identity', 'scope'}: raise ValueError('Missing evaluation identity')
            elif set(payload) == {'decision'}:
                verify_record(payload['decision']); count += 1
            elif set(payload) != {'failure'} or payload['failure'] not in (
                    'timeout-ambiguous', 'protocol-error', 'incompatible-game', 'resource-stop'):
                raise ValueError('Invalid external journal payload')
            previous = expected
    if rows == 0 or (expected_tail is not None and expected_tail != previous):
        raise ValueError('Missing journal or anchored tail differs')
    return {'decisions': count, 'tail_sha256': previous}
