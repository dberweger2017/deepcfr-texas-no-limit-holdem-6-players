"""Read-only current-policy/visit audit; never construct resumable trainer state."""

import gzip
import json
from hashlib import sha256
from math import isfinite
from pathlib import Path

from src.arena.catalog import Checkpoint
from src.blueprint.artifact import FrozenBlueprint, _checked_schema, HU20_UNCAPPED_FORMAT
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA, HU20_NATIVE_SCHEMAS
from src.blueprint.solver import regret_match


def file_hash(path):
    result = sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def load_saved(spec, inputs, guard=lambda: None, *, expected_schema=HU20_UNCAPPED_SCHEMA):
    # Default callers retain the original strict v1 contract. An experiment must
    # explicitly pin another registered schema; never infer it from the file.
    if expected_schema not in HU20_NATIVE_SCHEMAS:
        raise ValueError('Unknown native HU20 experiment schema')
    policy_path = Path(inputs) / spec['path']
    checkpoint_path = Path(inputs) / spec['checkpoint_path']
    if file_hash(checkpoint_path) != spec['checkpoint_sha256']:
        raise ValueError('Training checkpoint hash mismatch')
    source = FrozenBlueprint(Checkpoint(spec['name'], str(policy_path), spec['sha256'],
                                       spec['format']), policy_path)
    if (source.players != 2 or source.raise_cap is not None
            or source.abstraction != expected_schema
            or spec['format'] != HU20_UNCAPPED_FORMAT
            or source.description['strategy'] != 'current'
            or source.description['training_seed'] != spec['seed']):
        raise ValueError('Saved policy does not match the frozen HU20 lineage')
    visits = {}
    with gzip.open(checkpoint_path, 'rt') as handle:
        header = json.loads(handle.readline())
        if (_checked_schema(header) != source.abstraction or header.get('kind') != 'training'
                or header.get('identity') != source.identity
                or header.get('iteration') != source.description['iteration']
                or header['config']['seed'] != spec['seed']):
            raise ValueError('Checkpoint lineage differs from current policy')
        if header.get('checkpoint_format') == 'jsonl-v2':
            rows = (json.loads(line) for line in handle)
        elif 'checkpoint_format' not in header:
            rows = ([key, *value] for key, value in header['nodes'].items())
        else:
            raise ValueError('Unknown checkpoint format')
        for index, (key, names, regrets, average, count) in enumerate(rows):
            if index % 8192 == 0:
                guard()
            if (key in visits or type(count) is not int or count < 0
                    or len(names) != len(regrets) or len(names) != len(average)
                    or not all(isfinite(v) for v in (*regrets, *average))
                    or source.entries.get(key) != (tuple(names), regret_match(tuple(regrets)))):
                raise ValueError('Checkpoint node differs from current extraction')
            visits[key] = count
    if len(visits) != len(source.entries):
        raise ValueError('Checkpoint coverage differs from inference export')
    return source, visits
