"""Every-row accumulator/current audit for strictly sorted campaign outputs.

Sorted uniqueness permits a three-stream comparison without a table-sized
SQLite ledger. All model/header/number validators are the existing auditors'.
"""
from collections import Counter
import gzip
from itertools import zip_longest
import json
from math import fsum

from src.blueprint.artifact import _checked_policy_node, _checked_schema
from src.blueprint.average import EXTRACTIONS, average_rule, checked_header, checked_row, format_for_schema, zero_mass_rule
from src.blueprint.solver import BlueprintTrainer, PilotConfig, regret_match
from src.blueprint.streaming import current_rows
from src.diagnostics.cfr_average import exact_numbers, zero_mass_policy
from src.game.hand import Table
from src.policies.files import file_hash


def audit_sorted(checkpoint, current_path, average_path, spec, average_sha, *, expected_schema):
    """Refuse unordered/duplicate streams; check every row and gzip EOF/CRC."""
    if (file_hash(checkpoint) != spec['checkpoint_sha256'] or file_hash(current_path) != spec['sha256']
            or file_hash(average_path) != average_sha):
        raise ValueError('Extraction audit input hash differs')
    counts = Counter()
    tv = maximum = 0.0
    previous = ''
    current_metadata = {}
    with gzip.open(checkpoint, 'rt') as raw, gzip.open(average_path, 'rt') as exported, \
            gzip.open(current_path, 'rt') as current:
        header = json.loads(next(raw))
        checked_header(header, spec, expected_schema=expected_schema)
        metadata = json.loads(next(exported))
        rule, zero_mass = average_rule(header), zero_mass_rule(metadata)
        if (metadata.get('format') != format_for_schema(expected_schema)
                or metadata.get('kind') != 'diagnostic-inference'
                or metadata.get('extraction') != EXTRACTIONS[rule]
                or metadata.get('checkpoint_header') != header
                or metadata.get('source_checkpoint_sha256') != spec['checkpoint_sha256']):
            raise ValueError('Extraction lineage differs')
        for original, emitted, current_item in zip_longest(raw, exported, current_rows(current, current_metadata)):
            if original is None or emitted is None or current_item is None:
                raise ValueError('Current/average checkpoint coverage differs')
            key, names, regrets, p, total, visits = checked_row(
                json.loads(original), header['iteration'], rule == 'traverser-reach')
            if key <= previous:
                raise ValueError('Checkpoint keys not strictly sorted and unique')
            previous = key
            p = zero_mass_policy(p, total, regrets, zero_mass)
            if not exact_numbers(json.loads(emitted), [key, list(names), list(p), total, visits]):
                raise ValueError('Normalized accumulator differs')
            current_key, row = current_item
            _checked_policy_node(current_key, *row)
            current_p = regret_match(regrets)
            if current_key != key or not exact_numbers(row, [list(names), list(current_p)]):
                raise ValueError('Current export differs from checkpoint regrets or key order')
            counts['entries'] += 1
            counts['positive_mass' if total else 'zero_mass'] += 1
            if counts['entries'] > header['config']['max_entries']:
                raise ValueError('Checkpoint exceeds entry cap')
            distance = fsum(abs(a - b) for a, b in zip(p, current_p)) / 2
            tv += distance
            maximum = max(maximum, distance)
        if (_checked_schema(current_metadata) != expected_schema
                or current_metadata.get('kind') != 'inference'
                or current_metadata.get('strategy') != 'current'
                or current_metadata.get('iteration') != header['iteration']
                or current_metadata['config']['seed'] != spec['seed']):
            raise ValueError('Extraction lineage differs')
        table = current_metadata['table']
        BlueprintTrainer(Table(tuple(table['player_ids']), tuple(table['stacks']), table['button'],
            table['small_blind'], table['big_blind'], table['chip_unit']), PilotConfig(**current_metadata['config']))
    count = counts['entries']
    return dict(status='verified', all_nodes_verified=count, counts=dict(counts),
        unweighted_mean_current_average_tv=tv / count if count else None, maximum_tv=maximum,
        scope='all stored totals and current regrets checked; historical increments not reconstructed',
        current_sha256=spec['sha256'], average_sha256=average_sha, checkpoint_sha256=spec['checkpoint_sha256'])
