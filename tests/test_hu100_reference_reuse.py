"""Retained uniform traces own their original schema, even through reuse chains."""
from dataclasses import asdict
import gzip
import json
from pathlib import Path

import pytest

from scripts import evaluate_native_hu100_baseline as play, audit_native_hu100_baseline as replay
from src.arena.schedule import build_schedule
from src.blueprint import equity_buckets as buckets
from src.blueprint.abstraction import (HU100_SCHEMA, HU100_MENU_V2_SCHEMA,
    HU100_EQUITY_SCHEMA, HU100_EQUITY_MENU_V2_SCHEMA, EQUITY_SCHEMAS,
    information_key, schema_choices)
from src.blueprint.artifact import _identity, _format
from src.blueprint.average import EXTRACTIONS, ZERO_MASS_RULES, format_for_schema
from src.blueprint.solver import PilotConfig, HU100_GAME
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind
from src.policies.files import file_hash
from tests.native_equity_tables import write_tables
from tests.test_v2_consumers import dev_source


def uniform_model(path, schema, tables, cards):
    config = PilotConfig(seed=47, raise_cap=None, abstraction=schema, game=HU100_GAME)
    table = Table(('player-0', 'player-1'), (10000, 10000))
    header = {'format': _format(config), 'kind': 'training', 'abstraction': schema,
              'config': asdict(config), 'identity': _identity(config), 'iteration': 10,
              'checkpoint_format': 'jsonl-v2', 'average_rule': 'opponent-sampled',
              'table': asdict(table)}
    metadata = {'format': format_for_schema(schema), 'kind': 'diagnostic-inference',
                'checkpoint_header': header, 'source_checkpoint_sha256': 'a' * 64,
                'extraction': EXTRACTIONS['opponent-sampled'], 'zero_mass_rule': ZERO_MASS_RULES['uniform']}
    view = Hand.start(table, hand_id='fixture', seed=3).observe(0)
    menu = schema_choices(view, schema)
    row = [information_key(view, menu, schema=schema, equity_cards=cards),
           [c.name for c in menu], [1/len(menu)]*len(menu), 10., 1]
    with gzip.open(path, 'wt') as stream:
        stream.write(json.dumps(metadata)+'\n'+json.dumps(row)+'\n')
    spec = dict(name=path.stem, path=str(path), sha256=file_hash(path),
                format=format_for_schema(schema), abstraction=schema, bytes=path.stat().st_size,
                entries=1, iteration=10, source_checkpoint_sha256='a'*64)
    if schema in EQUITY_SCHEMAS:
        spec['card_buckets'] = str(tables)
    return spec


@pytest.mark.parametrize('schemas', [
    (HU100_EQUITY_SCHEMA, HU100_EQUITY_MENU_V2_SCHEMA, HU100_SCHEMA),
    (HU100_EQUITY_MENU_V2_SCHEMA, HU100_EQUITY_SCHEMA, HU100_MENU_V2_SCHEMA),
])
def test_mixed_schema_retained_reference_and_chained_reuse(tmp_path, monkeypatch, schemas):
    settings = json.loads(Path('configs/arena/hu100-playing-baseline-v1.json').read_text())
    # Cover the same fixed deals to river for both seats, independent of the actions played.
    situations = []
    for opponent in play.OPPONENTS:
        for block in build_schedule(play.make_plan(settings, opponent, 2, 12345)):
            hand = Hand.start(Table(('player-0', 'player-1'), (10000, 10000), block.button),
                              hand_id='tables', seed=block.deal_seeds[0])
            while not hand.finished:
                view = hand.observe(hand.actor)
                if view.board:
                    situations.extend((hand.observe(seat).hole_cards, view.board) for seat in (0, 1))
                kind = ActionKind.CALL if view.legal_actions.call_amount else ActionKind.CHECK
                hand = hand.apply(Action(kind))
    tables = tmp_path/'tables'
    tables.mkdir()
    write_tables(tables, situations)
    cards = buckets.EquityCards(tables, pinned=False)
    monkeypatch.setattr(buckets, 'EQUITY_K50_SHA256', cards.sha256)
    monkeypatch.setattr(buckets, '_registered', None)
    source = dev_source(monkeypatch)
    roots = []
    original_rows = {}
    for index, schema in enumerate(schemas):
        settings['model'] = uniform_model(tmp_path/f'model-{index}.gz', schema, tables, cards)
        settings['action_translation'] = None
        config = tmp_path/f'config-{index}.json'
        config.write_text(json.dumps(settings))
        root = tmp_path/f'run-{index}'
        done = play.execute(config, root, 2, 12345, source,
                            reference_run=roots[-1] if roots else None)
        assert done['hands'] == 40
        assert done['hands_actually_played'] == (20 if roots else 40)
        for opponent in play.OPPONENTS:
            for filename in ('hands.jsonl', 'decisions.jsonl.gz'):
                path = root/opponent/filename
                opener = gzip.open if path.suffix == '.gz' else open
                with opener(path, 'rb') as stream:
                    rows = [line for line in stream if json.loads(line)['arm'] == 'baseline']
                key = opponent, filename
                assert rows == original_rows.setdefault(key, rows)
        result = replay.audit(root, tmp_path/f'audit-{index}.json')
        assert result['status'] == 'verified' and result['hands_replayed'] == 40
        roots.append(root)
    # The oldest identity is pinned by the intermediate run, not just the newest link.
    with (roots[0]/'inputs.json').open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='Reused reference identity changed'):
        replay.audit(roots[-1], tmp_path/'tampered-audit.json')


def test_reference_provenance_rejects_cycles(tmp_path):
    inputs = {'reference_run': str(tmp_path)}
    with pytest.raises(ValueError, match='Cyclic reused reference provenance'):
        replay.baseline_inputs(tmp_path, inputs)


@pytest.mark.parametrize('changed', ['root', 'source', 'opponents'])
def test_reference_provenance_rejects_hash_pinned_schedule_mismatch(tmp_path, changed):
    original = {'root': 12345, 'blocks_per_opponent': 2, 'source': 'a'*40,
                'config': {'opponents': list(play.OPPONENTS), 'reference': play.REFERENCE},
                'reference_run': None}
    path = tmp_path/'inputs.json'
    path.write_text(json.dumps(original))
    inputs = json.loads(json.dumps(original))
    inputs.update(reference_run=str(tmp_path), reference_inputs_sha256=file_hash(path))
    if changed == 'opponents':
        inputs['config']['opponents'] = ['check_call']
    else:
        inputs[changed] = 54321 if changed == 'root' else 'b'*40
    with pytest.raises(ValueError, match='Reused reference schedule/source differs'):
        replay.baseline_inputs(tmp_path/'new', inputs)
