"""K50 evaluation owns tables and schema across mixed-policy and public probes."""
import gzip
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts.evaluate_hu20_cfr_average import play
from scripts.evaluate_hu20_cfr_plus_direct import PolicyRival
from scripts.evaluate_hu20_v041_arena import load
from scripts.native_parity_fixtures import record, DECK
from src.arena.catalog import Checkpoint
from src.arena.registry import load_frozen, PolicyRegistry
from src.arena.schedule import Plan, Scenario
from src.blueprint import equity_buckets as buckets
from src.blueprint.abstraction import (
    choices, information_key, HU20_EQUITY_SCHEMA, HU100_EQUITY_SCHEMA,
    HU20_UNCAPPED_SCHEMA, HU100_SCHEMA,
)
from src.blueprint.artifact import _identity, _format, load_training, save_training, FrozenBlueprint
from src.blueprint.average import AveragePolicy, EXTRACTIONS, ZERO_MASS_RULES, format_for_schema
from src.blueprint.solver import BlueprintTrainer, PilotConfig, HU20_UNCAPPED_GAME, HU100_GAME
from src.diagnostics.cached_lbr import SharedProbabilityCache
from src.diagnostics.robustness import LocalBestResponse
from src.game.hand import Hand, Table
from src.game.observation import replay
from src.game.types import Action, ActionKind
from src.policies.files import file_hash
from tests.native_equity_tables import write_tables

BINARY = Path(os.environ.get('K50_NATIVE_BINARY', 'native/hu20-trainer/target/release/hu20-trainer'))


def views(fixture, stack=2000):
    hand = Hand.from_deck(Table(('a', 'b'), (stack,) * 2, button=fixture['button']),
                          hand_id='parity', deck=fixture['deck'])
    for kind, amount in fixture['actions']:
        yield hand.observe(hand.actor)
        hand = hand.apply(Action(ActionKind(kind), amount))


def write_model(path, schema, cards, observations, *, average=True):
    bb = 100 if schema in (HU100_SCHEMA, HU100_EQUITY_SCHEMA) else 20
    config = PilotConfig(seed=47, raise_cap=None, abstraction=schema,
                         game=HU100_GAME if bb == 100 else HU20_UNCAPPED_GAME)
    from dataclasses import asdict
    header = {'format': _format(config), 'kind': 'training', 'abstraction': schema,
              'config': asdict(config), 'identity': _identity(config), 'iteration': 10,
              'checkpoint_format': 'jsonl-v2', 'average_rule': 'opponent-sampled',
              'table': {'player_ids': ['a', 'b'], 'stacks': [bb * 100] * 2,
                        'button': 0, 'small_blind': 50, 'big_blind': 100, 'chip_unit': '0.01'}}
    rows = {}
    for view in observations:
        menu = choices(view, raise_cap=None, free_fold=False)
        key = information_key(view, menu, schema=schema, equity_cards=cards)
        names = [c.name for c in menu]
        p = [1.] + [0.] * (len(menu) - 1)
        rows[key] = [key, names, p, 10., 1] if average else [key, names, [0.] * len(menu), [10.] + [0.] * (len(menu)-1), 1]
    metadata = {'format': format_for_schema(schema), 'kind': 'diagnostic-inference',
                'checkpoint_header': header, 'source_checkpoint_sha256': 'a' * 64,
                'extraction': EXTRACTIONS['opponent-sampled'], 'zero_mass_rule': ZERO_MASS_RULES['uniform']}
    with gzip.open(path, 'wt') as stream:
        stream.write(json.dumps(metadata if average else header) + '\n')
        for row in rows.values():
            stream.write(json.dumps(row) + '\n')
    return header


@pytest.fixture(params=[20, 100])
def model(tmp_path, monkeypatch, request):
    bb = request.param
    fixture = record(543, 0.1, 1., stack_bb=bb)
    observations = list(views(fixture, bb * 100))
    assert any(len(v.board) == 5 for v in observations)
    situations = [(pair, v.board) for v in observations if v.board
                  for pair in combinations([c for c in DECK if c not in v.board], 2)]
    tables = write_tables(tmp_path, situations)
    cards = buckets.EquityCards(tables, pinned=False)
    # Production API has no unpinned escape hatch; substitute the schema pins only in fixtures.
    monkeypatch.setattr(buckets, 'EQUITY_K50_SHA256', cards.sha256)
    schema = HU20_EQUITY_SCHEMA if bb == 20 else HU100_EQUITY_SCHEMA
    path = tmp_path / 'average.gz'
    write_model(path, schema, cards, observations)
    return bb, schema, tables, cards, path, observations


def test_loaders_own_mapped_tables_and_ignore_registration(model, tmp_path, monkeypatch):
    bb, schema, tables, cards, path, observations = model
    policy = AveragePolicy(path, file_hash(path), expected_schema=schema, card_buckets=tables)
    checkpoint = tmp_path / 'checkpoint.gz'
    write_model(checkpoint, schema, cards, observations, average=False)
    trainer = load_training(checkpoint, card_buckets=tables)
    assert trainer.average_rule == 'opponent-sampled'
    # Loading a checkpoint never licenses a Python continuation that would mix averages.
    with pytest.raises(ValueError, match='evaluation only'):
        trainer.step()
    with pytest.raises(ValueError, match='Evaluation-only'):
        save_training(trainer, tmp_path / 'bad-save.gz')
    monkeypatch.setattr(buckets, '_registered', None)
    for view in observations:
        menu, p, known = policy.distribution(view)
        assert known and p[0] == 1.
        assert trainer.frozen().distribution(view, menu) == (1 / len(menu),) * len(menu)
        assert policy.information_key(view, menu) == information_key(view, menu, schema=schema, equity_cards=cards)
    for table in policy.equity_cards.tables.values():
        assert isinstance(table.hashes.base, __import__('numpy').memmap)
        assert not table.hashes.flags.writeable
    assert policy.description['card_tables'] == cards.sha256


def test_refusal_cases(model, tmp_path):
    bb, schema, tables, cards, path, observations = model
    v1 = HU20_UNCAPPED_SCHEMA if bb == 20 else HU100_SCHEMA
    checkpoint = tmp_path / 'checkpoint.gz'
    header = write_model(checkpoint, schema, cards, observations, average=False)
    for read in (lambda: AveragePolicy(path, file_hash(path), expected_schema=schema), lambda: load_training(checkpoint)):
        with pytest.raises(ValueError, match='explicit card_buckets'):
            read()
    with pytest.raises(ValueError, match='identity|schema'):
        AveragePolicy(path, file_hash(path), expected_schema=v1, card_buckets=tables)
    vp = tmp_path / 'v1.gz'
    write_model(vp, v1, None, observations)
    with pytest.raises(ValueError, match='K50 schema'):
        AveragePolicy(vp, file_hash(vp), expected_schema=v1, card_buckets=tables)
    with pytest.raises(ValueError, match='identity|schema'):
        AveragePolicy(vp, file_hash(vp), expected_schema=schema, card_buckets=tables)
    # A recorded hash mismatch is refused even when the supplied files are correct.
    with gzip.open(checkpoint, 'rt') as f:
        rows = list(f)
    header['identity']['card_tables']['flop'] = 'f' * 64
    rows[0] = json.dumps(header) + '\n'
    with gzip.open(checkpoint, 'wt') as f:
        f.writelines(rows)
    with pytest.raises(ValueError, match='schema'):
        load_training(checkpoint, card_buckets=tables)
    # Correct metadata with changed table bytes also fails the production pin.
    (tables / 'flop-k50.bin').write_bytes((tables / 'flop-k50.bin').read_bytes() + b'x')
    with pytest.raises(ValueError, match='#163'):
        AveragePolicy(path, file_hash(path), expected_schema=schema, card_buckets=tables)


def test_lbr_hypotheticals_use_targets_schema_and_only_legal_observation(model, monkeypatch):
    bb, schema, tables, cards, path, observations = model
    policy = AveragePolicy(path, file_hash(path), expected_schema=schema, card_buckets=tables)
    monkeypatch.setattr(buckets, '_registered', None)
    view = next(v for v in observations if len(v.board) == 4)
    pair = tuple(c for c in DECK if c not in (*view.board, *view.hole_cards))[:2]
    hypothetical = replay(view.history, view.seat, pair)
    expected = policy.distribution(hypothetical)
    assert LocalBestResponse(policy, 1).probabilities(view.history, view.seat, pair) == expected
    assert SharedProbabilityCache(policy).probabilities(view.history, view.seat, pair) == expected
    # Distinct hidden decks and opponent hands give identical target inputs and distributions.
    fixture = record(543, 0.1, 1., stack_bb=bb)
    alternative = dict(fixture, deck=list(fixture['deck']))
    a = next(views(fixture, bb*100))
    hidden_index = next(i for i in range(4) if fixture['deck'][i] not in a.hole_cards)
    alternative['deck'][hidden_index], alternative['deck'][20] = alternative['deck'][20], alternative['deck'][hidden_index]
    for a, b in zip(views(fixture, bb*100), views(alternative, bb*100), strict=True):
        if a.seat == observations[0].seat:
            assert a == b
            assert policy.distribution(a) == policy.distribution(b)



def test_search_and_web_refuse_before_work(model, tmp_path):
    bb, schema, tables, cards, path, observations = model
    policy = AveragePolicy(path, file_hash(path), expected_schema=schema, card_buckets=tables)
    from src.blueprint.hu20_turn_search import HU20TurnSearchPolicy
    from src.blueprint.search import SearchPlayer
    from src.play_api.service import PlayService
    for construct in (lambda: HU20TurnSearchPolicy(policy, None),
                      lambda: SearchPlayer(policy, 1),
                      lambda: PlayService(tmp_path / 'play.db', policy)):
        with pytest.raises(ValueError, match='K50 models are not yet supported'):
            construct()
    assert not (tmp_path / 'play.db').exists()


def test_arena_registration_and_direct_duplicate_replay(model, tmp_path, monkeypatch):
    bb, schema, tables, cards, path, observations = model
    spec = Checkpoint('k50', str(path), file_hash(path), format_for_schema(schema), schema, str(tables))
    policy = load_frozen(spec, path)
    v1_schema = HU20_UNCAPPED_SCHEMA if bb == 20 else HU100_SCHEMA
    vp = tmp_path / 'v1.gz'
    write_model(vp, v1_schema, None, observations)
    vs = Checkpoint('v1', str(vp), file_hash(vp), format_for_schema(v1_schema))
    plan = Plan((Scenario('mixed', (bb*100,) * 2),), blocks=1, candidate='k50', baseline='v1',
                opponents=('check_call',), models=(spec, vs))
    registry = PolicyRegistry(plan)
    assert registry.models['k50'].abstraction == schema
    assert registry.models['v1'].abstraction == v1_schema
    assert registry.models['k50'].distribution(observations[-1])[2]
    if bb != 20:
        return
    spec_dict = {'name': 'k50', 'path': str(path), 'sha256': file_hash(path), 'bytes': path.stat().st_size,
                 'abstraction': schema, 'card_buckets': str(tables), 'seed': 47, 'iteration': 10,
                 'strategy': 'average', 'checkpoint_sha256': 'a'*64}
    assert load(spec_dict, tmp_path).abstraction == schema
    # A single direct pair can reach a board absent from this fixture. Cover its actual paired deal.
    from src.arena.schedule import stream_seed
    hand = Hand.start(Table(('seat0','seat1'), (2000,2000)), hand_id='cover',
                      seed=stream_seed(66, 'test', 'deal', 2, 0))
    situations = [(v.hole_cards, v.board) for v in observations if v.board]
    while not hand.finished:
        v = hand.observe(hand.actor)
        if v.board:
            situations.append((v.hole_cards, v.board))
        menu = choices(v, raise_cap=None, free_fold=False)
        quiet = next(c.action for c in menu if c.name in ('call', 'check'))
        hand = hand.apply(quiet)
    tables2 = tmp_path / 'direct-tables'; tables2.mkdir()
    write_tables(tables2, situations)
    monkeypatch.setattr(buckets, 'EQUITY_K50_SHA256', buckets.EquityCards(tables2, pinned=False).sha256)
    write_model(path, schema, buckets.EquityCards(tables2, pinned=False), observations)
    policy = AveragePolicy(path, file_hash(path), expected_schema=schema, card_buckets=tables2)
    reference = AveragePolicy(vp, file_hash(vp))
    for rotation in (0, 1):
        row = play(policy, spec_dict, {'name':'direct','contract':'frozen-policy','rule':'frozen-policy'},
                   66, 0, rotation, rival=PolicyRival(reference, 5))
        assert row['native_replay_verified']
        assert sum(row['net_chips_by_seat']) == 0


@pytest.mark.skipif(not BINARY.exists(), reason='build native trainer first or set K50_NATIVE_BINARY')
@pytest.mark.parametrize('bb', [20, 100])
def test_loaded_native_exports_key_parity(tmp_path, monkeypatch, bb):
    from src.blueprint import solver
    v1 = HU20_UNCAPPED_SCHEMA if bb == 20 else HU100_SCHEMA
    schema = HU20_EQUITY_SCHEMA if bb == 20 else HU100_EQUITY_SCHEMA
    config = PilotConfig(seed=3, raise_cap=None, abstraction=v1,
                         game=HU20_UNCAPPED_GAME if bb == 20 else HU100_GAME,
                         max_nodes=10**9, max_entries=10**9, max_seconds=900)
    observations, traverser_views, active_seat = [], [], [None]
    collect = solver._collect_root
    def collect_root(*args, **kwargs):
        active_seat[0] = args[4]
        return collect(*args, **kwargs)
    monkeypatch.setattr(solver, "_collect_root", collect_root)
    original = solver.information_key
    def capture(view, menu, **kwargs):
        observations.append(view)
        if view.seat == active_seat[0]:
            traverser_views.append(view)
        return original(view, menu, **kwargs)
    monkeypatch.setattr(solver, 'information_key', capture)
    reference = BlueprintTrainer(Table(('player-0','player-1'), (bb*100,) * 2), config)
    reference.step()
    monkeypatch.setattr(solver, 'information_key', original)
    write_tables(tmp_path, [(v.hole_cards, v.board) for v in observations if v.board])
    cards = buckets.EquityCards(tmp_path, pinned=False)
    monkeypatch.setattr(buckets, 'EQUITY_K50_SHA256', cards.sha256)
    checkpoint, average, current = (tmp_path / n for n in ('checkpoint.gz','average.gz','current.gz'))
    subprocess.run([str(BINARY), 'train', '--stack-bb', str(bb), '--nodes', str(10**9), '--iterations', '1',
                    '--seed', '3', '--card-buckets', str(tmp_path), '--card-tables-unpinned', '--out', str(checkpoint)],
                   check=True, capture_output=True)
    subprocess.run([str(BINARY), 'export', str(checkpoint), '--average', str(average), '--current', str(current)],
                   check=True, capture_output=True)
    trainer = load_training(checkpoint, card_buckets=tmp_path)
    policy = AveragePolicy(average, file_hash(average), expected_schema=schema, card_buckets=tmp_path)
    frozen = FrozenBlueprint(Checkpoint('native', str(current), file_hash(current), _format(config)),
                             current, card_buckets=tmp_path)
    monkeypatch.setattr(buckets, '_registered', None)
    assert observations and any(v.board for v in observations)
    retained = set()
    for view in traverser_views:
        menu, p, trained = policy.distribution(view)
        retained.add(policy.information_key(view, menu))
        assert trained  # Python keys exist in the actual native export, including all streets.
        key = policy.information_key(view, menu)
        assert key in trainer.nodes and key in frozen.entries
        assert trainer.nodes[key].names == tuple(c.name for c in menu)
        assert frozen.distribution(view) == (menu, trainer.frozen().distribution(view, menu), True)
    assert retained == set(trainer.nodes) == set(policy.entries)


def test_optional_real_tables(tmp_path):
    directory = os.environ.get('K50_REAL_TABLES')
    if directory is None:
        pytest.skip('set K50_REAL_TABLES to restored #163 tables')
    cards = buckets.EquityCards(directory)
    fixture = record(543, 0.1, 1.)
    observations = list(views(fixture))
    path = tmp_path / 'real-average.gz'
    write_model(path, HU20_EQUITY_SCHEMA, cards, observations)
    policy = AveragePolicy(path, file_hash(path), expected_schema=HU20_EQUITY_SCHEMA, card_buckets=directory)
    assert all(policy.distribution(v)[2] for v in observations)
