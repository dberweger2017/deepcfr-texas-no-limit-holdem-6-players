"""Diagnostic correctness: off-menu sizes, abstract support and stored joins."""

from dataclasses import asdict
import fcntl
import gzip
import json

import pytest

from scripts.diagnose_native_hu100 import Cell, joined, signature, support_witness, verify
from src.blueprint.abstraction import HU100_SCHEMA, choices, information_key
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind


def initial():
    return Hand.start(Table(('a', 'b'), (10000, 10000)), hand_id='diagnostic', seed=7)


def target_after(action):
    root = initial(); hand = root.apply(action); view = hand.observe(hand.actor)
    return root, view, choices(view, raise_cap=None, free_fold=False)


def test_off_menu_raise_can_have_an_exact_supported_abstract_key():
    root, view, menu = target_after(Action(ActionKind.RAISE, 400))
    assert Action(ActionKind.RAISE, 400) not in [c.action for c in choices(root.observe(root.actor), raise_cap=None, free_fold=False)]
    proof = support_witness(root, signature(view, menu))
    assert proof['support'] == 'supported'
    replay = root
    for a in proof['witness_actions']:
        action = Action(ActionKind(a['kind']), a['raise_to'])
        assert action in [c.action for c in choices(replay.observe(replay.actor), raise_cap=None, free_fold=False)]
        replay = replay.apply(action)
    v = replay.observe(replay.actor); m = choices(v, raise_cap=None, free_fold=False)
    assert information_key(v, m, schema=HU100_SCHEMA) == information_key(view, menu, schema=HU100_SCHEMA)
    assert v.pot != view.pot


def test_open_jam_history_requires_exhaustion_before_unsupported_label():
    root, view, menu = target_after(Action(ActionKind.RAISE, 10000))
    proof = support_witness(root, signature(view, menu))
    assert proof['support'] == 'unsupported-abstract-history-or-menu'
    assert proof['reason'] == 'exhaustive-supported-token-tree'


@pytest.mark.parametrize('limit,deadline', [(0, float('inf')), (5000, 0)])
def test_capped_search_stays_unresolved(limit, deadline):
    root, view, menu = target_after(Action(ActionKind.RAISE, 400))
    assert support_witness(root, signature(view, menu), max_states=limit, deadline=deadline)['support'] == 'unresolved'


def test_positive_mass_is_not_the_same_as_traverser_visits():
    root = initial(); view = root.observe(root.actor); menu = choices(view, raise_cap=None, free_fold=False)
    d = {'key': information_key(view, menu, schema=HU100_SCHEMA), 'lookup': 'positive-mass-known-key',
         'menu': [asdict(c) for c in menu], 'probabilities': [1 / len(menu)] * len(menu)}
    table = {d['key']: [[c.name for c in menu], d['probabilities'], 17., 0]}
    assert joined(d, table) == (17., 0)
    d['lookup'] = 'zero-mass'
    with pytest.raises(ValueError, match='classification'):
        joined(d, table)


def test_probability_and_menu_discrepancies_fail_closed():
    d = {'key': 'a', 'lookup': 'positive-mass-known-key', 'menu': [{'name': 'call'}], 'probabilities': [1.]}
    with pytest.raises(ValueError, match='probabilities'):
        joined(d, {'a': [['call'], [.9], 1., 1]})
    with pytest.raises(ValueError, match='menu'):
        joined(d, {'a': [['fold'], [1.], 1., 1]})


def test_missing_rows_do_not_dilute_covered_mass_or_visit_means():
    cell = Cell()
    base = {'block': 0, 'rotation': 0, 'net_bb': 1, 'fold_probability': .5,
            'passive_probability': .5, 'raise_probability': 0}
    cell.add({**base, 'mass': None, 'visits': None})
    assert cell.result()['mean_mass'] is None
    cell.add({**base, 'mass': 100, 'visits': 2})
    result = cell.result()
    assert result['mean_mass'] == 100
    assert result['mean_visits'] == 2
    assert result['covered_decisions'] == 1
    assert result['decisions'] == 2


def test_analysis_reserves_closeout_and_archive_requires_terminal_work(tmp_path, monkeypatch):
    from scripts import run_native_hu100_diagnosis as launcher
    budget = {'source': 'revision', 'deadline': 1800, 'swap_baseline': 'baseline',
              'limits': {'rss_gib': 4, 'disk_gib': 20, 'swap_gib': .25}}
    (tmp_path / 'budget.json').write_text(json.dumps(budget))
    monkeypatch.setattr(launcher.subprocess, 'check_output', lambda command, **kw:
                        'revision\n' if command[1] == 'rev-parse' else '')
    captured = []
    monkeypatch.setattr(launcher, 'run', lambda jobs, out, deadline, **kw:
                        captured.append(deadline) or {'status': 'complete'})
    launcher.guard(tmp_path, 'analysis', [])
    assert captured == [1560]
    with pytest.raises(RuntimeError, match='terminal'):
        launcher.guard(tmp_path, 'archive', [])
    prior = tmp_path / 'guard-analysis'; prior.mkdir()
    (prior / 'campaign.json').write_text(json.dumps({'status': 'running'}))
    with pytest.raises(RuntimeError, match='terminal'):
        launcher.guard(tmp_path, 'archive', [])
    (prior / 'campaign.json').write_text(json.dumps({'status': 'incomplete'}))
    launcher.guard(tmp_path, 'archive', [])
    assert captured == [1560, 1800]


def test_root_lock_prevents_overlapping_science_and_archive(tmp_path):
    from scripts.run_native_hu100_diagnosis import guard
    with (tmp_path / 'phase.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            guard(tmp_path, 'archive', [])


def test_independent_recount_rejects_changed_cell_mean(tmp_path):
    # Mixed missing/covered rows exercise the former diluted-denominator failure.
    base = {'nodes': 1, 'opponent': 'fixture', 'block': 0, 'rotation': 0, 'net_bb': 1,
            'street': 'preflop', 'position': 'BB', 'pot_band': '4-16',
            'fold_probability': .2, 'passive_probability': .3, 'raise_probability': .5}
    selected = [{**base, 'lookup': 'positive-mass-known-key', 'support': 'stored-training-witness',
                 'mass': 100, 'visits': 2, 'mass_band': '100-10k', 'visit_band': '2-9'},
                {**base, 'lookup': 'missing-key', 'support': 'supported-observed-menu-path',
                 'mass': None, 'visits': None, 'mass_band': 'missing', 'visit_band': 'missing'}]
    with gzip.open(tmp_path / 'features.jsonl.gz', 'wt') as out:
        out.write(''.join(json.dumps(r) + '\n' for r in selected))
    p = {'nodes': 1, 'opponent': 'fixture', 'target_decisions': 2, 'hands': 1,
         'bb_per_100': 100, 'missing_support': {'supported-observed-menu-path': 1},
         'hand_exposures': {'ever-missing': {'hands': 1, 'net_bb': 1, 'mean_net_bb': 1,
                                            'contribution_bb_per_100': 100}}}
    (tmp_path / 'summary.json').write_text(json.dumps({'target_decisions': 2, 'hands': 1, 'panels': [p]}))
    (tmp_path / 'hands.jsonl').write_text(json.dumps({**base, 'exposure': 'ever-missing'}) + '\n')
    (tmp_path / 'native-fixtures.jsonl').write_text(json.dumps({'final_stacks': [10100, 9900]}) + '\n')
    groups = {}
    dims = (('street', 'position', 'pot_band', 'lookup'), ('street', 'visit_band'),
            ('street', 'mass_band'), ('lookup',), ('street',))
    for r in selected:
        for names in dims:
            key = names, tuple(r[n] for n in names)
            groups.setdefault(key, Cell()).add(r)
    cells = [{'nodes': 1, 'opponent': 'fixture', 'dimensions': list(names),
              **dict(zip(names, values)), **cell.result()} for (names, values), cell in groups.items()]
    (tmp_path / 'cells.json').write_text(json.dumps(cells))
    verify(tmp_path)
    mixed = next(c for c in cells if c['dimensions'] == ['street'])
    mixed['mean_mass'] = 50  # The former error passed total-denominator checks.
    (tmp_path / 'cells.json').write_text(json.dumps(cells))
    with pytest.raises(ValueError, match='Stratum mean differs: mean_mass'):
        verify(tmp_path)
