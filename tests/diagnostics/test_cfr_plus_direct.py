"""A direct frozen-policy rival and block inference use the same information boundary."""
import gzip
import json

import pytest

from scripts.evaluate_hu20_cfr_plus_direct import PolicyRival, report
from src.arena.schedule import digest
from src.blueprint.abstraction import choices
from src.game.hand import Hand, Table


def test_rival_uses_its_own_observation_and_reproducible_stream():
    seen = []
    class Source:
        def distribution(self, view):
            seen.append(view)
            menu = choices(view, raise_cap=None, free_fold=False)
            return menu, [1 / len(menu)] * len(menu), False
    hand = Hand.start(Table(('a', 'b'), (2000, 2000)), hand_id='direct-test', seed=11)
    view = hand.observe(hand.actor)
    left, right = PolicyRival(Source(), 3), PolicyRival(Source(), 3)
    assert [left.choose_action(view) for _ in range(8)] == [right.choose_action(view) for _ in range(8)]
    assert all(v is view for v in seen)


def test_direct_report_counts_deals_not_positions_or_lineages(tmp_path):
    plan = {'root': 11, 'blocks': 32, 'pairs': [{}, {}, {}], 'primary_pairs': [1, 2, 3]}
    for lineage in (1, 2, 3):
        name = f'direct-lineage-{lineage}'
        (tmp_path / f'{name}.result.json').write_text(json.dumps({'status': 'complete', 'plan_sha256': digest(plan)}))
        with gzip.open(tmp_path / f'{name}.hands.jsonl.gz', 'wt') as f:
            for b in range(32):
                for r in (0, 1):
                    f.write(json.dumps({'block': b, 'rotation': r, 'lineage': lineage, 'root_seed': 11,
                                        'target_chips': 10 + b + lineage + r}) + '\n')
    result = report(plan, tmp_path)
    assert result['hands'] == 192
    assert result['overall']['blocks'] == 32
    assert result['overall']['bb_per_100'] == pytest.approx(28)
    assert result['overall']['ci95'][0] > 0
    assert result['primary_overall']['label'] == 'better'
    assert result['primary_overall']['blocks'] == 32
    with gzip.open(tmp_path / 'direct-lineage-1.hands.jsonl.gz', 'at') as f:
        f.write(json.dumps({'block': 0, 'rotation': 0, 'lineage': 1, 'root_seed': 11, 'target_chips': 0}) + '\n')
    with pytest.raises(ValueError, match='duplicate'):
        report(plan, tmp_path)


def test_shipped_reference_primary_does_not_average_other_references(tmp_path):
    plan = {'root': 12, 'blocks': 32, 'pairs': [{} for _ in range(9)], 'primary_pairs': [1, 4, 7]}
    for pair in range(1, 10):
        name = f'direct-lineage-{pair}'
        (tmp_path / f'{name}.result.json').write_text(json.dumps({'status': 'complete', 'plan_sha256': digest(plan)}))
        with gzip.open(tmp_path / f'{name}.hands.jsonl.gz', 'wt') as stream:
            for block in range(32):
                for rotation in (0, 1):
                    chips = -100 + block if pair in plan['primary_pairs'] else 1000 + block
                    stream.write(json.dumps({'block': block, 'rotation': rotation, 'lineage': pair,
                                             'root_seed': 12, 'target_chips': chips}) + '\n')
    result = report(plan, tmp_path)
    assert result['hands'] == 576
    assert result['primary_overall']['bb_per_100'] == -84.5
    assert result['primary_overall']['label'] == 'worse'
    assert result['overall']['bb_per_100'] > 0
    assert set(result['primary_lineages']) == {'1', '4', '7'}
