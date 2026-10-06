"""Full-game conversion preserves stored probabilities and native fallback coverage."""
import pytest
from scripts.evaluate_hu20_full_export_turn_loss import groups_for_board


def test_conversion_uses_exact_full_game_rows_and_keeps_fallback_mass():
    class Source:
        entries = {'a': (('check', 'raise'), (0.25, 0.75)), 'b': (('check', 'raise'), (0.5, 0.5))}
        zero_mass = {'b'}
    compact = {'pool_keys': {'v1': {'t': {'1': 'a', '2': 'b', '3': 'c'}}},
               'tables': {'t': {'names': ['check', 'raise']}}}
    groups, counts = groups_for_board(Source(), compact, 1)
    by_key = {r['key']: r for r in groups}
    assert by_key['a']['probabilities'] == [0.25, 0.75] and by_key['a']['mass'] == 1
    assert by_key['b']['mass'] == by_key['c']['mass'] == 0
    assert by_key['c']['probabilities'] == [0.5, 0.5]
    assert counts == {'stored': 1, 'zero_mass': 1, 'missing': 1}
    compact['tables']['t']['names'] = ['fold', 'check']
    with pytest.raises(ValueError, match='menu differs'):
        groups_for_board(Source(), compact, 1)
