import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('public_tables', Path('docs/reports/v055-confirmation-diagnostics/public_tables.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
hands = [
 {'client_pos': 0, 'action': 'b300c/kb300f', 'winnings': -300},
 {'client_pos': 0, 'action': 'b200b800c/b800f', 'winnings': 800},
 {'client_pos': 0, 'action': 'f', 'winnings': 50},
 {'client_pos': 0, 'action': 'ck/kb100c/kk/kk', 'winnings': -200},
 {'client_pos': 1, 'action': 'b300c/kb300b1200f', 'winnings': -600},
]
r = m.tables('fixture', hands)
assert {x['open'] for x in r['bb-opens']} == {'2 BB', '3 BB', 'fold', 'limp'}
assert next(x for x in r['bb-responses'] if x['open']=='3 BB')['response'] == 'call'
assert next(x for x in r['bb-responses'] if x['open']=='3 BB')['chips'] == -300
assert next(x for x in r['bb-responses'] if x['open']=='2 BB')['response'] == 'raise'
assert sum(x['chips'] for x in r['flop-fold-net']) == -100
assert r['reconciliation'][0]['bb_rate'] == 87.5
assert sum(x['chips'] for x in r['bb-endings'] if x['ending'].startswith('preflop:')) == 50
raise_fold = next(x for x in r['flop-folds'] if x['spot'] == 'raise')
assert (raise_fold['size'], raise_fold['position'], raise_fold['chips']) == ('~⅔–¾', 'BTN (IP)', -600)
# An all-in is a response raise and a separate size bucket, including as an open.
a = m.tables('all-in', [{'client_pos':0, 'action':'b20000f','winnings':-100}])
assert a['bb-responses'][0]['open'] == '200 BB all-in'
print('Public-action fixture checks passed: opening totals, raise sizes, positions, folds and denominators.')
