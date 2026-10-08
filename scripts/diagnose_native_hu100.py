"""Read-only, full-sample HU100 diagnostics; never trains or chooses a policy."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
from hashlib import sha256
from itertools import zip_longest
import json
from math import isclose
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

from src.arena.runner import public_events
from src.arena.schedule import Plan, build_schedule, canonical, digest
from src.blueprint.abstraction import HU100_SCHEMA, _history, choices, information_key
from src.game.hand import Hand, Table, card_name
from src.game.types import Action, ActionKind
from src.policies.files import file_hash

INDEX = Path('docs/reports/native-hu100-learning-curves-artifacts/model-retrieval-index.json')
ARCHIVE = Path.home() / 'Local/Research-Cloud/PR-200-HU100-learning-curves/hu100-learning-curves-20261008.zip'
PREFIX = 'research/hu100-learning-curves/run-01/'
OPPONENTS = ('random', 'check_call', 'tight_aggressive', 'loose_aggressive', 'pot_pressure')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def rows(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        for line in f:
            yield json.loads(line)


def retrieve(out):
    """Stream only required immutable members into a new ignored working root."""
    index = json.loads(INDEX.read_text())
    if ARCHIVE.stat().st_size != index['archive_bytes'] or file_hash(ARCHIVE) != index['archive_sha256']:
        raise ValueError('Whole indexed archive differs')
    out.mkdir(parents=True, exist_ok=False)
    selected = []
    with ZipFile(ARCHIVE) as z:
        encoded = z.read(index['manifest_member'])
        raw = gzip.decompress(encoded)
        if (sha256(encoded).hexdigest() != index['manifest_member_sha256']
                or sha256(raw).hexdigest() != index['manifest_sha256']):
            raise ValueError('Indexed manifest differs')
        members = {m['path']: m for m in json.loads(raw)['members']}
        needed = {m['archive_member'] for m in index['models']}
        for name in members:
            relative = name.removeprefix(PREFIX)
            if name.startswith(PREFIX) and relative.split('/')[0] in ('pilot', 'final'):
                if Path(name).name in ('hands.jsonl', 'decisions.jsonl.gz', 'manifest.json', 'report.json'):
                    needed.add(name)
        for name in sorted(needed):
            path = out / name
            if not path.resolve().is_relative_to(out.resolve()):
                raise ValueError('Unsafe archive path')
            path.parent.mkdir(parents=True, exist_ok=True)
            h = sha256(); size = 0
            with z.open(name) as source, path.open('xb') as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk); h.update(chunk); size += len(chunk)
            spec = members[name]
            if size != spec['bytes'] or h.hexdigest() != spec['sha256']:
                raise ValueError('Selected member differs: ' + name)
            selected.append(spec)
        for m in index['models']:
            path = out / m['archive_member']
            with gzip.open(path, 'rt') as f:
                header = json.loads(f.readline())
            if (file_hash(path) != m['sha256'] or path.stat().st_size != m['bytes']
                    or header['source_checkpoint_sha256'] != m['source_checkpoint_sha256']
                    or header['checkpoint_header']['iteration'] != m['iteration']
                    or header['checkpoint_header']['native_state']['completed_nodes'] != m['actual_completed_nodes']):
                raise ValueError('Checkpoint average identity differs')
    write(out / 'retrieval.json', {'status': 'verified', 'archive': str(ARCHIVE),
          'index': index, 'selected_members': selected, 'remote_bytes_downloaded': False,
          'method': 'read-only local accepted synced archive; full ZIP and selected member SHA256',
          'owning_pr_status_at_retrieval': 'MERGED; checked live before retrieval'})


def scan_models(inputs, stage):
    """Only retain queried rows; memory does not grow with the 3.25M-entry table."""
    index = json.loads(INDEX.read_text())
    wanted = set()
    for m in index['models']:
        for op in OPPONENTS:
            p = inputs / PREFIX / stage / str(m['actual_completed_nodes']) / op / 'decisions.jsonl.gz'
            wanted.update(d['key'] for d in rows(p) if d['arm'] == 'candidate' and d['logical_player'] == 0)
    found = {}; present = set(); started = perf_counter()
    for m in index['models']:
        table = {}; count = 0
        stream = rows(inputs / m['archive_member']); next(stream)
        for row in stream:
            count += 1
            if row[0] in wanted:
                if row[0] in table:
                    raise ValueError('Duplicate queried model key')
                table[row[0]] = row[1:]
                present.add(row[0])
        if count != m['entries']:
            raise ValueError('Checkpoint entry count differs')
        found[m['actual_completed_nodes']] = table
    return found, present, perf_counter() - started


def signature(view, menu):
    return (_history(view), (view.seat - view.button) % 2, view.street.value,
            tuple((view.players[(view.button + p) % 2].folded,
                   view.players[(view.button + p) % 2].all_in) for p in range(2)),
            tuple(c.name for c in menu))


def support_witness(initial, target, *, max_states=5000, deadline=float('inf')):
    """Exhaust the finite menu tree matching tokens, rather than equating sizes."""
    expected = target[0]
    todo = [(initial, [])]; examined = 0
    while todo:
        if examined >= max_states or perf_counter() >= deadline:
            return {'support': 'unresolved', 'examined': examined, 'reason': 'bounded-search-cap'}
        hand, path = todo.pop(); examined += 1
        if hand.finished:
            continue
        view = hand.observe(hand.actor); menu = choices(view, raise_cap=None, free_fold=False)
        tokens = _history(view)
        if len(tokens) > len(expected) or tokens != expected[:len(tokens)]:
            continue
        if len(tokens) == len(expected):
            if signature(view, menu) == target:
                return {'support': 'supported', 'examined': examined, 'witness_actions': path}
            continue
        for c in menu:
            todo.append((hand.apply(c.action), path + [asdict(c.action)]))
    return {'support': 'unsupported-abstract-history-or-menu', 'examined': examined,
            'reason': 'exhaustive-supported-token-tree'}


def band(value, cuts, names):
    return names[sum(value >= c for c in cuts)]


def joined(d, table):
    stored = table.get(d['key'])
    expected = 'missing-key' if stored is None else 'zero-mass' if not stored[2] else 'positive-mass-known-key'
    if expected != d['lookup']:
        raise ValueError('Lookup classification differs from checkpoint')
    if stored is None:
        p = [1 / len(d['menu'])] * len(d['menu']); mass = visits = None
    else:
        names, p, mass, visits = stored
        if names != [c['name'] for c in d['menu']]:
            raise ValueError('Stored menu names differ')
    if len(p) != len(d['probabilities']) or any(not isclose(a, b, rel_tol=0, abs_tol=1e-12)
                                               for a, b in zip(p, d['probabilities'], strict=True)):
        raise ValueError('Recorded probabilities differ from checkpoint')
    return mass, visits


class Cell:
    def __init__(self):
        self.n = self.net = self.wins = self.losses = 0
        self.sums = Counter(); self.valid = Counter(); self.hands = set()

    def add(self, row):
        self.n += 1; self.net += row['net_bb']
        self.wins += row['net_bb'] > 0; self.losses += row['net_bb'] < 0
        self.hands.add((row['block'], row['rotation']))
        for name in ('fold_probability', 'passive_probability', 'raise_probability', 'mass', 'visits'):
            if row[name] is not None:
                self.sums[name] += row[name]; self.valid[name] += 1

    def result(self):
        return {'decisions': self.n, 'distinct_hands': len(self.hands), 'wins': self.wins,
                'losses': self.losses, 'ties': self.n - self.wins - self.losses,
                'mean_final_net_bb_per_decision': self.net / self.n,
                'covered_decisions': self.valid['mass'],
                **{'mean_' + k: self.sums[k] / self.valid[k] if self.valid[k] else None
                   for k in ('fold_probability', 'passive_probability', 'raise_probability', 'mass', 'visits')}}


def replay(inputs, out, stage, *, search_seconds=120):
    out.mkdir(parents=True, exist_ok=False)
    started = perf_counter(); tables, present, scans = scan_models(inputs, stage)
    cache = {}; search_used = 0.; total_hands = total_actions = total_target = 0
    results = []; representatives = {}; all_cells = []
    index = json.loads(INDEX.read_text())
    with (out / 'native-fixtures.jsonl').open('x') as fixtures, gzip.open(out / 'features.jsonl.gz', 'wt') as features, (out / 'hands.jsonl').open('x') as hand_rows:
        for m in index['models']:
            nodes = m['actual_completed_nodes']
            for op in OPPONENTS:
                panel = inputs / PREFIX / stage / str(nodes) / op
                manifest = json.loads((panel / 'manifest.json').read_text()); plan = Plan.from_dict(manifest['plan'])
                blocks = {b.index: b for b in build_schedule(plan)}
                seen = set(); misses = Counter(); offmenu = Counter(); exposures = defaultdict(lambda: [0, 0.])
                cells = defaultdict(Cell); target_n = 0; raw_net = 0; support_totals = Counter()
                decisions = iter(rows(panel / 'decisions.jsonl.gz'))
                for row in rows(panel / 'hands.jsonl'):
                    events = [e for e in row['events'] if e['event'] == 'ActionTaken']
                    trace = [next(decisions) for _ in events]
                    if row['arm'] != 'candidate':
                        continue
                    coord = row['block'], row['rotation']
                    if (coord in seen or row['status'] != 'completed'
                            or row['outcome_sha256'] != digest({k: v for k, v in row.items() if k != 'outcome_sha256'})):
                        raise ValueError('Duplicate, failed or changed hand')
                    seen.add(coord); b = blocks[coord[0]]; rotation = coord[1]
                    ids = tuple(f'player-{(s - rotation) % 2}' for s in (0, 1))
                    table = Table(ids, (10000, 10000), b.button, 50, 100, '0.01')
                    initial = Hand.start(table, hand_id=f'table-0/{b.index}/{rotation}/0', seed=b.deal_seeds[0])
                    # The simulator's holes/deck are used only in offline replay fixtures.
                    holes = initial._state.players_state
                    first = 1 - b.button
                    deck = [card_name(holes[s].hand[r]) for r in (0, 1) for s in (first, b.button)]
                    deck += [card_name(c) for c in initial._state.deck]
                    hand = initial; fixture_d = []; fixture_a = []; own_rows = []
                    prior_off = False; opponent_off = False
                    net_bb = row['candidate_chips'] / 100
                    for event, d in zip(events, trace, strict=True):
                        view = hand.observe(hand.actor); legal = view.legal_actions
                        menu = choices(view, raise_cap=None, free_fold=False)
                        action = Action(ActionKind(event['action']['kind']), event['action']['raise_to'])
                        key = information_key(view, menu, schema=HU100_SCHEMA)
                        role = int(view.player_id != 'player-0')
                        if (d['action'] != event['action'] or d['block'] != coord[0] or d['rotation'] != rotation
                                or d['arm'] != 'candidate' or d['logical_player'] != role
                                or d['seat'] != hand.actor or d['street'] != view.street.value
                                or d['hand_id'] != view.hand_id or d['opponent'] != op):
                            raise ValueError('Decision coordinates differ')
                        fixture_d.append({'actor': hand.actor, 'street': view.street.value, 'pot': view.pot,
                             'kinds': [k.value for k in legal.kinds], 'call': legal.call_amount,
                             'min_raise_to': legal.min_raise_to, 'max_raise_to': legal.max_raise_to,
                             'menu': [[c.name, c.action.raise_to] for c in menu], 'key': key})
                        fixture_a.append([action.kind.value, action.raise_to])
                        is_off = action not in [c.action for c in menu]
                        if role == 0:
                            if d['key'] != key or d['menu'] != json.loads(canonical([asdict(c) for c in menu])) or is_off:
                                raise ValueError('Python key/menu/action differs')
                            mass, visits = joined(d, tables[nodes])
                            support = 'stored-training-witness'
                            proof = None
                            if d['lookup'] == 'missing-key':
                                misses[view.street.value] += 1
                                if not prior_off:
                                    support = 'supported-observed-menu-path'
                                elif key in present:
                                    support = 'supported-other-checkpoint'
                                else:
                                    sig = signature(view, menu)
                                    if sig not in cache:
                                        tick = perf_counter()
                                        cache[sig] = support_witness(initial, sig,
                                            deadline=tick + max(0, search_seconds - search_used))
                                        search_used += perf_counter() - tick
                                    proof = cache[sig]; support = proof['support']
                                support_totals[support] += 1
                            p = d['probabilities']
                            feature = {'nodes': nodes, 'opponent': op, 'block': coord[0], 'rotation': rotation,
                                'decision': len(fixture_a) - 1, 'key': key, 'street': view.street.value,
                                'position': 'BTN/SB' if view.seat == view.button else 'BB',
                                'pot_bb': view.pot / 100, 'lookup': d['lookup'], 'support': support,
                                'prior_off_menu': prior_off, 'prior_opponent_off_menu': opponent_off,
                                'net_bb': net_bb, 'mass': mass, 'visits': visits,
                                'fold_probability': sum(x for c, x in zip(menu, p) if c.action.kind == ActionKind.FOLD),
                                'passive_probability': sum(x for c, x in zip(menu, p) if c.action.kind in (ActionKind.CALL, ActionKind.CHECK)),
                                'raise_probability': sum(x for c, x in zip(menu, p) if c.action.kind == ActionKind.RAISE)}
                            feature['pot_band'] = band(feature['pot_bb'], (4, 16, 64), ('<4', '4-16', '16-64', '64+'))
                            feature['visit_band'] = 'missing' if visits is None else band(visits, (1, 2, 10, 100), ('0', '1', '2-9', '10-99', '100+'))
                            feature['mass_band'] = 'missing' if mass is None else band(mass, (1e-300, 100, 10000), ('0', '(0,100)', '100-10k', '10k+'))
                            features.write(json.dumps(feature, separators=(',', ':')) + '\n')
                            for dims in (('street', 'position', 'pot_band', 'lookup'), ('street', 'visit_band'), ('street', 'mass_band'), ('lookup',), ('street',)):
                                cell_key = dims, tuple(feature[k] for k in dims)
                                cells[cell_key].add(feature)
                            own_rows.append(feature); target_n += 1
                            rep_key = (nodes, op, support if d['lookup'] == 'missing-key' else d['lookup'], net_bb < 0)
                            if rep_key not in representatives:
                                representatives[rep_key] = {'feature': feature, 'proof': proof, 'events': row['events'],
                                    'trace': trace, 'deck': deck, 'button': b.button,
                                    'target_signature': json.loads(canonical(signature(view, menu)))}
                        offmenu['opponent' if role else 'target'] += is_off
                        prior_off |= is_off; opponent_off |= is_off and bool(role)
                        hand = hand.apply(action); total_actions += 1
                    if (not hand.finished or canonical(public_events(hand.events)) != canonical(row['events'])
                            or list(hand.events[-1].stacks) != [10000 + n for n in row['net_chips']]
                            or row['candidate_chips'] != row['net_chips'][rotation] or sum(row['net_chips']) != 0):
                        raise ValueError('Action/event/settlement replay differs')
                    fixture = {'stack_bb': 100, 'seed': b.deal_seeds[0], 'button': b.button, 'deck': deck,
                               'actions': fixture_a, 'decisions': fixture_d, 'final_stacks': list(hand.events[-1].stacks)}
                    fixtures.write(json.dumps(fixture, separators=(',', ':')) + '\n')
                    exposure = ('no-target-decision' if not own_rows else 'ever-missing' if any(x['lookup'] == 'missing-key' for x in own_rows)
                                else 'any-zero-no-missing' if any(x['lookup'] == 'zero-mass' for x in own_rows) else 'all-positive')
                    hand_rows.write(json.dumps({'nodes': nodes, 'opponent': op, 'block': coord[0],
                        'rotation': rotation, 'net_bb': net_bb, 'exposure': exposure}) + '\n')
                    exposures[exposure][0] += 1; exposures[exposure][1] += net_bb
                    raw_net += net_bb; total_hands += 1
                if next(decisions, None) is not None or seen != {(b, r) for b in blocks for r in (0, 1)}:
                    raise ValueError('Incomplete candidate sample/trace')
                total_target += target_n
                n = len(seen); report = json.loads((panel / 'report.json').read_text())
                recorded_rate = report['scenarios'][op]['comparison']['candidate']['bb_per_100']
                if not isclose(100 * raw_net / n, recorded_rate, abs_tol=1e-9):
                    raise ValueError('Archived panel arithmetic differs')
                result = {'nodes': nodes, 'opponent': op, 'hands': n, 'target_decisions': target_n,
                    'bb_per_100': 100 * raw_net / n, 'missing_by_street': dict(misses),
                    'missing_support': dict(support_totals), 'off_menu_actions': dict(offmenu),
                    'hand_exposures': {k: {'hands': v[0], 'net_bb': v[1], 'mean_net_bb': v[1] / v[0],
                                         'contribution_bb_per_100': 100 * v[1] / n} for k, v in exposures.items()}}
                results.append(result)
                for (dims, values), cell in sorted(cells.items()):
                    all_cells.append({'nodes': nodes, 'opponent': op, 'dimensions': list(dims),
                                      **dict(zip(dims, values)), **cell.result()})
                write(out / f'panel-{nodes}-{op}.json', result)
                print(f'completed {stage} {nodes} {op}: {n} hands', flush=True)
    result = {'status': 'complete', 'stage': stage, 'hands': total_hands, 'actions': total_actions,
              'target_decisions': total_target, 'model_scan_seconds': scans,
              'seconds': perf_counter() - started, 'search_seconds': search_used,
              'search_signatures': len(cache), 'panels': results}
    write(out / 'summary.json', result); write(out / 'cells.json', all_cells)
    write(out / 'representatives.json', list(representatives.values()))
    write(out / 'support-searches.json', [{'signature': json.loads(canonical(k)), **v} for k, v in cache.items()])
    return result


def verify(root):
    """Independent streaming denominator, group, probability and payoff accounting."""
    summary = json.loads((root / 'summary.json').read_text())
    tally = defaultdict(Counter); by_hand = defaultdict(set); hand_nets = {}; n = 0
    dimensions = (('street', 'position', 'pot_band', 'lookup'), ('street', 'visit_band'),
                  ('street', 'mass_band'), ('lookup',), ('street',))
    recount = {}
    for r in rows(root / 'features.jsonl.gz'):
        n += 1; key = r['nodes'], r['opponent']; g = tally[key]
        g['decisions'] += 1; g[r['lookup']] += 1
        hand_key = (*key, r['block'], r['rotation']); by_hand[hand_key].add(r['lookup'])
        if hand_key in hand_nets and hand_nets[hand_key] != r['net_bb']:
            raise ValueError('Features disagree on hand payoff')
        hand_nets[hand_key] = r['net_bb']
        if r['lookup'] == 'missing-key':
            g['support:' + r['support']] += 1
        for dims in dimensions:
            k = (*key, dims, tuple(r[x] for x in dims))
            a = recount.setdefault(k, {'n': 0, 'wins': 0, 'losses': 0, 'net': 0.,
                                      'hands': set(), 'sums': Counter(), 'valid': Counter()})
            a['n'] += 1; a['wins'] += r['net_bb'] > 0; a['losses'] += r['net_bb'] < 0
            a['net'] += r['net_bb']; a['hands'].add((r['block'], r['rotation']))
            for name in ('fold_probability', 'passive_probability', 'raise_probability', 'mass', 'visits'):
                if r[name] is not None:
                    a['sums'][name] += r[name]; a['valid'][name] += 1
        if not isclose(sum(r[x] for x in ('fold_probability', 'passive_probability', 'raise_probability')), 1, abs_tol=1e-10):
            raise ValueError('Action-kind probabilities do not sum to one')
    if n != summary['target_decisions']:
        raise ValueError('Feature denominator differs')
    exposures = defaultdict(lambda: defaultdict(lambda: [0, 0.])); hands = 0; unique = set()
    for h, fixture in zip_longest(rows(root / 'hands.jsonl'), rows(root / 'native-fixtures.jsonl')):
        if h is None or fixture is None:
            raise ValueError('Hand/native fixture cardinality differs')
        hk = (h['nodes'], h['opponent'], h['block'], h['rotation'])
        if hk in unique: raise ValueError('Duplicate hand accounting row')
        unique.add(hk); lookups = by_hand.get(hk, set())
        expected = ('no-target-decision' if not lookups else 'ever-missing' if 'missing-key' in lookups
                    else 'any-zero-no-missing' if 'zero-mass' in lookups else 'all-positive')
        payoff = (fixture['final_stacks'][h['rotation']] - 10000) / 100
        if (expected != h['exposure'] or payoff != h['net_bb']
                or hk in hand_nets and hand_nets[hk] != payoff):
            raise ValueError('Independent hand exposure/payoff differs')
        a = exposures[h['nodes'], h['opponent']][expected]; a[0] += 1; a[1] += payoff
        hands += 1
    if hands != summary['hands'] or set(by_hand) - unique:
        raise ValueError('Incomplete hand/feature accounting')
    for p in summary['panels']:
        if tally[p['nodes'], p['opponent']]['decisions'] != p['target_decisions']:
            raise ValueError('Panel target denominator differs')
        if sum(x['hands'] for x in p['hand_exposures'].values()) != p['hands']:
            raise ValueError('Hand partitions are incomplete')
        if not isclose(sum(x['contribution_bb_per_100'] for x in p['hand_exposures'].values()), p['bb_per_100'], abs_tol=1e-10):
            raise ValueError('Hand contributions do not add up')
        if sum(p['missing_support'].values()) != tally[p['nodes'], p['opponent']]['missing-key']:
            raise ValueError('Missing classifications are incomplete')
        for k, count in p['missing_support'].items():
            if tally[p['nodes'], p['opponent']]['support:' + k] != count:
                raise ValueError('Missing support cell differs')
        for k, values in exposures[p['nodes'], p['opponent']].items():
            recorded = p['hand_exposures'][k]
            if (values[0] != recorded['hands'] or not isclose(values[1], recorded['net_bb'], abs_tol=1e-9)
                    or not isclose(values[1] / values[0], recorded['mean_net_bb'], abs_tol=1e-10)
                    or not isclose(100 * values[1] / p['hands'], recorded['contribution_bb_per_100'], abs_tol=1e-10)):
                raise ValueError('Exposure cell accounting differs')
    cells = json.loads((root / 'cells.json').read_text())
    cell_keys = set()
    for c in cells:
        dims = tuple(c['dimensions']); key = (c['nodes'], c['opponent'], dims, tuple(c[x] for x in dims))
        if key in cell_keys or key not in recount: raise ValueError('Duplicate or unexpected stratum')
        cell_keys.add(key); a = recount[key]
        for name, value in {'decisions': a['n'], 'distinct_hands': len(a['hands']), 'wins': a['wins'],
                            'losses': a['losses'], 'ties': a['n'] - a['wins'] - a['losses'],
                            'covered_decisions': a['valid']['mass']}.items():
            if c[name] != value: raise ValueError('Stratum count differs: ' + name)
        expected_means = {'mean_final_net_bb_per_decision': a['net'] / a['n'],
                          **{'mean_' + k: a['sums'][k] / a['valid'][k] if a['valid'][k] else None
                             for k in ('fold_probability', 'passive_probability', 'raise_probability', 'mass', 'visits')}}
        for name, value in expected_means.items():
            if (value is None) != (c[name] is None) or value is not None and not isclose(value, c[name], rel_tol=1e-12, abs_tol=1e-10):
                raise ValueError('Stratum mean differs: ' + name)
    if cell_keys != set(recount): raise ValueError('Missing stratum')
    for p in summary['panels']:
        for dims in {tuple(c['dimensions']) for c in cells}:
            selected = [c for c in cells if c['nodes'] == p['nodes'] and c['opponent'] == p['opponent'] and tuple(c['dimensions']) == dims]
            if sum(c['decisions'] for c in selected) != p['target_decisions']:
                raise ValueError('Stratum denominator differs')
    write(root / 'verification.json', {'status': 'verified', 'feature_rows': n,
          'panels': len(summary['panels']), 'all_hand_partitions_add_up': True,
          'scope': 'independent feature/stratum counts and payoff partition arithmetic; replay performed separately'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('retrieve', 'pilot', 'analyse', 'verify'))
    p.add_argument('--inputs', type=Path); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'retrieve': retrieve(a.out)
    elif a.command == 'verify': verify(a.out)
    else: replay(a.inputs, a.out, 'pilot' if a.command == 'pilot' else 'final')


if __name__ == '__main__':
    main()
