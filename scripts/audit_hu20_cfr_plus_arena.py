"""Independent raw-hand audit; never imports the production arena reporter."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from scipy.stats import t
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind
from src.arena.runner import public_events
from src.blueprint.abstraction import choices, information_key, HU20_UNCAPPED_SCHEMA

ARMS = 'ROCT'
CONTRASTS = [('O', 'R'), ('C', 'R'), ('O', 'C'), ('O', 'T')]
STREETS = ('preflop', 'flop', 'turn', 'river')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def stats(values):
    # A chip is .01 BB, hence mean chips/hand equals BB/100 here.
    n = len(values)
    mean = math.fsum(values) / n
    variance = math.fsum((x - mean) ** 2 for x in values) / (n - 1)
    margin = float(t.ppf(.975, n - 1)) * math.sqrt(variance / n)
    return {'blocks': n, 'bb_per_100': mean, 'ci95': [mean - margin, mean + margin] if variance else None}

def close(a, b):
    assert a['blocks'] == b['blocks']
    assert math.isclose(a['bb_per_100'], b['bb_per_100'], abs_tol=1e-9)
    assert (a['ci95'] is None) == (b['ci95'] is None)
    if a['ci95'] is not None:
        assert all(math.isclose(x, y, abs_tol=1e-9) for x, y in zip(a['ci95'], b['ci95'], strict=True))

def audit(plan_path, run, out, expected_sha256, source):
    started = time.monotonic()
    p = json.loads(plan_path.read_text())
    assert digest(p) == expected_sha256
    arms = tuple(dict.fromkeys(spec['arm'] for spec in p['models']))
    assert set(arms) in ({'R', 'O'}, set(ARMS))
    contrasts_to_check = [(a, b) for a, b in CONTRASTS if a in arms and b in arms]
    summary = json.loads((run / 'summary.json').read_text())
    outcomes = {}
    cells = defaultdict(lambda: defaultdict(Counter))
    terminal_streets = defaultdict(Counter)
    files = []
    resources = []
    replayed = decisions = 0
    for spec in p['models']:
        name = spec['name']
        result_path = run / (name + '.result.json')
        hands_path = run / (name + '.hands.jsonl.gz')
        result = json.loads(result_path.read_text())
        assert result['model'] == name and result['arm'] == spec['arm']
        assert result['status'] == 'complete' and result['failure'] is None
        assert result['plan_sha256'] == digest(p)
        assert result['source'] == source
        resources.append(result)
        expected = {(panel['name'], block, rotation) for panel in p['panels'] for block in range(panel['blocks']) for rotation in (0, 1)}
        seen = set()
        lineage = int(str(spec['seed'])[-1])
        for path in (result_path, hands_path):
            files.append({'path': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)})
        with gzip.open(hands_path, 'rt') as stream:
            for line in stream:
                r = json.loads(line)
                coordinate = (r['panel'], r['block'], r['rotation'])
                assert coordinate in expected and coordinate not in seen
                seen.add(coordinate)
                assert r['status'] == 'complete' and r['native_replay_verified'] is True
                assert r['policy'] == name and r['arm'] == spec['arm'] and r['seed'] == spec['seed'] and r['strategy'] == spec['strategy']
                assert r['root_seed'] == p['root'] and r['players'] == 2 and r['button'] == r['block'] % 2
                panel = next(panel for panel in p['panels'] if panel['name'] == r['panel'])
                assert r['contract'] == panel['contract']
                seed_payload = (2, p['root'], 'deal', (2, r['block']))
                deal_seed = (2 << 62) | (int(digest(seed_payload)[:16], 16) & (2**62 - 1))
                assert r['deal_seed'] == deal_seed
                assert r['hand_id'] == f"cfr-average/{r['panel']}/{r['block']}/{r['rotation']}"
                hand = Hand.start(Table(('seat0', 'seat1'), (2000, 2000), button=r['button']), hand_id=r['hand_id'], seed=deal_seed)
                coverage = Counter()
                for index, a in enumerate(r['actions']):
                    assert not hand.finished and hand.actor == a['seat'] and a['index'] == index
                    view = hand.observe(hand.actor)
                    assert view.street.value == a['street']
                    logical = int(hand.actor != r['rotation'])
                    assert a['logical_player'] == logical
                    o = a['observation']
                    assert o['seat'] == view.seat and o['street'] == view.street.value
                    assert o['hole_cards'] == list(view.hole_cards) and o['board'] == list(view.board)
                    assert o['pot'] == view.pot and o['call_amount'] == view.legal_actions.call_amount
                    assert o['position'] == ('button' if view.seat == view.button else 'big_blind')
                    assert o['stack'] == view.players[view.seat].stack and o['street_bet'] == view.players[view.seat].street_bet
                    if not logical:
                        key = information_key(view, choices(view, raise_cap=None, free_fold=False), schema=HU20_UNCAPPED_SCHEMA)
                        assert a['target_key'] == key
                        status = a['average_mass_status']
                        assert status in ('missing', 'zero_mass', 'positive_mass', 'current')
                        assert bool(o['trained']) == (status != 'missing')
                        probabilities = o['probabilities']
                        assert all(math.isfinite(v) and v >= 0 for v in probabilities) and math.isclose(math.fsum(probabilities), 1, abs_tol=1e-9)
                        if status in ('missing', 'zero_mass'):
                            assert all(math.isclose(v, 1 / len(probabilities), abs_tol=1e-9) for v in probabilities)
                        coverage[status] += 1
                        coverage[a['street'] + ':' + status] += 1
                        cells[(spec['arm'], lineage, r['panel'])][a['street']]['decisions'] += 1
                        cells[(spec['arm'], lineage, r['panel'])][a['street']][status] += 1
                    if 'lbr' in a:
                        telemetry = a['lbr']
                        c = cells[(spec['arm'], lineage, r['panel'])][a['street']]
                        c['lbr_decisions'] += 1
                        c['lbr_incomplete'] += not telemetry['completed']
                        c['lbr_over_soft_budget'] += telemetry['over_soft_budget']
                    action = Action(ActionKind(a['kind']), a['raise_to'])
                    view.legal_actions.validate(action)
                    hand = hand.apply(action)
                    decisions += 1
                assert dict(coverage) == r['coverage']
                assert hand.finished
                chips = [player.stack - 2000 for player in hand.observe(0).players]
                assert chips == r['net_chips_by_seat'] and sum(chips) == 0
                assert type(r['target_chips']) is int and r['target_chips'] == chips[r['rotation']]
                assert digest(public_events(hand.events)) == r['public_events_sha256']
                position = 'button' if r['rotation'] == r['button'] else 'big_blind'
                outcomes[(spec['arm'], lineage, r['panel'], r['block'], position)] = r['target_chips']
                terminal_streets[(spec['arm'], lineage, r['panel'])][r['actions'][-1]['street']] += 1
                replayed += 1
        assert seen == expected and result['hands'] == len(seen)
        print(json.dumps({'model': name, 'replayed': len(seen), 'total': replayed}), flush=True)
    assert replayed == p['expected_hands'] == summary['hands']
    contrasts = {}
    absolute = {}
    details = []
    for panel in p['panels']:
        name, blocks = panel['name'], range(panel['blocks'])
        series = {}
        for arm in arms:
            series[arm] = [math.fsum(outcomes[arm, lineage, name, block, position] for lineage in (1, 2, 3) for position in ('button', 'big_blind')) / 6 for block in blocks]
            absolute.setdefault(arm, {})[name] = stats(series[arm])
            close(absolute[arm][name], summary['absolute_bb_per_100'][arm][name])
        for a, b in contrasts_to_check:
            label = a + '-' + b
            result = stats([x - y for x, y in zip(series[a], series[b], strict=True)])
            contrasts.setdefault(label, {})[name] = result
            close(result, summary['contrasts_bb_per_100'][label][name])
            for lineage in (None, 1, 2, 3):
                lineages = (1, 2, 3) if lineage is None else (lineage,)
                for position in (None, 'button', 'big_blind'):
                    positions = ('button', 'big_blind') if position is None else (position,)
                    values = [math.fsum(outcomes[a, l, name, block, pos] - outcomes[b, l, name, block, pos] for l in lineages for pos in positions) / (len(lineages) * len(positions)) for block in blocks]
                    details.append({'contrast': label, 'panel': name, 'lineage': lineage, 'position': position, **stats(values)})
    common_blocks = min(panel['blocks'] for panel in p['panels'])
    overall_series = {arm: [math.fsum(outcomes[arm, lineage, panel['name'], block, position]
                                      for panel in p['panels'] for lineage in (1, 2, 3)
                                      for position in ('button', 'big_blind')) / (6 * len(p['panels']))
                           for block in range(common_blocks)] for arm in arms}
    overall = {'blocks': common_blocks, 'weighting': 'equal panel, common paired blocks only',
               'absolute_bb_per_100': {arm: stats(values) for arm, values in overall_series.items()},
               'contrasts_bb_per_100': {a + '-' + b: stats([x - y for x, y in zip(overall_series[a], overall_series[b], strict=True)])
                                        for a, b in contrasts_to_check}}
    primary = contrasts['O-R']
    lbr_ci = primary['lbr']['ci95']
    pressure_ci = primary['native-pressure']['ci95']
    checks = {'lbr_lower_above_0': lbr_ci is not None and lbr_ci[0] > 0,
              'native_pressure_lower_above_minus_10': pressure_ci is not None and pressure_ci[0] > -10,
              'no_severe_regression': all(primary[panel['name']]['ci95'] is None or primary[panel['name']]['ci95'][1] >= -20
                                         for panel in p['panels'] if panel['name'] not in ('lbr', 'native-pressure'))}
    assert checks == summary['release_checks'] and all(checks.values()) == summary['release_rule_passed']
    diagnostic = [{'arm': arm, 'lineage': lineage, 'panel': panel, 'streets': {street: dict(counts[street]) for street in STREETS}, 'terminal_street_hands': dict(terminal_streets[arm, lineage, panel])} for (arm, lineage, panel), counts in sorted(cells.items())]
    result = {'status': 'verified', 'plan_file_sha256': sha(plan_path), 'plan_canonical_sha256': digest(p), 'hands_replayed': replayed, 'decisions_checked': decisions, 'independent_arithmetic_matches': True, 'release_checks': checks, 'release_rule_passed': all(checks.values()), 'absolute_bb_per_100': absolute, 'contrasts_bb_per_100': contrasts, 'lineage_position_contrasts': details, 'coverage_and_lbr_by_street': diagnostic, 'raw_files': files, 'worker_results': resources, 'audit_seconds': time.monotonic() - started, 'overall_equal_panel_common_blocks': overall}
    out.write_text(json.dumps(result, sort_keys=True, indent=1, allow_nan=False) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    audit(args.plan, args.run, args.out, args.plan_sha256, args.source)
