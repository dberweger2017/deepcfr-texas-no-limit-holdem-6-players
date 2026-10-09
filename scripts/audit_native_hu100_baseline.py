"""Independently replay every HU100 action/settlement and check block arithmetic."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import gzip
import json
from math import isclose, isfinite, sqrt
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

from scipy.stats import t

from src.arena.runner import public_events
from src.arena.schedule import Plan, build_schedule, canonical, digest, schedule_document, stream_seed
from src.blueprint.abstraction import HU100_SCHEMA, choices, information_key
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind
from src.policies.files import file_hash


def independent_estimate(values):
    rate = mean(values)
    interval = None
    if len(values) >= 30 and stdev(values):
        margin = float(t.ppf(.975, len(values) - 1)) * stdev(values) / sqrt(len(values))
        interval = [rate - margin, rate + margin]
    return rate, interval


def audit(root, output):
    started = perf_counter()
    inputs = json.loads((root / 'inputs.json').read_text())
    complete = json.loads((root / 'complete.json').read_text())
    if complete['status'] != 'complete':
        raise ValueError('Incomplete play cannot be audited as complete')
    files = json.loads((root / 'output-files.json').read_text())
    model_hash_seconds = 0.0
    for name, spec in files.items():
        hash_started = perf_counter()
        path = root / name
        if path.stat().st_size != spec['bytes'] or file_hash(path) != spec['sha256']:
            raise ValueError('Changed output: ' + name)
        if name.startswith('models/'):
            model_hash_seconds += perf_counter() - hash_started
    reference = Path(inputs['reference_run']) if inputs.get('reference_run') else None
    if reference and file_hash(reference / 'inputs.json') != inputs['reference_inputs_sha256']:
        raise ValueError('Reused reference identity changed')
    hands = actions = 0
    comparisons = {}
    for opponent in inputs['config']['opponents']:
        panel = root / opponent
        if reference:
            # Exact retained baseline rows are part of pairing, not another evaluation.
            for filename in ('hands.jsonl', 'decisions.jsonl.gz'):
                def baseline_rows(path):
                    opener = gzip.open if path.suffix == '.gz' else open
                    with opener(path, 'rt') as f:
                        return [json.loads(line) for line in f if json.loads(line)['arm'] == 'baseline']
                if baseline_rows(panel / filename) != baseline_rows(reference / opponent / filename):
                    raise ValueError('Reused uniform evidence differs')
        m = json.loads((panel / 'manifest.json').read_text())
        plan = Plan.from_dict(m['plan'])
        if (plan.root_seed != inputs['root'] or plan.blocks != inputs['blocks_per_opponent']
                or plan.opponents != (opponent,) or plan.baseline != 'native_hu100_uniform'
                or plan.candidate != inputs['config']['model']['name']
                or plan.scenarios[0].stacks != (10000, 10000) or m['revision'] != inputs['source']
                or canonical(json.loads((panel / 'schedule.json').read_text())) != canonical(schedule_document(plan))):
            raise ValueError('Pinned schedule/identity differs')
        blocks = {b.index: b for b in build_schedule(plan)}
        expected = {(b, r, arm) for b in blocks for r in (0, 1) for arm in ('candidate', 'baseline')}
        seen = set(); values = defaultdict(lambda: defaultdict(list)); raw_decisions = []
        with (panel / 'hands.jsonl').open() as rows, gzip.open(panel / 'decisions.jsonl.gz', 'rt') as traces:
            for line in rows:
                row = json.loads(line); coord = row['block'], row['rotation'], row['arm']
                if (coord not in expected or coord in seen or row['status'] != 'completed'
                        or row['outcome_sha256'] != digest({k: v for k, v in row.items() if k != 'outcome_sha256'})):
                    raise ValueError('Changed, failed or duplicate hand')
                seen.add(coord); block = blocks[row['block']]; rotation = row['rotation']
                ids = tuple(f'player-{(seat - rotation) % 2}' for seat in (0, 1))
                start = row['events'][0]
                hand = Hand.start(Table(ids, (10000, 10000), block.button, 50, 100, '0.01'),
                                  hand_id=f'table-0/{block.index}/{rotation}/0', seed=block.deal_seeds[0])
                if canonical(public_events(hand.events)[0]) != canonical(start):
                    raise ValueError('Hand start/deal or stacks differ')
                for event in row['events']:
                    if event['event'] != 'ActionTaken':
                        continue
                    if hand.finished or hand.actor != event['seat']:
                        raise ValueError('Replay actor differs')
                    view = hand.observe(hand.actor)
                    d = json.loads(next(traces)); raw_decisions.append(d)
                    role = int(view.player_id != 'player-0')
                    seed = stream_seed(plan.root_seed, 'test', 'action', 'pr197-hu100-v1',
                                       opponent, block.index, rotation, row['arm'], role)
                    if (d['opponent'] != opponent or d['block'] != block.index or d['rotation'] != rotation
                            or d['arm'] != row['arm'] or d['logical_player'] != role
                            or d['seed'] != seed or d['seat'] != hand.actor or d['hand_id'] != view.hand_id
                            or d['street'] != view.street.value or d['action'] != event['action']
                            or not isfinite(d['seconds']) or d['seconds'] < 0):
                        raise ValueError('Decision/stream coordinates differ')
                    action = Action(ActionKind(event['action']['kind']), event['action']['raise_to'])
                    view.legal_actions.validate(action)
                    if role == 0:
                        menu = choices(view, raise_cap=None, free_fold=False)
                        if (d['menu'] != json.loads(canonical([asdict(c) for c in menu]))
                                or d['key'] != information_key(view, menu, schema=HU100_SCHEMA)
                                or d['lookup'] not in ('positive-mass-known-key', 'zero-mass', 'missing-key')
                                or len(d['probabilities']) != len(menu)
                                or any(not isfinite(p) or p < 0 for p in d['probabilities'])
                                or not isclose(sum(d['probabilities']), 1, abs_tol=1e-8)
                                or not any(c.action == action and p > 0 for c, p in zip(menu, d['probabilities']))):
                            raise ValueError('Native menu/key/policy trace differs')
                        translation = d.get('translation', {})
                        if translation.get('mode') == 'translated':
                            if (row['arm'] != 'candidate' or d['lookup'] != 'missing-key'
                                    or not 0 < translation['states'] <= 4096
                                    or translation['selected_key'] != information_key(view, menu, schema=HU100_SCHEMA,
                                        history_label_overrides={int(i): label for i, label in translation['overrides']})):
                                raise ValueError('Translated key trace differs')
                        if row['arm'] == 'baseline' or (d['lookup'] != 'positive-mass-known-key'
                                and translation.get('mode') != 'translated'):
                            if d['probabilities'] != [1 / len(menu)] * len(menu):
                                raise ValueError('Uniform reference/fallback changed')
                    elif opponent == 'random' and action.kind == ActionKind.RAISE:
                        if action.raise_to not in {view.legal_actions.min_raise_to, view.legal_actions.max_raise_to}:
                            raise ValueError('Random min-raise/all-in behavior changed')
                    hand = hand.apply(action); actions += 1
                net = [p.stack - 10000 for p in hand.observe(0).players]
                if (not hand.finished or canonical(public_events(hand.events)) != canonical(row['events'])
                        or net != row['net_chips'] or sum(net) != 0 or row['candidate_chips'] != net[rotation]
                        or row['participants'] != list(ids)):
                    raise ValueError('Replay events/settlement differs')
                values[row['arm']][block.index].append(net[rotation]); hands += 1
            if next(traces, None) is not None or seen != expected:
                raise ValueError('Incomplete or extra hand/decision traces')
        report = json.loads((panel / 'report.json').read_text())
        computed = {arm: [100 * sum(values[arm][b]) / (2 * 100) for b in range(plan.blocks)]
                    for arm in ('candidate', 'baseline')}
        computed['paired_difference'] = [a - b for a, b in zip(computed['candidate'], computed['baseline'], strict=True)]
        recorded = report['scenarios'][opponent]['comparison']
        for arm, series in computed.items():
            rate, interval = independent_estimate(series); found = recorded[arm]
            if (found['blocks'] != plan.blocks or not isclose(rate, found['bb_per_100'], abs_tol=1e-8)
                    or (interval is None) != (found['ci95'] is None)
                    or interval is not None and any(not isclose(a, b, abs_tol=1e-8) for a, b in zip(interval, found['ci95'], strict=True))):
                raise ValueError('Independent block-based arithmetic differs')
        # Recount coverage/action frequencies without invoking production diagnostics.
        for g in report['diagnostics']:
            selected = [d for d in raw_decisions if (d['arm'], d['logical_player'], d['street']) ==
                        (g['arm'], g['logical_player'], g['street'])]
            if len(selected) != g['decisions']:
                raise ValueError('Decision rate denominator differs')
            for k, n in g['actions'].items():
                if n != sum(d['action']['kind'] == k for d in selected) or g['action_rates'][k] != n / len(selected):
                    raise ValueError('Action frequency differs')
            if g['lookup_rates'] is not None:
                for k, rate in g['lookup_rates'].items():
                    if rate != sum(d['lookup'] == k for d in selected) / len(selected):
                        raise ValueError('Lookup decision rate differs')
        comparisons[opponent] = {arm: recorded[arm] for arm in computed}
    if hands != complete['hands'] or actions != complete['decisions']:
        raise ValueError('Incomplete totals')
    result = {'status': 'verified', 'hands_replayed': hands, 'actions_replayed': actions,
              'all_settlements_replayed': True, 'independent_block_arithmetic': True,
              'coverage_and_action_rates_recounted': True, 'source': inputs['source'],
              'seconds': perf_counter() - started, 'model_hash_seconds': model_hash_seconds, 'panels': comparisons}
    with output.open('x') as f:
        f.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    result = audit(a.run, a.out)
    print(json.dumps({k: v for k, v in result.items() if k != 'panels'}))
