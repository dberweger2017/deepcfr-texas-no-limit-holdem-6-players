"""Analysis of a verified Slumbot journal: results, all-in equity, endings and decisions.

All-in adjustment: when both players are all-in before the river, the remaining board is pure
luck. Counting the client's exact equity share of the pot instead of the actual runout removes
that luck without changing the expected result. Equity is exact, over every possible runout;
river all-ins have no runout and keep their result.
"""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import math
from pathlib import Path
import statistics

from src.arena.external.journal import verify_journal
from src.arena.external.session import ending
from src.arena.external.slumbot import TOKEN, verify_record
from src.game.showdown import hand_value

STACK = 20000
DECK = [r + s for r in '23456789TJQKA' for s in 'cdhs']
STREETS = ('preflop', 'flop', 'turn', 'river')


def allin_street(action: str) -> int | None:
    """The street on which both players were all-in, or None for folds and hands that weren't."""
    if action.endswith('f'):
        return None
    paid, street_bet, actor, street = [100, 50], [100, 50], 1, 0
    for token in TOKEN.findall(action):
        if token == '/':
            street_bet, actor, street = [0, 0], 0, street + 1
            continue
        if token == 'c':
            paid[actor] += street_bet[1 - actor] - street_bet[actor]
            street_bet[actor] = street_bet[1 - actor]
        elif token != 'k':
            paid[actor] += int(token[1:]) - street_bet[actor]
            street_bet[actor] = int(token[1:])
        if paid[0] == paid[1] == STACK:
            return street
        actor = 1 - actor
    return None


def allin_equity(hole, bot, known_board) -> float:
    """The client's exact share of the pot over every runout of the remaining board."""
    dead = set(hole) | set(bot) | set(known_board)
    rest = [c for c in DECK if c not in dead]
    share = total = 0
    for runout in itertools.combinations(rest, 5 - len(known_board)):
        board = tuple(known_board) + runout
        mine, theirs = hand_value(tuple(hole) + board), hand_value(tuple(bot) + board)
        share += 2 if mine > theirs else 1 if mine == theirs else 0
        total += 2
    return share / total


def _equity_job(args):
    return allin_equity(*args)


def interval(values):
    n = len(values)
    if n < 2:
        return {'mean': statistics.fmean(values) if values else None, 'half_width_95': None, 'n': n}
    return {'mean': statistics.fmean(values), 'half_width_95': 1.96 * statistics.stdev(values) / math.sqrt(n), 'n': n}


def bet_bucket(view) -> str:
    """Slumbot's bet or raise as a fraction of the pot before it, from the client's view."""
    call = view.legal_actions.call_amount
    own = next(p for p in view.players if p.seat == view.seat)
    if call >= own.stack:
        return 'all-in'
    fraction = call / max(1, view.pot - call)
    for bound, name in ((0.55, '≤½ pot'), (0.85, '½–¾ pot'), (1.2, '~pot'), (2.05, '1.2–2× pot')):
        if fraction <= bound:
            return name
    return '>2× pot'


def response(selected: str, view) -> str:
    if selected == 'f':
        return 'fold'
    if selected in ('k', 'c'):
        return {'k': 'check', 'c': 'call'}[selected]
    return 'all-in' if int(selected[1:]) == view.legal_actions.max_raise_to else 'raise'


def analyze(journal, *, model=None, workers: int = 4, missing_key_fallback=None, hands: int | None = None) -> dict:
    """One journal, or several runs of the same player pooled, each verified on its own.

    `hands` keeps only the first that many hands in play order, with the decisions before them.
    """
    if model is not None and missing_key_fallback is not None and missing_key_fallback != model.missing_key_fallback:
        raise ValueError('Model fallback differs from analysis option')
    journals = [Path(journal)] if isinstance(journal, (str, Path)) else [Path(j) for j in journal]
    checked, parts, identities = [], [], []
    decisions, terminals, failures = [], [], Counter()
    translated = Counter()
    lookup_times = []
    for path in journals:
        if hands is not None and len(terminals) == hands:
            break
        checked.append(verify_journal(path))
        part = []
        for line in path.open():
            if hands is not None and len(terminals) + len(part) == hands:
                break
            payload = json.loads(line)['payload']
            if 'identity' in payload:
                identity = payload['identity']; identities.append(identity)
                expected = missing_key_fallback if missing_key_fallback is not None else (model.missing_key_fallback if model is not None else None)
                if model is not None and identity.get('action_translation') != model.description.get('action_translation'):
                    raise ValueError('Journal translation differs from analysis model')
                if model is not None and 'model_sha256' in identity and identity['model_sha256'] != model.description['weights_sha256']:
                    raise ValueError('Journal model differs from analysis model')
                played = identity.get('missing_key_fallback', 'uniform')
                if played not in ('uniform', 'check-call') or (expected is not None and played != expected):
                    raise ValueError('Journal fallback differs from analysis option')
            elif 'decision' in payload:
                decisions.append(payload['decision'])
            elif 'terminal' in payload:
                part.append(payload['terminal'])
            elif 'failure' in payload:
                failures[payload['failure']] += 1
        terminals += part
        parts.append({'journal': str(path), 'hands': len(part), 'bb_per_100': interval([t['winnings'] for t in part])})

    if hands is not None and len(terminals) < hands:
        raise ValueError(f'Only {len(terminals)} hands, fewer than the {hands} requested')

    # All-in equity, exact; preflop runouts are many, so hands run in parallel.
    jobs = []
    for i, t in enumerate(terminals):
        street = allin_street(t['action'])
        if street is not None and street < 3:
            jobs.append((i, (t['hole_cards'], t['bot_hole_cards'], t['board'][:(0, 3, 4)[street]])))
    with ProcessPoolExecutor(workers) as pool:
        equities = dict(zip((i for i, _ in jobs), pool.map(_equity_job, [a for _, a in jobs])))
    adjusted = [round(equities[i] * 2 * STACK - STACK, 2) if i in equities else t['winnings'] for i, t in enumerate(terminals)]

    raw = [t['winnings'] for t in terminals]
    by_seat = {name: [t['winnings'] for t in terminals if t['client_pos'] == seat]
               for seat, name in ((0, 'big_blind'), (1, 'small_blind_button'))}
    by_seat_adjusted = {name: [a for a, t in zip(adjusted, terminals) if t['client_pos'] == seat]
                        for seat, name in ((0, 'big_blind'), (1, 'small_blind_button'))}

    endings = defaultdict(lambda: {'hands': 0, 'chips': 0})
    for t in terminals:
        kind = ending(t)
        street = STREETS[t['action'].count('/')] if kind != 'showdown' else (
            'all-in before river' if (allin_street(t['action']) or 3) < 3 else 'showdown')
        key = f'{kind} {street}' if kind != 'showdown' else street
        endings[key]['hands'] += 1
        endings[key]['chips'] += t['winnings']

    actions = defaultdict(Counter)
    facing = defaultdict(Counter)
    known = Counter()
    coverage = Counter()
    for record in decisions:
        view = verify_record(record)
        street = view.street.value if hasattr(view.street, 'value') else str(view.street)
        choice = response(record['selected'], view)
        actions[street][choice] += 1
        if view.legal_actions.call_amount > 0 and not (street == 'preflop' and view.pot <= 150):
            facing[bet_bucket(view)][choice] += 1
        if model is not None:
            if getattr(model, 'translation', None) is not None:
                menu, _, stored, reproduced = model.distribution_with_telemetry(view)
                info = record.get('policy_telemetry')
                stable = lambda value: json.loads(json.dumps({k: v for k, v in value.items() if k != 'lookup_seconds'}))
                if info is None or stable(info) != stable(dict(reproduced, street=street)):
                    raise ValueError('Journal translation telemetry differs from model')
                translated[street, reproduced['mode']] += 1
                lookup_times.append(info['lookup_seconds'])
            else:
                menu, _, stored = model.distribution(view)
            key = model.information_key(view, menu)
            lookup = 'missing' if not stored else 'zero_mass' if key in model.zero_mass else 'positive_mass'
            known[(street, stored)] += 1
            coverage[street, lookup] += 1
            info = record.get('policy_telemetry')
            if model.missing_key_fallback != 'uniform' and info is None:
                raise ValueError('Check-call journal is missing decision telemetry')
            applied = model.missing_key_fallback != 'uniform' and lookup == 'missing' and (getattr(model, 'translation', None) is None or reproduced['selected_key'] is None)
            if info is not None and 'fallback_applied' in info and info['fallback_applied'] != applied:
                raise ValueError('Journal fallback coverage differs from model')

    swings = sorted(zip(raw, adjusted, terminals), key=lambda x: x[0])
    show = lambda rows: [{'winnings': w, 'all_in_adjusted': a, 'action': t['action'], 'client_pos': t['client_pos'],
                          'hole': t['hole_cards'], 'slumbot': t['bot_hole_cards'], 'board': t['board']} for w, a, t in rows]
    result = {
        'missing_key_fallback': identities[0].get('missing_key_fallback', 'uniform'),
        'run_identities': identities, 'journal': checked[0] if len(checked) == 1 else checked, 'hands': len(terminals), 'decisions': len(decisions), 'failures': dict(failures),
        'bb_per_100': interval(raw), 'all_in_adjusted_bb_per_100': interval(adjusted),
        'all_in_hands': {STREETS[s]: sum(1 for t in terminals if allin_street(t['action']) == s) for s in range(4)},
        'by_seat': {k: interval(v) for k, v in by_seat.items()},
        'by_seat_all_in_adjusted': {k: interval(v) for k, v in by_seat_adjusted.items()},
        'endings': dict(sorted(endings.items(), key=lambda kv: -kv[1]['hands'])),
        'actions_by_street': {s: dict(actions[s]) for s in STREETS if s in actions},
        'responses_by_bet_size': {b: dict(facing[b]) for b in ('≤½ pot', '½–¾ pot', '~pot', '1.2–2× pot', '>2× pot', 'all-in') if b in facing},
        'biggest_losses': show(swings[:10]), 'biggest_wins': show(swings[-5:][::-1]),
    }
    if len({json.dumps(i.get('action_translation'), sort_keys=True) for i in identities}) > 1:
        raise ValueError('Pooled journals have different translation options')
    if identities[0].get('action_translation') is not None:
        result['action_translation'] = identities[0]['action_translation']
    if lookup_times:
        lookup_times.sort()
        result['translation_by_street'] = {s: {k: translated[s, k] for k in ('exact', 'translated', 'uniform')} for s in STREETS}
        result['lookup_seconds'] = {'median': statistics.median(lookup_times), 'p90': lookup_times[int(.9*len(lookup_times))], 'max': max(lookup_times)}
    if len(parts) > 1:
        result['parts'] = parts
    if model is not None:
        result['missing_key_fallback'] = model.missing_key_fallback
        result['coverage_by_street'] = {s: {k: coverage[s, k] for k in ('missing', 'zero_mass', 'positive_mass')} for s in STREETS}
        result['fallback_decisions_by_street'] = {s: coverage[s, 'missing'] if model.missing_key_fallback == 'check-call' else 0 for s in STREETS}
        result['stored_policy_by_street'] = {s: {'stored': known[(s, True)], ('uniform_fallback' if model.missing_key_fallback == 'uniform' else 'check_call_fallback'): known[(s, False)]}
                                             for s in STREETS if known[(s, True)] + known[(s, False)]}
    if model is not None and getattr(model, 'translation', None) is not None:
        result['stored_policy_by_street'] = {s: {'exact_stored': translated[s,'exact'],
            'translated_stored': translated[s,'translated'], 'unresolved_fallback': translated[s,'uniform']}
            for s in STREETS if sum(translated[s,k] for k in ('exact','translated','uniform'))}
    return result
