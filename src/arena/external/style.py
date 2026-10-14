"""How a player and Slumbot played, from verified terminal records alone.

Every terminal carries the full public action string, so the same profile works for any player,
including ones without decision records (the random floor). It reports where hands ended, how
long they lasted (streets, actions, matched pot) and what each kind was worth, and for each side
the usual heads-up statistics: VPIP, PFR, small-blind limps and opens, 3-bets, flop c-bets,
aggression by street, showdowns, bet sizes and responses to the other side's bets.

Sizes are the raise beyond a call as a fraction of the pot after that call, so a half-pot bet is
0.5 and a pot-sized raise is 1.0 on any street.
"""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from src.arena.external.journal import verify_journal
from src.arena.external.slumbot import TOKEN

STACK = 20000
STREETS = ('preflop', 'flop', 'turn', 'river')
SIZES = ((0.4, '<0.4'), (0.6, '~½'), (0.85, '~⅔–¾'), (1.2, '~pot'), (2.5, '1.2–2.5×'), (math.inf, '>2.5×'))
SIZE_ORDER = [name for _, name in SIZES] + ['all-in']
POTS = ((10, '<10 BB'), (30, '10–30 BB'), (100, '30–100 BB'), (399, '100–399 BB'), (math.inf, 'all-in, 400 BB'))
SIDES = ('player', 'slumbot')


def terminals(runs: list[Path], hands: int | None = None) -> list[dict]:
    """Verified terminal records of one player's runs in play order, optionally only the first `hands`."""
    found = []
    for run in runs:
        verify_journal(run / 'journal.jsonl')
        for line in (run / 'journal.jsonl').open():
            payload = json.loads(line)['payload']
            if 'terminal' in payload:
                found.append(payload['terminal'])
    if hands is not None and len(found) < hands:
        raise ValueError(f'Only {len(found)} hands, fewer than the {hands} requested')
    return found[:hands]


def size_label(fraction: float, all_in: bool) -> str:
    return 'all-in' if all_in else next(name for bound, name in SIZES if fraction <= bound)


def replay(terminal) -> tuple[list[dict], list[int]]:
    """The hand's actions in order (street, side, kind, chips to call, size) and each seat's total paid."""
    paid, street_bet, actor, street = [100, 50], [100, 50], 1, 0
    events = []
    for token in TOKEN.findall(terminal['action']):
        if token == '/':
            street_bet, actor, street = [0, 0], 0, street + 1
            continue
        call = street_bet[1 - actor] - street_bet[actor]
        event = {'street': street, 'side': 'player' if actor == terminal['client_pos'] else 'slumbot', 'facing': call,
                 'kind': {'f': 'fold', 'k': 'check', 'c': 'call'}.get(token)}
        if token == 'c':
            paid[actor] += call
            street_bet[actor] = street_bet[1 - actor]
        elif event['kind'] is None:
            to = int(token[1:])
            event['kind'] = 'raise' if street_bet[1 - actor] > 0 else 'bet'
            event['size'] = (to - street_bet[1 - actor]) / (sum(paid) + call)
            paid[actor] += to - street_bet[actor]
            street_bet[actor] = to
            event['all_in'] = paid[actor] == STACK
        events.append(event)
        actor = 1 - actor
    return events, paid


def interval(values) -> dict:
    """Chips per hand as BB/100 with its 95% half-width."""
    n = len(values)
    mean = statistics.fmean(values) if values else None
    half = 1.96 * statistics.stdev(values) / math.sqrt(n) if n > 1 else None
    return {'hands': n, 'bb_per_100': mean, 'half_width_95': half}


def group(values: list[int], total: int) -> dict:
    """A group of hands: its own BB/100, its share of hands and what it adds to the overall BB/100.

    The added amount's interval treats every other hand as adding zero, so the groups' additions
    sum to the overall result.
    """
    adds = sum(values) / total
    variance = (sum(v * v for v in values) - total * adds * adds) / (total - 1) if total > 1 else None
    return {**interval(values), 'share': len(values) / total, 'adds_bb_per_100': adds,
            'adds_half_width_95': 1.96 * math.sqrt(variance / total) if variance is not None else None}


def rate(hits: int, chances: int) -> dict:
    return {'rate': hits / chances if chances else None, 'of': chances}


def profile(terminals: list[dict]) -> dict:
    if not terminals:
        raise ValueError('No hands to profile')
    total = len(terminals)
    endings, lengths, pots, actions = defaultdict(list), defaultdict(list), defaultdict(list), []
    count = {side: Counter() for side in SIDES}
    verbs = {side: defaultdict(Counter) for side in SIDES}
    sizes = {side: defaultdict(Counter) for side in SIDES}
    answers = {side: defaultdict(Counter) for side in SIDES}

    for t in terminals:
        events, paid = replay(t)
        won = t['winnings']
        folded = events[-1]['kind'] == 'fold'
        last_street = events[-1]['street']
        saw_flop = '/' in t['action']
        actions.append(len(events))
        if folded:
            who = 'player' if events[-1]['side'] == 'player' else 'slumbot'
            endings[f'{who} folded {STREETS[last_street]}'].append(won)
            lengths[f'ended {STREETS[last_street]}'].append(won)
        else:
            endings['showdown'].append(won)
            lengths['showdown'].append(won)
        matched = 2 * min(paid) / 100
        pots[next(name for bound, name in POTS if matched <= bound)].append(won)

        # Preflop: VPIP, PFR, the small blind's first action, 3-bets and folds to them.
        raises, opener = 0, None
        hand = {side: Counter() for side in SIDES}
        for e in (e for e in events if e['street'] == 0):
            h = hand[e['side']]
            h['vpip'] |= e['kind'] == 'raise' or (e['kind'] == 'call' and e['facing'] > 0)
            h['pfr'] |= e['kind'] == 'raise'
            if raises == 1 and e['facing'] > 0 and e['side'] != opener:
                h['3bet_chance'] = 1
                h['3bet'] |= e['kind'] == 'raise'
            if raises == 2 and e['facing'] > 0 and e['side'] == opener:
                h['face_3bet'] = 1
                h['fold_3bet'] |= e['kind'] == 'fold'
            if e['kind'] == 'raise':
                raises += 1
                opener = opener or e['side']
                aggressor = e['side']
        sb = events[0]['side']
        count[sb]['sb_hands'] += 1
        count[sb][f'sb_{events[0]["kind"]}'] += 1

        # Flop continuation bets by the last preflop raiser, and the answers to them.
        flop = [e for e in events if e['street'] == 1]
        if flop and raises:
            first = next((i for i, e in enumerate(flop) if e['side'] == aggressor), None)
            if first is not None and flop[first]['facing'] == 0:
                count[aggressor]['cbet_chance'] += 1
                if flop[first]['kind'] == 'bet':
                    count[aggressor]['cbet'] += 1
                    if first + 1 < len(flop):
                        defender = flop[first + 1]
                        count[defender['side']]['face_cbet'] += 1
                        count[defender['side']][f'cbet_{defender["kind"]}'] += 1

        for side in SIDES:
            c = count[side]
            c['hands'] += 1
            c.update(hand[side])
            if saw_flop:
                c['saw_flop'] += 1
                c['won_after_flop'] += (won if side == 'player' else -won) > 0
                if not folded:
                    c['showdown'] += 1
                    c['won_showdown'] += (won if side == 'player' else -won) > 0

        for e in events:
            verbs[e['side']][STREETS[e['street']]][e['kind']] += 1
            if e['kind'] in ('bet', 'raise'):
                sizes[e['side']][STREETS[e['street']]][size_label(e['size'], e['all_in'])] += 1
        for bet, answer in zip(events, events[1:]):
            if bet['kind'] in ('bet', 'raise') and answer['street'] == bet['street']:
                answers[answer['side']][STREETS[bet['street']], size_label(bet['size'], bet['all_in'])][answer['kind']] += 1

    def side(name):
        c = count[name]
        by_street = {}
        for street in STREETS:
            v = verbs[name][street]
            aggressive, n = v['bet'] + v['raise'], sum(v.values())
            by_street[street] = {**{k: v[k] for k in ('fold', 'check', 'call', 'bet', 'raise')},
                                 'aggression_factor': aggressive / v['call'] if v['call'] else None,
                                 'aggression_frequency': aggressive / n if n else None}
        return {
            'vpip': rate(c['vpip'], c['hands']), 'pfr': rate(c['pfr'], c['hands']),
            'small_blind_first_action': {k: rate(c[f'sb_{k}'], c['sb_hands']) for k in ('fold', 'call', 'raise')},
            '3bet': rate(c['3bet'], c['3bet_chance']), 'fold_to_3bet': rate(c['fold_3bet'], c['face_3bet']),
            'flop_cbet': rate(c['cbet'], c['cbet_chance']),
            'against_flop_cbet': {k: rate(c[f'cbet_{k}'], c['face_cbet']) for k in ('fold', 'call', 'raise')},
            'saw_flop': rate(c['saw_flop'], c['hands']),
            'showdown_after_flop': rate(c['showdown'], c['saw_flop']),
            'won_at_showdown': rate(c['won_showdown'], c['showdown']),
            'won_after_flop': rate(c['won_after_flop'], c['saw_flop']),
            'by_street': by_street,
            'bet_sizes': {s: {k: sizes[name][s][k] for k in SIZE_ORDER if sizes[name][s][k]} for s in STREETS if sizes[name][s]},
            'answers_to_bets': {f'{s} {k}': dict(answers[name][s, k]) for s in STREETS for k in SIZE_ORDER if answers[name][s, k]},
        }

    return {
        'hands': total,
        'overall': interval([t['winnings'] for t in terminals]),
        'endings': {k: group(v, total) for k, v in sorted(endings.items(), key=lambda kv: -len(kv[1]))},
        'streets_reached': {k: group(lengths[k], total) for k in [f'ended {s}' for s in STREETS] + ['showdown'] if lengths[k]},
        'matched_pot': {name: group(pots[name], total) for _, name in POTS if pots[name]},
        'actions_per_hand': {'mean': statistics.fmean(actions), 'median': statistics.median(actions), 'max': max(actions)},
        'player': side('player'),
        'slumbot': side('slumbot'),
    }
