"""Describe saved arena actions without replaying policies or judging EV from wins."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path


def describe_hand(row):
    if row['status'] != 'completed' or row['mode'] != 'fixed':
        raise ValueError('Expected completed fixed-stack hands')
    hero, bb = row['rotation'], row['big_blind']
    events = row['events']
    start = events[0]
    if start['event'] != 'HandStarted' or start['player_ids'][hero] != 'player-0':
        raise ValueError('Candidate seat disagrees with the recorded lineup')
    remaining = start['stacks'][hero]
    actions, pending = [], None
    pot = 0
    for event in events:
        kind = event['event']
        if kind == 'BlindPosted':
            pot += event['amount']
            if event['seat'] == hero:
                remaining -= event['amount']
        elif kind == 'Decision' and event['seat'] == hero:
            pending = event['legal_actions']
        elif kind == 'ActionTaken':
            if event['seat'] == hero:
                if pending is None or event['paid'] > remaining:
                    raise ValueError('Missing decision bounds or impossible contribution')
                actions.append({
                    'street': event['street'], 'kind': event['action']['kind'],
                    'paid_bb': event['paid'] / bb,
                    'call_bb': pending['call_amount'] / bb,
                    'pot_before_bb': pot / bb,
                    'all_in': event['paid'] > 0 and event['paid'] == remaining,
                    'raise_to_bb': (event['action']['raise_to'] / bb
                                    if event['action']['raise_to'] is not None else None),
                })
                remaining -= event['paid']
                pending = None
            pot += event['paid']
    if row['candidate_chips'] != row['net_chips'][hero]:
        raise ValueError('Incomplete candidate hand or inconsistent payoff')
    preflop_all_in = any(a['street'] == 'preflop' and a['all_in'] for a in actions)
    all_in = any(a['all_in'] for a in actions)
    first = actions[0] if actions else None
    first_shove = (first is not None and first['street'] == 'preflop'
                   and first['kind'] == 'raise' and first['all_in'])
    return {
        'block': row['block'], 'rotation': hero, 'hand': row['hand'],
        'net_bb': row['candidate_chips'] / bb,
        'shown_hero_cards': next((e['cards'] for e in events
                                  if e['event'] == 'CardsShown' and e['seat'] == hero), None),
        'actions': actions,
        'first_action_shove': first_shove,
        'first_action_shove_facing_at_most_one_bb': first_shove and first['call_bb'] <= 1,
        'commitment': 'preflop_all_in' if preflop_all_in else 'later_all_in' if all_in else 'never_all_in',
    }


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['arm']].append(describe_hand(row))
    summary = {}
    for arm, hands in groups.items():
        counts = Counter((a['street'], a['kind'], a['all_in']) for h in hands for a in h['actions'])
        categories = {}
        for category in ('preflop_all_in', 'later_all_in', 'never_all_in'):
            members = [h for h in hands if h['commitment'] == category]
            net = sum(h['net_bb'] for h in members)
            categories[category] = {
                'hands': len(members), 'fraction': len(members) / len(hands),
                'net_bb': net, 'contribution_bb_per_100_all_hands': 100 * net / len(hands),
            }
        summary[arm] = {
            'hands': len(hands),
            'profit_bb_per_100': 100 * sum(h['net_bb'] for h in hands) / len(hands),
            'first_action_shoves': sum(h['first_action_shove'] for h in hands),
            'first_action_shoves_facing_at_most_one_bb': sum(h['first_action_shove_facing_at_most_one_bb'] for h in hands),
            'actions': [{'street': s, 'kind': k, 'all_in': all_in, 'count': count}
                        for (s, k, all_in), count in sorted(counts.items())],
            'commitment': categories,
            # This roster illustrates recorded behavior; selecting a losing hand
            # would invite hindsight to stand in for a counterfactual action value.
            'first_shove_examples': [h for h in hands if h['first_action_shove']][:5],
            'low_unpaired_shown_shove_examples': [
                h for h in hands if h['first_action_shove'] and h['shown_hero_cards']
                and h['shown_hero_cards'][0][0] != h['shown_hero_cards'][1][0]
                and all(c[0] not in 'AKQ' for c in h['shown_hero_cards'])
            ][:3],
        }
    return summary


def run(inputs, output):
    result = {
        'format': 'saved-play-description-v1',
        'interpretation': 'Descriptive saved outcomes. Commitment groups are selected by policy actions; their returns are not causal effects or action EVs. No confidence intervals treat seat rotations as independent. Shown-card examples are a showdown-selected subset, not a population estimate of dealt holdings.',
        'inputs': [],
    }
    for path in inputs:
        raw = path.read_bytes()
        result['inputs'].append({
            'path': str(path), 'sha256': sha256(raw).hexdigest(),
            'summary': summarize(json.loads(raw)),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.out)


if __name__ == '__main__':
    main()
