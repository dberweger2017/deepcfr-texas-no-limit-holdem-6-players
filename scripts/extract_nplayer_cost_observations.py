"""Select outcome-independent acting-seat observations from a cost-only trace."""

import argparse
import hashlib
import json
from pathlib import Path


def acting_observation(hand, index):
    actor = hand['actors'][index]
    state = hand['states'][index]
    if state['actor'] != actor or state['finished']:
        raise ValueError('Expected a live acting-seat state')
    # Explicit fields prevent replay-only cards, future boards, decks and
    # rewards from entering the search adapter's eventual input.
    return dict(actor=actor, hole=state['holes'][actor], board=state['board'],
                street=state['street'], pot=state['pot'], wager=state['wager'],
                min_raise=state['min_raise'], players=state['players'],
                legal=state['legal'], starting_stacks=hand['stacks'],
                button=hand['button'], small_blind=hand['sb'], big_blind=hand['bb'],
                public_actions=[dict(street=hand['states'][i]['street'],
                    actor=hand['actors'][i], action=hand['actions'][i])
                    for i in range(index)])


def extract(trace, per_stratum):
    if not 1 <= per_stratum <= 64:
        raise ValueError('per_stratum must be 1..64')
    groups = {}
    opportunities = {}
    with trace.open() as stream:
        header = json.loads(next(stream))
        if header['format'] != 'nplayer-privileged-policy-replay-v1' or len(header['policies']) != 3:
            raise ValueError('Expected the three-player native replay trace')
        for line in stream:
            hand = json.loads(line)
            for index in range(len(hand['actors'])):
                observation = acting_observation(hand, index)
                if observation['street'] == 'preflop':
                    continue
                unfolded = sum(not player[3] for player in observation['players'])
                key = f"{observation['street']}/{unfolded}-unfolded"
                opportunities[key] = opportunities.get(key, 0) + 1
                encoded = json.dumps(observation, sort_keys=True, separators=(',', ':')).encode()
                identity = hashlib.sha256(encoded).hexdigest()
                group = groups.setdefault(key, {})
                group.setdefault(identity, dict(observation_sha256=identity,
                    source_hand=hand['hand'], source_decision=index,
                    observation=observation))
                if len(group) > per_stratum:
                    del group[max(group)]
    rows = [dict(stratum=key, **groups[key][identity])
            for key in sorted(groups) for identity in sorted(groups[key])]
    return dict(format='nplayer-cost-observations-v1',
        purpose='adapter qualification and cost-only search screening; no reference action values',
        selection='lowest observation SHA256 per street/unfolded-player stratum; no rewards',
        per_stratum=per_stratum, opportunities=opportunities,
        selected=len(rows), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--per-stratum', type=int, default=8)
    args = parser.parse_args()
    result = extract(args.trace, args.per_stratum)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
