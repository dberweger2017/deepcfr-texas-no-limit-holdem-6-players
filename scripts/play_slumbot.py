"""Play bounded live hands against Slumbot's public API (200BB, blinds 50/100).

    python -m scripts.play_slumbot --hands 20 --out DIR
    python -m scripts.play_slumbot --hands 1000 --model AVERAGE.gz --model-sha256 SHA --out DIR
    python -m scripts.play_slumbot ... --login        # account session; asks for the password

The default player is check/call. A trained HU200 average plays with its stored policy and
uniform play where it has none. Each run writes a fresh hash-chained journal of every decision
and verified settlement, plus `summary.json`. Any failure stops the run; nothing is retried.
With --login the username comes from SLUMBOT_USERNAME or a prompt and the password from
SLUMBOT_PASSWORD or a hidden prompt; neither is stored or printed.
"""

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import time

from src.arena.external.http import AmbiguousRequest, HttpTransport, ServiceError
from src.arena.external.journal import PrefixJournal, verify_journal
from src.arena.external.session import CheckCall, play_hand, summarize
from src.arena.external.slumbot import CONTRACT, SlumbotConnection

SERVICE = 'https://slumbot.com'


def load_player(args):
    if args.model is None:
        player = CheckCall()
        return player, player, {'player': player.name}
    from src.blueprint.abstraction import HU200_SCHEMA
    from src.blueprint.average import AveragePolicy
    model = AveragePolicy(args.model, args.model_sha256, expected_schema=HU200_SCHEMA)
    return model, model.policy(args.seed), {'player': 'hu200-average', 'model_sha256': args.model_sha256,
                                            'model_bytes': args.model.stat().st_size, 'policy_seed': args.seed,
                                            'description': model.description}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hands', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--model-sha256')
    parser.add_argument('--seed', type=int, default=1, help='policy sampling seed')
    parser.add_argument('--max-requests', type=int, default=None, help='default: 60 per hand')
    parser.add_argument('--max-seconds', type=float, default=None)
    parser.add_argument('--min-interval', type=float, default=0.5, help='seconds between requests')
    parser.add_argument('--timeout', type=float, default=10.0)
    parser.add_argument('--login', action='store_true')
    args = parser.parse_args()
    if args.hands < 1 or (args.model is None) != (args.model_sha256 is None):
        parser.error('Give a positive --hands, and --model together with --model-sha256')
    args.out.mkdir(parents=True)
    policy, player, identity = load_player(args)
    transport = HttpTransport(SERVICE, timeout=args.timeout, min_interval=args.min_interval,
                              max_requests=args.max_requests or 60 * args.hands)
    connection = SlumbotConnection(transport, policy)
    if args.login:
        username = os.environ.get('SLUMBOT_USERNAME') or input('Slumbot username: ')
        connection.login(username, os.environ.get('SLUMBOT_PASSWORD') or getpass.getpass('Slumbot password: '))
    run = {'service': SERVICE, 'contract': CONTRACT.__dict__, 'account_session': args.login,
           'hands_requested': args.hands, 'max_requests': transport.max_requests,
           'min_interval_seconds': args.min_interval, 'timeout_seconds': args.timeout, **identity}
    journal = PrefixJournal(args.out / 'journal.jsonl', run)
    hands, stop, started = [], None, time.monotonic()
    for index in range(args.hands):
        if args.max_seconds is not None and time.monotonic() - started > args.max_seconds:
            stop = 'time limit'
            break
        try:
            hands.append(play_hand(connection, player, f'slumbot-{index:06d}', journal))
        except AmbiguousRequest:
            journal.failure('timeout-ambiguous'); stop = 'ambiguous request'
            break
        except ServiceError:
            journal.failure('service-error'); stop = 'service error'
            break
        except ValueError:
            journal.failure('protocol-error'); stop = 'protocol error'
            break
    journal.close()
    checked = verify_journal(args.out / 'journal.jsonl')
    latencies = sorted(transport.latencies)
    summary = {**summarize(hands), 'run': run, 'stopped': stop, 'requests': transport.requests,
               'seconds': time.monotonic() - started, 'journal': checked,
               'journal_sha256': hashlib.sha256((args.out / 'journal.jsonl').read_bytes()).hexdigest(),
               'latency_seconds': {'median': latencies[len(latencies) // 2], 'p90': latencies[int(len(latencies) * 0.9)],
                                   'max': latencies[-1]} if latencies else None}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: summary[k] for k in ('hands', 'bb_per_100', 'baseline_adjusted_bb_per_100', 'stopped', 'requests')}))
    if stop is not None and stop != 'time limit':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
