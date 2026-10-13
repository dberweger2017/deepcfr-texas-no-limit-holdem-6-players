"""Play bounded live hands against Slumbot's public API (200BB, blinds 50/100).

    python -m scripts.play_slumbot --hands 20 --out DIR
    python -m scripts.play_slumbot --hands 1000 --model AVERAGE.gz --model-sha256 SHA --out DIR
    python -m scripts.play_slumbot ... --card-buckets TABLES   # a K50 average and its pinned tables
    python -m scripts.play_slumbot ... --login        # account session; asks for the password

The default player is check/call. A trained HU200 average, v1 or K50 cards, plays with its
stored policy and uniform play where it has none. Each run writes a fresh hash-chained journal of every decision
and verified settlement, plus `summary.json`. Nothing is retried. A request without a complete
answer abandons only its hand (Slumbot's session total still counts it); more than
--max-abandoned such hands, or any other failure, stops the run.
Progress goes to stderr: a line per stage, then a bar with BB/100 so far. With --login the username and password come from the environment variables SLUMBOT_USERNAME and
SLUMBOT_PASSWORD, else from the repository's git-ignored `.env` (see `.env.example`), else from
prompts, the password hidden. Neither is stored or printed.
"""

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

from tqdm import tqdm

from src.arena.external.http import AmbiguousRequest, HttpTransport, ServiceError
from src.arena.external.journal import PrefixJournal, verify_journal
from src.arena.external.session import CheckCall, hu200_average, play_hand, session_accounting, summarize
from src.arena.external.slumbot import CONTRACT, SlumbotConnection

SERVICE = 'https://slumbot.com'
ENV_FILE = Path(__file__).resolve().parents[1] / '.env'
CREDENTIALS = ('SLUMBOT_USERNAME', 'SLUMBOT_PASSWORD')


def read_env(path: Path) -> dict:
    """The Slumbot credentials in a KEY=VALUE file; every other line is ignored."""
    if not path.is_file():
        return {}
    found = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition('=')
        if sep and key.strip() in CREDENTIALS and value.strip().strip('"\''):
            found[key.strip()] = value.strip().strip('"\'')
    return found


def load_player(args):
    if args.model is None:
        player = CheckCall()
        return player, player, {'player': player.name}
    model = hu200_average(args.model, args.model_sha256, args.card_buckets)
    return model, model.policy(args.seed), {'player': 'hu200-average', 'model_sha256': args.model_sha256,
                                            'model_bytes': args.model.stat().st_size, 'policy_seed': args.seed,
                                            'abstraction': model.abstraction, 'description': model.description}


def say(message):
    print(time.strftime('%H:%M:%S'), message, file=sys.stderr, flush=True)


def progress(hands, abandoned):
    """BB/100 so far with its 95% half-width."""
    raw = [h.terminal['winnings'] for h in hands]
    if len(raw) < 2:
        return f'BB/100 - | abandoned {abandoned}'
    return (f'BB/100 {statistics.fmean(raw):+.0f}±{1.96 * statistics.stdev(raw) / len(raw) ** 0.5:.0f}'
            f' | abandoned {abandoned}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hands', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--model-sha256')
    parser.add_argument('--card-buckets', type=Path, help="a K50 model's pinned table directory")
    parser.add_argument('--seed', type=int, default=1, help='policy sampling seed')
    parser.add_argument('--max-requests', type=int, default=None, help='default: 60 per hand')
    parser.add_argument('--max-seconds', type=float, default=None)
    parser.add_argument('--min-interval', type=float, default=0.0,
                        help='extra seconds between requests; one is ever in flight')
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--max-abandoned', type=int, default=5,
                        help='hands given up after an ambiguous request before the run stops')
    parser.add_argument('--login', action='store_true')
    args = parser.parse_args()
    if args.hands < 1 or (args.model is None) != (args.model_sha256 is None):
        parser.error('Give a positive --hands, and --model together with --model-sha256')
    if args.card_buckets is not None and args.model is None:
        parser.error('--card-buckets needs a K50 --model')
    args.out.mkdir(parents=True)
    # Ask and check the credentials before the model loads, so nobody waits minutes to learn of
    # a typo; the playing session logs in again after the load, so it starts fresh.
    if args.login:
        stored = {**read_env(ENV_FILE), **{k: os.environ[k] for k in CREDENTIALS if os.environ.get(k)}}
        username = stored.get('SLUMBOT_USERNAME') or input('Slumbot username: ')
        password = stored.get('SLUMBOT_PASSWORD') or getpass.getpass('Slumbot password: ')
        say('Credentials read' + (' from .env or the environment' if 'SLUMBOT_PASSWORD' in stored else ''))
        try:
            SlumbotConnection(HttpTransport(SERVICE, timeout=args.timeout, max_requests=1), CheckCall()).login(username, password)
        except (ValueError, AmbiguousRequest, ServiceError) as error:
            # Messages name the endpoint, status or Slumbot's reason, never the password.
            say(f'Login check failed: {error}')
            raise SystemExit(1)
        say('Credentials accepted')
    say(f'Loading {args.model.name} (a few minutes for large models)' if args.model else 'Using the check/call player')
    loading = time.monotonic()
    policy, player, identity = load_player(args)
    if args.model:
        say(f'Model loaded in {time.monotonic() - loading:.0f}s')
    transport = HttpTransport(SERVICE, timeout=args.timeout, min_interval=args.min_interval,
                              max_requests=args.max_requests or 60 * args.hands)
    connection = SlumbotConnection(transport, policy)
    if args.login:
        say('Logging in to Slumbot')
        connection.login(username, password)
        del password
        say('Logged in')
    say(f'Playing {args.hands} hands; journal {args.out / "journal.jsonl"}')
    run = {'service': SERVICE, 'contract': CONTRACT.__dict__, 'account_session': args.login,
           'max_abandoned': args.max_abandoned,
           'hands_requested': args.hands, 'max_requests': transport.max_requests,
           'min_interval_seconds': args.min_interval, 'timeout_seconds': args.timeout, **identity}
    journal = PrefixJournal(args.out / 'journal.jsonl', run)
    hands, stop, started, abandoned = [], None, time.monotonic(), 0
    bar = tqdm(total=args.hands, unit='hand', file=sys.stderr, dynamic_ncols=True)
    for index in range(args.hands):
        if args.max_seconds is not None and time.monotonic() - started > args.max_seconds:
            stop = 'time limit'
            break
        try:
            hands.append(play_hand(connection, player, f'slumbot-{index:06d}', journal))
        except AmbiguousRequest:
            # Give up this hand without resending anything, and continue with a new one.
            journal.failure('timeout-ambiguous'); abandoned += 1
            bar.write(f'Hand {index} abandoned: a request got no complete answer ({abandoned} so far)')
            if abandoned > args.max_abandoned:
                stop = 'too many ambiguous requests'
                break
            connection.abandon_hand()
        except ServiceError as error:
            # The message names only the endpoint and status: safe to keep.
            journal.failure('service-error'); stop = f'service error: {error}'
            break
        except ValueError:
            journal.failure('protocol-error'); stop = 'protocol error'
            break
        finally:
            bar.set_postfix_str(progress(hands, abandoned), refresh=False)
            bar.update(1)
    bar.close()
    if stop is not None:
        say(f'Stopped: {stop}')
    journal.close()
    checked = verify_journal(args.out / 'journal.jsonl')
    latencies = sorted(transport.latencies)
    summary = {**summarize(hands), 'run': run, 'stopped': stop, 'requests': transport.requests,
               'abandoned_hands': abandoned, 'slumbot_session': session_accounting(hands),
               'seconds': time.monotonic() - started, 'journal': checked,
               'journal_sha256': hashlib.sha256((args.out / 'journal.jsonl').read_bytes()).hexdigest(),
               'latency_seconds': {'median': latencies[len(latencies) // 2], 'p90': latencies[int(len(latencies) * 0.9)],
                                   'max': latencies[-1]} if latencies else None}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: summary[k] for k in ('hands', 'bb_per_100', 'stopped', 'requests', 'abandoned_hands')}))
    if stop is not None and stop != 'time limit':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
