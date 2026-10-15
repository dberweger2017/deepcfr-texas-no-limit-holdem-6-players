"""Play bounded live hands against Slumbot's public API (200BB, blinds 50/100).

    python -m scripts.play_slumbot --hands 20 --out DIR
    python -m scripts.play_slumbot --hands 1000 --model AVERAGE.gz --model-sha256 SHA --out DIR
    python -m scripts.play_slumbot ... --card-buckets TABLES   # a K50 average and its pinned tables
    python -m scripts.play_slumbot ... --login        # account session; asks for the password

The default player is check/call; --player random plays uniformly among legal actions. A trained HU200 average, v1 or K50 cards, plays with its
stored policy and uniform play where it has none. Each run writes a fresh hash-chained journal of every decision
and verified settlement, plus `summary.json`. Nothing is retried. A request without a complete
answer, or one Slumbot refuses with an error status, abandons only its hand (Slumbot's session
total still counts it); a refusal also starts a new session, logged in again with --login; more than --max-abandoned or --max-rejected such hands, or any other
failure, stops the run.
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
from src.arena.external.session import (CheckCall, RandomLegal, hu200_average, play_hand, session_accounting, summarize,
                                        without_decisions)
from src.arena.external.slumbot import CONTRACT, SlumbotConnection

from src.blueprint.translation_cli import add_translation_options, translation_options

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
    translation = translation_options(args)
    if translation is not None and args.model is None:
        raise ValueError('--action-translation needs --model')
    if args.model is None:
        player = RandomLegal(args.seed) if args.player == 'random' else CheckCall()
        return player, player, {'player': player.name, **({'policy_seed': args.seed} if args.player == 'random' else {})}
    model = hu200_average(args.model, args.model_sha256, args.card_buckets, args.policy_cache,
                          missing_key_fallback=args.missing_key_fallback, cache_read_only=args.cache_read_only, translation=translation)
    return model, model.policy(args.seed), {'player': 'hu200-average', 'model_sha256': args.model_sha256,
                                            'model_bytes': args.model.stat().st_size, 'policy_seed': args.seed,
                                            'abstraction': model.abstraction, 'description': model.description,
                                            **({'action_translation': model.description['action_translation']} if translation is not None else {}),
                                            **({'missing_key_fallback': args.missing_key_fallback} if args.missing_key_fallback != 'uniform' else {})}


def say(message):
    print(time.strftime('%H:%M:%S'), message, file=sys.stderr, flush=True)


def progress(hands, abandoned):
    """BB/100 so far with its 95% half-width."""
    raw = [h.terminal['winnings'] for h in hands]
    if len(raw) < 2:
        return f'BB/100 - | abandoned {abandoned}'
    return (f'BB/100 {statistics.fmean(raw):+.0f}±{1.96 * statistics.stdev(raw) / len(raw) ** 0.5:.0f}'
            f' | abandoned {abandoned}')


def play_hands(connection, player, journal, target: int, *, max_abandoned: int, max_rejected: int,
               deadline: float | None = None, new_session=None):
    """Play until `target` hands have completed; abandoned and refused hands don't count toward it.

    Returns the completed hands, why play stopped early (None when the target was reached),
    the number of abandoned hands and the refusals. Abandoned and refused hands are bounded by
    their own limits, so the loop always ends.
    """
    hands, stop, abandoned, rejections, attempt = [], None, 0, [], 0
    bar = tqdm(total=target, unit='hand', file=sys.stderr, dynamic_ncols=True)
    while len(hands) < target:
        if deadline is not None and time.monotonic() > deadline:
            stop = 'time limit'
            break
        index, attempt = attempt, attempt + 1
        try:
            hands.append(without_decisions(play_hand(connection, player, f'slumbot-{index:06d}', journal)))
            bar.update(1)
        except AmbiguousRequest:
            # Give up this hand without resending anything, and continue with a new one.
            journal.failure('timeout-ambiguous'); abandoned += 1
            bar.write(f'Hand {index} abandoned: a request got no complete answer ({abandoned} so far)')
            if abandoned > max_abandoned:
                stop = 'too many ambiguous requests'
                break
            connection.abandon_hand()
        except ServiceError as error:
            # The message names the endpoint, status and Slumbot's reason without tokens: safe to keep.
            if error.status is None:
                journal.failure('service-error'); stop = f'service error: {error}'
                break
            # Slumbot answered and refused the request: the hand is lost, nothing is in doubt.
            journal.failure('service-rejected'); rejections.append({'hand': index, 'error': str(error)})
            bar.write(f'Hand {index} abandoned: Slumbot refused a request, {error} ({len(rejections)} so far)')
            if len(rejections) > max_rejected:
                stop = 'too many rejected requests'
                break
            # A refusal can mean the session expired: start a fresh one, logged in again for
            # account runs.
            try:
                (new_session or connection.new_session)()
            except (ValueError, AmbiguousRequest, ServiceError) as failure:
                stop = f'new session failed: {failure}'
                break
        except ValueError as error:
            # Messages are fixed sentences without response data or tokens, so the reason is safe to keep.
            journal.failure('protocol-error'); stop = f'protocol error: {error}'
            bar.write(f'Hand {index} stopped the run: {error}')
            break
        finally:
            bar.set_postfix_str(progress(hands, abandoned + len(rejections)), refresh=False)
    bar.close()
    return hands, stop, abandoned, rejections


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hands', type=int, required=True,
                        help='completed hands to play; abandoned and refused hands are replaced')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--player', choices=('check-call', 'random'), default='check-call',
                        help='scripted player when no --model is given; random uses --seed')
    parser.add_argument('--model-sha256')
    parser.add_argument('--card-buckets', type=Path, help="a K50 model's pinned table directory")
    parser.add_argument('--policy-cache', type=Path,
                        help='folder for a verified binary cache of the model: the first load writes it, later ones take seconds')
    parser.add_argument('--cache-read-only', action='store_true',
                        help='require an existing verified cache; never create it')
    parser.add_argument('--missing-key-fallback', choices=('uniform', 'check-call'), default='uniform',
                        help='missing keys only; omitted identity fields mean uniform')
    parser.add_argument('--seed', type=int, default=1, help='policy sampling seed')
    parser.add_argument('--max-requests', type=int, default=None, help='default: 60 per hand')
    parser.add_argument('--max-seconds', type=float, default=None)
    parser.add_argument('--min-interval', type=float, default=0.0,
                        help='extra seconds between requests; one is ever in flight')
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--max-abandoned', type=int, default=5,
                        help='hands given up after an ambiguous request before the run stops')
    parser.add_argument('--max-rejected', type=int, default=10,
                        help='hands given up after Slumbot refused a request before the run stops')
    parser.add_argument('--login', action='store_true')
    parser.add_argument('--account', help='with --login: the only Slumbot username this run may use')
    add_translation_options(parser)
    args = parser.parse_args()
    try:
        translation = translation_options(args)
    except ValueError as error:
        parser.error(str(error))
    if translation is not None and args.model is None:
        parser.error('--action-translation needs --model')
    if args.hands < 1 or (args.model is None) != (args.model_sha256 is None):
        parser.error('Give a positive --hands, and --model together with --model-sha256')
    if args.account is not None and not args.login:
        parser.error('--account needs --login')
    if args.model is not None and args.player != 'check-call':
        parser.error('--player is for scripted players; a --model plays itself')
    if args.card_buckets is not None and args.model is None:
        parser.error('--card-buckets needs a K50 --model')
    if args.missing_key_fallback != 'uniform' and args.model is None:
        parser.error('--missing-key-fallback needs --model')
    if args.cache_read_only and (args.model is None or args.policy_cache is None):
        parser.error('--cache-read-only needs --model and --policy-cache')
    args.out.mkdir(parents=True)
    # Ask and check the credentials before the model loads, so nobody waits minutes to learn of
    # a typo; the playing session logs in again after the load, so it starts fresh.
    if args.login:
        stored = {**read_env(ENV_FILE), **{k: os.environ[k] for k in CREDENTIALS if os.environ.get(k)}}
        username = stored.get('SLUMBOT_USERNAME') or input('Slumbot username: ')
        if args.account is not None and username.strip() != args.account:
            say(f'This run is for the Slumbot account {args.account}, not {username.strip()}')
            raise SystemExit(1)
        password = stored.get('SLUMBOT_PASSWORD') or getpass.getpass('Slumbot password: ')
        say('Credentials read' + (' from .env or the environment' if 'SLUMBOT_PASSWORD' in stored else ''))
        try:
            SlumbotConnection(HttpTransport(SERVICE, timeout=args.timeout, max_requests=1), CheckCall()).login(username, password)
        except (ValueError, AmbiguousRequest, ServiceError) as error:
            # Messages name the endpoint, status or Slumbot's reason, never the password.
            say(f'Login check failed: {error}')
            raise SystemExit(1)
        say('Credentials accepted')
    cached = args.model and args.policy_cache and any(args.policy_cache.glob(f'{args.model_sha256}-*'))
    say((f'Loading {args.model.name} from its cache' if cached else
         f'Loading {args.model.name} (a few minutes for large models)') if args.model else f'Using the {args.player} player')
    loading = time.monotonic()
    policy, player, identity = load_player(args)
    if args.model:
        say(f'Model loaded in {time.monotonic() - loading:.0f}s')
    transport = HttpTransport(SERVICE, timeout=args.timeout, min_interval=args.min_interval,
                              max_requests=args.max_requests or 60 * (args.hands + args.max_abandoned + args.max_rejected))
    connection = SlumbotConnection(transport, policy)
    if args.login:
        say('Logging in to Slumbot')
        # The password stays in memory only, to log in again if Slumbot refuses a request.
        connection.login(username, password)
        say('Logged in')
    say(f'Playing {args.hands} hands; journal {args.out / "journal.jsonl"}')
    run = {'service': SERVICE, 'contract': CONTRACT.__dict__, 'account_session': args.login,
           'account': args.account,
           'max_abandoned': args.max_abandoned, 'max_rejected': args.max_rejected,
           'hands_requested': args.hands, 'hands_counted': 'completed', 'max_requests': transport.max_requests,
           'min_interval_seconds': args.min_interval, 'timeout_seconds': args.timeout, **identity}
    journal = PrefixJournal(args.out / 'journal.jsonl', run)
    started = time.monotonic()
    hands, stop, abandoned, rejections = play_hands(
        connection, player, journal, args.hands, max_abandoned=args.max_abandoned, max_rejected=args.max_rejected,
        deadline=None if args.max_seconds is None else started + args.max_seconds,
        new_session=lambda: connection.new_session(*((username, password) if args.login else ())))
    if stop is not None:
        say(f'Stopped: {stop}')
    journal.close()
    checked = verify_journal(args.out / 'journal.jsonl')
    latencies = sorted(transport.latencies)
    summary = {**summarize(hands), 'run': run, 'stopped': stop, 'requests': transport.requests,
               'abandoned_hands': abandoned, 'rejected_hands': len(rejections), 'rejections': rejections,
               'slumbot_session': session_accounting(hands),
               'seconds': time.monotonic() - started, 'journal': checked,
               'journal_sha256': hashlib.sha256((args.out / 'journal.jsonl').read_bytes()).hexdigest(),
               'latency_seconds': {'median': latencies[len(latencies) // 2], 'p90': latencies[int(len(latencies) * 0.9)],
                                   'max': latencies[-1]} if latencies else None}
    if args.missing_key_fallback != 'uniform':
        summary['fallback_decisions_by_street'] = {s: getattr(player, 'fallback_decisions_by_street', {}).get(s, 0)
            for s in ('preflop', 'flop', 'turn', 'river')}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: summary[k] for k in ('hands', 'bb_per_100', 'stopped', 'requests', 'abandoned_hands', 'rejected_hands')}))
    if stop is not None and stop != 'time limit':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
