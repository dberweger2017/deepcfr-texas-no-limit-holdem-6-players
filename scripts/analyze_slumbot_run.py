"""Analyze a verified Slumbot run directory written by `scripts.play_slumbot`.

    python -m scripts.analyze_slumbot_run RUN_DIR [--model AVERAGE.gz --model-sha256 SHA [--card-buckets TABLES]]
    python -m scripts.analyze_slumbot_run RUN_DIR RUN_DIR ... --out ANALYSIS.json [...]   # pooled runs

Writes RUN_DIR/analysis.json, or --out for several runs of one player pooled (--hands N keeps the first N): raw and all-in-adjusted BB/100 overall and by seat (with --aivat, also AIVAT), how hands
ended, the client's actions by street and against each Slumbot bet size, the biggest swings,
and, with the model, how often each street's decisions had a stored policy.
"""

import argparse
import json
from pathlib import Path

from src.blueprint.translation_cli import add_translation_options, translation_options
from src.arena.external.analysis import analyze
from src.arena.external.session import hu200_average


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, help='required when pooling several runs')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--model-sha256')
    parser.add_argument('--card-buckets', type=Path, help="a K50 model's pinned table directory")
    parser.add_argument('--policy-cache', type=Path,
                        help='folder for a verified binary cache of the model: the first load writes it, later ones take seconds')
    parser.add_argument('--cache-read-only', action='store_true',
                        help='require an existing verified cache; never create it')
    parser.add_argument('--missing-key-fallback', choices=('uniform', 'check-call'), default='uniform')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--hands', type=int, help='only the first N hands in play order')
    parser.add_argument('--aivat', action='store_true',
                        help='add AIVAT results: card luck, plus our own action luck with --model; '
                             'about a second of CPU per hand, shared by --workers')
    add_translation_options(parser)
    args = parser.parse_args()
    try:
        translation = translation_options(args)
    except ValueError as error:
        parser.error(str(error))
    if translation is not None and args.model is None:
        parser.error('--action-translation needs --model')
    if len(args.runs) > 1 and args.out is None:
        parser.error('Pooling several runs needs --out')
    if args.cache_read_only and (args.model is None or args.policy_cache is None):
        parser.error('--cache-read-only needs --model and --policy-cache')
    model = hu200_average(args.model, args.model_sha256, args.card_buckets, args.policy_cache,
                          missing_key_fallback=args.missing_key_fallback, cache_read_only=args.cache_read_only, translation=translation) if args.model else None
    result = analyze([run / 'journal.jsonl' for run in args.runs], model=model, workers=args.workers,
                     missing_key_fallback=args.missing_key_fallback, hands=args.hands, aivat=args.aivat)
    with (args.out or args.runs[0] / 'analysis.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    keys = ('hands', 'bb_per_100', 'all_in_adjusted_bb_per_100', 'all_in_hands')
    summary = {k: result[k] for k in keys}
    if args.aivat:
        summary['aivat'] = {k: result['aivat'][k] for k in ('own_actions_corrected', 'bb_per_100', 'weighted_bb_per_100',
                                                             'plain_hands_worth', 'weighted_hands_worth')}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
