"""Analyze a verified Slumbot run directory written by `scripts.play_slumbot`.

    python -m scripts.analyze_slumbot_run RUN_DIR [--model AVERAGE.gz --model-sha256 SHA]

Writes RUN_DIR/analysis.json: raw and all-in-adjusted BB/100 overall and by seat, how hands
ended, the client's actions by street and against each Slumbot bet size, the biggest swings,
and, with the model, how often each street's decisions had a stored policy.
"""

import argparse
import json
from pathlib import Path

from src.arena.external.analysis import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--model-sha256')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    model = None
    if args.model:
        from src.blueprint.abstraction import HU200_SCHEMA
        from src.blueprint.average import AveragePolicy
        model = AveragePolicy(args.model, args.model_sha256, expected_schema=HU200_SCHEMA, sorted_rows=True)
    result = analyze(args.run / 'journal.jsonl', model=model, workers=args.workers)
    with (args.run / 'analysis.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: result[k] for k in ('hands', 'bb_per_100', 'all_in_adjusted_bb_per_100', 'all_in_hands')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
