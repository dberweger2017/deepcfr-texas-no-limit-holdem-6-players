"""Stream generated regression records into paired estimates and tail tables."""
import argparse
import gzip
import json
from pathlib import Path

from scripts.evaluate_hu20 import write_json
from src.diagnostics.stackoff_report import summarize


def records(folder):
    for path in sorted(folder.glob('*.hands.jsonl.gz')):
        with gzip.open(path, 'rt') as handle:
            for line in handle:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads((args.run / 'plan.json').read_text()), records(args.run))
    write_json(args.run / 'summary.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'requested_hands', 'attempted_hands', 'failed_hands')}))


if __name__ == '__main__':
    main()
