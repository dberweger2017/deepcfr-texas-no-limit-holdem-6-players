"""Derive post-hoc made-hand comparisons from existing packaged simulator data."""
import argparse
import json
from pathlib import Path

from scripts.check_hu20_stackoff import checked_rows, verified_manifest
from scripts.evaluate_hu20 import write_json
from scripts.report_hu20_stackoff import markdown as dashboard
from src.diagnostics.saved_hu20 import file_hash
from src.diagnostics.stackoff_made_hands import first_large_raise, summarize_events, markdown


def report(evidence):
    verified_manifest(evidence)
    plan = json.loads((evidence / 'plan.json').read_text())
    stats = {'native_replays': 0, 'raw_decisions': 0}
    events = [event for row in checked_rows(evidence, stats)
              if row['panel'] == 'Selective-stackoff-v1'
              and (event := first_large_raise(row)) is not None]
    result = summarize_events(events, plan)
    result['inputs'] = {name: file_hash(evidence / name) for name in
                        ('plan.json', 'generated-hands.jsonl.gz', 'decisions.csv.gz')}
    write_json(evidence / 'large-raise-made-hands.json', result)
    (evidence / 'large-raise-made-hands.rows.jsonl').write_text(
        ''.join(json.dumps(row, sort_keys=True) + '\n' for row in events))
    summary = json.loads((evidence / 'summary.json').read_text())
    (evidence / 'dashboard.md').write_text(dashboard(summary) + '\n' + markdown(result))
    manifest = json.loads((evidence / 'evidence-manifest.json').read_text())
    for name in ('large-raise-made-hands.json', 'large-raise-made-hands.rows.jsonl', 'dashboard.md'):
        path = evidence / name
        manifest['files'][name] = {'bytes': path.stat().st_size, 'sha256': file_hash(path)}
    write_json(evidence / 'evidence-manifest.json', manifest)
    return {**stats, 'first_large_raise_hands': len(events)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.evidence), sort_keys=True))


if __name__ == '__main__':
    main()
