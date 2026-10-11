"""One-use fresh 40-root scoring, sharing #222's unchanged evaluate/report."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from scripts import score_restored_hu20_equity_bench as scoring
from scripts.continue_restored_hu20_equity_bench import exclusive_json
from src.policies.files import file_hash


def guard_clear(base, phase):
    receipt = json.loads((base/'operations'/phase/'receipt.json').read_text())
    admission = json.loads((base/'operations'/phase/'admission.json').read_text())
    if (receipt['status'] != 'complete' or receipt['failure'] is not None
            or receipt['cleanup_error'] is not None or receipt['returncode'] != 0
            or admission['identity'] != ['Apple M4', str(16*1024**3), '10']):
        raise ValueError('Guard-clear M4 operation required: '+phase)
    return receipt


def checked_inputs(base):
    if (base/'campaign-failure.json').exists():
        raise ValueError('Stopped attempt; no retry')
    _, receipt, jobs = scoring.configure(base)
    for pin in receipt['members'] + receipt['aliases']:
        path = Path(pin['restored_path'])
        if path.stat().st_size != pin['bytes'] or file_hash(path) != pin['sha256']:
            raise ValueError('Restored member differs: '+str(path))
    review = json.loads(Path('docs/reports/hu20-equity-bench-scoring-m4-artifacts/source-review.json').read_text())
    if review['status'] != 'clear' or review['open_findings'] or not review['source_files']:
        raise ValueError('Independent scoring source review required')
    for pin in review['source_files']:
        if file_hash(Path(pin['path'])) != pin['sha256']:
            raise ValueError('Reviewed source differs: '+pin['path'])
    return jobs


def pilot(base):
    jobs = checked_inputs(base)
    if (base/'work/eval').exists() or (base/'continuation.json').exists():
        raise ValueError('Fresh empty final scoring required')
    exclusive_json(base/'pilot-claimed.json', {'job': jobs[0]['job']})
    scoring.bench.evaluate(pilot=True)


def quote(base):
    checked_inputs(base)
    receipt = guard_clear(base, 'pilot')
    timing = json.loads((base/'work/scoring-quote.json').read_text())
    if (timing['outcomes_used_for_quote'] is not False or timing['root_count'] != 40
            or timing['passes_per_root'] != 1):
        raise ValueError('Frozen timing-only quote required')
    samples = [json.loads(line) for line in (base/'operations/pilot/resources.jsonl').read_text().splitlines()]
    disk = shutil.disk_usage(base).free
    full = 40*timing['seconds']
    intent = json.loads((base/'operations/pilot/intent.json').read_text())
    value = {'created_utc': datetime.now(timezone.utc).isoformat(),
             'pilot_source': intent['source'],
             'pilot': timing, 'pilot_guard_seconds': receipt['seconds'],
             'peak_family_rss_bytes': receipt['peak_family_rss_bytes'],
             'max_swap_bytes': max(s['swap_bytes'] for s in samples),
             'min_free_percent': min(s['free_percent'] for s in samples),
             'min_disk_free_bytes': min(s['disk_free_bytes'] for s in samples),
             'final_roots': 40, 'reused_roots': 0, 'policies_per_fold': 21,
             'forecast_scoring_seconds': full, 'schedule_allowance_seconds': .25*full,
             'forecast_closeout_seconds': 1800, 'current_disk_free_bytes': disk,
             'output_originals_budget_bytes': 1024**3, 'zip_budget_bytes': 1024**3,
             'disk_floor_bytes': 16*1024**3,
             'new_pilot_outcomes_inspected': False, 'prior_completed_outcomes_inspected': 9,
             'science': 'unchanged #222 evaluate/report; same 40 roots and 2000 paired draws',
             'bootstrap': 'Random(202610050002); 2000 paired sorted40 draws; indices49/1949',
             'swap_ceiling_bytes': receipt['swap_ceiling_bytes'],
             'worker': 'one serial six-thread native worker; nice10; 4GiB arena; 7GiB family',
             'primary': 'pass: matched upper95% < -0.10BB and 10M point < 0; fail: matched lower > -0.10BB; otherwise inconclusive'}
    if disk-2*1024**3 <= 16*1024**3:
        raise ValueError('Storage quote fails 16 GiB floor')
    exclusive_json(base/'full-scoring-quote.json', value)
    return value


def evaluate(base):
    jobs = checked_inputs(base)
    guard_clear(base, 'pilot')
    posted = json.loads((base/'quote-posted.json').read_text())
    if (posted['quote_sha256'] != file_hash(base/'full-scoring-quote.json')
            or not posted['comment_url'].startswith('https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/')
            or '#issuecomment-' not in posted['comment_url']):
        raise ValueError('Full quote must be posted before scoring')
    if (base/'work/eval').exists() or (base/'continuation.json').exists():
        raise ValueError('Fresh empty final scoring required')
    expected = [job['job'] for job in jobs]
    exclusive_json(base/'evaluation-claimed.json', {'jobs': expected, 'reused_roots': 0})
    scoring.bench.evaluate()
    exclusive_json(base/'evaluation-complete.json', {'jobs': expected, 'new_roots': 40})


def report(base):
    jobs = checked_inputs(base)
    guard_clear(base, 'evaluate')
    expected = [job['job'] for job in jobs]
    complete = json.loads((base/'evaluation-complete.json').read_text())
    if (complete != {'jobs': expected, 'new_roots': 40}
            or {p.name for p in (base/'work/eval').iterdir()} != set(expected)
            or {p.parent.name for p in (base/'work/eval').glob('*/result.json')} != set(expected)):
        raise ValueError('All 40 fresh completed roots required')
    exclusive_json(base/'report-claimed.json', {'roots': 40})
    scoring.bench.report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('pilot', 'quote', 'evaluate', 'report'))
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    value = globals()[args.command](args.base.resolve())
    if value is not None:
        print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
