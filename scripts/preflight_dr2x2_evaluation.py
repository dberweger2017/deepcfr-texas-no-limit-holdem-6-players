"""Outcome-blind, sequential A/C evaluation runtime admission on M1."""

import argparse
import gc
import json
import shutil
from pathlib import Path
from time import perf_counter

from scripts.evaluate_hu20_cards_v2 import hand
from scripts.evaluate_hu20_stackoff import swap_bytes
from scripts.hu20_platform_pilot import peak_rss, write
from src.arena.schedule import digest
from src.diagnostics.saved_hu20 import file_hash, load_saved


def measure_model(spec, plan, guard):
    guard()
    started = perf_counter()
    source, visits = load_saved(spec, Path('/'), guard,
                                expected_schema=spec['abstraction'])
    load_seconds = perf_counter() - started
    panels = []
    for panel in plan['panels']:
        seconds = []
        # Validation phase deal streams differ from all playing test streams.
        timing_panel = {**panel, 'root': panel['timing_root']}
        for block in range(plan['timing_blocks']):
            for rotation in (0, 1):
                guard()
                began = perf_counter()
                row = hand(source, visits, spec, timing_panel, block, rotation,
                           plan, resource_only=True)
                if (row['status'] != 'complete' or row['target_chips'] is not None
                        or row['net_chips_by_seat'] is not None):
                    raise ValueError('Timing must not expose playing outcomes')
                seconds.append(perf_counter() - began)
        panels.append({'panel': panel['name'], 'hands': len(seconds),
                       'seconds': sum(seconds), 'max_hand_seconds': max(seconds),
                       'mean_hand_seconds': sum(seconds) / len(seconds)})
    del source, visits
    gc.collect()
    return {'cell': spec['cell'], 'seed': spec['seed'],
            'checkpoint_sha256': spec['checkpoint_sha256'],
            'policy_sha256': spec['sha256'], 'load_seconds': load_seconds,
            'panels': panels, 'outcome_blind': True}


def run(plan, out):
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'plan.json', plan)
    started = perf_counter()
    swap_before = swap_bytes()
    results = []
    failure = None
    def guard():
        limits = plan['limits']
        if perf_counter() - started >= limits['timing_max_seconds']:
            raise TimeoutError('Bounded timing admission reached its limit')
        if peak_rss() >= limits['max_rss_gib'] * 2**30:
            raise MemoryError('M1 timing RSS guard')
        if shutil.disk_usage(out).free < limits['min_free_disk_gib'] * 2**30:
            raise OSError('M1 timing disk guard')
        if swap_bytes() - swap_before > limits['max_swap_growth_gib'] * 2**30:
            raise MemoryError('M1 timing swap guard')
    try:
        for spec in plan['models']:
            result = measure_model(spec, plan, guard)
            results.append(result)
            write(out / f"{spec['cell']}-{spec['seed']}.json", result)
            write(out / 'progress.json', {'complete_models': len(results),
                  'seconds': perf_counter() - started, 'peak_rss_bytes': peak_rss()})
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
    projections = {}
    if not failure:
        for panel in plan['panels']:
            # Worst observed per-hand time per model, then a fixed 2x margin.
            projections[panel['name']] = plan['projection_multiplier'] * sum(
                next(p['max_hand_seconds'] for p in r['panels']
                     if p['panel'] == panel['name']) * panel['blocks'] * 2
                for r in results)
    summary = {'status': 'failed' if failure else 'complete', 'failure': failure,
               'plan_sha256': digest(plan), 'outcome_blind': True,
               'strength_outcomes_inspected': False, 'models': results,
               'projection_seconds_by_panel': projections,
               'projected_current_total_seconds': sum(projections.values()) +
                   plan['projection_multiplier'] * sum(r['load_seconds'] for r in results),
               'seconds': perf_counter() - started, 'peak_rss_bytes': peak_rss(),
               'swap_growth_bytes': swap_bytes() - swap_before}
    write(out / 'summary.json', summary)
    write(out / 'manifest.json', {p.name: {'bytes': p.stat().st_size,
          'sha256': file_hash(p)} for p in out.iterdir() if p.is_file()})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.plan.read_text()), args.out)
    print(json.dumps({'status': result['status'], 'seconds': result['seconds']}))
    raise SystemExit(result['status'] != 'complete')
