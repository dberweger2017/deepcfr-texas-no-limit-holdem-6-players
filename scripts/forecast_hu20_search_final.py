"""Timing-only final-stage forecasts from retained native screening receipts."""

from collections import defaultdict
from copy import deepcopy

import numpy as np


def requests_shareable(left, right):
    """Only identical unlocked requests may share a full solve across bot seats."""
    if left.get('locks') or right.get('locks'):
        return False
    return left == right


def budget_forecast(screen, *, used_seconds, reference_pairs=144, reserve_seconds=10800):
    groups = defaultdict(list)
    for row in screen:
        if row['config']['menu'] == 'native':
            groups[row['configuration_id']].append(row)
    cells = []
    for identity, rows in groups.items():
        if len(rows) != 8:
            raise ValueError('Complete eight-stratum native screen required')
        quality = [sum(r['seconds'] for r in row.get('receipts', [])[1:])
                   for row in rows if not row['fallback']]
        cells.append({'configuration_id': identity, 'config': deepcopy(rows[0]['config']),
            'cold_p95_seconds': float(np.percentile([r['cold_seconds'] for r in rows], 95)),
            'cold_median_seconds': float(np.median([r['cold_seconds'] for r in rows])),
            'quality_p95_seconds': float(np.percentile(quality, 95)) if quality else None,
            'quality_timing_samples': len(quality)})
    options = []
    for count in (2, 1):
        selected = []
        for floor in (0, .01):
            candidates = sorted((c for c in cells if c['config']['opponent_likelihood_floor'] == floor),
                key=lambda c: (c['cold_p95_seconds'], c['cold_median_seconds'], c['configuration_id']))
            if len(candidates) < count:
                raise ValueError('Incomplete native timing axes')
            selected.extend(candidates[:count])
        phases = []
        for deadline in (30, 120):
            entries = []
            for c in selected:
                counts = [n for n in (25, 50, 100, 200, 400)
                          if c['cold_p95_seconds']*n/100 <= deadline
                          or (deadline == 30 and n == 100)]
                if c['quality_p95_seconds'] is None:
                    raise ValueError('Selected setting needs measured quality timing')
                # Epsilon zero is seat-independent at an unlocked turn root.
                # Epsilon .01 is conservatively separate until complete requests compare equal.
                solves = reference_pairs*(1 if c['config']['opponent_likelihood_floor'] == 0 else 2)
                scaled_work = sum(counts)/100
                entries.append(dict(c, iterations=counts, unique_solves_per_iteration=solves,
                    play_seconds=solves*scaled_work*c['cold_p95_seconds'],
                    quality_seconds=solves*scaled_work*c['quality_p95_seconds'],
                    optimistic_all_seats_shared_play_seconds=reference_pairs*scaled_work*c['cold_p95_seconds']))
            phases.append({'deadline_seconds': deadline, 'entries': entries,
                'play_seconds': sum(c['play_seconds'] for c in entries),
                'quality_seconds': sum(c['quality_seconds'] for c in entries),
                'optimistic_all_seats_shared_play_seconds': sum(c['optimistic_all_seats_shared_play_seconds'] for c in entries)})
        total = sum(p['play_seconds']+p['quality_seconds'] for p in phases)+reserve_seconds
        options.append({'finalists_per_floor': count, 'phases': phases,
            'final_play_and_quality_seconds': total-reserve_seconds,
            'river_reserve_seconds': reserve_seconds, 'required_remaining_seconds': total,
            'fits_remaining_budget': total <= 86400-used_seconds,
            'optimistic_all_seats_shared_play_only_plus_river_seconds':
                sum(p['optimistic_all_seats_shared_play_seconds'] for p in phases)+reserve_seconds})
    return {'status': 'forecast-fits' if any(o['fits_remaining_budget'] for o in options) else 'owner-decision-needed',
        'used_seconds': used_seconds, 'remaining_seconds': 86400-used_seconds,
        'options': options, 'timing_cells': cells, 'outcome_fields_used': [],
        'forecast_scope': '30s final plus conditional 120s final with one/two finalists and three-hour river reserve',
        'not_included_nonnegative_costs': ['remaining eighteen reduced-menu screen rows',
            '120s new screen', 'preparation/load/parse outside solver receipts', 'verification/build/setup'],
        'scaling_limit': 'Linear screen-p95 planning estimate; timeout-censored roots may cost more; not a runtime guarantee',
        'restart_authorized_by_forecast': any(o['fits_remaining_budget'] for o in options)}
