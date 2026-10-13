"""Paired HU200 averages against seven scripted opponents, with full replay/reproduction."""
import argparse
import json
from pathlib import Path
from time import perf_counter

from scripts.evaluate_hu100_direct import PANEL_OPPONENTS, audit, execute, put


def run(specs, blocks, root, out, source, *, cache, pilot=False):
    out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    registry = None
    costs = {}
    for opponent in PANEL_OPPONENTS:
        panel = out / opponent
        registry, primary = execute(specs, blocks, root, 'panel', panel / 'play', source,
            registry, stack_bb=200, opponent=opponent, cache=cache)
        checked = audit(panel / 'play', panel / 'replay.json', report=not pilot, cost_only=pilot)
        _, repeat = execute(specs, blocks, root, 'panel', panel / 'reproduction', source,
            registry, panel / 'play', stack_bb=200, opponent=opponent, cache=cache)
        costs[opponent] = {'primary': primary, 'repeat': repeat, 'audit_seconds': checked['seconds'],
                           **({'paired_sd_bb_per_100': checked['paired_sd_bb_per_100'],
                              'arm_sd_bb_per_100': checked['arm_sd_bb_per_100']} if pilot else {})}
    result = {'status': 'verified', 'blocks_per_opponent': blocks, 'opponents': list(PANEL_OPPONENTS),
              'seconds': perf_counter() - started, 'panels': costs,
              'outcomes_inspected_for_quote': False, 'pilot': pilot}
    put(out / 'costs.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--specs', type=Path, required=True)
    p.add_argument('--blocks', type=int, required=True)
    p.add_argument('--root', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--policy-cache', type=Path, required=True)
    p.add_argument('--pilot', action='store_true')
    a = p.parse_args()
    run(json.loads(a.specs.read_text()), a.blocks, a.root, a.out, a.source,
        cache=a.policy_cache, pilot=a.pilot)


if __name__ == '__main__':
    main()
