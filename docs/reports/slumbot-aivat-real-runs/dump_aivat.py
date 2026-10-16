"""Run scripts.analyze_slumbot_run unchanged, and also save every hand's AIVAT corrections.

Usage (from the repo root): AIVAT_DUMP=out.jsonl PYTHONPATH=. python dump_aivat.py <analyze_slumbot_run args>
Each line: hand_id, client_pos, raw winnings (chips) and the 8 corrections in chips
(deal, flop, turn, river, then preflop/flop/turn/river own actions; zeros without --model).
The corrections are computed once and handed to the analyzer's own summary, so the summary is
exactly what the analyzer alone would report.
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor as RealPool
import src.arena.external.analysis as A

original = A.aivat_summary

def summary(raw, terminals, hand_decisions, workers, actions):
    with RealPool(workers) as pool:
        corrections = list(pool.map(A.hand_corrections, terminals, hand_decisions, chunksize=16))
    with open(os.environ['AIVAT_DUMP'], 'x') as out:
        for r, t, c in zip(raw, terminals, corrections):
            c = list(c) + [0.0] * (8 - len(c))
            out.write(json.dumps({'hand_id': t['hand_id'], 'client_pos': t['client_pos'], 'raw': r, 'corrections': c}) + '\n')

    class Cached:
        def __init__(self, *_): pass
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def map(self, *_args, **_kw): return iter(corrections)

    A.ProcessPoolExecutor = Cached
    try:
        return original(raw, terminals, hand_decisions, workers, actions)
    finally:
        A.ProcessPoolExecutor = RealPool

A.aivat_summary = summary
from scripts.analyze_slumbot_run import main
sys.argv = ['analyze_slumbot_run'] + sys.argv[1:]
main()
