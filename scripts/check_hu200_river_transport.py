"""Compare optimized HU200 outputs with the retained original fixed24 pilot."""

import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import subprocess

import numpy as np
import psutil

from scripts.evaluate_hu100_direct import put
from scripts.pilot_hu200_river_search import CONFIG, MODEL_SHA256, pin_earlier_blueprint
from src.arena.external.slumbot import SlumbotAdapter
from src.blueprint.abstraction import HU200_EQUITY_SCHEMA
from src.blueprint.action_translation import TranslationOptions
from src.blueprint.average import AveragePolicy
from src.blueprint.hu200_river_search import HU200RiverSearchPlayer, HU200RiverSearchPolicy
from src.blueprint.hu20_turn_solver import ExternalTurnSolver, parse_profiles, profile_rows
from src.blueprint.river_ledger import DecisionLedger, decode_matrix
from src.policies.files import file_hash


def same_numbers(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        raise ValueError("Numeric shape/unavailable mask differs")
    finite = ~np.isnan(a)
    if not np.array_equal(a[finite].view(np.uint64), b[finite].view(np.uint64)):
        raise ValueError("Finite output bits differ")


def scientific_request(request):
    return {k: v for k, v in request.items() if k not in
            ("mode", "dump_path", "seconds", "profile_format")}


def run(args):
    declaration = json.loads((Path(__file__).resolve().parents[1] /
        'docs/reports/hu200-river-search-artifacts/cost-declaration.json').read_text())
    if file_hash(args.plan) != declaration['full_plan_sha256']:
        raise ValueError('Fixed24 sample changed')
    plan = json.loads(args.plan.read_text())
    if plan['config'] != asdict(CONFIG) or len(plan['sample']) != 24:
        raise ValueError('Search work differs')
    baseline_summary = json.loads((args.baseline/'summary.json').read_text())
    if not baseline_summary['all_search_work_completed'] or baseline_summary['decisions'] != 24:
        raise ValueError('Incomplete original baseline')
    args.out.mkdir(parents=True, exist_ok=False)
    base = AveragePolicy(args.model, MODEL_SHA256, expected_schema=HU200_EQUITY_SCHEMA,
        card_buckets=args.tables, cache=args.cache, cache_read_only=True,
        translation=TranslationOptions(512, 128))
    put(args.out/'inputs.json',dict(plan_sha256=file_hash(args.plan),blueprint=base.description,
        baseline_inputs_sha256=file_hash(args.baseline/'inputs.json'),
        binary_sha256=file_hash(args.binary),no_strength_hands=True))
    rows = []
    for index, spot in enumerate(plan['sample']):
        folder = args.out/str(index)
        old = json.loads((args.baseline/str(index)/'decision.json').read_text())
        view = SlumbotAdapter().observation(spot['decision']['public'],spot['decision']['hand_id'])
        solver = ExternalTurnSolver(args.binary,folder/'native',compact_profiles=True)
        policy = HU200RiverSearchPolicy(base,solver,CONFIG,ledger=folder/'ledger')
        pin_earlier_blueprint(policy,view)
        player = HU200RiverSearchPlayer(policy,seed=20261016317000+index)
        action = player.choose_action(view)
        if player.last_decision['fallback']:
            raise ValueError('Parity search fell back')
        baseline_path = Path(old['native_records'][-1]['path'])
        manifest = json.loads((baseline_path/'manifest.json').read_text())
        for name in ('request.json','profile.jsonl'):
            if file_hash(baseline_path/name) != manifest[name]['sha256']:
                raise ValueError('Original baseline evidence changed')
        expected_request = json.loads((baseline_path/'request.json').read_text())
        current = policy.live_solutions[view.seat,view.history]
        if scientific_request(expected_request) != current.request:
            raise ValueError('Final search request/ranges/menu/locks differ')
        expected = parse_profiles(expected_request,baseline_path/'profile.jsonl')
        if expected.keys() != current.profiles.keys():
            raise ValueError('Public profile inventory differs')
        for key, matrix in expected.items():
            other = current.profiles[key]
            if matrix.menu != other.menu or matrix.holdings != other.holdings:
                raise ValueError('Profile action/holding identity differs')
            same_numbers(matrix.probabilities,other.probabilities)
            same_numbers(matrix.action_values_chips,other.action_values_chips)
        old_ledgers = list((args.baseline/str(index)/'ledger').glob('*.json'))
        if len(old_ledgers) != 1:
            raise ValueError('Ambiguous original decision ledger')
        old_ledger = DecisionLedger(old_ledgers[0].parent,replay=True).read(old_ledgers[0].stem)
        old_matrix = decode_matrix(old_ledger['matrix'])
        new_matrix = policy.played[view.seat,view.history][0]
        if (old_matrix.menu != new_matrix.menu or old_matrix.holdings != new_matrix.holdings
                or old_ledger['supported'] != sorted(map(list,policy.played[view.seat,view.history][3]['supported']))):
            raise ValueError('Played completion/support differs')
        same_numbers(old_matrix.probabilities,new_matrix.probabilities)
        if asdict(action) != old['search_action']:
            raise ValueError('Same-seed action differs')
        for field in ('action_values_chips','model_ev_chips','blueprint_ev_chips','model_ev_delta_chips'):
            same_numbers(old[field],player.last_decision[field])
        optimized_path = Path(solver.records[-1]['path'])
        # Also compare every raw exported diagnostic, beyond policy parser inputs.
        count = 0
        for before, after in zip(profile_rows(baseline_path/'profile.jsonl'),
                profile_rows(optimized_path/'profile.jsonl',compact=True),strict=True):
            for field in ('line','board','player','actions','holdings','opponent_holdings'):
                if before[field] != after[field]:
                    raise ValueError('Raw profile identity differs: '+field)
            for field in ('strategy','action_values_chips','own_weights','opponent_weights'):
                same_numbers(before[field],after[field])
            count += 1
        replay_solver = ExternalTurnSolver(args.binary,folder/'replay-native-forbidden',compact_profiles=True)
        replay = HU200RiverSearchPolicy(base,replay_solver,CONFIG,ledger=folder/'ledger',replay_ledger=True)
        pin_earlier_blueprint(replay,view)
        replay_action = HU200RiverSearchPlayer(replay,seed=20261016317000+index).choose_action(view)
        if replay_action != action or replay_solver.records:
            raise ValueError('Fresh replay differs or invokes solver')
        row = dict(index=index,status='exact-parity',nodes=count,
            baseline_solves=len(old['native_records']),optimized_solves=len(solver.records),
            scientific_request_identical=True,all_raw_and_normalized_finite_bits_identical=True,
            unavailable_masks_identical=True,played_matrix_support_action_ev_identical=True,
            fresh_replay_identical_without_solver=True,
            certified_ancestor_skips=policy.stats['certified_off_menu_ancestor_skips'])
        put(folder/'parity.json',row)
        rows.append(row)
        print(json.dumps(row),flush=True)
        del current,expected,policy,solver,replay,replay_solver
    result = dict(status='exact-parity',decisions=24,rows=rows,
        baseline_solves=sum(r['baseline_solves'] for r in rows),
        optimized_solves=sum(r['optimized_solves'] for r in rows),
        no_strength_hands=True,plan_sha256=file_hash(args.plan))
    put(args.out/'summary.json',result)
    print(json.dumps(result),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan','baseline','model','cache','tables','binary','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip() != 'Apple M4':
        raise ValueError('Scientific validation requires the authorized M4')
    if not any('scripts.guard_m4_research' in p.cmdline() for p in psutil.Process().parents()):
        raise ValueError('Scientific validation requires the resource guard')
    with (Path.home()/'Local/.hu20-m4-research.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == '__main__':
    main()
