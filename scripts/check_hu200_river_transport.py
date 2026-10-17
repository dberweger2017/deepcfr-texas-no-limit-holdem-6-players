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
from src.blueprint.river_evidence import COMPACT_RETENTION_RULE, header_semantic_sha256, read_request
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


class ReusedCompletedSolver:
    """Reparse a completed, hash-identical body after a checker-only stop.

    The old strict parser completed before its receipt was sealed. Its numeric
    body matches the retained baseline exactly; all node/holding/array metadata
    were validated against the identical request. Only process RSS was variable.
    This recovers completed work without invoking the native solver again.
    """
    def __init__(self, binary, original, baseline):
        self.expected_sha256 = file_hash(binary)
        proof = original/'reused-native-evidence.json'
        self.original = (json.loads(proof.read_text()) if proof.exists() else
            [json.loads(p.read_text()) for p in (original/'native').glob('*/receipt.json')])
        self.baseline = baseline
        self.records = []
        if len(self.original) != len(baseline):
            raise ValueError('Incomplete reusable native inventory')

    def solve(self, request, deadline, **kwargs):
        index = len(self.records)
        matched = [r for r in self.original if scientific_request(read_request(r['path'])) == request]
        if not matched:
            raise ValueError('No completed evidence matches the current request')
        record = matched[0]
        self.original.remove(record)
        source = Path(record['path'])
        baseline = Path(self.baseline[index]['path'])
        previous = json.loads((baseline/'manifest.json').read_text())
        if (record['status'] != 'completed' or record['executable_sha256'] != self.expected_sha256
                or scientific_request(read_request(source)) != request
                or scientific_request(read_request(baseline)) != request
                or record['profile_retention']['generated_files']['profile.f64'] != previous['profile.f64']
                or file_hash(baseline/'profile.f64') != previous['profile.f64']['sha256']):
            raise ValueError('Completed work cannot be reused')
        expected_semantics = header_semantic_sha256(baseline/'profile.jsonl')
        known_semantics = record['profile_retention'].get('header_semantic_sha256')
        if known_semantics is not None and known_semantics != expected_semantics:
            raise ValueError('Reusable header semantics differ')
        profiles = parse_profiles(read_request(baseline),baseline/'profile.jsonl',compact=True)
        record['profile_retention']['header_semantic_sha256'] = expected_semantics
        record['plumbing_reused_native_work'] = True
        record['reconstructed_header_semantics'] = 'prior strict request/metadata validation plus identical numeric body; process RSS excluded'
        self.records.append(record)
        return profiles


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
    if args.reuse_completed_root:
        reused = json.loads((args.reuse_completed_root/'summary.json').read_text())
        if reused['status'] != 'exact-parity' or reused['decisions'] != 24 or reused['plan_sha256'] != file_hash(args.plan):
            raise ValueError('Only a complete fixed24 parity root may be reused')
    args.out.mkdir(parents=True, exist_ok=False)
    base = AveragePolicy(args.model, MODEL_SHA256, expected_schema=HU200_EQUITY_SCHEMA,
        card_buckets=args.tables, cache=args.cache, cache_read_only=True,
        translation=TranslationOptions(512, 128))
    put(args.out/'inputs.json',dict(plan_sha256=file_hash(args.plan),blueprint=base.description,
        baseline_inputs_sha256=file_hash(args.baseline/'inputs.json'),
        binary_sha256=file_hash(args.binary),no_strength_hands=True,
        profile_retention=COMPACT_RETENTION_RULE if args.sample_native_bodies else None,
        source_tree=subprocess.check_output(['git','rev-parse','HEAD:src'],text=True).strip()))
    rows = []
    for index, spot in enumerate(plan['sample']):
        folder = args.out/str(index)
        old = json.loads((args.baseline/str(index)/'decision.json').read_text())
        view = SlumbotAdapter().observation(spot['decision']['public'],spot['decision']['hand_id'])
        reusable = (args.reuse_completed_root/str(index) if args.reuse_completed_root else
                    args.reuse_completed_spot if index == 0 else None)
        solver = (ReusedCompletedSolver(args.binary,reusable,old['native_records'])
            if reusable else
            ExternalTurnSolver(args.binary,folder/'native',compact_profiles=True,
                profile_retention=COMPACT_RETENTION_RULE if args.sample_native_bodies else None))
        policy = HU200RiverSearchPolicy(base,solver,CONFIG,ledger=folder/'ledger')
        pin_earlier_blueprint(policy,view)
        player = HU200RiverSearchPlayer(policy,seed=20261016317000+index)
        action = player.choose_action(view)
        if player.last_decision['fallback']:
            raise ValueError('Parity search fell back')
        baseline_path = Path(old['native_records'][-1]['path'])
        manifest = json.loads((baseline_path/'manifest.json').read_text())
        for name in ('request.json','profile.jsonl',*(['profile.f64'] if args.baseline_compact else [])):
            if file_hash(baseline_path/name) != manifest[name]['sha256']:
                raise ValueError('Original baseline evidence changed')
        expected_request = json.loads((baseline_path/'request.json').read_text())
        current = policy.live_solutions[view.seat,view.history]
        if scientific_request(expected_request) != current.request:
            raise ValueError('Final search request/ranges/menu/locks differ')
        expected = parse_profiles(expected_request,baseline_path/'profile.jsonl',compact=args.baseline_compact)
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
        if args.sample_native_bodies:
            if not args.baseline_compact or len(solver.records) != len(old['native_records']):
                raise ValueError('Retention parity requires the retained compact baseline')
            for generated, previous in zip(solver.records,old['native_records'],strict=True):
                previous_path = Path(previous['path'])
                previous_files = json.loads((previous_path/'manifest.json').read_text())
                request_now = read_request(generated['path'])
                if scientific_request(request_now) != scientific_request(read_request(previous_path)):
                    raise ValueError('An ancestor scientific request changed')
                for name,pin in generated['profile_retention']['generated_files'].items():
                    if name == 'profile.f64' and (pin != previous_files[name]
                            or file_hash(previous_path/name) != pin['sha256']):
                        raise ValueError('Generated numeric compact body bytes differ')
                semantics = generated['profile_retention'].get('header_semantic_sha256')
                if semantics != header_semantic_sha256(previous_path/'profile.jsonl'):
                    raise ValueError('Compact scientific header metadata differs')
                if set(generated['profile_retention']['generated_files']) != {'profile.jsonl','profile.f64'}:
                    raise ValueError('Missing complete raw checksum inventory')
            count = len(current.profiles)
        else:
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
        row['reused_native_solves'] = sum(r.get('plumbing_reused_native_work',False) for r in solver.records)
        if row['reused_native_solves']:
            put(folder/'reused-native-evidence.json',solver.records)
        put(folder/'parity.json',row)
        rows.append(row)
        print(json.dumps(row),flush=True)
        del current,expected,policy,solver,replay,replay_solver
    result = dict(status='exact-parity',decisions=24,rows=rows,
        baseline_solves=sum(r['baseline_solves'] for r in rows),
        optimized_solves=sum(r['optimized_solves'] for r in rows),
        no_strength_hands=True,plan_sha256=file_hash(args.plan),
        profile_retention=COMPACT_RETENTION_RULE if args.sample_native_bodies else None)
    put(args.out/'summary.json',result)
    print(json.dumps(result),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan','baseline','model','cache','tables','binary','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument("--sample-native-bodies",action="store_true")
    parser.add_argument("--baseline-compact",action="store_true")
    parser.add_argument("--reuse-completed-spot",type=Path)
    parser.add_argument("--reuse-completed-root",type=Path)
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
