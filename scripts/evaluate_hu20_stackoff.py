"""Frozen post-Luna saved-checkpoint regression: no training or model changes."""

import argparse
import gc
import gzip
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from time import time

from scripts.evaluate_hu20 import rss, system, write_json
from scripts.evaluate_robustness import play
from scripts.evaluate_hu20_reopening import UniformPlayer
from scripts.play_robustness import replay_row
from src.arena.policies import make_policy
from src.arena.schedule import digest, stream_seed
from src.diagnostics.robustness import LBRConfig, LocalBestResponse, ReactiveAttack
from src.diagnostics.saved_hu20 import file_hash, load_saved
from src.diagnostics.selective_stackoff import VERSION, SelectiveStackoff
from src.diagnostics.stackoff_tails import RecordingOpponent, RecordingTarget, attach_snapshots, hand_tails



def opponent(panel, source, block, plan):
    seed = stream_seed(panel['root'], 'test', 'action', 2, block, 1)
    rule = panel['rule']
    if rule == 'selective_stackoff':
        return SelectiveStackoff(seed)
    if rule == 'lbr':
        return LocalBestResponse(source, stream_seed(panel['root'], 'test', 'opponent', 2, block, 1),
                                 LBRConfig(plan['chance_samples'], plan['lbr_seconds']))
    if rule == 'hu20_uniform':
        return UniformPlayer(seed)
    if panel['contract'] == 'secondary':
        return make_policy(rule, seed)
    return ReactiveAttack(rule, panel['contract'])


def swap_bytes():
    value = system(['sysctl', 'vm.swapusage']) or ''
    match = re.search(r'used = ([\d.]+)([MG])', value)
    return float(match[1]) * (1024 ** (2 if match[2] == 'M' else 3)) if match else 0


class Guard:
    def __init__(self, plan, out, started=None):
        self.limits, self.out = plan['limits'], out
        self.deadline = (started or time()) + self.limits['max_seconds']
        self.swap_before = swap_bytes()
        self.last_swap_check = 0

    def __call__(self):
        if time() >= self.deadline:
            raise TimeoutError('Frozen absolute deadline')
        if rss() > self.limits['max_rss_gib'] * 1024**3:
            raise MemoryError('Frozen RSS guard')
        if shutil.disk_usage(self.out).free < self.limits['min_free_gib'] * 1024**3:
            raise OSError('Frozen free disk guard')
        if time() - self.last_swap_check >= 5:
            self.last_swap_check = time()
            if swap_bytes() - self.swap_before > self.limits['max_swap_growth_gib'] * 1024**3:
                raise MemoryError('Frozen swap growth guard')


class Contexts:
    """Lowest public-context hashes in fixed strata; never select by outcome."""
    def __init__(self, plan):
        self.plan, self.groups = plan, {}

    def consider(self, row, spec):
        if spec['milestone'] != self.plan['inspection_origin_milestone']:
            return
        for index, action in enumerate(row['actions']):
            view = action['observation']
            if (action['logical_player'] != 0 or action['street'] != 'river'
                    or not view['trained'] or view['visits'] < self.plan['minimum_visits']
                    or not 0 < view['call_amount'] <= self.plan['small_call_chips']
                    or 3 * view['call_amount'] > view['pot']):
                continue
            previous = row['actions'][index - 1] if index else None
            if not previous or previous['logical_player'] != 1 or previous['kind'] != 'raise':
                continue
            kind = 'small_raise' if view['street_bet'] else 'small_bet'
            group = (spec['seed'], view['position'], kind)
            candidates = self.groups.setdefault(group, {})
            context_id = view['public_context_id']
            candidates.setdefault(context_id, {
                'id': context_id, 'origin_seed': spec['seed'], 'kind': kind,
                'public_context': view['public_context'],
                'origin': {k: row[k] for k in ('players', 'rotation', 'button', 'phase', 'block', 'deal_seed')},
                'prefix': [{k: a[k] for k in ('seat', 'kind', 'raise_to')} for a in row['actions'][:index]]})
            for unwanted in sorted(candidates)[self.plan['contexts_per_seed_position_kind']:]:
                del candidates[unwanted]

    def document(self):
        return {'contexts': [value for group in sorted(self.groups)
                             for _, value in sorted(self.groups[group].items())],
                'empty_strata': [{'seed': seed, 'position': position, 'kind': kind}
                                 for seed in sorted({m['seed'] for m in self.plan['models']})
                                 for position in ('button', 'big_blind')
                                 for kind in ('small_bet', 'small_raise')
                                 if not self.groups.get((seed, position, kind))]}


def recorded_hand(source, visits, spec, panel, block, rotation, plan, guard=lambda: None):
    decisions, rows = [], []
    rival = opponent(panel, source, block, plan)
    target = RecordingTarget(source, visits, decisions, guard)
    wrapped = RecordingOpponent(rival, decisions, guard)
    def emit(row):
        row['panel'] = panel['name']
        try:
            # Preserve failed prefixes without treating uncommitted decisions as actions.
            attach_snapshots(row, decisions[:len(row['actions'])])
            if isinstance(rival, LocalBestResponse):
                rival_actions = [a for a in row['actions'] if a['logical_player'] == 1]
                for action, telemetry in zip(rival_actions, rival.telemetry):
                    action['lbr'] = telemetry
            if row['status'] == 'complete':
                replay_row(row)
                row['native_replay_verified'] = True
                row['tails'] = hand_tails(row)
        except Exception as exc:
            row['status'] = 'failed'
            row['error'] = f'Diagnostic audit {type(exc).__name__}: {exc}'
        rows.append(row)
    try:
        play(target, spec, (panel['rule'],), panel['contract'], block, rotation,
             panel['root'], 'stackoff-v1', LBRConfig(plan['chance_samples'], plan['lbr_seconds']),
             emit, opponent_policies={1: wrapped})
    except Exception:
        if not rows:
            raise
    return rows[0]


def run(plan, inputs, out):
    out.mkdir(parents=True, exist_ok=False)
    started = time()
    guard = Guard(plan, out, started)
    contexts = Contexts(plan)
    write_json(out / 'plan.json', plan)
    manifest = {'source_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'plan_sha256': digest(plan), 'started': started, 'deadline': guard.deadline,
                'opponent_version': VERSION, 'python': sys.version,
                'platform': sys.platform, 'swap_before_bytes': guard.swap_before,
                'inputs': [], 'generated_sessions_only': True}
    write_json(out / 'manifest.json', manifest)
    total, attempts, status, failure = 0, [], 'complete', None
    try:
        for spec in plan['models']:
            guard()
            source, visits = load_saved(spec, inputs, guard)
            manifest['inputs'].append({'name': spec['name'], **source.description,
                                       'checkpoint_sha256': spec['checkpoint_sha256']})
            write_json(out / 'manifest.json', manifest)
            with gzip.open(out / (spec['name'] + '.hands.jsonl.gz'), 'wt') as handle:
                for panel in plan['panels']:
                    attempt = {'model': spec['name'], 'panel': panel['name'], 'blocks': 0,
                               'requested_blocks': panel['blocks'], 'started': time()}
                    attempts.append(attempt)
                    for block in range(panel['blocks']):
                        guard()
                        for rotation in (0, 1):
                            row = recorded_hand(source, visits, spec, panel, block, rotation, plan, guard)
                            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                            handle.flush()
                            total += 1
                            if row['status'] != 'complete':
                                raise RuntimeError(row['error'])
                            if panel['rule'] == 'selective_stackoff':
                                contexts.consider(row, spec)
                        attempt['blocks'] += 1
                        if block % 16 == 0:
                            write_json(out / 'progress.json', {'hands': total, 'attempt': attempt,
                                       'seconds': time() - started, 'peak_rss_bytes': rss()})
                    attempt['seconds'] = time() - attempt['started']
            del source, visits
            gc.collect()
    except Exception as exc:
        status, failure = 'incomplete', f'{type(exc).__name__}: {exc}'
    finally:
        write_json(out / 'contexts.json', contexts.document())
        write_json(out / 'attempts.json', attempts)
        result = {'status': status, 'failure': failure, 'hands': total, 'seconds': time() - started,
                  'peak_rss_bytes': rss(), 'swap_after_bytes': swap_bytes()}
        write_json(out / 'result.json', result)
        manifest['outputs'] = {p.name: {'bytes': p.stat().st_size, 'sha256': file_hash(p)}
                               for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}
        write_json(out / 'manifest.json', manifest)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.plan.read_text()), args.inputs, args.out)
    print(json.dumps(result))
    return result['status'] != 'complete'


if __name__ == '__main__':
    raise SystemExit(main())
