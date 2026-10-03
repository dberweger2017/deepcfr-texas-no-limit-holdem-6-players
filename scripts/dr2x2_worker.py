"""From-zero C/D lineages; science fixed, backups require destination acknowledgements."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.hu20_platform_pilot import canonical, environment, write
from src.blueprint.solver import BlueprintTrainer, PilotConfig
from src.game.hand import Table
from scripts.benchmark_observation_reuse import fingerprint, install_trace
from src.blueprint.artifact import export_policy, load_training, save_training
from src.blueprint.solver import _seed
from src.blueprint.windowed import _hash


def append(path, value):
    with path.open('ab') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def scientific_config(config):
    return {k: v for k, v in asdict(config).items() if k != 'max_entries'}


def export_fingerprint(path):
    result = fingerprint(path)
    normalized = sha256()
    with path.open('rb') as stream:
        header = stream.read(10)
        if len(header) != 10 or header[9] not in (3, 19):
            raise ValueError('Unexpected gzip export header')
        normalized.update(header[:9]+b'\0')
        for chunk in iter(lambda: stream.read(1048576), b''):
            normalized.update(chunk)
    result['os_normalized_sha256'] = normalized.hexdigest()
    return result


def next_boundary(total, boundaries):
    return next((value for value in boundaries if value > total), None)


def acknowledged(row, out):
    path = out / 'ack' / (row['id'] + '.json')
    if not path.exists():
        return False
    ack = json.loads(path.read_text())
    return ack.get('id') == row['id'] and ack.get('files') == row['files']


def rotate(rows, out):
    """Keep two verified recoveries, all unverified files, all permanent saves."""
    verified = [r for r in rows if acknowledged(r, out)]
    keep = {r['id'] for r in verified[-2:]}
    for row in verified:
        if row['permanent'] or row['id'] in keep:
            continue
        for item in row['files']:
            path = out / item['name']
            if path.exists():
                if _hash(path) != item['sha256']:
                    raise ValueError('Refuse to rotate changed artifact')
                path.unlink()


def checkpoint(trainer, plan, parent, out, requested, total, chain, counters,
               *, policy=False, permanent=False, label=None):
    ident = label or f'{requested}-{trainer.iteration}'
    cp = out / ('checkpoint-' + ident + '.json.gz')
    before = time.monotonic()
    h = save_training(trainer, cp)
    seconds = time.monotonic() - before
    row = dict(id=ident, requested_total_nodes=requested, completed_nodes=total,
               overshoot_nodes=max(0, total-requested), iteration=trainer.iteration,
               entries=len(trainer.nodes), cell=parent['cell'], seed=parent['seed'], permanent=permanent,
               original_parent_sha256=parent['checkpoint_sha256'],
               original_parent_nodes=parent['completed_nodes'],
               config=asdict(trainer.config), checkpoint_seconds=seconds,
               work_chain_sha256=chain, cumulative_work=dict(counters),
               published=time.time(), files=[dict(name=cp.name, sha256=h, bytes=cp.stat().st_size)],
               checkpoint_path=str(cp.resolve()), checkpoint_sha256=h)
    if policy:
        p = out / ('current-' + ident + '.json.gz')
        before = time.monotonic()
        ph = export_policy(trainer, p)
        row.update(export_seconds=time.monotonic()-before,
                   policy_sha256=ph)
        row['files'].append(dict(name=p.name, sha256=ph, bytes=p.stat().st_size))
    directory = os.open(out, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    write(out / (ident + '.record.json'), row)
    append(out / 'saved.jsonl', row)
    return row


def run(plan, parent, out, control, resume=None):
    out.mkdir(parents=True, exist_ok=False)
    (out / 'ack').mkdir()
    stopping = []
    old_term = signal.signal(signal.SIGTERM, lambda signum, frame: stopping.append('SIGTERM'))
    old_int = signal.signal(signal.SIGINT, lambda signum, frame: stopping.append('SIGINT'))
    if resume:
        previous = json.loads(resume.read_text())
        path = Path(previous['checkpoint_path'])
        if _hash(path) != previous['checkpoint_sha256']:
            raise ValueError('Resume checkpoint differs from verified record')
        trainer = load_training(path)
        if (trainer.iteration != previous['iteration'] or len(trainer.nodes) != previous['entries']
                or previous['seed'] != parent['seed']):
            raise ValueError('Resume counters differ from checkpoint')
        total = previous['completed_nodes']
        chain = previous['work_chain_sha256']
        counters = Counter(previous['cumulative_work'])
        if counters['nodes'] != total-parent['completed_nodes']:
            raise ValueError('Resume lifetime work counter mismatch')
        if previous['original_parent_sha256'] != parent['checkpoint_sha256']:
            raise ValueError('Recovery has another original parent')
    else:
        trainer = make_trainer(plan, parent)
        total = parent['completed_nodes']
        chain = sha256(canonical(parent)).hexdigest()
        counters = Counter()
    original = expected_config(plan, parent)
    if scientific_config(trainer.config) != original or trainer.config.seed != parent['seed']:
        raise ValueError('Scientific recipe changed on continuation')
    started = time.time()
    state = dict(status='running', started=started, parent=parent, initial_nodes=total,
                 initial_iteration=trainer.iteration, environment=environment(),
                 plan_sha256=sha256(canonical(plan)).hexdigest(), pid=os.getpid())
    write(out / 'attempt.json', state)
    rows = []
    next_save = next_boundary(total, plan['recovery_totals'])
    exports = set(plan['export_totals'])
    permanent = set(plan['permanent_totals'])
    training_seconds = 0.0
    unpublished = False
    last_progress = 0
    last_control_check = 0.0
    control_cancelled = False

    def cancelled():
        nonlocal last_control_check, control_cancelled
        if stopping:
            return True
        # This callback is invoked at every traversal node. File reads belong
        # to the control cadence, not the poker hot path.
        now = time.monotonic()
        if now-last_control_check < 1:
            return control_cancelled
        last_control_check = now
        try:
            c = json.loads(control.read_text())
            control_cancelled = c.get('stop') is not None or time.time() >= c['lease_until']
        except (OSError, ValueError, KeyError):
            control_cancelled = True
        return control_cancelled

    try:
        with (out / 'iterations.jsonl.gz').open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as log:
            while total < plan['target_total_nodes']:
                if rows and not acknowledged(rows[-1], out):
                    while not acknowledged(rows[-1], out):
                        if cancelled():
                            raise InterruptedError('Stop during backup acknowledgement')
                        time.sleep(2)
                if cancelled():
                    raise InterruptedError('Controller lease/stop request')
                # Entry count is an engineering bound. Memory is independently
                # guarded; no key is evicted and no scientific field is changed.
                if len(trainer.nodes) + trainer.config.max_nodes >= trainer.config.max_entries:
                    old = trainer.config.max_entries
                    trainer.config = replace(trainer.config, max_entries=max(old*2, len(trainer.nodes)+1000000))
                    append(out / 'engineering-incidents.jsonl', dict(time=time.time(),
                           kind='entry-bound-expansion', before=old, after=trainer.config.max_entries,
                           entries=len(trainer.nodes), iteration=trainer.iteration))
                unpublished = True
                before = time.monotonic()
                step = trainer.step(workers=1, cancelled=cancelled)
                training_seconds += time.monotonic()-before
                unpublished = False
                total += step.nodes
                row = asdict(step)
                row['updated_keys'] = None
                row['lifetime_completed_nodes'] = total
                stable = {k: v for k, v in row.items() if k not in ('elapsed_seconds', 'replay_seconds', 'worker_rss_sum_bytes')}
                chain = sha256(bytes.fromhex(chain) + canonical(stable)).hexdigest()
                row['work_chain_sha256'] = chain
                log.write(canonical(row)+b'\n')
                for field, value in stable.items():
                    if type(value) in (int, float) and field not in ('iteration', 'entries', 'lifetime_completed_nodes'):
                        counters[field] += value
                    elif isinstance(value, dict):
                        for street, amount in value.items():
                            if type(amount) in (int, float):
                                counters[field+':'+street] += amount
                if next_save is not None and total >= next_save:
                    log.flush()
                    saved = checkpoint(trainer, plan, parent, out, next_save, total, chain, counters,
                                       policy=next_save in exports, permanent=next_save in permanent)
                    rows.append(saved)
                    rotate(rows, out)
                    next_save = next_boundary(total, plan['recovery_totals'])
                if time.time()-last_progress >= 10:
                    last_progress = time.time()
                    write(out / 'progress.json', dict(completed_nodes=total, iteration=trainer.iteration,
                        entries=len(trainer.nodes), training_seconds=training_seconds,
                        additional_nodes=total-parent['completed_nodes'], heartbeat=time.time(),
                        nodes_per_second=(total-state['initial_nodes'])/max(.001, training_seconds)))
        if rows and not acknowledged(rows[-1], out):
            while not acknowledged(rows[-1], out):
                if cancelled():
                    raise InterruptedError('Stop before final backup acknowledgement')
                time.sleep(2)
        state['status'] = 'complete'
    except Exception as error:
        state.update(status='interrupted', failure=f'{type(error).__name__}: {error}',
                     discarded_nodes=trainer.last_attempt_nodes if unpublished else 0,
                     discarded_work=trainer.last_attempt_work if unpublished else {})
        rows.append(checkpoint(trainer, plan, parent, out, total, total, chain, counters,
                               label='partial-'+str(trainer.iteration), permanent=True))
    state.update(finished=time.time(), completed_nodes=total, iteration=trainer.iteration,
                 entries=len(trainer.nodes), training_seconds=training_seconds,
                 work_chain_sha256=chain, cumulative_work=dict(counters), saves=len(rows))
    write(out / 'result.json', state)
    signal.signal(signal.SIGTERM, old_term)
    signal.signal(signal.SIGINT, old_int)
    return state


def expected_config(plan, parent):
    return dict(plan['trainer'], seed=parent['seed'], abstraction=plan['cells'][parent['cell']]['abstraction'])


def make_trainer(plan, parent):
    if parent['cell'] not in plan['cells'] or parent['seed'] not in plan['seeds']:
        raise ValueError('Undeclared cell/seed')
    config = dict(expected_config(plan, parent), max_entries=plan['cells'][parent['cell']]['max_entries'])
    return BlueprintTrainer(Table(('player-0', 'player-1'), (2000, 2000)), PilotConfig(**config))


def preflight(plan, parent, out, resume=None):
    """Trace the identical post-midpoint suffix in independent processes."""
    out.mkdir(parents=True, exist_ok=False)
    if resume:
        prior = json.loads((resume / 'result.json').read_text())
        if fingerprint(resume / 'midpoint.json.gz') != prior['midpoint']:
            raise ValueError('Hash-before-load midpoint admission failed')
        trainer = load_training(resume / 'midpoint.json.gz')
        completed = prior['midpoint_added_nodes']
        if save_training(trainer, out / 'reload.json.gz') != prior['midpoint']['sha256']:
            raise ValueError('Fresh-process reload bytes differ')
    else:
        trainer = make_trainer(plan, parent)
        completed = 0
    if scientific_config(trainer.config) != expected_config(plan, parent):
        raise ValueError('Preflight scientific configuration mismatch')
    result = dict(origin=parent, source=environment(), status='running')
    trace = install_trace() if resume else None
    work = sha256()
    while completed < plan['preflight_nodes']:
        step = trainer.step(workers=1)
        completed += step.nodes
        if trace:
            stable = asdict(step)
            for key in ('updated_keys', 'elapsed_seconds', 'replay_seconds', 'worker_rss_sum_bytes'):
                stable.pop(key)
            work.update(canonical(stable)+b'\n')
        if not resume and completed >= plan['preflight_nodes']//2 and 'midpoint_added_nodes' not in result:
            save_training(trainer, out / 'midpoint.json.gz')
            result.update(midpoint_added_nodes=completed, midpoint=fingerprint(out / 'midpoint.json.gz'))
            trace = install_trace()
    save_training(trainer, out / 'final.json.gz')
    export_policy(trainer, out / 'current.json.gz')
    result.update(status='complete', added_nodes=completed, iteration=trainer.iteration,
                  final=fingerprint(out / 'final.json.gz'), current=export_fingerprint(out / 'current.json.gz'),
                  suffix_work_sha256=work.hexdigest(), suffix_trace=trace(),
                  next_streams=[_seed(trainer.config.seed, trainer.iteration+1, seat, 0, 'actions') for seat in (0, 1)])
    write(out / 'result.json', result)
    return result


def verify_saved(plan, parent, out, record):
    out.mkdir(parents=True, exist_ok=False)
    saved = json.loads(record.read_text())
    path = Path(saved['checkpoint_path'])
    if _hash(path) != saved['checkpoint_sha256']:
        raise ValueError('Final checkpoint transport mismatch')
    trainer = load_training(path)
    if (scientific_config(trainer.config) != expected_config(plan, parent)
            or trainer.iteration != saved['iteration'] or len(trainer.nodes) != saved['entries']):
        raise ValueError('Final reload scientific state / counters mismatch')
    h = save_training(trainer, out/'reloaded.json.gz')
    ph = export_policy(trainer, out/'current.json.gz')
    if h != saved['checkpoint_sha256'] or ph != saved['files'][1]['sha256']:
        raise ValueError('Final fresh-process checkpoint/export bytes differ')
    result = dict(status='complete', checkpoint_sha256=h, policy_sha256=ph,
                  iteration=trainer.iteration, entries=len(trainer.nodes),
                  completed_nodes=saved['completed_nodes'],
                  next_streams=[dict(seat=seat, deal=_seed(trainer.config.seed, trainer.iteration+1, seat, 0, 'deal'),
                       actions=_seed(trainer.config.seed, trainer.iteration+1, seat, 0, 'actions')) for seat in (0,1)])
    step = asdict(trainer.step(workers=1))
    for key in ('updated_keys', 'elapsed_seconds', 'replay_seconds', 'worker_rss_sum_bytes'):
        step.pop(key)
    result.update(next_work=step, next_checkpoint_sha256=save_training(trainer, out/'next.json.gz'),
                  next_policy_sha256=export_policy(trainer, out/'next-current.json.gz'))
    write(out/'result.json', result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('train', 'preflight', 'verify-final'))
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--control', type=Path)
    p.add_argument('--resume', type=Path)
    a = p.parse_args()
    plan, parent = json.loads(a.plan.read_text()), json.loads(a.parent.read_text())
    if a.phase == 'train':
        if a.control is None:
            p.error('Control lease required')
        result = run(plan, parent, a.out, a.control, a.resume)
    elif a.phase == 'preflight':
        result = preflight(plan, parent, a.out, a.resume)
    else:
        result = verify_saved(plan, parent, a.out, a.resume)
    print(json.dumps({k: result[k] for k in ('status',) if k in result}))
    return result['status'] != 'complete'


if __name__ == '__main__':
    raise SystemExit(main())
