"""Continue immutable HU20 lineages; backups require destination acknowledgements."""

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
from scripts.hu20_scaling_common import parent_trainer, specification
from scripts.benchmark_observation_reuse import fingerprint
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
               entries=len(trainer.nodes), seed=parent['seed'], permanent=permanent,
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
                   spec=specification(parent['seed'], requested, trainer.iteration, cp, p, h, ph))
        row['files'].append(dict(name=p.name, sha256=ph, bytes=p.stat().st_size))
    write(out / (ident + '.record.json'), row)
    append(out / 'saved.jsonl', row)
    return row


def run(plan, parent, out, control, resume=None):
    out.mkdir(parents=True, exist_ok=False)
    (out / 'ack').mkdir()
    stopping = []
    signal.signal(signal.SIGTERM, lambda signum, frame: stopping.append('SIGTERM'))
    signal.signal(signal.SIGINT, lambda signum, frame: stopping.append('SIGINT'))
    if resume:
        previous = json.loads(resume.read_text())
        path = Path(previous['checkpoint_path'])
        if _hash(path) != previous['checkpoint_sha256']:
            raise ValueError('Resume checkpoint differs from verified record')
        trainer = load_training(path)
        total = previous['completed_nodes']
        chain = previous['work_chain_sha256']
        counters = Counter(previous['cumulative_work'])
        if previous['original_parent_sha256'] != parent['checkpoint_sha256']:
            raise ValueError('Recovery has another original parent')
    else:
        trainer = parent_trainer(parent, plan['engineering']['initial_max_entries'])
        total = parent['completed_nodes']
        chain = parent['checkpoint_sha256']
        counters = Counter()
    original = parent_trainer_config(parent)
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

    def cancelled():
        if stopping:
            return True
        try:
            c = json.loads(control.read_text())
            return (c.get('stop') is not None or time.time() >= c['lease_until'])
        except (OSError, ValueError, KeyError):
            return True

    try:
        with gzip.GzipFile(filename='', mode='wb', fileobj=(out / 'iterations.jsonl.gz').open('wb'), mtime=0) as log:
            while total < plan['target_total_nodes']:
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
                    if type(value) in (int, float) and field not in ('iteration', 'lifetime_completed_nodes'):
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
    return state


def parent_trainer_config(parent):
    # Expected scientific fields are frozen independently of artifact metadata.
    return dict(seed=parent['seed'], raise_cap=None, roots_per_seat=1,
                max_nodes=250000, max_seconds=300, postflop_replicates=1,
                abstraction='hu20-native-reopening-ordered-history-card-v1',
                game='hu20-native-reopening-20bb-52card-no-ante-rake-v1')


def preflight(plan, parent, out, resume=None):
    """Outcome-free, fresh-process suffix parity at identical whole iterations."""
    out.mkdir(parents=True, exist_ok=False)
    if resume:
        prior = json.loads((resume / 'result.json').read_text())
        trainer = load_training(resume / 'midpoint.json.gz')
        completed = prior['midpoint_added_nodes']
        if save_training(trainer, out / 'reload.json.gz') != prior['midpoint']['sha256']:
            raise ValueError('Fresh-process reload bytes differ')
    else:
        trainer = parent_trainer(parent, plan['engineering']['initial_max_entries'])
        completed = 0
    if scientific_config(trainer.config) != parent_trainer_config(parent):
        raise ValueError('Preflight scientific configuration mismatch')
    result = dict(origin=parent, source=environment(), status='running')
    while completed < 25000:
        completed += trainer.step(workers=1).nodes
        if not resume and completed >= 12500 and 'midpoint_added_nodes' not in result:
            save_training(trainer, out / 'midpoint.json.gz')
            result.update(midpoint_added_nodes=completed, midpoint=fingerprint(out / 'midpoint.json.gz'))
    save_training(trainer, out / 'final.json.gz')
    export_policy(trainer, out / 'current.json.gz')
    result.update(status='complete', added_nodes=completed, iteration=trainer.iteration,
                  final=fingerprint(out / 'final.json.gz'), current=fingerprint(out / 'current.json.gz'),
                  next_streams=[_seed(trainer.config.seed, trainer.iteration+1, seat, 0, 'actions') for seat in (0, 1)])
    write(out / 'result.json', result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('train', 'preflight'))
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
    else:
        result = preflight(plan, parent, a.out, a.resume)
    print(json.dumps({k: result[k] for k in ('status',) if k in result}))
    return result['status'] != 'complete'


if __name__ == '__main__':
    raise SystemExit(main())
