"""Prospectively frozen density continuation; no terminal-payoff reporting."""
import argparse
from collections import Counter
from dataclasses import asdict
import gc
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
from random import Random
import shutil
import subprocess
import sys
import time

from scripts.hu20_platform_pilot import environment
from scripts.preflight_hu20_history import (
    canonical, distribution, file_hash, report, rss, validate_runtime, write,
)
from src.blueprint.abstraction import choices, information_key
from src.blueprint.artifact import export_policy, load_training, save_training
from src.game.hand import Hand, Table


def checked_json(path, expected):
    if file_hash(path) != expected:
        raise ValueError(f"Input hash changed: {path}")
    return json.loads(path.read_text())


def parent(plan, inputs, seed, cell):
    """Hash before deserialization; checkpoint counters alone do not give nodes."""
    spec = plan['parents'][f'{cell}-{seed}']
    result = checked_json(inputs/spec['result'], spec['result_sha256'])
    cp = inputs/spec['checkpoint']
    if file_hash(cp) != spec['checkpoint_sha256']:
        raise ValueError('Parent checkpoint hash changed before load')
    if (result['status'] != 'complete' or result['completed_nodes'] != spec['completed_nodes']
            or result['iterations'] != spec['iteration']
            or result['milestones'][-1]['checkpoint_sha256'] != spec['checkpoint_sha256']):
        raise ValueError('Parent work/iteration receipt changed')
    trainer = load_training(cp)
    config = {**plan['trainer'], 'seed': seed, 'abstraction': plan['schemas'][cell]}
    if trainer.iteration != spec['iteration'] or asdict(trainer.config) != config:
        raise ValueError('Parent trainer configuration/iteration mismatch')
    return trainer, result, spec


def read_rows(path, expected):
    if file_hash(path) != expected:
        raise ValueError('Decision corpus hash changed')
    with gzip.open(path, 'rt') as source:
        return [json.loads(line) for line in source]


def weighted_index(rng, weights):
    threshold = rng.random()
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if threshold < cumulative:
            return index
    return len(weights)-1


def reach_corpus(plan, inputs, out):
    out.mkdir(parents=True, exist_ok=False)
    files = {}; counts = {}
    for seed in plan['seeds']:
        for reference in plan['schemas']:
            trainer, _, spec = parent(plan, inputs, seed, reference)
            policy = trainer.frozen()
            del trainer
            path = out/f'{reference}-{seed}.jsonl.gz'
            digest = sha256(); streets = Counter(); fallback = 0
            with path.open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as target:
                for block in range(plan['policy_reach_probe']['blocks']):
                    for button in (0, 1):
                        label = f"dr2x2-reach/{plan['policy_reach_probe']['root_seed']}/{seed}/{block}/{button}"
                        derive = lambda tag: int.from_bytes(sha256((label+'/'+tag).encode()).digest()[:8], 'big')
                        target_rng = Random(derive('target'))
                        opponent_rng = Random(derive('opponent'))
                        hand = Hand.start(Table(('player-0','player-1'), (2000,2000), button), hand_id=label, seed=derive('deal'))
                        while not hand.finished:
                            view = hand.observe(hand.actor)
                            menu = choices(view, raise_cap=None, free_fold=False)
                            if hand.actor == 0:
                                keys = {cell: information_key(view, menu, schema=schema) for cell,schema in plan['schemas'].items()}
                                index = weighted_index(target_rng, policy.distribution(view, menu))
                                missing = keys[reference] not in policy.entries
                                row = dict(block=block, button=button, street=view.street.value, keys=keys,
                                           reference=reference, seed=seed, fallback=missing,
                                           action=asdict(menu[index].action), menu=[x.name for x in menu])
                                encoded = canonical(row)+b'\n'; target.write(encoded); digest.update(encoded)
                                streets[view.street.value] += 1; fallback += missing
                            else:
                                index = opponent_rng.randrange(len(menu))
                            hand = hand.apply(menu[index].action)
            files[path.name] = dict(sha256=file_hash(path), bytes=path.stat().st_size)
            counts[f'{reference}-{seed}'] = dict(counts_by_street=dict(streets), fallback=fallback,
                canonical_rows_sha256=digest.hexdigest(), parent_checkpoint_sha256=spec['checkpoint_sha256'])
            del policy; gc.collect()
    result = dict(plan_sha256=sha256(canonical(plan)).hexdigest(), files=files, counts=counts,
                  terminal_payoffs_recorded=False, strength_outcomes_inspected=False)
    write(out/'manifest.json', result)
    return result


def density(trainer, classified, probes, cell):
    stored = {s: Counter() for s in ('preflop','flop','turn','river')}
    unknown = Counter()
    for key, node in trainer.nodes.items():
        street = classified.get(key)
        if street is None:
            unknown[node.visits] += 1
        else:
            stored[street][node.visits] += 1
    encounters = {s: Counter() for s in stored}; unique = {s: {} for s in stored}
    for row in probes:
        node = trainer.nodes.get(row['keys'][cell]); visits = node.visits if node else 0
        encounters[row['street']][visits] += 1
        unique[row['street']][row['keys'][cell]] = visits
    return dict(stored_classified_only={s: distribution(v) for s,v in stored.items()},
                stored_unclassified=distribution(unknown),
                common_encounters={s: distribution(v) for s,v in encounters.items()},
                common_unique_keys={s: distribution(Counter(v.values())) for s,v in unique.items()})


def worker(plan, inputs, seed, cell, reach, out, campaign_deadline):
    import src.blueprint.solver as solver
    runtime = environment(); validate_runtime(runtime, sys.version_info[:3])
    start = time.time(); deadline = min(campaign_deadline, start+plan['limits']['worker_seconds'])
    out.mkdir(parents=True, exist_ok=False)
    trainer, original_result, spec = parent(plan, inputs, seed, cell)
    parent_nodes = total = spec['completed_nodes']; continuation_seconds = 0.0
    probes = read_rows(inputs/plan['probe']['decisions'], plan['probe']['decisions_sha256'])
    reach_manifest = json.loads((reach/'manifest.json').read_text())
    if reach_manifest['plan_sha256'] != sha256(canonical(plan)).hexdigest():
        raise ValueError('Policy-reach plan mismatch')
    families = {ref: read_rows(reach/f'{ref}-{seed}.jsonl.gz', reach_manifest['files'][f'{ref}-{seed}.jsonl.gz']['sha256']) for ref in plan['schemas']}
    classified = {}
    for row in probes + [r for rows in families.values() for r in rows]:
        key = row['keys'][cell]
        if classified.setdefault(key, row['street']) != row['street']:
            raise ValueError('Probe key aliases streets')
    original_key = solver.information_key
    def record(view, menu, **kwargs):
        key = original_key(view, menu, **kwargs)
        if classified.setdefault(key, view.street.value) != view.street.value:
            raise ValueError('Training key aliases streets')
        return key
    result = dict(status='running', seed=seed, cell=cell, environment=runtime,
        source=runtime['source'], plan_sha256=sha256(canonical(plan)).hexdigest(), config=asdict(trainer.config),
        parent=spec, completed_nodes=total, continuation_nodes=0, iterations=trainer.iteration,
        milestones=[], recoveries=[], failure=None, strength_outcomes_inspected=False)
    result['parent_policy_reach'] = {ref: density(trainer, classified, rows, cell)['common_encounters'] for ref,rows in families.items()}
    write(out/'attempt.json', result)
    solver.information_key = record
    interval = plan['recovery_interval_nodes']; next_save = (total//interval+1)*interval
    try:
        with (out/'iterations.jsonl').open('x') as log:
            while total < plan['milestones'][-1]:
                if time.time() >= deadline or rss() > plan['limits']['max_rss_gib']*2**30 or shutil.disk_usage(out).free < plan['limits']['min_free_gib']*2**30:
                    raise RuntimeError('Continuation time/RSS/disk guard')
                step = trainer.step(workers=1, cancelled=lambda: time.time() >= deadline)
                total += step.nodes; continuation_seconds += step.elapsed_seconds
                row = asdict(step); row.pop('updated_keys'); row['completed_nodes'] = total
                log.write(json.dumps(row, sort_keys=True)+'\n')
                if total >= next_save:
                    before = time.time(); cp = out/f'checkpoint-{next_save}.json.gz'
                    digest = save_training(trainer, cp)
                    if file_hash(cp) != digest: raise ValueError('Recovery checkpoint hash mismatch')
                    receipt = dict(requested_nodes=next_save, completed_nodes=total, overshoot=total-next_save,
                        iteration=trainer.iteration, entries=len(trainer.nodes), checkpoint_sha256=digest,
                        checkpoint_bytes=cp.stat().st_size, save_seconds=time.time()-before,
                        continuation_seconds=continuation_seconds, continuation_nodes=total-parent_nodes,
                        nodes_per_continuation_second=(total-parent_nodes)/continuation_seconds,
                        wall_seconds=time.time()-start, peak_rss_bytes=rss())
                    result['recoveries'].append(receipt)
                    if next_save in plan['milestones']:
                        result['milestones'].append({**receipt, 'density': density(trainer, classified, probes, cell),
                            'policy_reach': {ref: density(trainer, classified, rows, cell)['common_encounters'] for ref,rows in families.items()}})
                    next_save += interval
                    result.update(completed_nodes=total, continuation_nodes=total-parent_nodes, iterations=trainer.iteration)
                    write(out/'progress.json', result)
            before = time.time(); exp = out/'current.json.gz'; digest = export_policy(trainer, exp)
            result.update(status='complete', export_sha256=digest, export_bytes=exp.stat().st_size, export_seconds=time.time()-before)
            write(out/'classified-streets.json', classified)
    except Exception as exc:
        result.update(status='failed', failure=f'{type(exc).__name__}: {exc}', discarded_work=trainer.last_attempt_work,
                      partial_checkpoint_sha256=save_training(trainer, out/'partial-last-completed.json.gz'))
    finally:
        solver.information_key = original_key
    result.update(completed_nodes=total, continuation_nodes=total-parent_nodes, iterations=trainer.iteration,
                  peak_rss_bytes=rss(), wall_seconds=time.time()-start)
    write(out/'result.json', result)
    return result


def run(plan, plan_path, inputs, out, expected_source):
    runtime = environment(); validate_runtime(runtime, sys.version_info[:3])
    if runtime['source'] != expected_source: raise ValueError('Frozen source changed')
    checked_json(inputs/plan['original_plan'], plan['original_plan_sha256'])
    checked_json(inputs/plan['probe']['manifest'], plan['probe']['manifest_sha256'])
    out.mkdir(parents=True, exist_ok=False)
    state = dict(pid=os.getpid(), status='running', source=expected_source, plan_sha256=sha256(canonical(plan)).hexdigest(),
                 started=time.time(), completed=[], active=None, child_pid=None)
    deadline = state['started']+plan['limits']['campaign_seconds']
    shutil.copyfile(plan_path, out/'frozen-plan.json')
    def phase(name, args):
        if environment()['source'] != expected_source: raise ValueError('Frozen source changed between phases')
        state.update(active=name); write(out/'supervisor.json', state)
        with (out/f'{name}.log').open('xb') as log:
            child = subprocess.Popen([sys.executable, '-m', 'scripts.continue_hu20_history', *args], stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid; write(out/'supervisor.json', state)
            while child.poll() is None:
                if time.time() >= deadline+60:
                    child.terminate()
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
                    raise RuntimeError('Campaign hard guard; latest atomic recoveries retained')
                time.sleep(5)
            state['completed'].append(dict(task=name, exit_code=child.returncode)); state['child_pid'] = None
            write(out/'supervisor.json', state)
            if child.returncode: raise RuntimeError(f'Preserved failed phase {name}; no retry')
    common = ['--plan',str(plan_path),'--inputs',str(inputs)]
    try:
        phase('policy-reach', ['reach',*common,'--out',str(out/'policy-reach')])
        for seed in plan['seeds']:
            for cell in plan['schemas']:
                name = f'{cell}-{seed}'
                phase(name, ['worker',*common,'--seed',str(seed),'--cell',cell,'--reach',str(out/'policy-reach'),
                             '--out',str(out/name),'--deadline',str(deadline)])
                d = json.loads((out/name/'result.json').read_text())
                if d['status'] != 'complete': raise RuntimeError(f'Preserved failed worker {name}')
        state.update(active='density-report'); write(out/'supervisor.json', state)
        gate = report(plan, out)
        state.update(status='complete', density_gate=gate['status'])
    except Exception as exc:
        state.update(status='blocked', failure=f'{type(exc).__name__}: {exc}')
    finally:
        state.update(active=None, child_pid=None, finished=time.time()); write(out/'supervisor.json', state)
    return state


def main():
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=('run','reach','worker'))
    p.add_argument('--plan',type=Path,required=True); p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--source'); p.add_argument('--reach',type=Path)
    p.add_argument('--seed',type=int); p.add_argument('--cell',choices=('full','compressed')); p.add_argument('--deadline',type=float)
    a = p.parse_args(); plan = json.loads(a.plan.read_text())
    if a.phase == 'run': result = run(plan,a.plan.resolve(),a.inputs.resolve(),a.out.resolve(),a.source)
    elif a.phase == 'reach':
        validate_runtime(environment(),sys.version_info[:3]); result = reach_corpus(plan,a.inputs,a.out)
    else: result = worker(plan,a.inputs,a.seed,a.cell,a.reach,a.out,a.deadline)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','failure','completed_nodes','density_gate')}))
    return result.get('status') in ('failed','blocked')


if __name__ == '__main__': raise SystemExit(main())
