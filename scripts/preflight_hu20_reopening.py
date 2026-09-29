"""Outcome-free complete-iteration A/B resource measurement; never skip a failed root."""

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from time import perf_counter, time

from scripts.evaluate_robustness import play
from scripts.tp20_common import append, guard, interruptible, seal
from scripts.train_hu20 import rss, system, write_json
from src.arena.schedule import digest
from src.blueprint.abstraction import HU20_SCHEMA, HU20_UNCAPPED_SCHEMA
from src.blueprint.artifact import export_policy, load_training, save_training
from src.blueprint.lookup import TableDistribution
from src.blueprint.solver import (BlueprintTrainer, CollectionLimitExceeded, HU20_GAME,
                                  HU20_UNCAPPED_GAME, PilotConfig)
from src.blueprint.windowed import _hash
from src.diagnostics.robustness import LBRConfig
from src.game.hand import Table


def configuration(plan, seed, arm):
    return PilotConfig(seed=seed, raise_cap=2 if arm=='A' else None,
        abstraction=HU20_SCHEMA if arm=='A' else HU20_UNCAPPED_SCHEMA,
        game=HU20_GAME if arm=='A' else HU20_UNCAPPED_GAME,
        roots_per_seat=plan['roots_per_seat'], postflop_replicates=1,
        max_nodes=plan['limits']['max_nodes_per_iteration'],
        max_entries=plan['limits']['max_entries'],
        max_seconds=plan['limits']['max_seconds_per_iteration'])


class PreflightSource(TableDistribution):
    def __init__(self, trainer):
        super().__init__(trainer)
        self.raise_cap=trainer.config.raise_cap


def probe(trainer, plan, seed, arm, label, out):
    source=PreflightSource(trainer)
    spec={'name':f'{arm}-{seed}-{label}','players':2,'abstraction':trainer.config.abstraction,
          'dual_menu_telemetry':True}
    started=perf_counter();hands=0
    # Identical visible schedules for both target arms, outcomes suppressed.
    root=plan['preflight_root'] + plan['preflight_seeds'].index(seed)*10 + (label=='trained')
    with gzip.open(out/f'{label}-timing-hands.jsonl.gz','wt') as rows:
        def emit(row):
            nonlocal hands
            assert row.get('target_chips') is None
            rows.write(json.dumps(row,sort_keys=True)+'\n');rows.flush();hands+=1
        for rules,contract,blocks in [(('pressure',),'menu',4),(('pressure',),'native',4),(('lbr',),'menu',2)]:
            t=perf_counter()
            for b in range(blocks):
                for rotation in range(2):
                    play(source,spec,rules,contract,b,rotation,root,label,
                         LBRConfig(plan['chance_samples'],plan['lbr_seconds']),emit,True)
            append(out/'inference-timing.jsonl',{'profile':label,'rule':rules,'contract':contract,
                'blocks':blocks,'hands':blocks*2,'seconds':perf_counter()-t})
    return {'profile':label,'hands':hands,'seconds':perf_counter()-started}


def run(plan, seed, arm, out, deadline):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);interruptible()
    config=configuration(plan,seed,arm)
    trainer=BlueprintTrainer(Table(('player-0','player-1'),(2000,2000)),config)
    start=perf_counter();total=0
    result={'status':'incomplete','arm':arm,'seed':seed,'config':asdict(config),
            'plan_digest':digest(plan),'started':time(),'requested_nodes':plan['preflight_nodes'],
            'initial_regret_entries':0,'failure':None,'probes':[]}
    save_training(trainer,out/'last-completed.json.gz')
    durations=[];updates=Counter();nodes_by_street=Counter()
    try:
        guard(plan,out,deadline)
        result['probes'].append(probe(trainer,plan,seed,arm,'uniform',out))
        while total<plan['preflight_nodes']:
            guard(plan,out,deadline)
            t=perf_counter()
            report=trainer.step(cancelled=lambda:time()>=deadline)
            total+=report.nodes;durations.append(perf_counter()-t)
            updates.update(report.traverser_visits_by_street)
            nodes_by_street.update(report.attempted_work['nodes_by_street'])
            append(out/'iterations.jsonl',{**asdict(report),'updated_keys':None,
                'completed_nodes':total,'rss_bytes':rss()})
            # A complete checkpoint at the first iteration and regular work intervals.
            if trainer.iteration==1 or total>=result.get('last_saved_nodes',0)+100000:
                t=perf_counter();h=save_training(trainer,out/'last-completed.json.gz')
                append(out/'checkpoint-times.jsonl',{'iteration':trainer.iteration,'nodes':total,
                    'sha256':h,'seconds':perf_counter()-t})
                result['last_saved_nodes']=total
        result['probes'].append(probe(trainer,plan,seed,arm,'trained',out))
        t=perf_counter();checkpoint=save_training(trainer,out/'last-completed.json.gz')
        result['checkpoint_seconds']=perf_counter()-t
        t=perf_counter();result['policy_sha256']=export_policy(trainer,out/'current.json.gz')
        result['export_seconds']=perf_counter()-t
        # Deliberate bounded cancellation is a recovery check, not a discarded hard traversal.
        calls=0
        def cancel():
            nonlocal calls
            calls+=1
            return calls>2
        cancelled=False
        try:trainer.step(cancelled=cancel)
        except CollectionLimitExceeded:cancelled=True
        if not cancelled:raise RuntimeError('Cancellation did not interrupt a live iteration')
        after=save_training(trainer,out/'after-cancellation.json.gz')
        if checkpoint!=after:raise RuntimeError('Partial iteration changed saved regrets')
        resumed=load_training(out/'last-completed.json.gz')
        replay=save_training(resumed,out/'recovered.json.gz')
        if replay!=checkpoint:raise RuntimeError('Last complete checkpoint does not recover byte-identically')
        cancelled_nodes=trainer.last_attempt_nodes
        cancelled_work=trainer.last_attempt_work
        original_next=trainer.step(cancelled=lambda:time()>=deadline)
        resumed_next=resumed.step(cancelled=lambda:time()>=deadline)
        next_original=save_training(trainer,out/'recovery-next-original.json.gz')
        next_resumed=save_training(resumed,out/'recovery-next-resumed.json.gz')
        if next_original!=next_resumed:raise RuntimeError('Fresh checkpoint resume diverged at the next complete iteration')
        result['recovery']={'cancelled_nodes':cancelled_nodes,
            'discarded_work':cancelled_work,'before_sha256':checkpoint,
            'next_original_sha256':next_original,'next_resumed_sha256':next_resumed,
            'extra_replay_validation_nodes':original_next.nodes+resumed_next.nodes,
            'after_sha256':after,'recovered_sha256':replay,'published_iteration':trainer.iteration-1}
        trainer=load_training(out/'last-completed.json.gz')
        result['status']='complete'
    except Exception as exc:
        result['failure']=f'{type(exc).__name__}: {exc}'
        result['failed_iteration']=trainer.iteration+1
        result['discarded_nodes']=trainer.last_attempt_nodes
        result['discarded_work']=trainer.last_attempt_work
        # Preserve all completed updates, never turn a resource boundary into a payoff.
        result['last_checkpoint_sha256']=save_training(trainer,out/'last-completed.json.gz')
    result.update(completed_nodes=total,overshoot_nodes=max(0,total-plan['preflight_nodes']),
        completed_iterations=trainer.iteration,entries=len(trainer.nodes),
        iteration_seconds=durations,nodes_by_street=dict(nodes_by_street),updates_by_street=dict(updates),
        elapsed_seconds=perf_counter()-start,peak_rss_bytes=rss(),
        swap_after=system(['sysctl','vm.swapusage']),memory_pressure_after=system(['memory_pressure','-Q']))
    write_json(out/'result.json',result);seal(out)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',choices=('A','B'),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();r=run(json.loads(a.plan.read_text()),a.seed,a.arm,a.out,a.deadline)
    print(json.dumps(r,sort_keys=True));return r['status']!='complete'

if __name__=='__main__':raise SystemExit(main())
