"""Deployment failures and post-publication recovery use the production runner."""

from pathlib import Path
from time import time

import pytest

from scripts.hu20_reopening_common import write_cases
from scripts.hu20_scaling_common import specification
from scripts.train_hu20_scaling import run
from src.blueprint.artifact import export_policy, save_training, load_training
from src.blueprint.solver import BlueprintTrainer, PilotConfig, HU20_UNCAPPED_GAME
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
from src.blueprint.windowed import _hash
from src.game.hand import Table


def fixture(tmp_path):
    trainer = BlueprintTrainer(Table(('player-0', 'player-1'), (2000, 2000)),
        PilotConfig(seed=819, raise_cap=None, max_entries=10000, max_nodes=250000,
                    abstraction=HU20_UNCAPPED_SCHEMA, game=HU20_UNCAPPED_GAME))
    nodes = sum(trainer.step().nodes for _ in range(3))
    cp=tmp_path/'parent.gz';h=save_training(trainer,cp)
    observations=tmp_path/'observations.gz'
    write_cases({'independent_blocks':1,'independent_root':2026158001},observations)
    parent={'seed':819,'entries':len(trainer.nodes),'iteration':trainer.iteration,
            'completed_nodes':nodes,'checkpoint_path':str(cp),'checkpoint_sha256':h}
    plan={'limits':{'max_entries':10000,'max_rss_gib':10.5,'min_free_gib':8},
          'training_total_nodes':nodes+100,'milestones':[nodes+100],
          'recovery_nodes':1000,'recovery_seconds':900,
          'independent_path':str(observations),'independent_sha256':_hash(observations)}
    return trainer,parent,plan


@pytest.mark.parametrize('fault',['missing','hash'])
def test_missing_or_corrupt_fixture_fails_before_parent_load(tmp_path,monkeypatch,fault):
    _,parent,plan=fixture(tmp_path)
    if fault=='missing':Path(plan['independent_path']).unlink()
    else:plan['independent_sha256']='0'*64
    def forbidden(*args,**kwargs):raise AssertionError('Collection parent loaded before validation')
    monkeypatch.setattr('scripts.train_hu20_scaling.parent_trainer',forbidden)
    with pytest.raises((FileNotFoundError,ValueError)):
        run(plan,parent,tmp_path/'worker',time()+60)
    assert not (tmp_path/'worker/iterations.jsonl').exists()


def test_density_failure_does_not_discard_published_work(tmp_path,monkeypatch):
    _,parent,plan=fixture(tmp_path)
    def fail(*args,**kwargs):raise RuntimeError('injected post-publication density failure')
    monkeypatch.setattr('scripts.train_hu20_scaling.independent_density',fail)
    r=run(plan,parent,tmp_path/'worker',time()+60)
    assert r['status']=='incomplete' and r['failure_phase']=='milestone'
    assert r['discarded_nodes']==0 and r['discarded_work']=={} and r['failed_iteration'] is None
    assert r['completed_nodes']>=plan['milestones'][0]
    assert load_training(tmp_path/'worker/partial-last-completed.json.gz').iteration==r['completed_iterations']
    stages=(tmp_path/'worker/milestone-stages.jsonl').read_text()
    assert 'export_complete' in stages and 'density_started' in stages and 'density_complete' not in stages


def test_recovered_milestone_publishes_exact_original_before_any_step(tmp_path):
    trainer,parent,plan=fixture(tmp_path)
    policy=tmp_path/'policy.gz';ph=export_policy(trainer,policy)
    parent['recovered_milestones']=[{'requested_total_nodes':parent['completed_nodes'],
         'checkpoint_path':parent['checkpoint_path'],'checkpoint_sha256':parent['checkpoint_sha256'],
         'policy_path':str(policy),'policy_sha256':ph}]
    plan['milestones']=[parent['completed_nodes'],plan['training_total_nodes']]
    r=run(plan,parent,tmp_path/'worker',time()+60)
    assert r['status']=='complete'
    m=r['milestones'][0]
    assert m['recovered'] and m['milestone_training_steps']==0
    assert m['checkpoint_sha256']==parent['checkpoint_sha256'] and m['policy_sha256']==ph
    assert m['iteration']==parent['iteration'] and m['completed_nodes']==parent['completed_nodes']
    assert not (tmp_path/'worker'/f"checkpoint-{parent['completed_nodes']}.json.gz").exists()
    while trainer.iteration<r['completed_iterations']:trainer.step()
    assert save_training(trainer,tmp_path/'expected.gz')==r['milestones'][-1]['checkpoint_sha256']


def test_primary_and_diagnostic_tasks_partition_frozen_schedule():
    from scripts.evaluate_hu20_scaling import tasks
    plan={'cheap_blocks':4096,'lbr_blocks':2048,'secondary_blocks':256,'stress_root':1,
          'lbr_root':2,'secondary_root':3,'training_total_nodes':100000000}
    specs=[{'name':'B-final','arm':'B','milestone':100000000},
           {'name':'B-40','arm':'B','milestone':40000000},
           {'name':'A-reference','arm':'A','milestone':20000000}]
    full=list(tasks(plan,specs))
    primary=list(tasks({**plan,'panel_filter':'primary'},specs))
    diagnostic=list(tasks({**plan,'panel_filter':'diagnostic'},specs))
    assert len(primary)+len(diagnostic)==len(full)
    assert {row[1] for row in primary}=={'LBR-original-cap2','Pressure-native'}
    assert not {row[0]['name']+'--'+row[1] for row in primary}&{row[0]['name']+'--'+row[1] for row in diagnostic}


def test_m4_primary_replay_reports_pending_diagnostics(tmp_path):
    import json
    from scripts.evaluate_hu20_scaling import run as evaluate
    from scripts.report_hu20_scaling import audit,combine
    from scripts.finish_hu20_scaling import independent
    specs=[]
    for seed in (121,122,123):
        trainer=BlueprintTrainer(Table(('player-0','player-1'),(2000,2000)),
             PilotConfig(seed=seed,raise_cap=None,max_entries=10000,max_nodes=250000,
                         abstraction=HU20_UNCAPPED_SCHEMA,game=HU20_UNCAPPED_GAME))
        for m in (20000000,100000000):
            trainer.step();cp=tmp_path/f'cp-{seed}-{m}.gz';policy=tmp_path/f'policy-{seed}-{m}.gz'
            specs.append(specification(seed,m,trainer.iteration,cp,policy,save_training(trainer,cp),export_policy(trainer,policy)))
    models=tmp_path/'models.json';models.write_text(json.dumps(specs))
    plan={'training_total_nodes':100000000,'cheap_blocks':2,'lbr_blocks':2,'secondary_blocks':2,
          'stress_root':2026158101,'lbr_root':2026158102,'secondary_root':2026158103,
          'training_seeds':[121,122,123],'block_host_cycle':['m4'],'milestones':[100000000],
          'coordinator_models':str(models),'limits':{'max_rss_gib':10.5,'min_free_gib':8}}
    primary={**plan,'panel_filter':'primary'}
    r=evaluate(primary,specs,'m4',tmp_path/'eval',time()+120)
    assert r['status']=='complete' and r['hands']==48
    a=audit(primary,tmp_path/'eval',tmp_path/'audit')
    assert a['status']=='complete' and a['native_replayed_hands']==48
    report=combine(plan,[tmp_path/'audit/results.json'],tmp_path/'combined.json')
    assert report['status']=='incomplete' and report['pending_panels']
    assert report['primary']['LBR-original-cap2']['status']=='available'
    reference=independent(list((tmp_path/'eval').glob('*.jsonl.gz')),[121,122,123],100000000)
    assert reference['primary']['LBR-original-cap2']['bb100']==pytest.approx(report['primary']['LBR-original-cap2']['long_minus_20M']['bb100'])
