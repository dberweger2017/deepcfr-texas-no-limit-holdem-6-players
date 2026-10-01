"""Density denominators and trainer observation do not alter learning."""
from collections import Counter
import json
import pytest

import scripts.preflight_hu20_history as preflight
from scripts.preflight_hu20_history import corpus, distribution, passes, snapshot, worker
from src.blueprint.artifact import load_training, save_training
from src.blueprint.solver import BlueprintTrainer, PilotConfig
from src.game.hand import Table


def plan():
    value=json.load(open('configs/blueprint/dr2x2-history-preflight.json'))
    value['probe']['blocks']=2
    value['milestones']=[250,500,1000,2000]
    return value


def test_histogram_and_materiality_gate():
    full=distribution(Counter({0:80,1:20}));compressed=distribution(Counter({100:90,2:10}))
    gate=plan()['gate']
    assert full['median']==0 and passes(full,compressed,gate)
    assert not passes(full,distribution(Counter({1:100})),gate)
    assert not passes(full,distribution(Counter({100:89,2:10})),gate)
    assert not passes(distribution(Counter()),compressed,gate)


@pytest.mark.parametrize('version,engine',[
    ((3,11,16),'5db20e3d5d6862b32a7402035c1340b622d3b005'),
    ((3,11,14),'different-engine'),
])
def test_worker_rejects_unpinned_runtime_before_training(tmp_path,monkeypatch,version,engine):
    runtime={'engine_origin':json.dumps({'vcs_info':{'commit_id':engine}})}
    monkeypatch.setattr(preflight,'environment',lambda:runtime)
    monkeypatch.setattr(preflight.sys,'version_info',version)
    with pytest.raises(ValueError,match='requires pinned Python'):
        worker(plan(),2026093001,'compressed',tmp_path/'missing-probe',tmp_path/'worker')
    assert not (tmp_path/'worker'/'attempt.json').exists()


def test_pinned_runtime_is_accepted():
    runtime={'engine_origin':json.dumps({'vcs_info':{'commit_id':'5db20e3d5d6862b32a7402035c1340b622d3b005'}})}
    preflight.validate_runtime(runtime,(3,11,14))


def test_worker_observer_preserves_complete_trainer_state(tmp_path,monkeypatch):
    # This unit test compares learning state on the CI interpreter. Runtime
    # admission is tested separately above; actual campaign workers stay pinned.
    checked=[]
    monkeypatch.setattr(preflight,'validate_runtime',lambda runtime,version:checked.append((runtime,version)))
    value=plan();probe=tmp_path/'probe';out=tmp_path/'worker'
    corpus(value,probe)
    result=worker(value,2026093001,'compressed',probe,out)
    assert len(checked)==1 and checked[0][0]==result['environment']
    assert result['status']=='complete' and len(result['milestones'])==4
    loaded=load_training(out/'checkpoint-2000.json.gz')
    direct=BlueprintTrainer(Table(('player-0','player-1'),(2000,2000)),loaded.config)
    for _ in range(loaded.iteration): direct.step()
    assert direct.nodes==loaded.nodes
    assert save_training(direct,tmp_path/'direct.gz')==result['milestones'][-1]['checkpoint_sha256']
    encounters=result['milestones'][-1]['density']['common_encounters']
    assert sum(row['count'] for row in encounters.values())==json.loads((probe/'manifest.json').read_text())['decisions']
    assert result['strength_outcomes_inspected'] is False
