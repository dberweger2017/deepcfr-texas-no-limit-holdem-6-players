"""Resume identity, honest street denominators and fixed reach corpora."""
from dataclasses import asdict
import json
import subprocess
import sys
import time

import pytest

import scripts.continue_hu20_history as continuation
from scripts.preflight_hu20_history import corpus, file_hash, write
from src.blueprint.artifact import load_training, save_training
from src.blueprint.solver import BlueprintTrainer, PilotConfig
from src.game.hand import Table


def fixture(tmp_path):
    plan = json.load(open('configs/blueprint/dr2x2-history-10m-followup.json'))
    plan['seeds'] = [7]; plan['milestones'] = [3000,4000,5000]
    plan['recovery_interval_nodes'] = 1000
    plan['policy_reach_probe']['blocks'] = 2
    parents = {}
    for cell in plan['schemas']:
        trainer = BlueprintTrainer(Table(('player-0','player-1'),(2000,2000)),
                                   PilotConfig(**plan['trainer'],seed=7,abstraction=plan['schemas'][cell]))
        total = 0
        while total < 2000: total += trainer.step().nodes
        cp = tmp_path/f'{cell}.gz'; digest = save_training(trainer,cp)
        result_path = tmp_path/f'{cell}.json'
        write(result_path,dict(status='complete',completed_nodes=total,iterations=trainer.iteration,
                               milestones=[dict(checkpoint_sha256=digest)]))
        parents[f'{cell}-7'] = dict(checkpoint=cp.name,checkpoint_sha256=digest,result=result_path.name,
            result_sha256=file_hash(result_path),completed_nodes=total,iteration=trainer.iteration)
    plan['parents'] = parents
    old = json.load(open('configs/blueprint/dr2x2-history-preflight.json'))
    old['probe']['blocks'] = 2
    corpus(old,tmp_path/'corpus')
    plan['probe']['decisions'] = 'corpus/decisions.jsonl.gz'
    plan['probe']['decisions_sha256'] = file_hash(tmp_path/'corpus/decisions.jsonl.gz')
    return plan


def test_parent_hash_is_checked_before_deserialization(tmp_path,monkeypatch):
    plan = fixture(tmp_path)
    plan['parents']['compressed-7']['checkpoint_sha256'] = 'wrong'
    monkeypatch.setattr(continuation,'load_training',lambda _: pytest.fail('Loaded a corrupt parent'))
    with pytest.raises(ValueError,match='before load'):
        continuation.parent(plan,tmp_path,7,'compressed')


def test_unknown_retained_streets_remain_explicit(tmp_path):
    plan = fixture(tmp_path); trainer,_,_ = continuation.parent(plan,tmp_path,7,'compressed')
    rows = continuation.read_rows(tmp_path/plan['probe']['decisions'],plan['probe']['decisions_sha256'])
    d = continuation.density(trainer,{},rows,'compressed')
    assert d['stored_unclassified']['count'] == len(trainer.nodes)
    assert all(v['count']==0 for v in d['stored_classified_only'].values())
    assert sum(v['count'] for v in d['common_encounters'].values()) == len(rows)


def test_frozen_reach_corpus_is_deterministic_and_has_no_payoffs(tmp_path):
    plan = fixture(tmp_path)
    first = continuation.reach_corpus(plan,tmp_path,tmp_path/'reach1')
    second = continuation.reach_corpus(plan,tmp_path,tmp_path/'reach2')
    assert first == second and first['terminal_payoffs_recorded'] is False
    for name,meta in first['files'].items():
        rows = continuation.read_rows(tmp_path/'reach1'/name,meta['sha256'])
        assert rows and {r['button'] for r in rows} == {0,1}
        assert all(set(row)=={'block','button','street','keys','reference','seed','fallback','action','menu'} for row in rows)


def test_continuation_matches_direct_work_and_fresh_process_resume(tmp_path,monkeypatch):
    plan = fixture(tmp_path)
    continuation.reach_corpus(plan,tmp_path,tmp_path/'reach')
    original = load_training(tmp_path/'compressed.gz')
    parent_spec = plan['parents']['compressed-7']
    total = parent_spec['completed_nodes']
    while total < 5000: total += original.step().nodes
    expected = save_training(original,tmp_path/'expected.gz')
    # Portable learning-state test; strict production runtime admission has
    # separate rejection coverage in test_hu20_history_preflight.
    monkeypatch.setattr(continuation,'validate_runtime',lambda *_: None)
    result = continuation.worker(plan,tmp_path,7,'compressed',tmp_path/'reach',tmp_path/'out',time.time()+60)
    assert result['status'] == 'complete'
    assert result['completed_nodes'] == total
    assert result['continuation_nodes'] == total-parent_spec['completed_nodes']
    assert result['milestones'][-1]['checkpoint_sha256'] == expected
    assert file_hash(tmp_path/'compressed.gz') == parent_spec['checkpoint_sha256']
    loaded = load_training(tmp_path/'out/checkpoint-5000.json.gz')
    original.step(); loaded.step()
    next_hash = save_training(original,tmp_path/'next.gz')
    assert save_training(loaded,tmp_path/'loaded-next.gz') == next_hash
    script = ('from pathlib import Path; from src.blueprint.artifact import load_training,save_training; '
              'import sys; t=load_training(Path(sys.argv[1]));t.step();print(save_training(t,Path(sys.argv[2])))')
    digest = subprocess.check_output([sys.executable,'-c',script,str(tmp_path/'out/checkpoint-5000.json.gz'),str(tmp_path/'fresh.gz')],text=True).strip()
    assert digest == next_hash
