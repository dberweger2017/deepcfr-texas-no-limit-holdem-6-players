"""Readout requires all fixed contrasts and intact replay/reproduction receipts."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from scripts import run_nplayer_phase2_evaluation as runner
from scripts.nplayer_phase2_matrix import matrix


def test_evaluation_refuses_changed_matrix_before_any_play(tmp_path):
    manifest={'format':'nplayer-phase2-evaluation-v1','cases':matrix()}
    manifest['cases'][0]['blocks']+=1
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    with patch.object(runner.subprocess,'run') as execute:
        with pytest.raises(ValueError,match='predeclared matrix'):
            runner.run(path,'unused',tmp_path,tmp_path,tmp_path,tmp_path/'out')
        execute.assert_not_called()


def test_final_readout_refuses_missing_reproduction(tmp_path):
    from scripts.analyze_nplayer_arena import verified_readout
    trace=tmp_path/'trace';trace.write_text('original')
    other=tmp_path/'repeat';other.write_text('different')
    receipt=tmp_path/'receipt.json'
    receipt.write_text(json.dumps({'status':'passed','trace_sha256':runner.sha(trace)}))
    with pytest.raises(ValueError,match='exact reproduction required'):
        verified_readout(trace,receipt,other)


def test_manifest_preparation_requires_all_three_training_guards(tmp_path):
    from scripts.prepare_nplayer_phase2_evaluation import prepare
    training=tmp_path/'training';seed=training/'2026101541';seed.mkdir(parents=True)
    (seed/'result.json').write_text(json.dumps({'status':'passed'}))
    guard=tmp_path/'guards/seed-2026101541';guard.mkdir(parents=True)
    (guard/'result.json').write_text(json.dumps({'status':'failed'}))
    costs=tmp_path/'costs.json';costs.write_text(json.dumps({'status':'passed'}))
    with pytest.raises(ValueError,match='training/guard failure'):
        prepare(training,tmp_path/'unused',costs,tmp_path/'manifest.json')
    assert not (tmp_path/'manifest.json').exists()


def test_complete_controller_and_readout_on_native_script_fixture(tmp_path,monkeypatch):
    import os
    from scripts import read_nplayer_phase2_results as reader
    binary=Path(os.environ.get('NPLAYER_ARENA_BINARY','native/hu20-trainer/target/release/examples/nplayer_arena')).resolve()
    if not binary.exists(): pytest.skip('build nplayer_arena')
    case={'name':'fixture','singleton':'bot=random','pair':'bot=check-call','primary':True,
          'target_role':'singleton','blocks':3,'root':7714}
    manifest={'format':'nplayer-phase2-evaluation-v1','cases':[case],
              'arena_binary_sha256':runner.sha(binary),'models':{}}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    monkeypatch.setattr(runner,'matrix',lambda:[case])
    monkeypatch.setattr(reader,'matrix',lambda:[case])
    monkeypatch.setenv('NPLAYER_STOP_FILE',str(tmp_path/'stop'))
    runner.run(path,'fixture',binary.parent,tmp_path,tmp_path,tmp_path/'eval/fixture')
    guard=tmp_path/'guards/fixture';guard.mkdir(parents=True)
    (guard/'result.json').write_text(json.dumps({'status':'passed'}))
    result=reader.read_results(path,tmp_path/'eval',tmp_path/'guards')
    assert result['final_hands']==18
    assert result['primary_count']==1
    assert result['learning_rule_passed']==(result['rows'][0]['singleton']['lower95_bb100']>0)
    (guard/'result.json').write_text(json.dumps({'status':'failed'}))
    with pytest.raises(ValueError,match='unsettled contrast'):
        reader.read_results(path,tmp_path/'eval',tmp_path/'guards')
