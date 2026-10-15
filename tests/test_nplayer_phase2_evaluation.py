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
