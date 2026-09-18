import gzip
import json
from pathlib import Path
import subprocess
import sys

from src.holdem.checkpoint import load_training, save_training


def test_continuous_stop_recovery_and_rolling_retention(tmp_path):
    root=Path(__file__).resolve().parents[1]
    recipe=json.loads((root/'configs/holdem/continuous-m4.json').read_text())
    assert 'iterations' not in recipe and 'max_seconds' not in recipe
    recipe.update(seeds=[2026091999],blocks=2,save_every=1,evaluate_every=1)
    recipe['training'].update(capacity=8,traversals_per_player=1)
    recipe['training']['fit'].update(width=4,steps=1,batch_size=2,diagnostic_samples=2)
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(recipe))
    out=tmp_path/'run'
    subprocess.run([sys.executable,'-m','scripts.continuous_holdem','--worker',
                    '--plan',str(plan),'--out',str(out),'--stop-after','3'],
                   cwd=root,check=True,timeout=60)
    assert json.loads((out/'status.json').read_text())['state']=='stopped'
    job=out/'scenario-0-seed-2026091999'
    assert sorted(p.name for p in job.glob('training-*.pt'))==['training-2.pt','training-3.pt']
    assert sorted(p.name for p in job.glob('average-*.pt'))==['average-2.pt','average-3.pt']
    assert len(list(job.glob('outcomes-*.json.gz')))==6
    assert not list(job.glob('outcomes-*.json'))
    with gzip.open(job/'outcomes-1.json.gz','rt') as f:
        assert len(json.load(f))==24
    provenance=json.loads((out/'manifest.json').read_text())
    marker=json.loads((job/'training-3.json').read_text())
    trainer=load_training(job/'training-3.pt',marker['sha256'],manifest=provenance)
    assert trainer.iteration==3
    assert save_training(trainer,tmp_path/'recovered.pt',manifest=provenance)==marker['sha256']
    assert len(json.loads((job/'learning-curve.json').read_text()))==6
