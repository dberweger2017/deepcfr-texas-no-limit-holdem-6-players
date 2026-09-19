import gzip
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.continuous_holdem import compact_outputs, pin_comparison

from src.holdem.checkpoint import load_training, save_training


@pytest.mark.parametrize("arm", ["", "-branching", "-replay"])
def test_continuous_stop_recovery_and_rolling_retention(tmp_path, arm):
    root=Path(__file__).resolve().parents[1]
    recipe=json.loads((root/f'configs/holdem/continuous-m4{arm}.json').read_text())
    assert 'iterations' not in recipe and 'max_seconds' not in recipe
    recipe.update(seeds=[2026091999],blocks=2,save_every=1,evaluate_every=1)
    recipe['training'].update(traversals_per_player=1)
    if arm != '-replay':
        recipe['training']['capacity']=8
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


def test_pinned_comparison_survives_retirement(tmp_path):
    directory = tmp_path/'scenario-0-seed-1'
    directory.mkdir()
    for iteration in (1024, 1088, 1152):
        (directory/f'training-{iteration}.json').write_text('{}')
        (directory/f'training-{iteration}.pt').write_bytes(b'checkpoint')
        (directory/f'average-{iteration}.pt').write_bytes(b'policy')
    pin_comparison(tmp_path)
    pin_comparison(tmp_path)
    compact_outputs(directory)
    assert not (directory/'training-1024.pt').exists()
    assert (directory/'pinned/1024/training-1024.pt').read_bytes() == b'checkpoint'
    assert (directory/'pinned/1024/average-1024.pt').read_bytes() == b'policy'


def test_adopt_worker_without_restart(tmp_path):
    import time

    root = Path(__file__).resolve().parents[1]
    recipe = json.loads((root/'configs/holdem/continuous-m4.json').read_text())
    recipe.update(seeds=[2026091998], blocks=2, save_every=64, evaluate_every=64)
    recipe['training'].update(capacity=8, traversals_per_player=1)
    recipe['training']['fit'].update(width=4, steps=1, batch_size=2, diagnostic_samples=2)
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps(recipe))
    out = tmp_path/'run'
    child = subprocess.Popen([sys.executable, '-m', 'scripts.continuous_holdem',
                              '--worker', '--plan', str(plan), '--out', str(out)],
                             cwd=root, start_new_session=True)
    try:
        deadline = time.monotonic()+15
        while not (out/'status.json').exists():
            assert child.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        supervisor = subprocess.Popen([sys.executable, '-m', 'scripts.continuous_holdem',
                                       '--attach-worker', str(child.pid),
                                       '--plan', str(plan), '--out', str(out)], cwd=root)
        try:
            record_path = out.with_name('run-supervisor.json')
            while not record_path.exists():
                assert supervisor.poll() is None and time.monotonic() < deadline
                time.sleep(0.05)
            record = json.loads(record_path.read_text())
            assert record['worker_pid'] == child.pid
            assert record['memory_limit_bytes'] is None
            (out/'STOP').touch()
            assert child.wait(timeout=30) == 0
            assert supervisor.wait(timeout=15) == 0
        finally:
            if supervisor.poll() is None:
                supervisor.kill()
                supervisor.wait()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_output_budget_counts_pinned_hard_links_once(tmp_path):
    import os
    from scripts.local_fullgame import used_bytes

    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'x' * 1024)
    pinned = tmp_path / 'pinned'
    pinned.mkdir()
    os.link(checkpoint, pinned / checkpoint.name)
    assert used_bytes(tmp_path) == 1024
    checkpoint.unlink()
    assert used_bytes(tmp_path) == 1024
    (tmp_path / 'another.pt').write_bytes(b'x' * 1024)
    assert used_bytes(tmp_path) == 2048
