from dataclasses import replace
from itertools import permutations
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.game.hand import Hand
from src.game.observation import BoardDealt, replay
from src.game.types import Street
from src.holdem.actions import bet_candidates
from src.holdem.betting import BettingNetwork
from src.holdem.checkpoint import load_policy, load_training, save_policy, save_training
from src.holdem.fitting import FitConfig
from src.holdem.training import HoldemTrainer, SampledTrainConfig
from src.solver.neural.network import deterministic_cpu
from tests.test_hand_observations import table
from tests.test_holdem_encoding import advance, change_suits


@pytest.fixture(autouse=True)
def cpu():
    with deterministic_cpu(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(39)
        yield


def test_paper_cards_and_public_history():
    hand = advance(Hand.start(table(6), hand_id='paper', seed=8), Street.TURN)
    view = hand.observe(hand.actor)
    original = bet_candidates(view)
    model = BettingNetwork(64, 'paper')
    expected = model([original])[0]
    assert len(expected.regrets) == len(original.actions)
    assert torch.isfinite(expected.values).all()
    for order in permutations('cdhs'):
        changed = bet_candidates(change_suits(view, dict(zip('cdhs', order))))
        torch.testing.assert_close(model([changed])[0].regrets, expected.regrets, rtol=0, atol=0)
    events = tuple(replace(e, cards=tuple(reversed(e.cards)))
                   if isinstance(e, BoardDealt) and e.street == Street.FLOP else e for e in view.history)
    changed = bet_candidates(replay(events, view.seat, tuple(reversed(view.hole_cards))))
    torch.testing.assert_close(model([changed])[0].regrets, expected.regrets, rtol=0, atol=0)
    # Card indicators in the event pathway cannot affect this architecture.
    changed = replace(original, decision=replace(original.decision, events=tuple(
        row[:-52] + tuple(1-x for x in row[-52:]) for row in original.decision.events)))
    torch.testing.assert_close(model([changed])[0].regrets, expected.regrets, rtol=0, atol=0)
    (expected.regrets.square().sum()+expected.values.square().sum()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_paper_recovery_and_resume(tmp_path):
    config = SampledTrainConfig(seed=701, capacity=16, traversals_per_player=1,
                               max_seconds=60, fit=FitConfig(width=64, steps=3,
                               batch_size=4, architecture='paper'))
    trainer = HoldemTrainer(table(4, (10,)*4), config)
    trainer.step()
    digest = save_training(trainer, tmp_path/'first.pt', manifest={})
    restored = load_training(tmp_path/'first.pt', digest, manifest={})
    assert save_training(restored, tmp_path/'copy.pt', manifest={}) == digest
    trainer.step(); restored.step()
    assert save_training(trainer, tmp_path/'next.pt', manifest={}) == save_training(restored, tmp_path/'next-copy.pt', manifest={})
    exported = save_policy(trainer, tmp_path/'policy.pt', manifest={})
    policy, _ = load_policy(tmp_path/'policy.pt', exported)
    hand = Hand.start(trainer.table, hand_id='recovered', seed=2)
    view = hand.observe(hand.actor)
    assert policy.distribution(view) == trainer.average_policy().distribution(view)


def test_day_campaign_finishes_fresh_final_and_retains_one(tmp_path):
    root = Path(__file__).resolve().parents[1]
    recipe = json.loads((root/'configs/holdem/day-paper-16k.json').read_text())
    recipe.update(seeds=[2026091996], save_every=1, blocks=2)
    recipe['training'].update(capacity=8, traversals_per_player=1)
    recipe['training']['fit'].update(steps=1, batch_size=2, diagnostic_samples=2)
    recipe['campaign'].update(iterations=3, evaluate_at=[1,2,3], random_at=[3], final_blocks=2)
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(recipe));out=tmp_path/'run'
    subprocess.run([sys.executable,'-m','scripts.continuous_holdem','--worker',
                    '--plan',str(plan),'--out',str(out)],cwd=root,check=True,timeout=90)
    assert json.loads((out/'result.json').read_text())['complete']
    job=out/'scenario-0-seed-2026091996'
    assert [p.name for p in job.glob('training-*.pt')] == ['training-3.pt']
    assert (job/'final/average-3.pt').stat().st_ino == (job/'average-3.pt').stat().st_ino
    assert json.loads((job/'final/evaluation-3.json').read_text())['status'] == 'valid'
