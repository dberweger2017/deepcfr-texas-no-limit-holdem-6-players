"""Schema-aware visits and disjoint, outcome-blind paired admission."""
from pathlib import Path
import pytest

from scripts.evaluate_hu20_cards_v2 import hand
from scripts.preflight_dr2x2_evaluation import measure_model
from src.blueprint.abstraction import HU20_COMPRESSED_SCHEMA, HU20_UNCAPPED_SCHEMA
from src.blueprint.artifact import export_policy, save_training, HU20_UNCAPPED_FORMAT
from src.blueprint.solver import BlueprintTrainer, PilotConfig, HU20_UNCAPPED_GAME
from src.diagnostics.saved_hu20 import load_saved
from src.diagnostics.stackoff_tails import RecordingTarget
from src.game.hand import Hand, Table


def fixture(tmp_path, schema, cell):
    table = Table(('seat0', 'seat1'), (2000, 2000))
    trainer = BlueprintTrainer(table, PilotConfig(seed=13, abstraction=schema,
                              game=HU20_UNCAPPED_GAME, raise_cap=None))
    trainer.step()
    cp = tmp_path / (cell + '-checkpoint.gz')
    policy = tmp_path / (cell + '-current.gz')
    spec = {'name': cell, 'cell': cell, 'version': cell, 'seed': 13,
            'players': 2, 'abstraction': schema, 'format': HU20_UNCAPPED_FORMAT,
            'checkpoint_path': str(cp), 'checkpoint_sha256': save_training(trainer, cp),
            'path': str(policy), 'sha256': export_policy(trainer, policy)}
    return spec, trainer


def test_compressed_visits_use_its_own_key_and_default_contract_stays_strict(tmp_path):
    spec, trainer = fixture(tmp_path, HU20_COMPRESSED_SCHEMA, 'C')
    with pytest.raises(ValueError, match='lineage'):
        load_saved(spec, Path('/'))
    source, visits = load_saved(spec, Path('/'), expected_schema=HU20_COMPRESSED_SCHEMA)
    decisions = []
    view = Hand.start(trainer.table, hand_id='visits', seed=19).observe(0)
    target = RecordingTarget(source, visits, decisions)
    target.distribution(view)
    row = decisions[0]
    assert row['visits'] == visits.get(row['key'], 0)
    assert row['trained'] == (row['key'] in source.entries)
    damaged = {**spec, 'checkpoint_sha256': '0' * 64}
    with pytest.raises(ValueError, match='hash'):
        load_saved(damaged, Path('/'), expected_schema=HU20_COMPRESSED_SCHEMA)


def test_arms_and_rotations_share_deals_but_timing_cannot_expose_payoffs(tmp_path):
    rows = []
    for schema, cell in ((HU20_UNCAPPED_SCHEMA, 'A'), (HU20_COMPRESSED_SCHEMA, 'C')):
        spec, _ = fixture(tmp_path, schema, cell)
        source, visits = load_saved(spec, Path('/'), expected_schema=schema)
        panel = {'name': 'uniform', 'rule': 'hu20_uniform', 'contract': 'menu', 'root': 202610120001}
        for rotation in (0, 1):
            rows.append(hand(source, visits, spec, panel, 0, rotation,
                        {'chance_samples': 4, 'lbr_seconds': 5}, resource_only=True))
        playing = hand(source, visits, spec, panel, 0, 0,
                       {'chance_samples': 4, 'lbr_seconds': 5}, resource_only=False)
        assert playing['deal_seed'] != rows[-1]['deal_seed']
        assert playing['native_replay_verified']
    assert len({row['deal_seed'] for row in rows}) == 1
    assert all(row['target_chips'] is None and row['net_chips_by_seat'] is None for row in rows)


def test_admission_only_returns_timing_and_input_identities(tmp_path):
    spec, _ = fixture(tmp_path, HU20_COMPRESSED_SCHEMA, 'C')
    plan = {'panels': [{'name': 'uniform', 'rule': 'hu20_uniform', 'contract': 'menu',
                       'timing_root': 202610120002}], 'timing_blocks': 1,
            'chance_samples': 4, 'lbr_seconds': 5}
    result = measure_model(spec, plan, lambda: None)
    assert result['outcome_blind'] and result['panels'][0]['hands'] == 2
    assert not any('chips' in k or 'payoff' in k for k in result)
