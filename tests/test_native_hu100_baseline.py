"""Small fixtures for private streams, unchanged sampling and HU100 legality."""

from dataclasses import replace
import gzip
import json

import pytest

from scripts.evaluate_native_hu100_baseline import Probe, action_seed, execute, OPPONENTS
from scripts.audit_native_hu100_baseline import audit
from src.arena.policies import NativeHU100Uniform, make_policy
from src.arena.runner import run_schedule
from src.arena.schedule import Plan, Scenario, build_schedule
from src.blueprint.abstraction import HU100_SCHEMA, choices, information_key
from src.blueprint.artifact import FrozenBlueprint, save_training
from src.blueprint.average import AveragePolicy
from src.blueprint.solver import BlueprintTrainer, HU100_GAME, Node, PilotConfig
from src.diagnostics.cfr_average import extract
from src.game.hand import Hand, Table
from src.game.play import RandomPolicy
from src.game.types import ActionKind
from src.policies.files import file_hash


def test_quote_is_outcome_blind_reduces_counts_and_reserves_full_reproduction():
    from scripts.run_native_hu100_baseline import quote
    pilot = {'panel_costs': [{'seconds': 10}] * 5, 'wall_seconds': 60,
             'blocks_per_opponent': 16, 'bb_per_100': -99999}
    reproduction = {'panel_costs': [{'seconds': 10}] * 5, 'wall_seconds': 60}
    replay = {'seconds': 20, 'panels': {'payoff': -99999}}
    count = quote(pilot, replay, reproduction, 1800)
    assert count['blocks_per_opponent'] == 96
    assert count['final_hands'] == 1920
    assert count['predicted_seconds'] + count['closeout_reserve_seconds'] <= 1800
    pilot['bb_per_100'] = 99999; replay['panels']['payoff'] = 99999
    assert quote(pilot, replay, reproduction, 1800) == count
    assert quote(pilot, replay, reproduction, 100)['blocks_per_opponent'] == 0
    with pytest.raises(ValueError): quote(pilot, replay, reproduction, float('nan'))


def model_fixture(tmp_path):
    table = Table(('player-0', 'player-1'), (10000, 10000))
    hand = Hand.start(table, hand_id='fixture', seed=3); view = hand.observe(hand.actor)
    menu = choices(view, raise_cap=None, free_fold=False)
    trainer = BlueprintTrainer(table, PilotConfig(seed=123, raise_cap=None,
                               abstraction=HU100_SCHEMA, game=HU100_GAME))
    trainer.iteration = 2
    key = information_key(view, menu, schema=HU100_SCHEMA)
    trainer.nodes[key] = Node(tuple(c.name for c in menu), [0.] * len(menu),
                              [float(i + 1) for i in range(len(menu))], 10)
    checkpoint = tmp_path / 'training.gz'; save_training(trainer, checkpoint)
    average = tmp_path / 'average.gz'
    extract(checkpoint, {'name': 'fixture', 'seed': 123, 'iteration': 2,
                        'checkpoint_sha256': file_hash(checkpoint)}, average, expected_schema=HU100_SCHEMA)
    return hand, AveragePolicy(average, file_hash(average), expected_schema=HU100_SCHEMA), average


def test_telemetry_preserves_exact_actions_and_random_state(tmp_path):
    hand, model, _ = model_fixture(tmp_path)
    raw, measured = model.policy(71), model.policy(71)
    captured = []
    probe = Probe(measured, model, {'logical_player': 0, 'arm': 'candidate'}, captured.append)
    for _ in range(50):
        view = hand.observe(hand.actor)
        assert raw.choose_action(view) == probe.choose_action(view)
        assert raw.random.getstate() == measured.random.getstate()
    assert all(r['lookup'] == 'positive-mass-known-key' for r in captured)
    # Classification does not confuse a known uniform zero-mass key with a missing key.
    model.zero_mass = {captured[0]['key']}
    probe.choose_action(view); assert captured[-1]['lookup'] == 'zero-mass'
    model.entries = {}; probe.choose_action(view); assert captured[-1]['lookup'] == 'missing-key'


def test_uniform_reference_uses_exact_native_menu_and_weighted_sampler(tmp_path):
    hand, model, _ = model_fixture(tmp_path); model.entries = {}
    uniform, missing = NativeHU100Uniform(51), model.policy(51)
    for _ in range(100):
        assert uniform.choose_action(hand.observe(hand.actor)) == missing.choose_action(hand.observe(hand.actor))
        assert uniform._random.getstate() == missing.random.getstate()
    wrong = Hand.start(Table(('a', 'b'), (2000, 2000)), hand_id='wrong', seed=1)
    with pytest.raises(ValueError): uniform.choose_action(wrong.observe(wrong.actor))


def test_random_keeps_minraise_allin_beyond_native_reference_menu():
    hand = Hand.start(Table(('a', 'b'), (10000, 10000)), hand_id='random', seed=1)
    view = hand.observe(hand.actor); menu = choices(view, raise_cap=None, free_fold=False)
    raises = [(a, p) for a, p in RandomPolicy(1).distribution(view) if a.kind == ActionKind.RAISE]
    assert {a.raise_to for a, _ in raises} == {view.legal_actions.min_raise_to, view.legal_actions.max_raise_to}
    assert raises[0][1] == raises[1][1] == 1 / len(view.legal_actions.kinds) / 2
    assert not any(c.action.raise_to == view.legal_actions.max_raise_to for c in menu)


@pytest.mark.parametrize('opponent', OPPONENTS)
def test_unchanged_opponents_complete_hu100_paired_hands(opponent):
    plan = Plan((Scenario(opponent, (10000, 10000)),), candidate='native_hu100_uniform',
                baseline='native_hu100_uniform', opponents=(opponent,), blocks=2, root_seed=193)
    seeds = []
    def factory(name, seed):
        seeds.append(seed)
        return make_policy(name, seed)
    rows = []
    assert run_schedule(plan, build_schedule(plan), lambda r, t: rows.append(r), factory=factory,
                        seed_factory=lambda b, r, a, i: action_seed(plan.root_seed, b, r, a, i))
    assert len(rows) == 8 and len(seeds) == len(set(seeds)) == 16
    assert all(r['status'] == 'completed' and sum(r['net_chips']) == 0 for r in rows)


def test_full_small_adapter_replay_and_deterministic_reproduction(tmp_path):
    _, model, path = model_fixture(tmp_path)
    config = json.loads(open('configs/arena/hu100-playing-baseline-v1.json').read())
    config['model'].update(name='fixture', path=str(path), sha256=file_hash(path),
                           bytes=path.stat().st_size, entries=1, iteration=2,
                           source_checkpoint_sha256=model.description['source_checkpoint_sha256'])
    config_path = tmp_path / 'config.json'; config_path.write_text(json.dumps(config))
    import subprocess
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    # Qualification tests run on a committed checkout, as the real adapter requires.
    first = tmp_path / 'first'; second = tmp_path / 'second'
    result = execute(config_path, first, 2, 12345, source)
    assert result['hands'] == 40
    verified = audit(first, tmp_path / 'audit.json')
    assert verified['hands_replayed'] == 40 and verified['actions_replayed'] == result['decisions']
    assert execute(config_path, second, 2, 12345, source, reproduce=first)['reproduced_all_hands_and_decisions']
    # A recomputed outer file inventory cannot turn an illegal settlement into valid evidence.
    hand_file = first / 'random/hands.jsonl'
    rows = [json.loads(line) for line in hand_file.read_text().splitlines()]
    rows[0]['net_chips'][0] += 1; hand_file.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    manifest = json.loads((first / 'output-files.json').read_text())
    manifest['random/hands.jsonl'].update(bytes=hand_file.stat().st_size, sha256=file_hash(hand_file))
    (first / 'output-files.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError): audit(first, tmp_path / 'bad-audit.json')
