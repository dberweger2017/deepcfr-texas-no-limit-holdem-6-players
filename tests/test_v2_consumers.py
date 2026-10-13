"""Schema-owned menus must agree during training, play, tracing and independent replay."""
from pathlib import Path
import gzip
import json
import subprocess

import pytest

from scripts import evaluate_hu100_direct as direct, pot_pressure_diagnostic as pressure
from src.blueprint.abstraction import HU100_SCHEMA, HU100_MENU_V2_SCHEMA, information_key, schema_choices
from src.blueprint.artifact import save_training
from src.blueprint.average import AveragePolicy
from src.blueprint.lookup import TableDistribution
from src.blueprint.solver import BlueprintTrainer, HU100_GAME, Node, PilotConfig
from src.diagnostics.cfr_average import extract
from src.game.hand import Hand, Table
from src.policies.files import file_hash


def synthetic(tmp_path, schema, name):
    tmp_path.mkdir()
    table = Table(('player-0', 'player-1'), (10000, 10000))
    hand = Hand.start(table, hand_id='fixture', seed=3)
    view = hand.observe(hand.actor)
    menu = schema_choices(view, schema)
    trainer = BlueprintTrainer(table, PilotConfig(seed=123, raise_cap=None, abstraction=schema, game=HU100_GAME))
    trainer.iteration = 2
    key = information_key(view, menu, schema=schema)
    trainer.nodes[key] = Node(tuple(c.name for c in menu), [0.] * len(menu), [float(i + 1) for i in range(len(menu))], 20)
    checkpoint, average = tmp_path / 'training.gz', tmp_path / 'average.gz'
    save_training(trainer, checkpoint)
    extract(checkpoint, {'name': name, 'seed': 123, 'iteration': 2, 'checkpoint_sha256': file_hash(checkpoint)},
            average, expected_schema=schema)
    model = AveragePolicy(average, file_hash(average), expected_schema=schema)
    return dict(name=name, path=str(average), sha256=file_hash(average),
                format='holdem-hu100-stored-cfr-average-research-v1', abstraction=schema,
                bytes=average.stat().st_size, entries=1, iteration=2,
                source_checkpoint_sha256=model.description['source_checkpoint_sha256'])


def dev_source(monkeypatch):
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    real = subprocess.check_output
    monkeypatch.setattr(subprocess, 'check_output', lambda cmd, **kw: '' if cmd[:2] == ['git', 'status'] else real(cmd, **kw))
    return source


@pytest.mark.parametrize('left,right', [(HU100_SCHEMA, HU100_SCHEMA), (HU100_MENU_V2_SCHEMA, HU100_MENU_V2_SCHEMA),
                                       (HU100_SCHEMA, HU100_MENU_V2_SCHEMA), (HU100_MENU_V2_SCHEMA, HU100_SCHEMA)])
def test_direct_play_trace_replay_and_reproduction_use_each_policy_schema(tmp_path, monkeypatch, left, right):
    specs = [synthetic(tmp_path / 'left', left, 'left'), synthetic(tmp_path / 'right', right, 'right')]
    source = dev_source(monkeypatch)
    _, done = direct.execute(specs, 4, 2026100922201, 'fixture', tmp_path / 'play', source)
    result = direct.audit(tmp_path / 'play', tmp_path / 'audit.json')
    assert result['hands_replayed'] == done['hands'] == 8
    _, repeat = direct.execute(specs, 4, 2026100922201, 'fixture', tmp_path / 'repeat', source, reproduce=tmp_path / 'play')
    assert repeat['reproduced_all_hands_and_decisions']
    with gzip.open(tmp_path / 'play/decisions.jsonl.gz', 'rt') as stream:
        traces = list(map(json.loads, stream))
    for spec in specs:
        names = {c['name'] for row in traces if row['policy'] == spec['name'] for c in row['menu']}
        if spec['abstraction'] == HU100_MENU_V2_SCHEMA:
            assert {'pot-1.5', 'pot-2'} <= names
        else:
            assert not {'half', 'pot-1.5', 'pot-2'} & names


@pytest.mark.parametrize('schema', [HU100_SCHEMA, HU100_MENU_V2_SCHEMA])
def test_pot_pressure_play_and_independent_replay(tmp_path, monkeypatch, schema):
    spec = synthetic(tmp_path / 'model', schema, 'candidate')
    source = dev_source(monkeypatch)
    _, done = pressure.execute(spec, 'v1', 4, 20261013100101, tmp_path / 'play', source)
    result = pressure.audit(tmp_path / 'play', tmp_path / 'audit.json')
    assert done['hands'] == 8
    assert result['status'] == 'verified'


def test_tiny_python_v2_training_stores_schema_menu_and_keys(monkeypatch):
    from src.blueprint import solver
    table = Table(('player-0', 'player-1'), (10000, 10000))
    config = PilotConfig(seed=7, raise_cap=None, abstraction=HU100_MENU_V2_SCHEMA, game=HU100_GAME,
                         max_nodes=30000, max_seconds=30)
    trainer = BlueprintTrainer(table, config)
    real = solver.choices
    witnessed = {}
    def capture(view, **kw):
        menu = real(view, **kw)
        assert menu == schema_choices(view, config.abstraction)
        witnessed[information_key(view, menu, schema=config.abstraction)] = tuple(c.name for c in menu)
        return menu
    monkeypatch.setattr(solver, 'choices', capture)
    trainer.step()
    assert trainer.nodes
    names = {name for node in trainer.nodes.values() for name in node.names}
    assert {'half', 'pot-1.5', 'pot-2'} <= names
    assert all(witnessed[key] == node.names for key, node in trainer.nodes.items())
    lookup = TableDistribution(trainer)
    initial = Hand.start(table, hand_id='lookup', seed=2).observe(0)
    # Opening actor is the button in heads-up.
    assert lookup.distribution(initial)[0] == schema_choices(initial, config.abstraction)


def test_unfinished_consumers_refuse_v2_before_work(tmp_path):
    from src.blueprint.windowed import NativePreflopAdapter, WindowedDistribution
    from src.diagnostics.subgame_bench import SubgameTrainer
    from scripts.preflight_hu20_history import corpus
    from scripts.continue_hu20_history import reach_corpus
    from src.blueprint.hu20_turn_search import HU20TurnSearchPolicy
    table = Table(('a', 'b'), (10000, 10000))
    trainer = BlueprintTrainer(table, PilotConfig(raise_cap=None, abstraction=HU100_MENU_V2_SCHEMA, game=HU100_GAME))
    with pytest.raises(ValueError, match='menu-v2'): NativePreflopAdapter(trainer)
    with pytest.raises(ValueError, match='menu-v2'): WindowedDistribution(tmp_path/'absent', {'abstraction':HU100_MENU_V2_SCHEMA}, 'C')
    with pytest.raises(ValueError, match='menu-v1'): SubgameTrainer([], seed=1, schema=HU100_MENU_V2_SCHEMA)
    plan = {'schemas': {'v2': HU100_MENU_V2_SCHEMA}}
    with pytest.raises(ValueError, match='menu-v2'): corpus(plan, tmp_path/'corpus')
    with pytest.raises(ValueError, match='menu-v2'): reach_corpus(plan, tmp_path, tmp_path/'reach')
    from types import SimpleNamespace
    with pytest.raises(ValueError, match='menu-v2'): HU20TurnSearchPolicy(SimpleNamespace(abstraction=HU100_MENU_V2_SCHEMA), None)


def test_hu200_witness_diagnosis_refuses_v2_identity():
    from types import SimpleNamespace
    from scripts.evaluate_hu200_diagnosis import play_hand
    from src.blueprint.abstraction import HU200_MENU_V2_SCHEMA
    with pytest.raises(ValueError, match="menu-v1"):
        play_hand(SimpleNamespace(abstraction=HU200_MENU_V2_SCHEMA), 1, "check_call", 0, 0, {}, [0.])


@pytest.mark.parametrize("schema", [
    "hu20-native-reopening-ordered-history-card-menu-v2",
    "hu100-native-reopening-ordered-history-card-menu-v2",
    "hu200-native-reopening-ordered-history-card-menu-v2",
])
def test_live_blueprint_rejects_actual_v2_trainer_before_search_wrappers(schema):
    from src.blueprint.abstraction import STACK_BY_SCHEMA
    from src.blueprint.search import LiveBlueprint
    from src.blueprint.solver import SHORTSTACK_GAMES
    stack = STACK_BY_SCHEMA[schema]
    trainer = BlueprintTrainer(Table(("a", "b"), (stack, stack)),
        PilotConfig(raise_cap=None, abstraction=schema, game=SHORTSTACK_GAMES[schema]))
    with pytest.raises(ValueError, match="menu-v2"):
        LiveBlueprint(trainer)


def test_live_blueprint_keeps_v1_trainer_distribution():
    from src.blueprint.search import LiveBlueprint
    from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
    from src.blueprint.solver import HU20_UNCAPPED_GAME
    table = Table(("a", "b"), (2000, 2000))
    trainer = BlueprintTrainer(table, PilotConfig(raise_cap=None, abstraction=HU20_UNCAPPED_SCHEMA,
                                                 game=HU20_UNCAPPED_GAME))
    wrapped = LiveBlueprint(trainer)
    view = Hand.start(table, hand_id="live", seed=17).observe(0)
    assert wrapped.distribution(view) == TableDistribution(trainer).distribution(view)


@pytest.mark.parametrize('schema', [HU100_SCHEMA, HU100_MENU_V2_SCHEMA])
def test_scripted_play_replays_candidate_and_v1_uniform_reference(tmp_path, monkeypatch, schema):
    from scripts import evaluate_native_hu100_baseline as scripted, audit_native_hu100_baseline as audit_scripted
    spec = synthetic(tmp_path/'model', schema, 'candidate')
    config = json.loads(Path('configs/arena/hu100-playing-baseline-v1.json').read_text())
    config['model'].update(spec)
    config['action_translation'] = None
    config_path = tmp_path/'config.json'
    config_path.write_text(json.dumps(config))
    source = dev_source(monkeypatch)
    done = scripted.execute(config_path, tmp_path/'play', 2, 12345, source)
    result = audit_scripted.audit(tmp_path/'play', tmp_path/'audit.json')
    assert result['status'] == 'verified'
    assert result['hands_replayed'] == done['hands'] == 40


def test_v1_k50_uniform_trace_keeps_candidate_key_context():
    from src.blueprint.abstraction import HU100_EQUITY_SCHEMA
    from scripts.evaluate_native_hu100_baseline import Probe
    from src.arena.policies import NativeHU100Uniform
    from types import SimpleNamespace
    table = Table(('a', 'b'), (10000, 10000))
    view = Hand.start(table, hand_id='legacy-context', seed=17).observe(0)
    menu = schema_choices(view, HU100_EQUITY_SCHEMA)
    model = SimpleNamespace(abstraction=HU100_EQUITY_SCHEMA, zero_mass=set(),
        distribution=lambda v: (menu, (1/len(menu),)*len(menu), False),
        information_key=lambda v,m: information_key(v,m,schema=HU100_EQUITY_SCHEMA))
    rows=[]
    Probe(NativeHU100Uniform(17), model, {'logical_player':0,'arm':'baseline'}, rows.append).choose_action(view)
    assert rows[0]['key'] == information_key(view, menu, schema=HU100_EQUITY_SCHEMA)
