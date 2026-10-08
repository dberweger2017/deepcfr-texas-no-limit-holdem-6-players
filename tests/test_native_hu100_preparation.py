"""Small deterministic game, information and serialization fixtures; no campaign runs."""
import gzip
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from src.arena.catalog import Checkpoint
from src.arena.registry import PolicyRegistry, load_frozen
from src.arena.schedule import Plan, Scenario
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA, HU100_SCHEMA, choices, information_key
from src.blueprint.artifact import HU100_FORMAT, save_training
from src.blueprint.average import AveragePolicy, HU100_FORMAT as AVERAGE_FORMAT
from src.blueprint.solver import BlueprintTrainer, HU20_UNCAPPED_GAME, HU100_GAME, Node, PilotConfig
from src.diagnostics.cfr_average import extract, audit
from src.game.hand import Hand, Table
from src.game.types import Action, ActionKind
from src.policies.files import file_hash

BINARY = Path('native/hu20-trainer/target/debug/hu20-trainer')
if not BINARY.exists():
    BINARY = Path('native/hu20-trainer/target/release/hu20-trainer')


def fixture(bb):
    schema, game = (HU100_SCHEMA, HU100_GAME) if bb == 100 else (HU20_UNCAPPED_SCHEMA, HU20_UNCAPPED_GAME)
    table = Table(('player-0', 'player-1'), (bb * 100,) * 2)
    hand = Hand.start(table, hand_id='fixture', seed=3)
    view = hand.observe(hand.actor)
    menu = choices(view, raise_cap=None, free_fold=False)
    config = PilotConfig(seed=123, raise_cap=None, abstraction=schema, game=game)
    trainer = BlueprintTrainer(table, config)
    trainer.iteration = 2
    key = information_key(view, menu, schema=schema)
    trainer.nodes[key] = Node(tuple(c.name for c in menu), [float(i) for i in range(len(menu))],
                              [float(i + 1) for i in range(len(menu))], 10)
    return hand, menu, trainer


def test_game_keys_are_disjoint_and_hidden_cards_do_not_enter_hu100_key():
    hand, menu, trainer = fixture(100)
    view = hand.observe(hand.actor)
    with pytest.raises(ValueError):
        information_key(view, menu, schema=HU20_UNCAPPED_SCHEMA)
    old, old_menu, _ = fixture(20)
    assert information_key(view, menu, schema=HU100_SCHEMA) != information_key(old.observe(old.actor), old_menu, schema=HU20_UNCAPPED_SCHEMA)
    # Swap only unrevealed opponent cards/deck suffix, preserving the owner's cards.
    deck = list('2c 2d Ac Ad 3c 4d 5h 6s 7c'.split()) + [r+s for r in '23456789TJQKA' for s in 'cdhs' if r+s not in '2c 2d Ac Ad 3c 4d 5h 6s 7c'.split()]
    a = Hand.from_deck(trainer.table, hand_id='private', deck=tuple(deck))
    deck[0], deck[8] = deck[8], deck[0]
    b = Hand.from_deck(trainer.table, hand_id='private', deck=tuple(deck))
    x, y = a.observe(a.actor), b.observe(b.actor)
    assert x.hole_cards == y.hole_cards
    assert information_key(x, choices(x, raise_cap=None, free_fold=False), schema=HU100_SCHEMA) == information_key(y, choices(y, raise_cap=None, free_fold=False), schema=HU100_SCHEMA)


@pytest.mark.parametrize('bb', [20, 100])
def test_native_and_python_exports_preserve_exact_fixture_rows(tmp_path, bb):
    if not BINARY.exists(): pytest.skip('build the native trainer')
    hand, menu, trainer = fixture(bb)
    checkpoint = tmp_path/'checkpoint.gz'
    save_training(trainer, checkpoint)
    current, average, python_average = (tmp_path/n for n in ('current.gz', 'average.gz', 'python.gz'))
    subprocess.run([str(BINARY), 'export', str(checkpoint), '--current', str(current), '--average', str(average)], check=True, capture_output=True)
    spec = {'name': 'fixture', 'seed': trainer.config.seed, 'iteration': trainer.iteration,
            'checkpoint_sha256': file_hash(checkpoint), 'sha256': file_hash(current)}
    extract(checkpoint, spec, python_average, expected_schema=trainer.config.abstraction)
    lines = lambda p: [json.loads(line) for line in gzip.open(p, 'rt')]
    assert lines(average) == lines(python_average)
    assert audit(checkpoint, current, average, spec, file_hash(average), expected_schema=trainer.config.abstraction)['status'] == 'verified'
    policy = AveragePolicy(average, file_hash(average), expected_schema=trainer.config.abstraction)
    assert policy.distribution(hand.observe(hand.actor))[2]
    other, _, _ = fixture(20 if bb == 100 else 100)
    with pytest.raises(ValueError): policy.distribution(other.observe(other.actor))
    if bb == 100:
        with pytest.raises(ValueError): AveragePolicy(average, file_hash(average))
        cp = Checkpoint('fixture', str(average), file_hash(average), AVERAGE_FORMAT)
        loaded = load_frozen(cp, average)
        assert loaded.game == HU100_GAME
        plan = Plan((Scenario('hu100', stacks=(10000,10000)),), candidate='fixture', models=(cp,))
        registry = PolicyRegistry(plan)
        out = tmp_path/'snapshot'; out.mkdir(); registry.snapshot(out)
        assert file_hash(out/'models'/f'{cp.sha256}.json.gz') == cp.sha256
        with pytest.raises(ValueError, match='game differs'):
            PolicyRegistry(replace(plan, scenarios=(Scenario('wrong', stacks=(2000,2000)),)))


@pytest.mark.parametrize('bb', [20, 100])
def test_tiny_native_rules_menu_key_parity(tmp_path, bb):
    if not BINARY.exists(): pytest.skip('build the native trainer')
    from scripts.native_parity_fixtures import record
    path = tmp_path/'hands.jsonl'
    # Twelve reproducible correctness hands, not an arena or resource pilot.
    path.write_text(''.join(json.dumps(record(101+i, .35, .7, bb))+'\n' for i in range(12)))
    result = subprocess.run([str(BINARY), 'parity', str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('bb', [20,100])
def test_explicit_refund_short_allin_and_split_parity(tmp_path, bb):
    if not BINARY.exists(): pytest.skip('build the native trainer')
    deck = list('2c 3d 4h 5s Tc Jd Qh Ks Ac'.split())
    deck += [r+s for r in '23456789TJQKA' for s in 'cdhs' if r+s not in deck]
    schema = HU100_SCHEMA if bb == 100 else HU20_UNCAPPED_SCHEMA
    stack = bb * 100
    raise_ = lambda n: Action(ActionKind.RAISE, n)
    call, check, fold = (Action(k) for k in (ActionKind.CALL, ActionKind.CHECK, ActionKind.FOLD))
    sequences = [[raise_(300), fold], [raise_(stack-100), raise_(stack), call],
                 [call, check, check, check, check, check, check, check]]
    rows = []
    for i, sequence in enumerate(sequences):
        hand = Hand.from_deck(Table(('a','b'), (stack,stack)), hand_id='fixture', deck=tuple(deck))
        decisions = []
        for action in sequence:
            view = hand.observe(hand.actor); legal = view.legal_actions
            menu = choices(view, raise_cap=None, free_fold=False)
            decisions.append({'actor': hand.actor, 'street': view.street.value, 'pot': view.pot,
                'kinds': [k.value for k in legal.kinds], 'call': legal.call_amount,
                'min_raise_to': legal.min_raise_to, 'max_raise_to': legal.max_raise_to,
                'menu': [[c.name,c.action.raise_to] for c in menu], 'key': information_key(view,menu,schema=schema)})
            hand = hand.apply(action)
        assert hand.finished
        final = hand.events[-1].stacks
        assert sum(final) == 2*stack
        if i == 1: assert 'raise' not in decisions[-1]['kinds']
        if i == 2: assert final == (stack,stack)
        rows.append({'stack_bb': bb, 'seed': i, 'button': 0, 'deck': deck,
                     'decisions': decisions, 'actions': [[a.kind.value,a.raise_to] for a in sequence], 'final_stacks': list(final)})
    path = tmp_path/'explicit.jsonl'; path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result = subprocess.run([str(BINARY),'parity',str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_mislabeled_hu100_artifacts_reject_before_rows(tmp_path):
    _, _, trainer = fixture(100)
    checkpoint = tmp_path/'checkpoint.gz'; save_training(trainer, checkpoint)
    from src.blueprint.average import checked_header
    header = json.loads(gzip.open(checkpoint,'rt').readline())
    for field, value in [('table', {**header['table'], 'stacks': [2000,2000]}),
                         ('abstraction', HU20_UNCAPPED_SCHEMA),
                         ('identity', {**header['identity'], 'action_menu': 'unknown'})]:
        with pytest.raises(ValueError): checked_header({**header, field: value}, {'seed':123,'iteration':2}, expected_schema=HU100_SCHEMA)


@pytest.mark.parametrize("schema,game", [
    (HU20_UNCAPPED_SCHEMA, HU20_UNCAPPED_GAME),
    ("tp20-ordered-history-card-baseline-v1", "tp20-20bb-52card-no-ante-rake-v1"),
])
def test_legacy_table_rejection_keeps_20bb_diagnostic(schema,game):
    config=PilotConfig(abstraction=schema,game=game,raise_cap=None if schema==HU20_UNCAPPED_SCHEMA else 2)
    with pytest.raises(ValueError,match="20BB table"):
        BlueprintTrainer(Table(("a","b"),(10000,10000)),config)
