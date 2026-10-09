"""HU200 fixed-game identity, complete training lineage and protocol admission."""
import gzip
import json
import subprocess
from pathlib import Path

import pytest

from src.blueprint.abstraction import HU100_SCHEMA, HU200_SCHEMA, information_key
from src.blueprint.average import AveragePolicy, HU200_FORMAT, checked_header
from src.blueprint.artifact import save_training
from src.blueprint.solver import BlueprintTrainer, HU100_GAME, HU200_GAME, PilotConfig
from src.game.hand import Hand, Table
from src.policies.files import file_hash
from tests.test_native_hu100_preparation import fixture

BINARY = Path('native/hu20-trainer/target/release/hu20-trainer')


def test_hu200_identity_and_private_information(tmp_path):
    hand, menu, trainer = fixture(200)
    key = information_key(hand.observe(hand.actor), menu, schema=HU200_SCHEMA)
    with pytest.raises(ValueError):
        information_key(hand.observe(hand.actor), menu, schema=HU100_SCHEMA)
    for table in (Table(('a', 'b'), (10000, 10000)), Table(('a', 'b'), (20000, 10000))):
        with pytest.raises(ValueError, match='200BB table'):
            BlueprintTrainer(table, trainer.config)
    # Own cards/public prefix stay fixed while unrevealed opponent cards change.
    deck = [r+s for r in '23456789TJQKA' for s in 'cdhs']
    a = Hand.from_deck(trainer.table, hand_id='private', deck=tuple(deck))
    deck[0], deck[10] = deck[10], deck[0]
    b = Hand.from_deck(trainer.table, hand_id='private', deck=tuple(deck))
    from src.blueprint.abstraction import choices
    x, y = a.observe(a.actor), b.observe(b.actor)
    assert x.hole_cards == y.hole_cards
    assert information_key(x, choices(x, raise_cap=None, free_fold=False), schema=HU200_SCHEMA) == information_key(y, choices(y, raise_cap=None, free_fold=False), schema=HU200_SCHEMA)
    path = tmp_path/'fixture.gz'; save_training(trainer, path)
    h = json.loads(gzip.open(path, 'rt').readline())
    for field, value in [('table', {**h['table'], 'stacks': [10000,10000]}),
                         ('abstraction', HU100_SCHEMA), ('format', 'holdem-hu100-native-reopening-blueprint-v1'),
                         ('identity', {**h['identity'], 'action_menu': 'hu100-min-pot-conditional-jam-native-reopening-v1'})]:
        bad = {**h, field: value}
        with pytest.raises(ValueError): checked_header(bad, {'seed':123, 'iteration':2}, expected_schema=HU200_SCHEMA)


@pytest.mark.skipif(not BINARY.exists(), reason='build native trainer')
def test_hu200_split_resume_is_byte_identical(tmp_path):
    full, part, resumed = [tmp_path/n for n in ('full.gz','part.gz','resumed.gz')]
    base = [str(BINARY), 'train', '--stack-bb', '200', '--seed', '2026100905', '--average-rule', 'opponent-sampled', '--nodes', '1000000000']
    for path, count, extra in [(full,80,[]),(part,31,[]),(resumed,80,[])]:
        # Bind the saved parent only after it exists.
        if path == resumed: extra = ['--resume',str(part),'--resume-sha256',file_hash(part)]
        subprocess.run([*base,'--iterations',str(count),'--out',str(path),*extra],check=True,capture_output=True)
    assert file_hash(full) == file_hash(resumed)
    refused = subprocess.run([str(BINARY),'train','--stack-bb','100','--nodes','1000000000','--resume',str(part),'--resume-sha256',file_hash(part),'--out',str(tmp_path/'wrong.gz')],capture_output=True,text=True)
    assert refused.returncode != 0 and 'resume game differs' in refused.stderr
    assert not (tmp_path/'wrong.gz').exists()


@pytest.mark.skipif(not BINARY.exists(), reason='build native trainer')
def test_hu200_python_native_training_rows_are_exact(tmp_path):
    # Traverser-reach reference verifies unchanged arithmetic at the new depth.
    config = PilotConfig(seed=913, raise_cap=None, abstraction=HU200_SCHEMA, game=HU200_GAME,
                         max_nodes=10**9,max_entries=10**9,max_seconds=900)
    t=BlueprintTrainer(Table(('player-0','player-1'),(20000,20000)),config)
    for _ in range(12): t.step()
    py=tmp_path/'python.gz'; native=tmp_path/'native.gz';save_training(t,py)
    subprocess.run([str(BINARY),'train','--stack-bb','200','--nodes','1000000000','--iterations','12','--seed','913','--out',str(native)],check=True,capture_output=True)
    def rows(path):
        with gzip.open(path,'rt') as f: return [json.loads(line) for line in f]
    a,b=rows(py),rows(native)
    b[0].pop('native_state')
    assert a == b


def test_matching_slumbot_contract_is_admitted_and_hu100_is_rejected():
    from types import SimpleNamespace
    from src.arena.external.interface import GameContract
    contract=GameContract(20000)
    contract.admit(SimpleNamespace(game=HU200_GAME, abstraction=HU200_SCHEMA, players=2, raise_cap=None))
    with pytest.raises(ValueError): contract.admit(SimpleNamespace(game=HU100_GAME, abstraction=HU100_SCHEMA))
