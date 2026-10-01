import gzip
import json

from scripts.evaluate_hu20_500m import execute_task
from src.blueprint.artifact import export_policy, save_training
from src.blueprint.solver import BlueprintTrainer, HU20_UNCAPPED_GAME, PilotConfig
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
from src.diagnostics.selective_stackoff import VERSION
from src.game.hand import Table


def test_queue_hand_path_replays_native_state_and_preserves_every_coordinate(tmp_path):
    trainer = BlueprintTrainer(Table(('player-0','player-1'), (2000,2000)),
        PilotConfig(seed=2026093001, raise_cap=None, abstraction=HU20_UNCAPPED_SCHEMA,
                    game=HU20_UNCAPPED_GAME))
    cp, policy = tmp_path/'training.gz', tmp_path/'current.gz'
    spec = dict(name='fixture', seed=2026093001, milestone=100000000, players=2, iteration=0,
                path=str(policy), sha256=export_policy(trainer,policy), checkpoint_path=str(cp),
                checkpoint_sha256=save_training(trainer,cp), format='holdem-hu20-native-reopening-blueprint-v1',
                abstraction=HU20_UNCAPPED_SCHEMA, dual_menu_telemetry=True)
    panels = [dict(name='random',rule='random',contract='secondary',blocks=2,root=8920101),
              dict(name='selective_stackoff',rule='selective_stackoff',contract='native',blocks=2,root=8920102)]
    plan = dict(opponent_version=VERSION, light_panels=panels, chance_samples=4, lbr_seconds=5)
    out=tmp_path/'evaluation'
    assert execute_task(plan,spec,'light',out)==0
    result=json.loads((out/'result.json').read_text())
    assert result['hands']==8 and result['status']=='complete'
    with gzip.open(out/'hands.jsonl.gz','rt') as source:
        rows=[json.loads(line) for line in source]
    expected={(p['name'],b,r) for p in panels for b in range(2) for r in (0,1)}
    assert {(r['panel'],r['block'],r['rotation']) for r in rows}==expected
    assert all(r['native_replay_verified'] and sum(r['net_chips_by_seat'])==0 for r in rows)
    assert all('large_calls' in r and 'allin_calls' in r and 'tails' in r for r in rows)
