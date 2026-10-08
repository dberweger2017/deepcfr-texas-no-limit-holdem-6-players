"""Fixed research/release tables and deterministic inference receipts."""

from dataclasses import asdict, dataclass
import json
from math import fsum, isfinite

from src.blueprint.abstraction import HU100_SCHEMA, HU20_UNCAPPED_SCHEMA
from src.blueprint.solver import HU100_GAME, HU20_UNCAPPED_GAME
from src.game.hand import Table


@dataclass(frozen=True)
class PlayTable:
    stack: int = 2000
    small_blind: int = 50
    big_blind: int = 100
    chip_unit: str = '0.01'

    def __post_init__(self):
        if (type(self.stack) is not int or self.stack not in (2000, 10000)
                or type(self.small_blind) is not int or self.small_blind != 50
                or type(self.big_blind) is not int or self.big_blind != 100
                or self.chip_unit != '0.01'):
            raise ValueError('Local play requires fixed HU20 or HU100 stacks and 50/100 blinds')

    def record(self):
        return asdict(self)

    def table(self, players, button):
        return Table(players, (self.stack, self.stack), button=button,
                     small_blind=self.small_blind, big_blind=self.big_blind, chip_unit=self.chip_unit)

    def validate_policy(self, policy):
        expected = (HU100_GAME, HU100_SCHEMA) if self.stack == 10000 else (HU20_UNCAPPED_GAME, HU20_UNCAPPED_SCHEMA)
        if ((policy.game, policy.abstraction) != expected
                or getattr(policy, 'players', 2) != 2 or getattr(policy, 'raise_cap', None) is not None):
            raise ValueError('Model and local table configuration are incompatible')


def policy_table(policy):
    table = PlayTable(stack=10000 if policy.game == HU100_GAME else 2000)
    table.validate_policy(policy)
    return table


def recorded_table(row):
    # Journals created before table configuration was recorded are HU20 only.
    if 'table' not in row:
        if row.get('game', row.get('modelGame', HU20_UNCAPPED_GAME)) != HU20_UNCAPPED_GAME:
            raise ValueError('Research journal is missing its table configuration')
        return PlayTable()
    if not isinstance(row['table'], dict) or set(row['table']) != set(PlayTable().record()):
        raise ValueError('Invalid journal table configuration')
    return PlayTable(**row['table'])


def inference_record(policy):
    return {'adapter': getattr(policy, 'adapter_id', 'direct-v1'),
            'translation': policy.description.get('action_translation')}


def distribution(policy, view):
    telemetry = None
    if policy.game == HU100_GAME and hasattr(policy, 'distribution_with_telemetry'):
        menu, probabilities, trained, telemetry = policy.distribution_with_telemetry(view)
        telemetry = json.loads(json.dumps({key: value for key, value in telemetry.items()
                                           if key != 'lookup_seconds'}, allow_nan=False))
    else:
        menu, probabilities, trained = policy.distribution(view)
    if (not menu or len(menu) != len(probabilities)
            or any(not isfinite(p) or p < 0 for p in probabilities)
            or abs(fsum(probabilities) - 1) > 1e-8):
        raise ValueError('Invalid action distribution')
    for item in menu:
        view.legal_actions.validate(item.action)
    return menu, probabilities, trained, telemetry


def spectator_identity_matches(record, expected):
    if not isinstance(record, dict) or not isinstance(expected, dict):
        return False
    if expected['game'] == HU100_GAME or 'inference' in record:
        return record == expected
    # Released HU20 spectator journals predate configuration fields. Compare
    # every retained pin and require the original identity fields.
    required = {'version', 'name', 'sha256', 'game', 'schema', 'format', 'strategy', 'adapter', 'benchmarkOnly'}
    return (required <= set(record)
            and all(expected.get(key) == value for key, value in record.items()))
