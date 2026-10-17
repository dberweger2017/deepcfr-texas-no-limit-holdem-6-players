"""Fixed research/release tables and deterministic inference receipts."""

from dataclasses import asdict, dataclass
import json
from math import fsum, isfinite

from src.blueprint.abstraction import HU100_SCHEMA, HU20_UNCAPPED_SCHEMA, EQUITY_SCHEMAS, HU100_EQUITY_MENU_V2_SCHEMA, HU200_EQUITY_SCHEMA
from src.blueprint.solver import HU100_GAME, HU20_UNCAPPED_GAME, HU200_GAME
from src.game.hand import Table


@dataclass(frozen=True)
class PlayTable:
    stack: int = 2000
    small_blind: int = 50
    big_blind: int = 100
    chip_unit: str = '0.01'

    def __post_init__(self):
        if (type(self.stack) is not int or self.stack not in (2000, 10000, 20000)
                or type(self.small_blind) is not int or self.small_blind != 50
                or type(self.big_blind) is not int or self.big_blind != 100
                or self.chip_unit != '0.01'):
            raise ValueError('Local play requires fixed HU20, HU100 or HU200 stacks and 50/100 blinds')

    def record(self):
        return asdict(self)

    def table(self, players, button):
        return Table(players, (self.stack, self.stack), button=button,
                     small_blind=self.small_blind, big_blind=self.big_blind, chip_unit=self.chip_unit)

    def validate_policy(self, policy):
        k50_release = ((self.stack == 10000 and policy.abstraction == HU100_EQUITY_MENU_V2_SCHEMA)
                       or (self.stack == 20000 and policy.abstraction == HU200_EQUITY_SCHEMA))
        if k50_release:
            from src.blueprint.equity_buckets import EQUITY_K50_SHA256
            from src.blueprint.action_translation import version_for_schema
            translation_version = version_for_schema(policy.abstraction)
            cards = getattr(policy, 'equity_cards', None)
            translation = policy.description.get('action_translation')
            if (cards is None or cards.sha256 != EQUITY_K50_SHA256
                    or policy.identity.get('card_tables') != EQUITY_K50_SHA256
                    or getattr(policy, 'adapter_id', None) != translation_version
                    or translation != {'version': translation_version, 'max_states': 512, 'max_events': 128}):
                raise ValueError('K50 release table or translation identity differs')
        elif policy.abstraction in EQUITY_SCHEMAS:
            raise ValueError("K50 models are not yet supported by the web runtime; use the evaluation harnesses")
        expected = {2000: (HU20_UNCAPPED_GAME, HU20_UNCAPPED_SCHEMA),
                    10000: (HU100_GAME, HU100_EQUITY_MENU_V2_SCHEMA if k50_release else HU100_SCHEMA),
                    20000: (HU200_GAME, HU200_EQUITY_SCHEMA)}[self.stack]
        if ((policy.game, policy.abstraction) != expected
                or getattr(policy, 'players', 2) != 2 or getattr(policy, 'raise_cap', None) is not None):
            raise ValueError('Model and local table configuration are incompatible')


def policy_table(policy):
    table = PlayTable(stack={HU100_GAME: 10000, HU200_GAME: 20000}.get(policy.game, 2000))
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
    if policy.game in (HU100_GAME, HU200_GAME) and hasattr(policy, 'distribution_with_telemetry'):
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
    if expected['game'] in (HU100_GAME, HU200_GAME) or 'inference' in record:
        return record == expected
    # Released HU20 spectator journals predate configuration fields. Compare
    # every retained pin and require the original identity fields.
    required = {'version', 'name', 'sha256', 'game', 'schema', 'format', 'strategy', 'adapter', 'benchmarkOnly'}
    return (required <= set(record)
            and all(expected.get(key) == value for key, value in record.items()))
