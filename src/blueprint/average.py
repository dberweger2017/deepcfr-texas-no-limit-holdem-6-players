"""Shared inference and validation for stored CFR averages.

The diagnostic-v1 format identifiers are retained verbatim for released assets.
Extraction and accumulator audits live in src.diagnostics.cfr_average.
"""

from gzip import open as gzip_open
import json
from math import fsum, isfinite
from dataclasses import asdict
from time import perf_counter

from src.blueprint.action_translation import TranslationOptions, version_for_schema, translate
from src.blueprint.abstraction import choices, information_key

from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA, HU20_COMPRESSED_SCHEMA, HU100_SCHEMA, HU200_SCHEMA, HU100_EQUITY_SCHEMA, HU200_EQUITY_SCHEMA, HU100_EQUITY_MENU_V2_SCHEMA, EQUITY_SCHEMAS, MENU_V2_SCHEMAS, STACK_BY_SCHEMA, menu_sizes
from src.blueprint.artifact import FrozenBlueprint, _checked_schema, _Player
from src.blueprint import policy_cache
from src.blueprint.compact_policy import CompactBuilder
from src.blueprint.solver import HU20_UNCAPPED_GAME, HU100_GAME, HU200_GAME
from src.policies.files import file_hash

FORMAT = 'holdem-hu20-stored-cfr-average-diagnostic-v1'
HU100_FORMAT = 'holdem-hu100-stored-cfr-average-research-v1'
HU200_FORMAT = 'holdem-hu200-stored-cfr-average-research-v1'


def format_for_schema(schema):
    return {10000: HU100_FORMAT, 20000: HU200_FORMAT}.get(STACK_BY_SCHEMA.get(schema), FORMAT)

EXTRACTION = 'normalize-lifetime-iteration-own-reach-accumulator-v1'
# Native checkpoints may name a non-production average in their header. Its accumulator
# sums t * policy over sampled opponent visits, so the traverser-visit bound doesn't apply.
EXTRACTIONS = {'traverser-reach': EXTRACTION,
               'opponent-sampled': 'normalize-lifetime-iteration-opponent-sampled-accumulator-v1'}


# How a key whose stored average has no mass plays. Uniform is the original rule. "current" plays the
# key's regret-matched policy instead: under traverser-reach averaging many trained keys get no mass
# (subtrees reached only through zero-probability own actions), and uniform discards their regrets.
ZERO_MASS_RULES = {'uniform': 'uniform in retained menu; reported separately from missing keys',
                   'current': 'current regret-matched policy in retained menu; reported separately from missing keys'}


def zero_mass_rule(metadata):
    names = {text: name for name, text in ZERO_MASS_RULES.items()}
    if metadata.get('zero_mass_rule') not in names:
        raise ValueError('Unknown zero-mass rule')
    return names[metadata['zero_mass_rule']]


def average_rule(header):
    rule = header.get('average_rule', 'traverser-reach')
    if rule not in EXTRACTIONS:
        raise ValueError('Unknown stored average rule')
    return rule


def checked_header(header, spec, *, expected_schema=HU20_UNCAPPED_SCHEMA):
    if expected_schema not in (HU20_UNCAPPED_SCHEMA, HU20_COMPRESSED_SCHEMA, HU100_SCHEMA, HU200_SCHEMA, *EQUITY_SCHEMAS, *MENU_V2_SCHEMAS):
        raise ValueError('Unknown A/C diagnostic schema')
    game = {20000: HU200_GAME, 10000: HU100_GAME}.get(STACK_BY_SCHEMA[expected_schema], HU20_UNCAPPED_GAME)
    stack = STACK_BY_SCHEMA[expected_schema]
    if (_checked_schema(header)!=expected_schema or header.get('kind')!='training'
        or type(header.get('iteration')) is not int or header['iteration']<1
        or header.get('checkpoint_format')!='jsonl-v2'
        or header['config']['game']!=game or header['config']['raise_cap'] is not None
        or header['config']['seed']!=spec['seed'] or header['iteration']!=spec['iteration']
        or header['table']['stacks']!=[stack,stack] or header['table']['small_blind']!=50
        or header['table']['big_blind']!=100 or header['table']['chip_unit']!='0.01'
        or len(header['table']['player_ids'])!=2 or len(set(header['table']['player_ids']))!=2
        or type(header['table']['button']) is not int or header['table']['button'] not in (0,1)):
        raise ValueError('Retained heads-up checkpoint identity differs')


def normalized(average, visits, iteration, bounded=True):
    if (type(visits) is not int or visits<0 or not average
        or any(type(x) not in (int,float) or not isfinite(x) or x<0 for x in average)):
        raise ValueError('Invalid stored average accumulator')
    total=fsum(average)
    if not isfinite(total) or (bounded and total>iteration*visits+1e-9*max(1,iteration*visits)):
        raise ValueError('Stored average violates the iteration/reach bound')
    return tuple(x/total for x in average) if total else (1/len(average),)*len(average),total


def checked_row(row, iteration, bounded=True):
    key,names,regrets,average,visits=row
    if (not isinstance(key,str) or len(key)!=32 or any(c not in '0123456789abcdef' for c in key)
        or not names or any(not isinstance(n,str) for n in names) or len(set(names))!=len(names)
        or len(names)!=len(regrets) or len(names)!=len(average)
        or any(type(x) not in (int,float) or not isfinite(x) for x in regrets)):
        raise ValueError('Invalid retained checkpoint node')
    probabilities,total=normalized(average,visits,iteration,bounded)
    return key,tuple(names),tuple(regrets),probabilities,total,visits



def missing_check_call(view, menu):
    """PR251's diagnostic_action(view, 'call'): check if legal, else call.

    Sampling this point mass still consumes one draw, as PR251's override did.
    Stored rows, including zero-mass rows, never use this distribution.
    """
    from src.game.types import Action, ActionKind
    kind = ActionKind.CHECK if ActionKind.CHECK in view.legal_actions.kinds else ActionKind.CALL
    action = Action(kind)
    if sum(c.action == action for c in menu) != 1:
        raise ValueError('Missing-key check/call action is absent or ambiguous')
    return tuple(float(c.action == action) for c in menu)


class AveragePolicy(FrozenBlueprint):
    """Read stored averages with the same observation/key/menu inference as current policies."""
    def __init__(self,path,expected_sha256, *, expected_schema=HU20_UNCAPPED_SCHEMA, translation: TranslationOptions | None = None, sorted_rows: bool = False, card_buckets=None, cache=None, missing_key_fallback="uniform", cache_read_only=False):
        """With `cache`, a folder: rows parse once into a verified binary cache there, and later loads
        map it instead (see policy_cache). Hash, header and tables are checked either way."""
        self.configure_missing_key_fallback(missing_key_fallback)
        if file_hash(path)!=expected_sha256:raise ValueError('Diagnostic average hash differs')
        # Compact storage: a 10B-node average needs about 50 bytes per key instead of about 800.
        rows=CompactBuilder();count=0
        with gzip_open(path,'rt') as source:
            metadata=json.loads(source.readline());header=metadata['checkpoint_header']
            checked_header(header,{'seed':header['config']['seed'],'iteration':header['iteration']},expected_schema=expected_schema)
            from src.blueprint.equity_buckets import load_cards
            self.equity_cards = load_cards(expected_schema, header["identity"], card_buckets)
            rule=average_rule(header);extraction=EXTRACTIONS[rule];zero_mass=zero_mass_rule(metadata)
            if (metadata.get('format')!=format_for_schema(expected_schema) or metadata.get('kind')!='diagnostic-inference'
                or metadata.get('extraction')!=extraction):raise ValueError('Unknown diagnostic extraction')
            cached=None if cache is None else policy_cache.load(cache,expected_sha256,expected_schema)
            if cache_read_only and cached is None:
                raise ValueError("Read-only policy cache is missing")
            for line in source if cached is None else ():
                key,names,p,total,visits=json.loads(line)
                checked_row([key,names,[0]*len(names),[total/len(names)]*len(names),visits],header['iteration'],rule=='traverser-reach')
                if (len(names)!=len(p)
                    or any(type(x) not in (int,float) or not isfinite(x) or x<0 for x in p)
                    or abs(fsum(p)-1)>1e-8 or (not total and zero_mass=='uniform' and p!=[1/len(names)]*len(names))):raise ValueError('Invalid diagnostic policy row')
                rows.add(key,names,[float(x) for x in p],visits,not total);count+=1
                if count>header['config']['max_entries']:raise ValueError('Diagnostic export exceeds entry cap')
        if cached is None:
            try:
                cached=rows.build_sorted() if sorted_rows else rows.build()
            except ValueError as duplicate:
                raise ValueError('Invalid diagnostic policy row') from duplicate
            del rows
            if cache is not None:
                # Continue from the mapped copy, so even this first load releases its parsed arrays.
                policy_cache.save(cache,expected_sha256,expected_schema,cached)
                cached=policy_cache.load(cache,expected_sha256,expected_schema)
        elif len(cached[0])>header['config']['max_entries']:raise ValueError('Diagnostic export exceeds entry cap')
        self.entries,self.zero_mass,self.visits=cached
        self.players=2;self.raise_cap=None;self.abstraction=expected_schema;self.game=header['config']['game'];self.identity=header['identity']
        self.description={'kind':metadata['format'],'weights_sha256':expected_sha256,'num_players':2,'iteration':header['iteration'],
            'training_seed':header['config']['seed'],'strategy':extraction,'abstraction':self.abstraction,
            'entries':len(self.entries),'zero_mass_entries':len(self.zero_mass),
            'source_checkpoint_sha256':metadata['source_checkpoint_sha256']}
        if self.equity_cards is not None:
            self.description["card_tables"] = dict(self.equity_cards.sha256)
        if zero_mass!='uniform':self.description['zero_mass_rule']=zero_mass
        self.configure_missing_key_fallback(missing_key_fallback)
        self.configure_translation(translation)


    def configure_missing_key_fallback(self, option: str):
        """Omitted metadata means uniform, preserving legacy default trace bytes."""
        if option not in ('uniform', 'check-call'):
            raise ValueError('Unknown missing-key fallback')
        self.missing_key_fallback = option
        if hasattr(self, 'description'):
            if option == 'uniform': self.description.pop('missing_key_fallback', None)
            else: self.description['missing_key_fallback'] = option


    def configure_translation(self, options: TranslationOptions | None):
        """Explicit research option; model bytes and release defaults stay unchanged."""
        if options is not None and (not isinstance(options, TranslationOptions)
                                    or self.abstraction not in (HU100_SCHEMA, HU200_SCHEMA, HU100_EQUITY_SCHEMA, HU200_EQUITY_SCHEMA, HU100_EQUITY_MENU_V2_SCHEMA)
                                    or self.game != {HU100_SCHEMA: HU100_GAME, HU200_SCHEMA: HU200_GAME,
                                                     HU100_EQUITY_SCHEMA: HU100_GAME, HU200_EQUITY_SCHEMA: HU200_GAME,
                                                     HU100_EQUITY_MENU_V2_SCHEMA: HU100_GAME}.get(self.abstraction)):
            raise ValueError("Action translation requires matching HU100/HU200 game and validated options")
        if options is not None and self.abstraction in EQUITY_SCHEMAS:
            from src.blueprint.equity_buckets import EQUITY_K50_SHA256
            cards = getattr(self, "equity_cards", None)
            if (cards is None or cards.sha256 != EQUITY_K50_SHA256
                    or cards.sha256 != self.identity.get("card_tables")):
                raise ValueError("K50 translation table identity differs from the policy")
        self.translation = options
        if options is not None:
            self.description['action_translation'] = {'version': version_for_schema(self.abstraction), **asdict(options)}
        else:
            self.description.pop('action_translation', None)

    def distribution_with_telemetry(self, view):
        started = perf_counter()
        if view.capacity != self.players:
            raise ValueError("Blueprint table size differs from the evaluation table")
        menu = choices(view, raise_cap=self.raise_cap, free_fold=False, menu_sizes=menu_sizes(self.abstraction))
        key = self.information_key(view, menu)
        saved = self.entries.get(key)
        known = saved is not None
        mode = 'exact' if known and key not in self.zero_mass else 'uniform'
        reason = 'positive-mass' if mode == 'exact' else 'zero-mass' if known else 'missing-key'
        info = {'mode': mode, 'reason': reason, 'exact_key': key, 'selected_key': key if known else None,
                'distance': 0., 'all_in_changes': 0, 'states': 0, 'bound_reached': False,
                'overrides': (), 'witness_raise_to': ()}
        if not known and self.translation is not None:
            result = translate(view, menu, self.entries, self.zero_mass, self.translation, schema=self.abstraction,
                               equity_cards=getattr(self, "equity_cards", None))
            info.update(asdict(result)); info.pop('key')
            if result.key is not None:
                saved = self.entries[result.key]
                info.update(mode='translated', reason='positive-mass-witness', selected_key=result.key)
        probabilities = (1 / len(menu),) * len(menu) if saved is None else saved[1]
        if saved is not None and saved[0] != tuple(c.name for c in menu):
            raise ValueError("Blueprint action labels differ from the observation")
        if self.missing_key_fallback != 'uniform':
            applied = saved is None
            if applied: probabilities = missing_check_call(view, menu)
            info.update(missing_key_fallback=self.missing_key_fallback, fallback_applied=applied)
        info['lookup_seconds'] = perf_counter() - started
        return menu, probabilities, known, info

    def policy(self, seed: int):
        if self.missing_key_fallback == "uniform" and self.translation is None and not getattr(self, "record_translation", False):
            return super().policy(seed)
        return _AveragePlayer(self, seed)

    def distribution(self, view):
        if self.missing_key_fallback == "uniform" and self.translation is None and not getattr(self, "record_translation", False):
            return super().distribution(view)
        menu, probabilities, known, _ = self.distribution_with_telemetry(view)
        return menu, probabilities, known


class _AveragePlayer(_Player):
    def choose_action(self, view):
        self.last_decision = self.blueprint.distribution_with_telemetry(view)
        menu, probabilities, _, info = self.last_decision
        if info.get('fallback_applied'):
            if not hasattr(self, 'fallback_decisions_by_street'): self.fallback_decisions_by_street = {}
            street = view.street.value
            self.fallback_decisions_by_street[street] = self.fallback_decisions_by_street.get(street, 0) + 1
        return self.random.choices(menu, weights=probabilities, k=1)[0].action
