"""Shared inference and validation for stored CFR averages.

The diagnostic-v1 format identifiers are retained verbatim for released assets.
Extraction and accumulator audits live in src.diagnostics.cfr_average.
"""

from gzip import open as gzip_open
import json
from math import fsum, isfinite
from dataclasses import asdict
from time import perf_counter

from src.blueprint.action_translation import TranslationOptions, VERSION, translate
from src.blueprint.abstraction import choices, information_key

from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA, HU20_COMPRESSED_SCHEMA, HU100_SCHEMA, HU200_SCHEMA, STACK_BY_SCHEMA
from src.blueprint.artifact import FrozenBlueprint, _checked_schema, _Player
from src.blueprint.compact_policy import CompactBuilder
from src.blueprint.solver import HU20_UNCAPPED_GAME, HU100_GAME, HU200_GAME
from src.policies.files import file_hash

FORMAT = 'holdem-hu20-stored-cfr-average-diagnostic-v1'
HU100_FORMAT = 'holdem-hu100-stored-cfr-average-research-v1'
HU200_FORMAT = 'holdem-hu200-stored-cfr-average-research-v1'


def format_for_schema(schema):
    return {HU100_SCHEMA: HU100_FORMAT, HU200_SCHEMA: HU200_FORMAT}.get(schema, FORMAT)

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
    if expected_schema not in (HU20_UNCAPPED_SCHEMA, HU20_COMPRESSED_SCHEMA, HU100_SCHEMA, HU200_SCHEMA):
        raise ValueError('Unknown A/C diagnostic schema')
    game = {HU100_SCHEMA: HU100_GAME, HU200_SCHEMA: HU200_GAME}.get(expected_schema, HU20_UNCAPPED_GAME)
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



class AveragePolicy(FrozenBlueprint):
    """Read stored averages with the same observation/key/menu inference as current policies."""
    def __init__(self,path,expected_sha256, *, expected_schema=HU20_UNCAPPED_SCHEMA, translation: TranslationOptions | None = None):
        if file_hash(path)!=expected_sha256:raise ValueError('Diagnostic average hash differs')
        # Compact storage: a 10B-node average needs about 50 bytes per key instead of about 800.
        rows=CompactBuilder();count=0
        with gzip_open(path,'rt') as source:
            metadata=json.loads(source.readline());header=metadata['checkpoint_header']
            checked_header(header,{'seed':header['config']['seed'],'iteration':header['iteration']},expected_schema=expected_schema)
            rule=average_rule(header);extraction=EXTRACTIONS[rule];zero_mass=zero_mass_rule(metadata)
            if (metadata.get('format')!=format_for_schema(expected_schema) or metadata.get('kind')!='diagnostic-inference'
                or metadata.get('extraction')!=extraction):raise ValueError('Unknown diagnostic extraction')
            for line in source:
                key,names,p,total,visits=json.loads(line)
                checked_row([key,names,[0]*len(names),[total/len(names)]*len(names),visits],header['iteration'],rule=='traverser-reach')
                if (len(names)!=len(p)
                    or any(type(x) not in (int,float) or not isfinite(x) or x<0 for x in p)
                    or abs(fsum(p)-1)>1e-8 or (not total and zero_mass=='uniform' and p!=[1/len(names)]*len(names))):raise ValueError('Invalid diagnostic policy row')
                rows.add(key,names,[float(x) for x in p],visits,not total);count+=1
                if count>header['config']['max_entries']:raise ValueError('Diagnostic export exceeds entry cap')
        try:
            self.entries,self.zero_mass,self.visits=rows.build()
        except ValueError as duplicate:
            raise ValueError('Invalid diagnostic policy row') from duplicate
        self.players=2;self.raise_cap=None;self.abstraction=expected_schema;self.game=header['config']['game'];self.identity=header['identity']
        self.description={'kind':metadata['format'],'weights_sha256':expected_sha256,'num_players':2,'iteration':header['iteration'],
            'training_seed':header['config']['seed'],'strategy':extraction,'abstraction':self.abstraction,
            'entries':len(self.entries),'zero_mass_entries':len(self.zero_mass),
            'source_checkpoint_sha256':metadata['source_checkpoint_sha256']}
        if zero_mass!='uniform':self.description['zero_mass_rule']=zero_mass
        self.configure_translation(translation)


    def configure_translation(self, options: TranslationOptions | None):
        """Explicit research option; model bytes and release defaults stay unchanged."""
        if options is not None and (not isinstance(options, TranslationOptions)
                                    or self.abstraction != HU100_SCHEMA):
            raise ValueError("Action translation requires HU100 and validated options")
        self.translation = options
        if options is not None:
            self.description['action_translation'] = {'version': VERSION, **asdict(options)}
        else:
            self.description.pop('action_translation', None)

    def distribution_with_telemetry(self, view):
        started = perf_counter()
        if view.capacity != self.players:
            raise ValueError("Blueprint table size differs from the evaluation table")
        menu = choices(view, raise_cap=self.raise_cap, free_fold=False)
        key = information_key(view, menu, schema=self.abstraction)
        saved = self.entries.get(key)
        known = saved is not None
        mode = 'exact' if known and key not in self.zero_mass else 'uniform'
        reason = 'positive-mass' if mode == 'exact' else 'zero-mass' if known else 'missing-key'
        info = {'mode': mode, 'reason': reason, 'exact_key': key, 'selected_key': key if known else None,
                'distance': 0., 'all_in_changes': 0, 'states': 0, 'bound_reached': False,
                'overrides': (), 'witness_raise_to': ()}
        if not known and self.translation is not None:
            result = translate(view, menu, self.entries, self.zero_mass, self.translation)
            info.update(asdict(result)); info.pop('key')
            if result.key is not None:
                saved = self.entries[result.key]
                info.update(mode='translated', reason='positive-mass-witness', selected_key=result.key)
        probabilities = (1 / len(menu),) * len(menu) if saved is None else saved[1]
        if saved is not None and saved[0] != tuple(c.name for c in menu):
            raise ValueError("Blueprint action labels differ from the observation")
        info['lookup_seconds'] = perf_counter() - started
        return menu, probabilities, known, info

    def policy(self, seed: int):
        if self.translation is None and not getattr(self, "record_translation", False):
            return super().policy(seed)
        return _AveragePlayer(self, seed)

    def distribution(self, view):
        if self.translation is None and not getattr(self, "record_translation", False):
            return super().distribution(view)
        menu, probabilities, known, _ = self.distribution_with_telemetry(view)
        return menu, probabilities, known


class _AveragePlayer(_Player):
    def choose_action(self, view):
        self.last_decision = self.blueprint.distribution_with_telemetry(view)
        menu, probabilities, _, _ = self.last_decision
        return self.random.choices(menu, weights=probabilities, k=1)[0].action
