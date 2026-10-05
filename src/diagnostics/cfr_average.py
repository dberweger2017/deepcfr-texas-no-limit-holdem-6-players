"""Versioned diagnostic extraction of retained HU20 CFR accumulators.

Production short-stack exports and loaders deliberately remain unchanged.
"""
from collections import Counter
from gzip import GzipFile, open as gzip_open
from io import TextIOWrapper
import json
from math import fsum, isfinite
from os import fsync, replace
from pathlib import Path
from itertools import zip_longest

from src.arena.catalog import Checkpoint
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA, HU20_COMPRESSED_SCHEMA
from src.blueprint.artifact import FrozenBlueprint, HU20_UNCAPPED_FORMAT, _checked_schema
from src.blueprint.solver import HU20_UNCAPPED_GAME, regret_match
from src.diagnostics.saved_hu20 import file_hash

FORMAT = 'holdem-hu20-stored-cfr-average-diagnostic-v1'
EXTRACTION = 'normalize-lifetime-iteration-own-reach-accumulator-v1'
# Native checkpoints may name a non-production average in their header. Its accumulator
# sums t * policy over sampled opponent visits, so the traverser-visit bound doesn't apply.
EXTRACTIONS = {'traverser-reach': EXTRACTION,
               'opponent-sampled': 'normalize-lifetime-iteration-opponent-sampled-accumulator-v1'}


def average_rule(header):
    rule = header.get('average_rule', 'traverser-reach')
    if rule not in EXTRACTIONS:
        raise ValueError('Unknown stored average rule')
    return rule


def checked_header(header, spec, *, expected_schema=HU20_UNCAPPED_SCHEMA):
    if expected_schema not in (HU20_UNCAPPED_SCHEMA, HU20_COMPRESSED_SCHEMA):
        raise ValueError('Unknown A/C diagnostic schema')
    if (_checked_schema(header)!=expected_schema or header.get('kind')!='training'
        or type(header.get('iteration')) is not int or header['iteration']<1
        or header.get('checkpoint_format')!='jsonl-v2'
        or header['config']['game']!=HU20_UNCAPPED_GAME or header['config']['raise_cap'] is not None
        or header['config']['seed']!=spec['seed'] or header['iteration']!=spec['iteration']
        or header['table']['stacks']!=[2000,2000] or header['table']['small_blind']!=50
        or header['table']['big_blind']!=100 or header['table']['chip_unit']!='0.01'
        or len(header['table']['player_ids'])!=2 or len(set(header['table']['player_ids']))!=2
        or type(header['table']['button']) is not int or header['table']['button'] not in (0,1)):
        raise ValueError('Retained HU20 checkpoint identity differs')


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


def extract(checkpoint, spec, output, *, expected_schema=HU20_UNCAPPED_SCHEMA):
    """Stream immutable checkpoint bytes into a distinct inference format."""
    if output.exists():raise FileExistsError('Preserve existing diagnostic export')
    if file_hash(checkpoint)!=spec['checkpoint_sha256']:raise ValueError('Checkpoint hash differs before extraction')
    output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_name(output.name+'.tmp')
    counts=Counter();seen=set();mass=0.0
    try:
        with gzip_open(checkpoint,'rt') as source, temporary.open('xb') as raw:
            header=json.loads(source.readline());checked_header(header,spec,expected_schema=expected_schema)
            rule=average_rule(header)
            metadata={'format':FORMAT,'kind':'diagnostic-inference','extraction':EXTRACTIONS[rule],
                'source_checkpoint_sha256':spec['checkpoint_sha256'],'checkpoint_header':header,
                'zero_mass_rule':'uniform in retained menu; reported separately from missing keys'}
            with GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as zipped, TextIOWrapper(zipped,encoding='utf-8') as target:
                target.write(json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
                for line in source:
                    key,names,regrets,p,total,visits=checked_row(json.loads(line),header['iteration'],rule=='traverser-reach')
                    if key in seen:raise ValueError('Duplicate retained node')
                    seen.add(key);counts['entries']+=1;counts['positive_mass' if total else 'zero_mass']+=1
                    counts['visits']+=visits;mass+=total
                    if len(seen)>header['config']['max_entries']:raise ValueError('Checkpoint exceeds entry cap')
                    target.write(json.dumps([key,names,p,total,visits],separators=(',',':'),allow_nan=False)+'\n')
            raw.flush();fsync(raw.fileno())
        replace(temporary,output)
    except Exception:
        temporary.unlink(missing_ok=True);raise
    return {'format':FORMAT,'sha256':file_hash(output),'bytes':output.stat().st_size,
            'counts':dict(counts),'sum_accumulator_mass':mass,'source_checkpoint_sha256':spec['checkpoint_sha256']}


def audit(checkpoint, current_path, average_path, spec, average_sha, *, expected_schema=HU20_UNCAPPED_SCHEMA):
    """Verify every output against accumulators and the paired current export."""
    if (file_hash(checkpoint)!=spec['checkpoint_sha256'] or file_hash(current_path)!=spec['sha256']
        or file_hash(average_path)!=average_sha):raise ValueError('Extraction audit input hash differs')
    current=FrozenBlueprint(Checkpoint(spec['name'],str(current_path),spec['sha256'],HU20_UNCAPPED_FORMAT),current_path)
    counts=Counter();seen=set();tv=0.0;max_tv=0.0
    with gzip_open(checkpoint,'rt') as raw,gzip_open(average_path,'rt') as exported:
        header=json.loads(raw.readline());checked_header(header,spec,expected_schema=expected_schema)
        metadata=json.loads(exported.readline());rule=average_rule(header)
        if (metadata.get('format')!=FORMAT or metadata.get('kind')!='diagnostic-inference'
            or metadata.get('extraction')!=EXTRACTIONS[rule] or metadata.get('checkpoint_header')!=header
            or metadata.get('source_checkpoint_sha256')!=spec['checkpoint_sha256']
            or current.abstraction!=expected_schema or current.description['strategy']!='current' or current.description['iteration']!=header['iteration']
            or current.description['training_seed']!=spec['seed']):raise ValueError('Extraction lineage differs')
        for original,emitted in zip_longest(raw,exported):
            if original is None or emitted is None:raise ValueError('Extraction node count differs')
            key,names,regrets,p,total,visits=checked_row(json.loads(original),header['iteration'],rule=='traverser-reach')
            row=json.loads(emitted)
            if row!=[key,list(names),list(p),total,visits] or key in seen:raise ValueError('Normalized accumulator differs')
            if current.entries.get(key)!=(names,regret_match(regrets)):raise ValueError('Current export differs from checkpoint regrets')
            seen.add(key);counts['entries']+=1;counts['positive_mass' if total else 'zero_mass']+=1
            distance=fsum(abs(a-b) for a,b in zip(p,current.entries[key][1]))/2
            tv+=distance;max_tv=max(max_tv,distance)
        if set(current.entries)!=seen:raise ValueError('Current/average checkpoint coverage differs')
    return {'status':'verified','all_nodes_verified':len(seen),'counts':dict(counts),
            'unweighted_mean_current_average_tv':tv/len(seen) if seen else None,'maximum_tv':max_tv,
            'scope':'all stored totals and current regrets checked; historical increments not reconstructed',
            'current_sha256':spec['sha256'],'average_sha256':average_sha,'checkpoint_sha256':spec['checkpoint_sha256']}


class DiagnosticAverage(FrozenBlueprint):
    """Reuse observation/key/menu inference, with a separate diagnostic reader."""
    def __init__(self,path,expected_sha256, *, expected_schema=HU20_UNCAPPED_SCHEMA):
        if file_hash(path)!=expected_sha256:raise ValueError('Diagnostic average hash differs')
        self.entries={};self.zero_mass=set();self.visits={}
        with gzip_open(path,'rt') as source:
            metadata=json.loads(source.readline());header=metadata['checkpoint_header']
            checked_header(header,{'seed':header['config']['seed'],'iteration':header['iteration']},expected_schema=expected_schema)
            rule=average_rule(header);extraction=EXTRACTIONS[rule]
            if (metadata.get('format')!=FORMAT or metadata.get('kind')!='diagnostic-inference'
                or metadata.get('extraction')!=extraction):raise ValueError('Unknown diagnostic extraction')
            for line in source:
                key,names,p,total,visits=json.loads(line)
                checked_row([key,names,[0]*len(names),[total/len(names)]*len(names),visits],header['iteration'],rule=='traverser-reach')
                if (key in self.entries or len(names)!=len(p)
                    or any(type(x) not in (int,float) or not isfinite(x) or x<0 for x in p)
                    or abs(fsum(p)-1)>1e-8 or (not total and p!=[1/len(names)]*len(names))):raise ValueError('Invalid diagnostic policy row')
                self.entries[key]=(tuple(names),tuple(p));self.visits[key]=visits
                if not total:self.zero_mass.add(key)
                if len(self.entries)>header['config']['max_entries']:raise ValueError('Diagnostic export exceeds entry cap')
        self.players=2;self.raise_cap=None;self.abstraction=expected_schema;self.game=HU20_UNCAPPED_GAME;self.identity=header['identity']
        self.description={'kind':FORMAT,'weights_sha256':expected_sha256,'num_players':2,'iteration':header['iteration'],
            'training_seed':header['config']['seed'],'strategy':extraction,'abstraction':self.abstraction,
            'entries':len(self.entries),'zero_mass_entries':len(self.zero_mass),
            'source_checkpoint_sha256':metadata['source_checkpoint_sha256']}
