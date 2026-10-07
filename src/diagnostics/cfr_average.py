"""Extract and independently audit stored HU20 CFR accumulators."""

from collections import Counter
from gzip import GzipFile, open as gzip_open
from io import TextIOWrapper
from itertools import zip_longest
import json
from math import fsum
from os import fsync, replace

from src.arena.catalog import Checkpoint
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA, HU100_SCHEMA
from src.blueprint.artifact import FrozenBlueprint, HU20_UNCAPPED_FORMAT, HU100_FORMAT as HU100_CURRENT_FORMAT
from src.blueprint.average import (
    FORMAT, HU100_FORMAT, EXTRACTIONS, ZERO_MASS_RULES, average_rule, checked_header,
    checked_row, zero_mass_rule,
)
from src.blueprint.solver import regret_match
from src.policies.files import file_hash

def zero_mass_policy(p, total, regrets, zero_mass):
    return tuple(regret_match(tuple(regrets))) if not total and zero_mass == 'current' else p



def extract(checkpoint, spec, output, *, expected_schema=HU20_UNCAPPED_SCHEMA, zero_mass='uniform'):
    """Stream immutable checkpoint bytes into a distinct inference format."""
    if output.exists():raise FileExistsError('Preserve existing diagnostic export')
    if file_hash(checkpoint)!=spec['checkpoint_sha256']:raise ValueError('Checkpoint hash differs before extraction')
    output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_name(output.name+'.tmp')
    counts=Counter();seen=set();mass=0.0
    try:
        with gzip_open(checkpoint,'rt') as source, temporary.open('xb') as raw:
            header=json.loads(source.readline());checked_header(header,spec,expected_schema=expected_schema)
            rule=average_rule(header)
            export_format = HU100_FORMAT if expected_schema == HU100_SCHEMA else FORMAT
            metadata={'format':export_format,'kind':'diagnostic-inference','extraction':EXTRACTIONS[rule],
                'source_checkpoint_sha256':spec['checkpoint_sha256'],'checkpoint_header':header,
                'zero_mass_rule':ZERO_MASS_RULES[zero_mass]}
            with GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as zipped, TextIOWrapper(zipped,encoding='utf-8') as target:
                target.write(json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
                for line in source:
                    key,names,regrets,p,total,visits=checked_row(json.loads(line),header['iteration'],rule=='traverser-reach')
                    if key in seen:raise ValueError('Duplicate retained node')
                    seen.add(key);counts['entries']+=1;counts['positive_mass' if total else 'zero_mass']+=1
                    counts['visits']+=visits;mass+=total
                    if len(seen)>header['config']['max_entries']:raise ValueError('Checkpoint exceeds entry cap')
                    p=zero_mass_policy(p,total,regrets,zero_mass)
                    target.write(json.dumps([key,names,p,total,visits],separators=(',',':'),allow_nan=False)+'\n')
            raw.flush();fsync(raw.fileno())
        replace(temporary,output)
    except Exception:
        temporary.unlink(missing_ok=True);raise
    return {'format':export_format,'sha256':file_hash(output),'bytes':output.stat().st_size,
            'counts':dict(counts),'sum_accumulator_mass':mass,'source_checkpoint_sha256':spec['checkpoint_sha256']}


def audit(checkpoint, current_path, average_path, spec, average_sha, *, expected_schema=HU20_UNCAPPED_SCHEMA):
    """Verify every output against accumulators and the paired current export."""
    if (file_hash(checkpoint)!=spec['checkpoint_sha256'] or file_hash(current_path)!=spec['sha256']
        or file_hash(average_path)!=average_sha):raise ValueError('Extraction audit input hash differs')
    current=FrozenBlueprint(Checkpoint(spec['name'],str(current_path),spec['sha256'],HU100_CURRENT_FORMAT if expected_schema == HU100_SCHEMA else HU20_UNCAPPED_FORMAT),current_path)
    counts=Counter();seen=set();tv=0.0;max_tv=0.0
    with gzip_open(checkpoint,'rt') as raw,gzip_open(average_path,'rt') as exported:
        header=json.loads(raw.readline());checked_header(header,spec,expected_schema=expected_schema)
        metadata=json.loads(exported.readline());rule=average_rule(header);zero_mass=zero_mass_rule(metadata)
        if (metadata.get('format')!=(HU100_FORMAT if expected_schema == HU100_SCHEMA else FORMAT) or metadata.get('kind')!='diagnostic-inference'
            or metadata.get('extraction')!=EXTRACTIONS[rule] or metadata.get('checkpoint_header')!=header
            or metadata.get('source_checkpoint_sha256')!=spec['checkpoint_sha256']
            or current.abstraction!=expected_schema or current.description['strategy']!='current' or current.description['iteration']!=header['iteration']
            or current.description['training_seed']!=spec['seed']):raise ValueError('Extraction lineage differs')
        for original,emitted in zip_longest(raw,exported):
            if original is None or emitted is None:raise ValueError('Extraction node count differs')
            key,names,regrets,p,total,visits=checked_row(json.loads(original),header['iteration'],rule=='traverser-reach')
            row=json.loads(emitted);p=zero_mass_policy(p,total,regrets,zero_mass)
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
