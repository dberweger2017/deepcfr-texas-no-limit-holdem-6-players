"""Hash real released-policy traces without reporting poker outcomes.

Run this same harness from clean baseline and candidate checkouts with identical
models, roots and read-only caches. Only elapsed telemetry is omitted from the
decision hash; every hand independently replays.
"""
import argparse
from dataclasses import asdict
import gc
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from scripts import evaluate_hu100_direct as direct
from src.arena.schedule import canonical
from src.blueprint.action_translation import TranslationOptions
from src.policies.hu100_bundle import RELEASES, verify_model

ROOTS={'v0.5.0':20261015530500,'v0.5.1':20261015530501,'v0.5.2':20261015530502}


def check(release, model, cache, out):
    pin=RELEASES[release]['model'];verify_model(model,release)
    spec=dict(name='released-on',path=str(model.resolve()),sha256=pin['sha256'],bytes=pin['bytes'],
        format=pin['format'],abstraction=pin['schema'],entries=pin['entries'],iteration=pin['iteration'],
        source_checkpoint_sha256=pin['checkpoint_sha256'],translation=asdict(TranslationOptions()))
    specs=[spec,dict(spec,name='released-control-on')]
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    registry,done=direct.execute(specs,64,ROOTS[release],'compatibility',out/'play',source,
        opponent='pot_pressure',cache=cache,cache_read_only=cache is not None)
    del registry;gc.collect()
    audit=direct.audit(out/'play',out/'replay.json')
    digests={};translated=0
    for name in ('hands','decisions'):
        digest=hashlib.sha256()
        with gzip.open(out/'play'/(name+'.jsonl.gz'),'rt') as stream:
            for row in map(json.loads,stream):
                if name=='decisions':
                    translated+=int(row.get('translation',{}).get('mode')=='translated')
                    row=direct.stable_decision(row)
                digest.update((canonical(row)+'\n').encode())
        digests[name]=digest.hexdigest()
    result=dict(status='verified',release=release,source=source,model_sha256=pin['sha256'],
        root=ROOTS[release],blocks=64,hands=done['hands'],decisions=done['decisions'],
        all_hands_independently_replayed=audit['hands_replayed']==done['hands'],translated=translated,
        trace_sha256=digests,outcomes_reported=False,excluded_fields=['decision_seconds','translation.lookup_seconds'])
    result['runtime_module_paths']={name:str(Path(sys.modules[name].__file__).resolve()) for name in (
        'scripts.evaluate_hu100_direct','src.blueprint.average','src.blueprint.action_translation','src.policies.hu100_bundle')}
    direct.put(out/'trace-hashes.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',choices=ROOTS,required=True);parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--cache',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(check(args.release,args.model,args.cache,args.out),sort_keys=True))


if __name__=='__main__':main()
