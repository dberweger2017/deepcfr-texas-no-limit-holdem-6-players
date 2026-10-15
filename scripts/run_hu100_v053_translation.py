"""Fixed-count M1 v0.5.3 experiment with a global replay/reproduction outcome barrier."""
import argparse
from dataclasses import asdict
import gc
import gzip
import json
from pathlib import Path
import subprocess
import sys
from time import time

from scripts import evaluate_hu100_direct as direct
from scripts.guard_v053_m1 import boundary_stop, guard
from scripts.run_hu200_frozen_evaluation import estimate, gh, read, revision, series
from scripts.run_hu200_k50_translation import telemetry
from src.blueprint.action_translation import TranslationOptions, version_for_schema
from src.blueprint.average import AveragePolicy
from src.blueprint.equity_buckets import EQUITY_K50_SHA256
from src.policies.hu100_bundle import RELEASES, verify_model
from src.policies.files import file_hash

ROOTS = dict(pilot_direct=20261015530101, pilot_panel=20261015530102,
             direct=20261015530103, panel=20261015530104)
COUNTS = dict(direct=131072, panel=16384)
RULE = 'Adopt translation iff no seven-opponent paired nominal Student-t95 upper bound is below -20 BB/100; all gains and direct descriptive'


def put(path, value):
    direct.put(Path(path), value)


def spec(path, release, name, tables=None, translation=True):
    pin = RELEASES[release]['model']; path = Path(path)
    verify_model(path, release)
    return dict(name=name, path=str(path.resolve()), sha256=pin['sha256'], bytes=pin['bytes'],
        entries=pin['entries'], iteration=pin['iteration'], format=pin['format'],
        abstraction=pin['schema'], source_checkpoint_sha256=pin['checkpoint_sha256'],
        translation=asdict(TranslationOptions()) if translation else None,
        **({'card_buckets': str(Path(tables).resolve())} if tables else {}))


def prepare(config, base):
    tables = Path(config['tables'])
    hashes = {street: file_hash(tables/(street+'-k50.bin')) for street in EQUITY_K50_SHA256}
    if hashes != EQUITY_K50_SHA256:
        raise ValueError('Pinned K50 tables differ')
    model = spec(config['model'], 'v0.5.3', 'v053-on', tables)
    release = spec(config['release'], 'v0.5.2', 'v052-on')
    specs = dict(panel=[model, dict(model, name='v053-off', translation=None)], direct=[model, release])
    owners = {str(n): json.loads(gh('pr','view',n,'--json','state,url,mergedAt')) for n in (163,240,272,296)}
    if any(owner['state'] != 'MERGED' for owner in owners.values()):
        raise ValueError('Owning input PR is not merged')
    inputs=dict(source=revision(), models=specs, tables=hashes, owners=owners,
        retrieval=config['retrieval'], machine='Apple M1', candidate_selection='owner-declared seed 2026100601')
    for name,value in (('inputs.json',inputs),('specs.json',specs)):
        if (base/name).exists():
            if read(base/name)!=value:
                raise ValueError('Resumed preparation identity differs')
        else:
            put(base/name,value)
    for item in (model, release):
        if boundary_stop(base):
            return
        started=time()
        policy=AveragePolicy(Path(item['path']),item['sha256'],expected_schema=item['abstraction'],
            card_buckets=Path(item['card_buckets']) if 'card_buckets' in item else None,
            sorted_rows=True, cache=Path(config['cache']))
        receipt=base/('cache-'+item['name']+'.json')
        if not receipt.exists():
            put(receipt, dict(seconds=time()-started, description=policy.description,
                source_sha256=item['sha256'], no_hands=True))
        del policy; gc.collect()


def pair(base, specs, blocks, root, opponent, cache, pilot, control):
    for label, reproduce in (('play', None), ('reproduction', base/'play')):
        out = base/label
        if not (out/'complete.json').exists():
            registry, done = direct.execute(specs,blocks,root,'translation',out,revision(),
                reproduce=reproduce,stack_bb=100,opponent=opponent,cache=cache,cache_read_only=True,
                boundary_stop=lambda n: boundary_stop(control), resume=(out/'resume.json').exists())
            del registry; gc.collect()
            if done['status'] == 'paused':
                return
        if label=='play' and not (base/'replay.json').exists():
            direct.audit(out,base/'replay.json',cost_only=pilot)
        if boundary_stop(control):
            return
    if not (base/'costs.json').exists():
        checked=read(base/'replay.json')
        put(base/'costs.json',dict(primary=read(base/'play/complete.json'),
            repeat=read(base/'reproduction/complete.json'),audit=checked,blocks=blocks,pilot=pilot,
            **({'paired_sd_bb_per_100':checked['paired_sd_bb_per_100']} if pilot else {})))


def measured_quote(costs):
    fixed = work = 0.
    for name, cost in costs.items():
        target=COUNTS['direct' if name=='direct' else 'panel']
        fixed += cost['primary']['load_or_validation_seconds'] + cost['repeat']['load_or_validation_seconds'] + cost['audit']['fixed_setup_seconds']
        work += target/cost['blocks']*(cost['primary']['play_seconds']+cost['repeat']['play_seconds']+cost['audit']['block_work_seconds'])
    return dict(fixed_seconds=fixed,scaled_block_seconds=work,margin=1.25,
        scientific_closeout_seconds=1800,active_work_budget_seconds=1.25*(fixed+work)+1800)


def freeze(base, costs):
    quote=measured_quote(costs)
    plan=dict(source=revision(),inputs=read(base/'inputs.json'),roots=ROOTS,counts=COUNTS,
        rule=RULE,quote=quote,pilots=costs,frozen_at=time(),
        translation={'version':version_for_schema(RELEASES['v0.5.3']['model']['schema']),**asdict(TranslationOptions())},
        total_original_final_hands=720896,global_replay_and_reproduction_before_outcomes=True,
        no_pilot_pooling=True,no_failed_hand_reruns=True,no_count_selection=True,
        budget_clock='Active guarded final work; battery pauses consume no work and retain the same roots/source/counts',
        guards=dict(machine='M1',sample_seconds=5,sustained_seconds=60,family_rss_gib=13,
            free_percent=8,critical_pressure=4,swap_growth_bytes=4_000_000_000,disk_floor_gib=16,
            battery_pause_at_percent=10,ac_required=False))
    put(base/'canonical-plan.json',plan)
    original=json.loads(gh('pr','view',298,'--json','body'))['body']
    # The full input/receipt plan must fit GitHub's PR-body limit. Preserve its
    # JSON values exactly; the archived canonical file remains pretty-printed.
    payload=json.dumps(plan,sort_keys=True,separators=(',',':'))
    body=original+'\n\n## Canonical translation plan before final hands\n\nSHA256 `'+file_hash(base/'canonical-plan.json')+'`.\n\nMeasured final science allowance: '+format(quote['active_work_budget_seconds']/3600,'.2f')+' active hours from launch, including replay/reproduction, margin and scientific closeout. Packaging/runtime/archive follow separately.\n\n```json\n'+payload+'\n```\n'
    path=base/'canonical-pr-body.md'; path.write_text(body)
    gh('pr','edit',298,'--body-file',path)
    if json.loads(gh('pr','view',298,'--json','body'))['body']!=body:
        raise ValueError('Canonical PR-description readback differs')
    put(base/'plan-posted.json',dict(at=time(),sha256=file_hash(base/'canonical-plan.json'),readback_verified=True))


def run_pairs(config,base,pilot):
    if pilot and (base/'plan-posted.json').exists():
        if file_hash(base/'canonical-plan.json')!=read(base/'plan-posted.json')['sha256']:
            raise ValueError('Frozen pilot plan changed')
        return
    if not pilot:
        plan=read(base/'canonical-plan.json')
        if (file_hash(base/'canonical-plan.json')!=read(base/'plan-posted.json')['sha256']
                or plan['source']!=revision() or plan['counts']!=COUNTS or plan['roots']!=ROOTS):
            raise ValueError('Frozen plan/source/counts/roots changed')
    specs=read(base/'specs.json'); costs={}
    if not pilot and (specs!=plan['inputs']['models'] or read(base/'inputs.json')!=plan['inputs']):
        raise ValueError('Frozen model/input specs changed')
    jobs=[('direct',None),*[('panel-'+op,op) for op in direct.PANEL_OPPONENTS]]
    for label,op in jobs:
        if boundary_stop(base):
            return
        folder=base/(('pilot-' if pilot else '')+label)
        blocks=(1024 if op is None else 64) if pilot else COUNTS['direct' if op is None else 'panel']
        root=ROOTS[('pilot_' if pilot else '')+('direct' if op is None else 'panel')]
        pair(folder,specs['direct' if op is None else 'panel'],blocks,root,op,Path(config['cache']),pilot,base)
        if boundary_stop(base):
            return
        costs[label]=read(folder/'costs.json')
    if pilot:
        put(base/'pilot-costs.json',costs);freeze(base,costs)
    else:
        verify_barrier(base)


def verify_barrier(base):
    comparisons={}
    plan=read(base/'canonical-plan.json')
    for label, expected in [('direct',262144), *[('panel-'+op,65536) for op in direct.PANEL_OPPONENTS]]:
        folder=base/label; checked=read(folder/'replay.json')
        if checked['status']!='verified' or checked['hands_replayed']!=expected:
            raise ValueError('Global replay barrier incomplete')
        for arm in ('play','reproduction'):
            inputs=read(folder/arm/'inputs.json')
            if (inputs['source']!=plan['source'] or inputs['models']!=plan['inputs']['models']['direct' if label=='direct' else 'panel']
                    or inputs['plan']['blocks']!=COUNTS['direct' if label=='direct' else 'panel']
                    or inputs['plan']['root_seed']!=ROOTS['direct' if label=='direct' else 'panel']):
                raise ValueError('Global input/source/root/count identity differs')
            done=read(folder/arm/'complete.json')
            if done['hands']!=expected or arm=='reproduction' and not done['reproduced_all_hands_and_decisions']:
                raise ValueError('Global reproduction barrier incomplete')
            for name,pin in done['files'].items():
                path=folder/arm/name
                if path.stat().st_size!=pin['bytes'] or file_hash(path)!=pin['sha256']:
                    raise ValueError('Verified final bytes changed')
        comparisons[label]=dict(hands=expected,independent_replay=True,exact_reproduction=True)
    if not (base/'all-final-verified.json').exists():
        put(base/'all-final-verified.json',dict(status='verified',at=time(),source=revision(),comparisons=comparisons))


def safeguards(panel):
    if set(panel)!=set(direct.PANEL_OPPONENTS):
        raise ValueError('All seven opponent contrasts required')
    flags={op:row['contrast']['ci95'][1]<-20 for op,row in panel.items()}
    return dict(adopt_translation=not any(flags.values()),severe_regressions=flags)


def readout(base):
    if not (base/'all-final-verified.json').exists():
        raise ValueError('All final hands must verify before outcomes')
    verify_barrier(base)
    result=dict(direct=estimate(series(base/'direct',COUNTS['direct'])['candidate']),panel={})
    for op in direct.PANEL_OPPONENTS:
        arms=series(base/('panel-'+op),COUNTS['panel'],panel=True)
        result['panel'][op]=dict(arms={name:estimate(values) for name,values in arms.items()},
            contrast=estimate([a-b for a,b in zip(arms['candidate'],arms['baseline'],strict=True)]))
    result.update(safeguards(result['panel']),rule=RULE,
        canonical_plan_sha256=file_hash(base/'canonical-plan.json'),
        all_final_hands_independently_replayed_and_exactly_reproduced=True,
        off_menu_bets={op:read(base/('panel-'+op)/'replay.json')['scripted_bets_against_candidate_menu']
            for op in direct.PANEL_OPPONENTS},
        telemetry={name:telemetry(base/name) for name in ('direct',*('panel-'+op for op in direct.PANEL_OPPONENTS))})
    put(base/'summary.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','pilots','final','readout','worker'))
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--operation',choices=('prepare','pilots','final','readout'))
    args=parser.parse_args();base=args.out.resolve();config=read(args.config)
    if args.stage=='worker':
        if args.operation=='prepare':prepare(config,base)
        elif args.operation=='readout':readout(base)
        else:run_pairs(config,base,args.operation=='pilots')
    else:
        budget=read(base/'canonical-plan.json')['quote']['active_work_budget_seconds'] if args.stage=='final' else 7200
        if args.stage=='final':
            spent=sum(read(p)['seconds'] for p in (base/'guards').glob('final-*/receipt.json'))
            budget-=spent
        print(json.dumps(guard(base,args.stage,[sys.executable,'-m','scripts.run_hu100_v053_translation',
            'worker','--operation',args.stage,'--config',str(args.config.resolve()),'--out',str(base)],active_budget=budget)))


if __name__=='__main__':main()
