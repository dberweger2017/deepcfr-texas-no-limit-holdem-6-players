"""Paired HU100 frozen-v2 panel with a cost-only pilot and global readout barrier."""
import argparse
import fcntl
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import time

import psutil

from scripts import evaluate_hu100_direct as direct
from scripts.archive_hu100_4b_seed_ladder import seal
from scripts.reserve_match_roots import reserve
from scripts.run_hu100_k50_v2 import header, pin
from scripts.run_hu200_frozen_evaluation import estimate, series, gh, revision, read
from scripts.train_hu100_v2_frozen_5b import BASE, BINARY, ROOT, TABLES, ENTRIES
from src.blueprint.average import AveragePolicy
from src.policies.files import file_hash

PR = 272
ROOTS = dict(pilot_direct=20261014272001,pilot_panel=20261014272002,
             panel=20261014272003,direct=20261014272004)
RULE = ('Recommend v0.5.3 candidate iff no primary panel paired nominal Student-t95 '
        'upper<-20 BB/100 AND pot_pressure lower>-20 BB/100; owner decides')
put = direct.put


def projected_cost(cost, blocks):
    """Only block work scales; model/table admission stays one payment per run."""
    fixed = cost['fixed_seconds']
    scaled = cost['block_work_seconds']*blocks/cost['blocks']
    return dict(fixed_seconds=fixed,scaled_block_seconds=scaled,total_seconds=fixed+scaled)


def select_counts(costs, elapsed):
    options = []
    for panel, count in ((16384,131072),(8192,65536)):
        parts = [projected_cost(costs['direct'],count)]
        parts += [projected_cost(c,panel) for c in costs['panel'].values()]
        fixed = sum(p['fixed_seconds'] for p in parts)
        scaled = sum(p['scaled_block_seconds'] for p in parts)
        total = elapsed+1.25*(fixed+scaled)+1800
        row = dict(panel=panel,direct=count,elapsed_seconds=elapsed,fixed_seconds=fixed,
            scaled_block_seconds=scaled,planning_margin=1.25,closeout_seconds=1800,
            projected_whole_seconds=total,fits=total<=7*3600)
        options.append(row)
        if row['fits']: return row,options
    raise ValueError('Neither pre-outcome count option fits seven hours')


def recommendation(panel):
    if set(panel) != set(direct.PANEL_OPPONENTS):
        raise ValueError('Every primary opponent is required')
    flags = {o:r['contrast']['ci95'][1] < -20 for o,r in panel.items()}
    return panel['pot_pressure']['contrast']['ci95'][0] > -20 and not any(flags.values()),flags


def prepare():
    preparation_started = time()
    reserve(list(ROOTS.values()),PR,BASE/'root-reservation.json')
    frozen = dict(read(BASE/'treatment.json'),translation=None)
    if read(BASE/'5b.audit.json')['all_nodes_verified'] != ENTRIES:
        raise ValueError('Full frozen audit required')
    prior = read(ROOT/'docs/reports/hu100-k50-v2-artifacts/model-index.json')
    parent_spec = dict(prior['terminal'],name='v2-unfrozen',translation=None)
    original = Path(parent_spec['path'])
    method = 'hash-verified retained average'
    if not original.exists():
        original = BASE/'parent/332894788.average.gz'
        original.parent.mkdir(exist_ok=True)
        parent = read(BASE/'parent-verified.json')['checkpoint']
        subprocess.run(list(map(str,[BINARY,'export',parent['path'],'--average',original,'--zero-mass','uniform'])),check=True)
        method = 'native average regenerated from byte-identical PR260 checkpoint'
    if (file_hash(original) != parent_spec['sha256'] or original.stat().st_size != parent_spec['bytes']):
        raise ValueError('Unfrozen average differs from audited PR260 average')
    parent_spec.update(path=str(original),card_buckets=str(TABLES))
    release = dict(read(ROOT/'docs/reports/hu100-k50-v2-artifacts/inputs.json')['models']['v052'],translation=None)
    if file_hash(Path(release['path'])) != release['sha256'] or Path(release['path']).stat().st_size != release['bytes']:
        raise ValueError('Published v0.5.2 model differs')
    # Published source verification is retained separately from inferred model names.
    bundle = Path(release['path']).parent
    verification = subprocess.check_output([sys.executable,str(bundle/'verify_hu100_bundle.py'),str(bundle),
        '--expect-source','3a3bc4218bab96637d6a27a69b73383ee8bb196c','--require-publication'],text=True)
    models = dict(frozen=frozen,unfrozen=parent_spec,release=release)
    pairs = dict(primary=[frozen,dict(release,translation=dict(max_states=512,max_events=128))],
                 learning=[frozen,parent_spec],direct=[frozen,release])
    put(BASE/'evaluation-inputs.json',dict(models=models,pairs=pairs,parent_average_method=method,
        parent_index=prior,release_verification=verification,source=revision()))
    input_seconds = time()-preparation_started
    cache = BASE/'policy-cache'
    for spec in models.values():
        started=time()
        model = AveragePolicy(spec['path'],spec['sha256'],expected_schema=spec['abstraction'],
            card_buckets=Path(spec['card_buckets']) if spec.get('card_buckets') else None,
            sorted_rows=True,cache=cache)
        put(BASE/('cache-'+spec['name']+'.json'),dict(seconds=time()-started,description=model.description,
            no_hands=True,model_sha256=spec['sha256']))
        del model; gc.collect()
    put(BASE/'evaluation-preparation-complete.json',dict(input_verification_and_parent_export_seconds=input_seconds,
        model_cache_builds={s['name']:read(BASE/('cache-'+s['name']+'.json')) for s in models.values()},
        whole_seconds=time()-preparation_started,no_hands=True,scaled_with_blocks=False))


def pair_worker(folder, specs, blocks, root, opponent, pilot):
    cache = BASE/'policy-cache'
    registry, original = direct.execute(specs,blocks,root,'hu100-v2-frozen',folder/'play',revision(),
        stack_bb=100,opponent=opponent,cache=cache,cache_read_only=True)
    del registry; gc.collect()
    checked = direct.audit(folder/'play',folder/'replay.json',cost_only=True)
    registry, repeat = direct.execute(specs,blocks,root,'hu100-v2-frozen',folder/'reproduction',revision(),
        reproduce=folder/'play',stack_bb=100,opponent=opponent,cache=cache,cache_read_only=True)
    del registry; gc.collect()
    if not repeat['reproduced_all_hands_and_decisions']:
        raise ValueError('Exact reproduction incomplete')
    fixed = original['load_or_validation_seconds']+repeat['load_or_validation_seconds']+checked['fixed_setup_seconds']
    block_work = original['seconds']+repeat['seconds']+checked['seconds']-fixed
    put(folder/'costs.json',dict(primary=original,repeat=repeat,audit_seconds=checked['seconds'],
        audit_fixed_seconds=checked['fixed_setup_seconds'],audit_block_work_seconds=checked['block_work_seconds'],
        fixed_seconds=fixed,block_work_seconds=block_work,blocks=blocks,pilot=pilot,
        paired_sd_bb_per_100=checked['paired_sd_bb_per_100']))


def run_pair(label, specs, blocks, root, opponent=None, pilot=False):
    path = BASE/(label+'-specs.json'); put(path,specs)
    command = [sys.executable,'-m','scripts.evaluate_hu100_v2_frozen_5b','pair',
        '--folder',BASE/label,'--specs',path,'--blocks',blocks,'--root',root]
    if opponent: command += ['--opponent',opponent]
    if pilot: command += ['--pilot']
    with (BASE/(label+'.log')).open('x') as log:
        subprocess.run(list(map(str,command)),check=True,stdout=log,stderr=subprocess.STDOUT,
            timeout=max(1,read(BASE/'budget.json')['deadline']-1800-time()))
    return read(BASE/label/'costs.json')


def evaluate():
    pairs = read(BASE/'evaluation-inputs.json')['pairs']
    costs = dict(direct=run_pair('pilot-direct',pairs['direct'],1024,ROOTS['pilot_direct'],pilot=True),panel={})
    for comparison in ('primary','learning'):
        for opponent in direct.PANEL_OPPONENTS:
            label = comparison+'-'+opponent
            costs['panel'][label] = run_pair('pilot-'+label,pairs[comparison],64,ROOTS['pilot_panel'],opponent,True)
    put(BASE/'pilot-costs.json',costs)
    budget = read(BASE/'budget.json')
    counts, options = select_counts(costs,time()-budget['started'])
    plan = dict(pr=PR,source=revision(),counts=counts,projection_options=options,pilot=costs,
        one_off_preparation=read(BASE/'evaluation-preparation-complete.json'),
        inputs=read(BASE/'evaluation-inputs.json'),roots=ROOTS,opponents=direct.PANEL_OPPONENTS,
        decision_rule=RULE,conditional_on='fixed models and training seed2026100601',
        direct_role='descriptive only; translation off; cross-menu missing support limits interpretation',
        learning_role='descriptive frozen-minus-unfrozen on same panel and shared deals',
        pairing='HU100 reset, duplicate rotated seats; shared deals/action-role RNGs across primary and learning',
        no_failed_hand_reruns=True,no_pilot_pooling=True,no_outcome_extension=True,
        every_final_hand_independently_replayed_and_exactly_reproduced_before_readout=True,
        frozen_at=time(),deadline=budget['deadline'],resource_guards='5s samples;60s RSS>13GiB/critical/free<8%/swap growth>4GB;disk<16GiB immediate',
        cost_scaling='One-off model/cache/table loading and model/table hashes stay fixed; play/trace hashing/replay/reproduction scale by blocks')
    put(BASE/'canonical-plan.json',plan)
    original = json.loads(gh('pr','view',PR,'--json','body'))['body']
    body = original+'\n\n## Canonical plan before final hands\n\nSHA256 `'+file_hash(BASE/'canonical-plan.json')+'`.\n\n```json\n'+json.dumps(plan,indent=2,sort_keys=True)+'\n```\n'
    path = BASE/'canonical-pr-body.md'; path.write_text(body)
    gh('pr','edit',PR,'--body-file',path)
    if json.loads(gh('pr','view',PR,'--json','body'))['body'] != body:
        raise ValueError('Canonical plan PR readback differs')
    put(BASE/'plan-posted.json',dict(sha256=file_hash(BASE/'canonical-plan.json'),at=time(),readback_verified=True))
    for comparison in ('primary','learning'):
        for opponent in direct.PANEL_OPPONENTS:
            run_pair(comparison+'-'+opponent,pairs[comparison],counts['panel'],ROOTS['panel'],opponent)
    run_pair('direct',pairs['direct'],counts['direct'],ROOTS['direct'])
    put(BASE/'all-final-verified.json',dict(status='verified',at=time(),source=revision()))
    readout(plan)


def readout(plan):
    if read(BASE/'all-final-verified.json')['status'] != 'verified':
        raise ValueError('Every final hand must verify before readout')
    folders = ['direct',*[c+'-'+o for c in ('primary','learning') for o in direct.PANEL_OPPONENTS]]
    coverage = {}
    for label in folders:
        folder = BASE/label
        if (read(folder/'replay.json')['status'] != 'verified'
                or not read(folder/'reproduction/complete.json')['reproduced_all_hands_and_decisions']):
            raise ValueError('Global verification barrier incomplete')
        for stage in ('play','reproduction'):
            for name, expected in read(folder/stage/'complete.json')['files'].items():
                p=folder/stage/name
                if p.stat().st_size != expected['bytes'] or file_hash(p) != expected['sha256']:
                    raise ValueError('Verified final bytes changed')
        coverage[label] = read(folder/'replay.json')['coverage']
    summary = dict(primary={},learning={},direct=estimate(series(BASE/'direct',plan['counts']['direct'])['candidate']),coverage=coverage)
    for comparison in ('primary','learning'):
        for opponent in direct.PANEL_OPPONENTS:
            arms = series(BASE/(comparison+'-'+opponent),plan['counts']['panel'],panel=True)
            summary[comparison][opponent] = dict(arms={a:estimate(v) for a,v in arms.items()},
                contrast=estimate([a-b for a,b in zip(arms['candidate'],arms['baseline'],strict=True)]))
            if comparison=='learning':
                first = series(BASE/('primary-'+opponent),plan['counts']['panel'],panel=True)['candidate']
                if first != arms['candidate']:
                    raise ValueError('Shared candidate blocks differ between panels')
    passes, flags = recommendation(summary['primary'])
    summary.update(status='verified',recommend_v053_candidate=passes,severe_primary_regressions=flags,
        rule=RULE,canonical_plan_sha256=file_hash(BASE/'canonical-plan.json'),
        all_final_hands_independently_replayed_and_exactly_reproduced=True)
    put(BASE/'summary.json',summary)


def archive():
    dest=Path.home()/f'Local/Research-Cloud/PR-{PR}-hu100-v2-frozen-5b';dest.mkdir(parents=True,exist_ok=True)
    # Byte-identical PR260 inputs already have accepted restoration archives.
    # Preserve the copies locally and reference their exact indexed members.
    excluded=('parent','policy-cache','guards/operations/archive')
    files=[(p,'research/'+p.relative_to(BASE).as_posix()) for p in sorted(BASE.rglob('*'))
        if p.is_file() and not p.is_symlink() and not any(p.is_relative_to(BASE/x) for x in excluded)
        and p.name not in ('guard-held.json','archive-receipt.json','average-receipt.json')]
    files += [(BINARY,'research/bin/hu20-trainer')]
    files += [(ROOT/p,'source/'+p) for p in ('scripts/train_hu100_v2_frozen_5b.py',
        'scripts/evaluate_hu100_v2_frozen_5b.py','scripts/evaluate_hu100_direct.py','docs/hu100-v2-frozen-5b-protocol.md')]
    files += [(p,'launch/'+p.relative_to(ROOT/'planning').as_posix()) for p in sorted((ROOT/'planning').rglob('*'))
        if p.is_file() and not p.is_symlink() and p.name!='evaluation-pipeline.log']
    average = Path(read(BASE/'treatment.json')['path'])
    if shutil.disk_usage(dest).free-sum(p.stat().st_size for p,_ in files)-average.stat().st_size < 16*1024**3:
        raise ValueError('Both archives would breach the disk floor; preserve originals')
    owner=json.loads(gh('pr','view',PR,'--json','state,url,headRefOid'))
    metadata=dict(source=revision(),owner=owner,originals_retained=True,
        indexed_pr260_inputs=read(BASE/'inputs.json')['parent_index'],
        omitted_rebuildable_policy_cache=str(BASE/'policy-cache'))
    put(BASE/'archive-receipt.json',seal(files,dest/'hu100-v2-frozen-5b-M4-20261014.zip',metadata))
    put(BASE/'average-receipt.json',seal([(average,'averages/5000000000.average.gz')],
        dest/'hu100-v2-frozen-5b-average-M4-20261014.zip',metadata))


def run():
    if not (BASE/'training-detached-complete.json').exists():
        raise ValueError('Detached training/export/audit must complete first')
    with (Path.home()/'Local/.hu20-m4-research.lock').open('a+') as one, (Path.home()/'Local/.hu100-m4-research.lock').open('a+') as two:
        for f in (one,two): fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        own={psutil.Process().pid,*[p.pid for p in psutil.Process().parents()]}
        for p in psutil.process_iter(['pid','name','cmdline'],ad_value=None):
            if p.pid in own: continue
            cmd=' '.join((p.info['cmdline'] or [])[1:])
            if p.info['name']=='hu20-trainer' or any(x in cmd for x in ('scripts.run_hu','scripts.evaluate_hu','campaign.py')):
                raise ValueError('Another campaign is active: '+str(p.pid))
        for stage in ('prepare','evaluate','archive'):
            deadline=read(BASE/'budget.json')['deadline']-(0 if stage=='archive' else 1800)
            subprocess.run(list(map(str,[sys.executable,'-m','scripts.guard_m4_research',
                '--stage',BASE/'guards','--name','evaluation-'+stage,'--deadline',deadline,
                '--',sys.executable,'-m','scripts.evaluate_hu100_v2_frozen_5b',stage])),check=True)
    put(BASE/'detached-complete.json',dict(at=time(),research_locks_released=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('run','prepare','pair','evaluate','archive'))
    parser.add_argument('--folder',type=Path);parser.add_argument('--specs',type=Path)
    parser.add_argument('--blocks',type=int);parser.add_argument('--root',type=int)
    parser.add_argument('--opponent',choices=direct.PANEL_OPPONENTS);parser.add_argument('--pilot',action='store_true')
    a=parser.parse_args()
    try:
        if a.stage=='pair':pair_worker(a.folder,read(a.specs),a.blocks,a.root,a.opponent,a.pilot)
        else:globals()[a.stage]()
    except BaseException as exc:
        put(BASE/(a.stage+'-failure.json'),dict(at=time(),failure=repr(exc)))
        raise


if __name__=='__main__':main()
