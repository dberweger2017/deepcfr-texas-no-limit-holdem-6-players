"""Owner-authorized HU100 pot-pressure diagnosis, frozen counts and paired readout."""
import argparse
import gc
import gzip
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from zipfile import ZipFile

from scripts import pot_pressure_diagnostic as diagnostic
from scripts.evaluate_hu100_direct import put
from scripts.report_native_hu100_learning_curves import interval
from scripts.reserve_match_roots import reserve
from src.blueprint.abstraction import HU100_SCHEMA, HU100_EQUITY_SCHEMA
from src.blueprint.equity_buckets import EQUITY_K50_SHA256
from src.policies.files import file_hash
from src.policies.hu100_bundle import RELEASES
from src.arena.schedule import canonical, digest, build_schedule

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/hu100-pot-pressure'
PILOT_ROOT, FINAL_ROOT = 20261013100101,20261013100102
REPO = 'dberweger2017/deepcfr-texas-no-limit-holdem-6-players'


def read(path): return json.loads(Path(path).read_text())


def revision():
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip(): raise ValueError('Clean source required')
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()


def pin(path): return {'path':str(path),'bytes':path.stat().st_size,'sha256':file_hash(path)}


def prepare(pr):
    BASE.mkdir(parents=True,exist_ok=True)
    reserve([PILOT_ROOT,FINAL_ROOT],pr,BASE/'root-reservation.json')
    models = {}; receipts = {}
    for label,tag,source in [('v052','v0.5.2','3a3bc4218bab96637d6a27a69b73383ee8bb196c'),
                             ('v051','v0.5.1','dca7547e6eca82ce77f3e42230b82fabc1c29a4c')]:
        release_owner = json.loads(subprocess.check_output(['gh','pr','view',str(RELEASES[tag]['provenance']['pr']),'--repo',REPO,'--json','state,mergedAt,url'],text=True))
        if release_owner['state']!='MERGED': raise ValueError('Released research model owner must be merged')
        dest = BASE/'releases'/tag; dest.mkdir(parents=True,exist_ok=False)
        subprocess.run(['gh','release','download',tag,'--repo',REPO,'--dir',str(dest)],check=True)
        subprocess.run([sys.executable,str(dest/'verify_hu100_bundle.py'),str(dest),'--expect-source',source,'--require-publication'],check=True)
        item = RELEASES[tag]['model']; path = dest/item['file']
        models[label] = dict(pin(path),name=label,format=item['format'],abstraction=HU100_SCHEMA,
                           entries=item['entries'],iteration=item['iteration'],source_checkpoint_sha256=item['checkpoint_sha256'])
        receipts[label] = {'owner':release_owner,'release':tag,'source':source,'verifier':pin(dest/'verify_hu100_bundle.py')}
    owner = json.loads(subprocess.check_output(['gh','pr','view','246','--repo',REPO,'--json','state,mergedAt,url'],text=True))
    if owner['state']!='MERGED': raise ValueError('PR246 must be merged before restoration')
    index = read(ROOT/'docs/reports/hu100-k50-2b-artifacts/model-index.json')
    asset = next(a for a in index['models'] if a['kind']=='average' and a['actual_nodes']>=2_000_000_000)
    archive = Path(index['archives']['averages']['path'])
    if file_hash(archive)!=index['archives']['averages']['sha256']: raise ValueError('K50 archive hash differs')
    dest = BASE/'k50'; dest.mkdir(exist_ok=False)
    with ZipFile(archive) as zipped:
        manifest = json.loads(zipped.read('ARCHIVE-MANIFEST.json'))
        recorded = manifest['members'][asset['member']]
        if recorded['sha256']!=asset['sha256'] or recorded['bytes']!=asset['bytes']: raise ValueError('K50 member index differs')
        path = dest/'average.gz'
        with zipped.open(asset['member']) as raw,path.open('xb') as output:
            import shutil; shutil.copyfileobj(raw,output,1024**2)
    if file_hash(path)!=asset['sha256'] or path.stat().st_size!=asset['bytes']: raise ValueError('K50 restored bytes differ')
    tables = BASE/'tables'; tables.mkdir(exist_ok=False)
    table_owner = json.loads(subprocess.check_output(['gh','pr','view','163','--repo',REPO,'--json','state,mergedAt,url'],text=True))
    if table_owner['state']!='MERGED': raise ValueError('PR163 must be merged')
    table_archive = Path.home()/'Local/Research-Cloud/PR-163-equity-buckets/hu20-equity-buckets-20261005.tar.gz'
    if file_hash(table_archive)!='03d4828f030082f213c8ce82e29fa0fa8fd97288bd8035ad0056de85bc0816ca': raise ValueError('K50 table archive differs')
    subprocess.run(['tar','-xzf',str(table_archive),'-C',str(tables),'--strip-components','1',
                    *[f'hu20-equity-buckets-20261005/{s}-k50.bin' for s in EQUITY_K50_SHA256]],check=True)
    table_pins = {s:pin(tables/(s+'-k50.bin')) for s in EQUITY_K50_SHA256}
    if any(table_pins[s]['sha256']!=sha for s,sha in EQUITY_K50_SHA256.items()): raise ValueError('K50 tables differ')
    models['k50'] = dict(pin(path),name='k50',format='holdem-hu100-stored-cfr-average-research-v1',abstraction=HU100_EQUITY_SCHEMA,
                         entries=asset['entries'],iteration=asset['iterations'],source_checkpoint_sha256=next(a['sha256'] for a in index['models'] if a['kind']=='checkpoint' and a['actual_nodes']==asset['actual_nodes']),card_buckets=str(tables))
    put(BASE/'inputs.json',{'models':models,'release_receipts':receipts,'k50_owner':owner,'k50_archive':index['archives']['averages'],
                           'k50_asset':asset,'tables':table_pins,'table_owner':table_owner,'source':revision()})


def specs(): return read(BASE/'inputs.json')['models']


def play(pilot=False):
    source = revision(); models = specs()
    if pilot:
        blocks, root = 128,PILOT_ROOT
    else:
        frozen = read(BASE/'canonical-plan.json'); posted = read(BASE/'plan-posted.json')
        if frozen['source']!=source or file_hash(BASE/'canonical-plan.json')!=posted['sha256'] or frozen['inputs_sha256']!=file_hash(BASE/'inputs.json'):
            raise ValueError('Frozen plan/source/inputs differ')
        blocks,root = frozen['blocks'],FINAL_ROOT
    quotes = {}; registry = None; previous_model = None
    for arm,(label,_,_,_) in diagnostic.ARMS.items():
        if previous_model!=label:
            registry = None; gc.collect()
        dest = BASE/('pilot' if pilot else 'final')/arm
        registry,done = diagnostic.execute(models[label],arm,blocks,root,dest/'play',source,registry)
        replay = diagnostic.audit(dest/'play',dest/'replay.json',cost_only=pilot)
        registry,repeat = diagnostic.execute(models[label],arm,blocks,root,dest/'reproduction',source,registry,dest/'play')
        quotes[arm] = {'load_seconds':done['load_seconds'],'play_seconds':done['play_seconds'],
                       'replay_seconds':replay['seconds'],'reproduction_seconds':repeat['play_seconds']}
        previous_model = label
    del registry; gc.collect()
    put(BASE/('pilot-costs.json' if pilot else 'all-final-verified.json'),{'status':'verified','blocks':blocks,'root':root,
          'quotes':quotes,'payoffs_inspected':False,'source':source})


def freeze(pr):
    costs = read(BASE/'pilot-costs.json'); n = 16384
    fixed = sum(q['load_seconds'] for q in costs['quotes'].values())
    rate = sum(q['play_seconds']+q['replay_seconds']+q['reproduction_seconds'] for q in costs['quotes'].values())/costs['blocks']
    while fixed+rate*n>2.5*3600 and n>1: n//=2
    reserve([FINAL_ROOT],pr,BASE/'final-root-reservation.json')
    # All arms intentionally share the same deal/opponent/action streams.
    plan = diagnostic.make_plan(specs()['v052'],n,FINAL_ROOT)
    pilot_seeds = {b.deal_seeds[0] for b in build_schedule(diagnostic.make_plan(specs()['v052'],128,PILOT_ROOT))}
    final_seeds = {b.deal_seeds[0] for b in build_schedule(plan)}
    if len(final_seeds)!=n or pilot_seeds&final_seeds: raise ValueError('Fresh schedule collision')
    document = {'pr':pr,'source':revision(),'inputs_sha256':file_hash(BASE/'inputs.json'),'blocks':n,
        'pilot_root':PILOT_ROOT,'root':FINAL_ROOT,'models':specs(),'arms':diagnostic.ARMS,
        'style_baseline':asdict_style(3),'style_control':asdict_style(2),'translation':{'max_states':512,'max_events':128},
        'missing_definition':'unresolved missing exact key after released translation; translated witnesses are a separate category',
        'pairing':'PR236 scripted action_seed and Plan; same deals/opponent seeds for every arm, rotations 0/1',
        'statistics':'nominal Student-t paired duplicate-block 95%, contrasts size1x-baseline,passive-baseline,folding-baseline,k50-v1',
        'pilot_costs':costs,'projected_seconds':fixed+rate*n,'no_outcome_driven_extension':True,'no_failed_hand_reruns':True,
        'deal_seeds_sha256':digest(sorted(final_seeds)), 'root_reservation_sha256':file_hash(BASE/'final-root-reservation.json')}
    put(BASE/'canonical-plan.json',document)


def asdict_style(n):
    from dataclasses import asdict
    from src.arena.heuristics import Style
    return asdict(Style(0.40,0.60,0.95,0.20,n))


def readout():
    final = read(BASE/'all-final-verified.json'); n = final['blocks']; models = specs()
    values = {}; audits = {}
    for arm in diagnostic.ARMS:
        audit = read(BASE/'final'/arm/'replay.json'); repeat = read(BASE/'final'/arm/'reproduction/complete.json')
        if audit['status']!='verified' or not repeat['reproduced_all_hands_and_decisions'] or audit['hands_replayed']!=2*n:
            raise ValueError('Every final hand must verify before readout')
        values[arm] = [(audit['values'][f'{b}/0']+audit['values'][f'{b}/1'])/2 for b in range(n)]
        audits[arm] = audit
    results = {arm:interval(series) for arm,series in values.items()}
    differences = {f'{a}-{b}':interval([x-y for x,y in zip(values[a],values[b],strict=True)])
                   for a,b in [('size1x','baseline'),('passive','baseline'),('folding','baseline'),('k50','v1')]}
    base = audits['baseline']; classes = {}
    for kind in ('exact_or_zero_mass','translated_only','missing'):
        contribution = [(sum(base['values'][f'{b}/{r}'] if base['hand_classes'][f'{b}/{r}']==kind else 0 for r in (0,1)))/2 for b in range(n)]
        selected = [v for coord,v in base['values'].items() if base['hand_classes'][coord]==kind]
        classes[kind] = {'hands':len(selected),'fraction':len(selected)/(2*n),'net_contribution_bb_per_100':interval(contribution),
                         'conditional_bb_per_100':sum(selected)/len(selected) if selected else None}
    positive_values = [v for coord,v in base['values'].items() if base['hand_positive_exact_only'][coord]]
    positive_contribution = [sum(base['values'][f'{b}/{r}'] if base['hand_positive_exact_only'][f'{b}/{r}'] else 0 for r in (0,1))/2 for b in range(n)]
    put(BASE/'summary.json',{'positive_mass_exact_only_hands':{'hands':len(positive_values),
         'conditional_bb_per_100':sum(positive_values)/len(positive_values) if positive_values else None,
         'net_contribution_bb_per_100':interval(positive_contribution)},'status':'verified','blocks':n,'results':results,'paired_differences':differences,'baseline_hand_classes':classes,
         'baseline_action_mix':base['action_mix'],'baseline_chips_committed_by_street':base['chips_committed_by_street'],
         'baseline_chips_committed_in_losing_hands_by_street':base['chips_committed_in_losing_hands_by_street'],
         'baseline_net_chips_by_terminal_street':base['net_chips_by_terminal_street'],
         'baseline_gross_chips_lost_by_terminal_street':base['gross_chips_lost_by_terminal_street'],
         'all_hands_replayed_and_reproduced':True,'total_final_hands':12*n,'inference_conditions_on_fixed_models':True})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('operation',choices=('prepare','pilot','freeze','final','readout')); p.add_argument('--pr',type=int)
    a = p.parse_args()
    if a.operation=='prepare': prepare(a.pr)
    elif a.operation=='freeze': freeze(a.pr)
    elif a.operation=='pilot': play(True)
    elif a.operation=='final': play(False)
    else: readout()


if __name__=='__main__': main()
