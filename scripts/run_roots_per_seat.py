"""Owner-authorized HU100 roots-per-seat comparison at fixed nodes and match counts."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys

from scripts import evaluate_hu100_direct as direct
from scripts.evaluate_hu100_direct import put
from scripts.reserve_match_roots import reserve
from src.blueprint.abstraction import HU100_SCHEMA
from src.diagnostics.cfr_average import audit
from src.policies.files import file_hash
from src.policies.hu100_bundle import RELEASES

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/hu100-roots-per-seat'
BINARY = ROOT/'native/hu20-trainer/target/release/hu20-trainer'
PILOT_ROOT, FINAL_ROOT = 20261013200101,20261013200102
SEED = 2026100601
SOURCE_BASE = '7b0090242767bfe93e24a7501a4a87664bef7f9d'
# Keep the native default far above the run; observed machine guards stop B.
CAP = 1_000_000_000
REPO = 'dberweger2017/deepcfr-texas-no-limit-holdem-6-players'


def read(path): return json.loads(Path(path).read_text())


def pin(path): return {'path':str(path),'bytes':path.stat().st_size,'sha256':file_hash(path)}


def revision():
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip(): raise ValueError('Clean committed source required')
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()


def prepare(pr):
    reserve([PILOT_ROOT,FINAL_ROOT],pr,BASE/'root-reservation.json')
    owner = json.loads(subprocess.check_output(['gh','pr','view','207','--repo',REPO,'--json','state,mergedAt,url'],text=True))
    if owner['state']!='MERGED': raise ValueError('Released reference owner must be merged')
    dest = BASE/'releases/v0.5.0'; dest.mkdir(parents=True,exist_ok=False)
    subprocess.run(['gh','release','download','v0.5.0','--repo',REPO,'--dir',str(dest)],check=True)
    subprocess.run([sys.executable,str(dest/'verify_v050_bundle.py'),str(dest),'--expect-source',
                    'b9c9bd161422e87371a2641ae4c60304c66eed79','--require-publication'],check=True)
    item = RELEASES['v0.5.0']['model']; path = dest/item['file']
    spec = dict(pin(path),name='v050-r1',format=item['format'],abstraction=HU100_SCHEMA,
                entries=item['entries'],iteration=item['iteration'],source_checkpoint_sha256=item['checkpoint_sha256'])
    put(BASE/'inputs.json',{'reference_owner':owner,'reference':spec,'release':'v0.5.0','source':revision(),'native_main_base':SOURCE_BASE,'verifier':pin(dest/'verify_v050_bundle.py'),
         'historical_r1_training_receipt':read(ROOT/'docs/reports/native-hu100-growth-1b-artifacts/training-result.json')})


def train_command(roots,nodes,dest):
    return [BINARY,'train','--stack-bb',100,'--seed',SEED,'--roots-per-seat',roots,
            '--average-rule','opponent-sampled','--nodes',nodes,'--max-entries',CAP,
            '--out',dest/'checkpoint.gz','--telemetry',dest/'telemetry.jsonl','--stop-file',BASE/'native-stop.json']


def timing():
    # Same current binary and machine, three timing-only 100M recipes; no poker scores.
    for roots in (1,4,8):
        dest = BASE/'timing'/str(roots); dest.mkdir(parents=True,exist_ok=False)
        subprocess.run(['/usr/bin/time','-l',*map(str,train_command(roots,100_000_000,dest))],check=True)
    put(BASE/'timing-complete.json',{'roots':(1,4,8),'nodes_per_run':100_000_000,'only_costs':True,'binary':pin(BINARY)})


def train(roots):
    dest = BASE/'training'/str(roots); dest.mkdir(parents=True,exist_ok=False)
    rc = subprocess.run(['/usr/bin/time','-l',*map(str,train_command(roots,1_000_000_000,dest))]).returncode
    if rc not in (0,3): raise RuntimeError('Native training failed '+str(rc))
    rows = [json.loads(s) for s in (dest/'telemetry.jsonl').read_text().splitlines()]
    if len(rows)!=1 or rows[0]['completed_nodes']<1_000_000_000:
        raise ValueError('Incomplete fixed 1B target; preserve partial and stop stage')
    if rows[0]['binary_sha256']!=file_hash(BINARY) or rows[0]['checkpoint_sha256']!=file_hash(dest/'checkpoint.gz'):
        raise ValueError('Binary/checkpoint pin differs')
    put(dest/'training-complete.json',rows[0])


def exports(roots):
    dest = BASE/'training'/str(roots); checkpoint = dest/'checkpoint.gz'; current = dest/'current.gz'; average = dest/'average.gz'
    subprocess.run([str(BINARY),'export',str(checkpoint),'--current',str(current),'--average',str(average),'--zero-mass','uniform'],check=True)
    with gzip.open(checkpoint,'rt') as source: h = json.loads(next(source))
    spec = {'seed':SEED,'iteration':h['iteration'],'checkpoint_sha256':file_hash(checkpoint),'sha256':file_hash(current)}
    result = audit(checkpoint,current,average,spec,file_hash(average),expected_schema=HU100_SCHEMA)
    entries = read(dest/'training-complete.json')['diagnostics']['entries']
    if result['all_nodes_verified']!=entries or h['config']['roots_per_seat']!=roots: raise ValueError('Incomplete audit or roots mismatch')
    put(dest/'audit.json',result)
    with gzip.open(average,'rt') as source: header = json.loads(next(source))
    model = dict(pin(average),name=f'r{roots}-1b',format=header['format'],abstraction=HU100_SCHEMA,
                  entries=entries,iteration=h['iteration'],source_checkpoint_sha256=file_hash(checkpoint))
    put(dest/'spec.json',model)


def pairs(): return {f'r{r}-r1':[read(BASE/'training'/str(r)/'spec.json'),read(BASE/'inputs.json')['reference']] for r in (4,8)}


def play(pilot=False):
    source = revision()
    if pilot: blocks,root=128,PILOT_ROOT
    else:
        freeze = read(BASE/'canonical-plan.json'); posted = read(BASE/'plan-posted.json')
        if (freeze['source']!=source or file_hash(BASE/'canonical-plan.json')!=posted['sha256']
                or freeze['pairs']!=pairs() or freeze['inputs_sha256']!=file_hash(BASE/'inputs.json')):
            raise ValueError('Final source/plan/inputs differ')
        blocks,root = 131072,FINAL_ROOT
    costs = {}
    for rung,pair in pairs().items():
        dest = BASE/('pilot' if pilot else 'final')/rung
        registry,done = direct.execute(pair,blocks,root,rung,dest/'play',source)
        checked = direct.audit(dest/'play',dest/'replay.json')
        registry,repeat = direct.execute(pair,blocks,root,rung,dest/'reproduction',source,registry,dest/'play')
        costs[rung] = {'load_seconds':done['load_or_validation_seconds'],'play_seconds':done['play_seconds'],
                       'replay_seconds':checked['seconds'],'repeat_seconds':repeat['play_seconds']}
        del registry
        import gc; gc.collect()
    put(BASE/('pilot-costs.json' if pilot else 'all-final-verified.json'),{'status':'verified','blocks':blocks,'root':root,
          'costs':costs,'payoffs_inspected':False,'source':source})


def freeze(pr):
    reserve([FINAL_ROOT],pr,BASE/'final-root-reservation.json')
    from src.arena.schedule import build_schedule,digest
    seen = set(); seed_hashes = {}
    for root,n in ((PILOT_ROOT,128),(FINAL_ROOT,131072)):
        for rung,pair in pairs().items():
            seeds = {b.deal_seeds[0] for b in build_schedule(direct.make_plan(pair,n,root,rung))}
            if len(seeds)!=n or seen&seeds: raise ValueError('Duplicate physical schedule')
            seen.update(seeds); seed_hashes[str(root)+'/'+rung] = digest(sorted(seeds))
    put(BASE/'canonical-plan.json',{'pr':pr,'source':revision(),'blocks':131072,'pilot_root':PILOT_ROOT,'root':FINAL_ROOT,
        'pairs':pairs(),'seed_hashes':seed_hashes,'translation':None,
        'inputs_sha256':file_hash(BASE/'inputs.json'),'native_build_receipt':read(BASE/'build-receipt.json'),
        'native_main_base':SOURCE_BASE,'seed':SEED,'roots_per_seat':[4,8],
        'recipe':{'stack_bb':100,'average_rule':'opponent-sampled','weighting':'linear','zero_mass':'uniform'},
        'reference_release':{'tag':'v0.5.0','verified_source':'b9c9bd161422e87371a2641ae4c60304c66eed79','require_publication':True},
        'root_reservation_sha256':file_hash(BASE/'final-root-reservation.json'),'statistics':'nominal paired duplicate-block Student-t 95%',
        'question':'R4/R8 learning per node relative to published R1; report upper loss bound = minus lower gain bound',
        'no_extension':True,'no_failed_hand_reruns':True,'pilot_costs':read(BASE/'pilot-costs.json')})


def readout():
    if read(BASE/'all-final-verified.json')['status']!='verified': raise ValueError('All final play must verify')
    results = {}
    for rung in pairs():
        dest = BASE/'final'/rung
        result = direct.audit(dest/'play',dest/'readout.json',report=True)
        if result['hands_replayed']!=262144 or not read(dest/'reproduction/complete.json')['reproduced_all_hands_and_decisions']:
            raise ValueError('Every final hand must replay/reproduce')
        result['nominal_95_loss_interval'] = [-result['ci95'][1],-result['ci95'][0]]
        result['upper_loss_bound_bb_per_100'] = -result['ci95'][0]
        results[rung] = result
    put(BASE/'summary.json',{'status':'verified','results':results,'all_hands_replayed_and_reproduced':True,
        'total_final_hands':524288,'inference_conditions_on_fixed_models':True})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('prepare','timing','train','exports','pilot','freeze','final','readout'));p.add_argument('--pr',type=int);p.add_argument('--roots',type=int,choices=(4,8))
    a=p.parse_args()
    if a.operation=='prepare': prepare(a.pr)
    elif a.operation in ('train','exports'): globals()[a.operation](a.roots)
    elif a.operation=='freeze': freeze(a.pr)
    elif a.operation=='pilot': play(True)
    elif a.operation=='final': play(False)
    else: globals()[a.operation]()


if __name__=='__main__':main()
