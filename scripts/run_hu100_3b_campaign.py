"""Sequential one-use M4 orchestration; posting gates precede final science."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import time

from scripts import run_hu100_3b_ladder as run
from scripts.evaluate_hu100_direct import put
from src.policies.files import file_hash

ROOT,OUT,BINARY=run.ROOT,run.OUT,run.BINARY
PYTHON=sys.executable
MODULE='scripts.run_hu100_3b_ladder'


def guarded(name,command,seconds=None,stop=None):
    args=[PYTHON,'-m','scripts.overnight_research_guard','--out',str(OUT/'guards'),'--name',name]
    if seconds:args+=['--seconds',str(seconds)]
    if stop:args+=['--stop-file',str(stop),'--accept-capacity-stop']
    subprocess.run([*args,'--',*map(str,command)],check=True,cwd=ROOT)


def trainer(target,resume=None,*,pilot=False):
    folder=OUT/('training-pilot' if pilot else 'training/'+str(target));folder.mkdir(parents=True,exist_ok=False)
    cap=run.GATE_ENTRY_CAP if target<=1_000_000_000 else run.ENTRY_CAP
    telemetry=OUT/'training'/f'{target}-telemetry.jsonl' if not pilot else folder/'telemetry.jsonl'
    command=[BINARY,'train','--stack-bb','100','--seed',str(run.SEED),'--roots-per-seat','1',
        '--average-rule','opponent-sampled','--nodes',str(target),'--max-entries',str(cap),
        '--out',folder/'checkpoint.gz','--telemetry',telemetry,'--stop-file',folder/'stop.json']
    if resume:command+=['--resume',resume,'--resume-sha256',file_hash(resume)]
    quote=None
    if not pilot:
        q=run.read(OUT/'training-quote.json')
        fraction=(target-(0 if resume is None else run.read(resume.parent/'gate.json')['actual_nodes']))/3_000_000_000
        quote=q['training_seconds']*max(fraction,.01)+q['save_seconds_per_entry']*q['entries'][str(target)]
        # Resume loading is charged from #207's measured parser cost per entry.
        if resume:quote+=2*923*run.read(resume.parent/'audit.json')['entries']/41_010_014
    guarded('pilot-train' if pilot else 'train-'+str(target),command,quote,folder/'stop.json')
    return folder


def pilot():
    guarded('prepare',[PYTHON,'-m',MODULE,'prepare'])
    folder=trainer(1_000_000,pilot=True)
    row=run.read_lines(folder/'telemetry.jsonl')[-1]
    speed=row['completed_nodes']/max(row['elapsed_seconds_including_writes']-row['write_seconds'],.001)
    index=run.read(ROOT/'docs/reports/native-hu100-growth-1b-artifacts/training-result.json')
    saves=index['saves']
    oldops=run.read(ROOT/'docs/reports/native-hu100-growth-1b-artifacts/resources.json')['operations']
    save_rate=2*max(s['write_seconds']/s['diagnostics']['entries'] for s in saves)
    tool_rate=2*max(sum(o['seconds'] for o in oldops if o['name'] in
        (f"save-{s['requested_nodes']}-export",f"save-{s['requested_nodes']}-audit"))/s['diagnostics']['entries'] for s in saves)
    entries={'500000000':30_027_422,'1000000000':41_010_014,'2000000000':55_000_000,'3000000000':run.ENTRY_CAP}
    training=2*3_000_000_000/min(speed,1_742_094)
    write_seconds=save_rate*sum(entries.values())
    tools=tool_rate*sum(entries.values())
    # Whole final originals remain local, but exact indexed 500M/1B models
    # need no second ZIP copies. Reserve new2B/3B archives and16GiB floor.
    bytes_per_entry=(1_789_504_352+1_173_264_021+1_033_507_823)/41_010_014
    originals=bytes_per_entry*sum(entries.values())
    archive=bytes_per_entry*(entries['2000000000']+entries['3000000000'])
    required=originals+archive+2*1024**3+16*1024**3
    available=shutil.disk_usage(OUT).free
    put(OUT/'training-quote.json',{'pilot_speed_nodes_per_second':speed,'training_seconds':training,
        'save_seconds_per_entry':save_rate,'tool_seconds_per_entry':tool_rate,'entries':entries,
        'save_seconds':write_seconds,'export_audit_seconds':tools,
        'total_seconds':training+write_seconds+tools,'required_additional_free_bytes':required,
        'available_bytes':available,'storage_admitted':available>=required,
        'memory_forecast_bytes':{k:110*v+100_000_000 for k,v in entries.items()},
        'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'basis':'fresh timing1M and #207 measured save/tool rates;2x measured headroom'})


def train():
    quote=run.read(OUT/'training-quote.json')
    posting=run.read(ROOT/'planning/training-quote-posted.json')
    if posting['quote_sha256']!=file_hash(OUT/'training-quote.json') or posting['pr']!=223:
        raise ValueError('Training pilot quote must be posted on owning PR')
    if not quote['storage_admitted'] or shutil.disk_usage(OUT).free<quote['required_additional_free_bytes']:
        raise ValueError('Training storage not admitted')
    resume=None
    for target in (500_000_000,1_000_000_000,2_000_000_000,3_000_000_000):
        folder=trainer(target,resume)
        guarded('check-'+str(target),[PYTHON,'-m',MODULE,'check','--target',target])
        guarded('export-'+str(target),[BINARY,'export',folder/'checkpoint.gz','--current',folder/'current.gz',
            '--average',folder/'average.gz','--zero-mass','uniform'],
            quote['tool_seconds_per_entry']*quote['entries'][str(target)])
        nodes=run.read(folder/'gate.json')['actual_nodes']
        guarded('audit-'+str(target),[PYTHON,'-m','scripts.audit_native_hu_checkpoint',
            '--checkpoint',folder/'checkpoint.gz','--current',folder/'current.gz','--average',folder/'average.gz',
            '--stack-bb','100','--target-nodes',nodes,'--out',folder/'audit.json'],
            quote['tool_seconds_per_entry']*quote['entries'][str(target)])
        guarded('spec-'+str(target),[PYTHON,'-m',MODULE,'spec','--target',target])
        if run.read(folder/'gate.json')['terminal_capacity_stop']:break
        resume=folder/'checkpoint.gz'
    guarded('pairs',[PYTHON,'-m',MODULE,'pairs'])


def calibrate(a_seconds):
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    # Memory admission precedes any dual-model loading; checked-sorted native
    # loading shares buffers rather than allocating global sorting copies.
    pairs=run.read(OUT/'pairs.json')
    memory={r:66*sum(s['entries'] for s in specs)+150_000_000 for r,specs in pairs.items()}
    put(OUT/'dual-model-admission.json',{'forecast_family_bytes':memory,'soft_bytes':7*1024**3,
        'basis':'66 B/entry checked-sorted buffers (native mean menu3.1:60B payload plus10% capacity headroom), plus150MB fixed pilot workspace'})
    if max(memory.values())>=7*1024**3:raise ValueError('Two-policy load memory not admitted')
    # Fresh physical schedules are checked before pilot hands too.
    put(OUT/'pilot-freshness.json',run.freshness(32))
    for rung in pairs:
        guarded('direct-pilot-'+rung,[PYTHON,'-m','scripts.evaluate_hu100_direct','--specs',OUT/'specs'/f'{rung}.json',
            '--blocks',32,'--root',run.PILOT_ROOT,'--rung',rung,'--out',OUT/'direct-pilot'/rung,
            '--source',source,'--pilot'])
    guarded('secondary-pilot',[PYTHON,'-m',MODULE,'secondary-pilot'])
    guarded('freeze',[PYTHON,'-m',MODULE,'freeze','--a-seconds',a_seconds])


def final():
    plan=run.read(OUT/'frozen-final.json');posting=run.read(ROOT/'planning/final-quote-posted.json')
    if posting['quote_sha256']!=file_hash(OUT/'frozen-final.json') or posting['pr']!=223:
        raise ValueError('Frozen timing-only sample/quote must be posted on owning PR')
    if shutil.disk_usage(OUT).free<plan['storage']['required_additional_free_bytes']:
        raise ValueError('Current final disk admission refused')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    # Reserve schedule/coordinate work at2KiB/block in addition to actual pilot
    # family high water, before taking the first final hand.
    peaks=plan['full_pair_pilot_family_peaks']
    if max(peaks[r] for r in plan['rungs'])+2048*plan['blocks_per_rung']>=7*1024**3:
        raise ValueError('Final schedule/replay memory not admitted')
    deadline=time()+plan['direct_quote_seconds']+(0 if plan['skip_secondary'] else plan['secondary_quote_seconds'])
    for rung in plan['rungs']:
        guarded('final-direct-'+rung,[PYTHON,'-m','scripts.evaluate_hu100_direct','--specs',OUT/'specs'/f'{rung}.json',
            '--blocks',plan['blocks_per_rung'],'--root',run.FINAL_ROOT,'--rung',rung,'--out',OUT/'final-direct'/rung,
            '--source',source],deadline-time())
    if not plan['skip_secondary']:
        guarded('final-secondary',[PYTHON,'-m',MODULE,'secondary'],deadline-time())
    put(OUT/'completed.json',{'status':'complete','source':source,'plan':plan,
        'primary':run.read(OUT/'final-direct/terminal-vs-1b/replay.json'),
        'rungs':{r:run.read(OUT/'final-direct'/r/'replay.json') for r in plan['rungs']},
        'all_final_hands_replayed_and_reproduced':True})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('pilot','train','calibrate','final'))
    p.add_argument('--a-seconds',type=float);a=p.parse_args()
    if a.command=='calibrate':calibrate(a.a_seconds)
    else:globals()[a.command]()

if __name__=='__main__':main()
