"""Guard one heavy evaluator; preserve the original deadline across freeze stages."""
import argparse,json,re,shutil,subprocess,sys
from pathlib import Path
from time import time,sleep,monotonic
from scripts.evaluate_hu20 import system,write_json
from scripts.tp20_common import append,seal
from scripts.run_tp20_campaign import swap_bytes
from src.arena.schedule import digest
from src.blueprint.windowed import _hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=('preflight','main'),required=True);a=p.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Source must be clean and committed')
    plan=json.loads(a.plan.read_text()); revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if a.stage=='preflight':
        if a.root.exists():raise FileExistsError(a.root)
        a.root.mkdir(parents=True);now=time()
        record={'started':now,'deadline':now+36000,'status':'preflight','preflight_revision':revision,
                'swap_baseline':system(['sysctl','vm.swapusage']),'attempts':[]}
    else:
        record=json.loads((a.root/'campaign.json').read_text())
        if record['status']!='awaiting_freeze' or plan.get('preflight_sha256')!=_hash(a.root/'preflight/result.json'):
            raise ValueError('Missing frozen preflight provenance')
        if plan['lbr_blocks'] not in (64,128,256,512) or not plan['lbr_targets'] or not 1<=plan['chance_samples']<=128:
            raise ValueError('Unfrozen resource decision')
        record.update(status='running',frozen_plan_digest=digest(plan),source_revision=revision)
    write_json(a.root/'campaign.json',record)
    phase='preflight' if a.stage=='preflight' else 'confirmation'
    deadline=record['deadline']-plan['limits']['report_reserve_seconds']
    cmd=[sys.executable,'-m','scripts.evaluate_robustness','--plan',str(a.plan),'--out',str(a.root/phase),'--phase',phase,'--deadline',str(deadline)]
    attempt={'command':cmd,'started':time(),'revision':revision,'phase':phase,'status':'running'};record['attempts'].append(attempt)
    with (a.root/f'{phase}.log').open('w') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);attempt['pid']=child.pid
        write_json(a.root/'campaign.json',record);next_system=0;failure=None;peak=0
        while child.poll() is None:
            sample=subprocess.run(['ps','-o','rss=','-p',str(child.pid)],capture_output=True,text=True)
            current=int(sample.stdout.strip() or 0)*1024;peak=max(peak,current)
            if current>=10.5*1024**3:failure='RSS ceiling'
            if time()>=deadline:failure='Absolute execution deadline'
            if shutil.disk_usage(a.root).free<8*1024**3:failure='Free disk ceiling'
            if monotonic()>=next_system:
                swap=system(['sysctl','vm.swapusage']);before=swap_bytes(record['swap_baseline']);after=swap_bytes(swap)
                if before is not None and after is not None and after-before>.5*1024**3:failure='Swap growth ceiling'
                append(a.root/'resources.jsonl',{'unix_seconds':time(),'rss_bytes':current,'swap':swap,
                    'memory_pressure':system(['memory_pressure','-Q']),'free_disk_bytes':shutil.disk_usage(a.root).free,'phase':phase})
                next_system=monotonic()+30
            if failure:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                break
            sleep(1)
        attempt.update(exit_code=child.wait(),finished=time(),peak_rss_bytes=peak,failure=failure)
    attempt['status']='complete' if attempt['exit_code']==0 and not failure else 'failed'
    record['status']='awaiting_freeze' if a.stage=='preflight' and attempt['status']=='complete' else 'evaluation_complete' if attempt['status']=='complete' else 'failed'
    write_json(a.root/'campaign.json',record);seal(a.root)
    print(json.dumps(record,indent=2));return attempt['status']!='complete'

if __name__=='__main__':raise SystemExit(main())
