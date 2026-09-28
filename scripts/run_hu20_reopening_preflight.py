"""Guard the sequential outcome-free cap A/B preflight and retain the first obstruction."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import monotonic, sleep, time

from scripts.run_tp20_campaign import swap_bytes
from scripts.tp20_common import append, seal
from scripts.train_hu20 import system, write_json
from src.arena.schedule import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Preflight source must be clean and committed')
    if a.root.exists():raise FileExistsError(a.root)
    a.root.mkdir(parents=True)
    plan=json.loads(a.plan.read_text());start=time()
    record={'status':'preflight','started':start,'deadline':start+plan['limits']['max_campaign_seconds'],
            'plan_digest':digest(plan),'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'swap_baseline':system(['sysctl','vm.swapusage']),'attempts':[]}
    write_json(a.root/'campaign.json',record);write_json(a.root/'preflight-plan.json',plan)
    deadline=record['deadline']-plan['limits']['minimum_evaluation_report_reserve_seconds']
    failed=False
    for seed in plan['preflight_seeds']:
        for arm in ('A','B'):
            out=a.root/'preflight'/f'{arm}-{seed}'
            cmd=[sys.executable,'-m','scripts.preflight_hu20_reopening','--plan',str(a.plan),
                 '--out',str(out),'--seed',str(seed),'--arm',arm,'--deadline',str(deadline)]
            attempt={'arm':arm,'seed':seed,'command':cmd,'started':time(),'status':'running'}
            record['attempts'].append(attempt);write_json(a.root/'campaign.json',record)
            with (a.root/f'{arm}-{seed}.log').open('w') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                attempt['pid']=child.pid;write_json(a.root/'campaign.json',record)
                next_system=0;peak=0;failure=None
                while child.poll() is None:
                    measured=subprocess.run(['ps','-o','rss=','-p',str(child.pid)],capture_output=True,text=True)
                    current=int(measured.stdout.strip() or 0)*1024;peak=max(peak,current)
                    if current>=plan['limits']['max_rss_gib']*1024**3:failure='RSS ceiling'
                    if time()>=deadline:failure='Original absolute deadline minus reserved evaluation/report time'
                    if shutil.disk_usage(a.root).free<plan['limits']['min_free_gib']*1024**3:failure='Free disk guard'
                    if monotonic()>=next_system:
                        swap=system(['sysctl','vm.swapusage'])
                        before,after=swap_bytes(record['swap_baseline']),swap_bytes(swap)
                        if before is not None and after is not None and after-before>plan['limits']['max_swap_growth_gib']*1024**3:
                            failure='Swap growth guard'
                        append(a.root/'resources.jsonl',{'unix_seconds':time(),'arm':arm,'seed':seed,
                            'rss_bytes':current,'swap':swap,'memory_pressure':system(['memory_pressure','-Q']),
                            'free_disk_bytes':shutil.disk_usage(a.root).free})
                        next_system=monotonic()+30
                    if failure:
                        child.terminate()
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        break
                    sleep(1)
                attempt.update(exit_code=child.wait(),finished=time(),peak_sampled_rss_bytes=peak,guard_failure=failure)
            attempt['status']='complete' if attempt['exit_code']==0 and not failure else 'failed'
            write_json(a.root/'campaign.json',record)
            if attempt['status']!='complete':failed=True;break
        if failed:break
    record['status']='infeasible' if failed else 'awaiting_resource_freeze'
    record['preflight_finished']=time()
    write_json(a.root/'campaign.json',record);seal(a.root)
    print(json.dumps(record,indent=2));return failed

if __name__=='__main__':raise SystemExit(main())
