"""Run the six frozen arms, comparisons and audit sequentially under the original deadline."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import sleep, time

from scripts.hu20_reopening_common import write_cases
from scripts.run_tp20_campaign import swap_bytes
from scripts.tp20_common import append, seal
from scripts.train_hu20 import system, write_json
from src.arena.schedule import digest
from src.blueprint.windowed import _hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Main source must be clean and committed')
    plan=json.loads(a.plan.read_text());root=a.root;record=json.loads((root/'campaign.json').read_text())
    if record['status']!='awaiting_resource_freeze' or plan['preflight_plan_digest']!=record['plan_digest']:
        raise ValueError('Missing completed preflight gate')
    if plan['training_nodes'] not in (5000000,10000000,20000000) or plan['lbr_blocks'] not in (512,1024,2048):
        raise ValueError('Unfrozen common work/count')
    if plan['chance_samples']!=4 or plan['noninferiority_margin_bb100']!=10 or plan['primary_interval_level']!=.975:
        raise ValueError('Changed quality/attacker contract')
    if _hash(root/'resource-summary.json')!=plan['resource_summary_sha256']:raise ValueError('Resource decision provenance')
    # Preserve the preflight supervisor's sealed record before updating it.
    shutil.copyfile(root/'campaign.json',root/'preflight-campaign-record.json')
    shutil.copyfile(root/'checksums.json',root/'preflight-checksums.json')
    shutil.copyfile(root/'resources.jsonl',root/'preflight-resources.jsonl')
    fixture=Path(plan['independent_path']);count=write_cases(plan,fixture)
    write_json(root/'independent-manifest.json',{'observations':count,'sha256':_hash(fixture),
        'root_seed':plan['independent_root'],'blocks':plan['independent_blocks'],
        'paths':['cap2-uniform','native-uniform','passive','later-repeated-minraise'],
        'selection':'all decisions; no trained model/outcome used'})
    write_json(root/'frozen-plan.json',plan)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    record.update(status='training',main_revision=revision,frozen_plan_digest=digest(plan),resource_choice=plan['resource_decision']['choice'])
    write_json(root/'campaign.json',record)
    deadline=record['deadline'];python=sys.executable
    def run(name,cmd,phase_deadline):
        attempt={'phase':name,'command':cmd,'started':time(),'status':'running','revision':revision}
        record['attempts'].append(attempt);write_json(root/'campaign.json',record)
        with (root/f'{name}.log').open('w') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            attempt['pid']=child.pid;write_json(root/'campaign.json',record)
            next_system=0;peak=0;reason=None
            while child.poll() is None:
                data=subprocess.run(['ps','-o','rss=','-p',str(child.pid)],capture_output=True,text=True)
                rss=int(data.stdout.strip() or 0)*1024;peak=max(peak,rss)
                if time()>=phase_deadline:reason='Original absolute deadline/reserved phase time'
                if rss>=10.5*1024**3:reason='RSS guard'
                if shutil.disk_usage(root).free<8*1024**3:reason='Free disk guard'
                if time()>=next_system:
                    swap=system(['sysctl','vm.swapusage']);before=swap_bytes(record['swap_baseline']);after=swap_bytes(swap)
                    if before is not None and after is not None and after-before>.5*1024**3:reason='Swap growth guard'
                    append(root/'resources.jsonl',{'unix_seconds':time(),'phase':name,'rss_bytes':rss,'swap':swap,
                        'memory_pressure':system(['memory_pressure','-Q']),'free_disk_bytes':shutil.disk_usage(root).free})
                    next_system=time()+30
                if reason:
                    child.terminate()
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    break
                sleep(1)
            attempt.update(exit_code=child.wait(),guard_failure=reason,peak_sampled_rss_bytes=peak,finished=time())
        attempt['status']='complete' if attempt['exit_code']==0 and not reason else 'failed'
        write_json(root/'campaign.json',record);return attempt['status']=='complete'
    ok=True
    for seed in plan['training_seeds']:
        for arm in ('A','B'):
            name=f'train-{arm}-{seed}';end=deadline-plan['evaluation_report_reserve_seconds']
            ok=run(name,[python,'-m','scripts.train_hu20_reopening','--plan',str(a.plan),'--seed',str(seed),
                '--arm',arm,'--out',str(root/'training'/f'{arm}-{seed}'),'--deadline',str(end)],end)
            if not ok:break
        if not ok:break
    if ok:
        record['status']='evaluation';write_json(root/'campaign.json',record);end=deadline-2100
        ok=run('evaluation',[python,'-m','scripts.evaluate_hu20_reopening','--plan',str(a.plan),'--root',str(root),
            '--out',str(root/'evaluation'),'--deadline',str(end)],end)
    record['status']='audit' if ok else 'failed';write_json(root/'campaign.json',record)
    audited=run('audit',[python,'-m','scripts.report_hu20_reopening','--root',str(root),'--out',str(root/'audit')],deadline-120)
    if ok and audited:
        audited=run('independent-summary',[python,'-m','scripts.check_hu20_reopening_summary','--root',str(root)],deadline-90)
    if ok and audited:
        seed=plan['training_seeds'][0];result=json.loads((root/'training'/f'B-{seed}'/'result.json').read_text())
        model=root/'training'/f'B-{seed}'/'current-3.json.gz';sha=result['milestones'][3]['policy_sha256']
        code=("from pathlib import Path; from scripts.play_hu20_native import play; from scripts.play_hu20 import replay_history; "
              "from scripts.train_hu20 import write_json; visible=[]; "
              "choose=lambda prompt: next(x.split('.')[0].strip() for x in reversed(visible) if x.startswith('  ') and ('check' in x or 'call' in x)); "
              f"r=play(Path({str(model)!r}),{sha!r},Path({str(root/'candidate-human-smoke.jsonl')!r}),seed={plan['demo_root']},max_hands=20,input_fn=choose,output=visible.append); "
              f"r['replayed']=replay_history(Path({str(root/'candidate-human-smoke.jsonl')!r})); "
              f"write_json(Path({str(root/'candidate-human-smoke-result.json')!r}),r); "
              f"Path({str(root/'candidate-human-transcript.txt')!r}).write_text('\\n'.join(visible)+'\\n')")
        ok=run('human-smoke',[python,'-c',code],deadline-60)
    record.update(status='complete' if ok and audited else 'incomplete',finished=time())
    write_json(root/'campaign.json',record);seal(root)
    # All child logs and supervisor metadata are now closed; preserve every file.
    inventory={'status':record['status'],'started':record['started'],'finished':time(),'deadline':deadline,
        'root':str(root.resolve()),'files':{str(f.resolve()):{'bytes':f.stat().st_size,'sha256':_hash(f)} for f in sorted(root.rglob('*')) if f.is_file()}}
    write_json(root.with_name(root.name+'-manifest.json'),inventory)
    print(json.dumps({'status':record['status'],'files':len(inventory['files']),'deadline':deadline,'finished':time()}));return record['status']!='complete'

if __name__=='__main__':raise SystemExit(main())
