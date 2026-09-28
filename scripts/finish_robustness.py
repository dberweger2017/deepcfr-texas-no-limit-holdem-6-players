"""Run audits/probes/demo sequentially under the campaign's original deadline."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
from time import time,sleep
from scripts.evaluate_hu20 import system,write_json
from scripts.tp20_common import append,seal
from scripts.run_tp20_campaign import swap_bytes
from src.blueprint.windowed import _hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;record=json.loads((root/'campaign.json').read_text());deadline=record['deadline']
    if record['status'] not in ('evaluation_complete','failed'):raise ValueError('Campaign is still active')
    # Do not write inside the sealed evaluation inventory until its reporter has
    # verified every entry. Audit records live in a sibling directory first.
    audit=root.with_name(root.name+'-audit');audit.mkdir(exist_ok=False)
    plan=root/'confirmation/plan.json';attempts=[]
    def run(name,cmd):
        attempt={'name':name,'command':cmd,'started':time(),'status':'running'};attempts.append(attempt);write_json(audit/'attempts.json',attempts)
        with (audit/f'{name}.log').open('w') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);reason=None;next_system=0;peak=0
            while child.poll() is None:
                data=subprocess.run(['ps','-o','rss=','-p',str(child.pid)],capture_output=True,text=True)
                rss=int(data.stdout.strip() or 0)*1024;peak=max(peak,rss)
                if time()>=deadline:reason='Original absolute campaign deadline'
                if rss>=10.5*1024**3:reason='Audit RSS guard'
                if shutil.disk_usage(root).free<8*1024**3:reason='Audit disk guard'
                if time()>=next_system:
                    swap=system(['sysctl','vm.swapusage']);baseline=swap_bytes(record['swap_baseline']);used=swap_bytes(swap)
                    if baseline is not None and used is not None and used-baseline>.5*1024**3:reason='Audit swap guard'
                    append(audit/'resources.jsonl',{'time':time(),'rss_bytes':rss,'swap':swap,'free_disk_bytes':shutil.disk_usage(root).free,'phase':name})
                    next_system=time()+30
                if reason:
                    child.terminate()
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    break
                sleep(1)
            attempt.update(exit_code=child.wait(),reason=reason,peak_rss_bytes=peak,finished=time())
        attempt['status']='complete' if not reason and attempt['exit_code']==0 else 'failed';write_json(audit/'attempts.json',attempts)
        return attempt['status']=='complete'
    python=sys.executable
    ok=run('report',[python,'-m','scripts.report_robustness','--root',str(root),'--out',str(audit/'report')])
    if ok:
        ok=run('river-probes',[python,'-m','scripts.probe_robustness_rivers','--plan',str(plan),'--hands',str(root/'confirmation/hands.jsonl.gz'),'--out',str(audit/'river-probes.json')])
    if ok:
        ok=run('diagnostic-demo',[python,'-m','scripts.play_robustness','--plan',str(plan),'--policy','2p-2026092801-20M','--rule','pressure','--contract','native','--history',str(audit/'diagnostic-demo.json')])
    if ok:
        ok=run('demo-replay',[python,'-m','scripts.play_robustness','--replay',str(audit/'diagnostic-demo.json')])
    seal(audit)
    manifest={'campaign_root':str(root.resolve()),'audit_root':str(audit.resolve()),'deadline':deadline,'finished':time(),
        'status':'complete' if ok else 'incomplete','files':{}}
    # Both directories are now quiet; inventory all artifacts including their
    # phase checksums. This inventory itself is written outside those roots.
    for folder in (root,audit):
        for file in sorted(folder.rglob('*')):
            if file.is_file():manifest['files'][str(file.resolve())]={'bytes':file.stat().st_size,'sha256':_hash(file)}
    write_json(root.with_name(root.name+'-manifest.json'),manifest)
    print(json.dumps({'status':manifest['status'],'artifact_files':len(manifest['files']),'finished':time()}));return not ok

if __name__=='__main__':raise SystemExit(main())
