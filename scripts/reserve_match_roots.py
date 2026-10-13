"""Reserve fresh roots against current main and every open PR's tracked plans."""
import json
import subprocess
from pathlib import Path
from time import time
from scripts.evaluate_hu100_direct import put


def reserve(roots, pr, destination):
    repo = 'dberweger2017/deepcfr-texas-no-limit-holdem-6-players'
    subprocess.run(['git','fetch','origin','main'],check=True)
    opened = json.loads(subprocess.check_output(['gh','pr','list','--repo',repo,'--state','open','--limit','100',
                      '--json','number,headRefName,headRefOid,url'],text=True))
    refs = {'main':subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip()}
    for item in opened:
        if item['number']==pr: continue
        subprocess.run(['git','fetch','origin',item['headRefOid']],check=True)
        refs['pr'+str(item['number'])] = item['headRefOid']
    matches = {}
    for label,rev in refs.items():
        for root in roots:
            result = subprocess.run(['git','grep','-n','-E',f'(^|[^0-9]){root}([^0-9]|$)',rev,'--',
                                     'configs','docs','scripts','planning','ROADMAP.md','RESULTS_INDEX.md'],capture_output=True,text=True)
            if result.returncode not in (0,1): raise RuntimeError('Root inventory failed')
            if result.returncode==0: matches[label+'/'+str(root)] = result.stdout
    record = {'at':time(),'roots':roots,'pr':pr,'source_refs':refs,'open_prs':opened,'matches':matches,
              'fresh':not matches,'scope':'exact numerical root scan of all tracked configs, reports, plans and scripts; main and current open heads'}
    put(destination,record)
    if matches: raise ValueError('Root collision; stop before any outcome')
    return record
