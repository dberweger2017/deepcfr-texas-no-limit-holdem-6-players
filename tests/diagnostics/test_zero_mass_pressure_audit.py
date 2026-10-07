"""Exercise the actual arena/replayer boundary, including a corrupted settlement."""
import gzip
import json

import pytest

from scripts.evaluate_hu20_cfr_average import play
from scripts.report_audit_hu20_zero_mass_pressure import analyze, ARMS
from src.arena.schedule import digest
from src.blueprint.abstraction import choices, information_key, HU20_UNCAPPED_SCHEMA
from src.blueprint.average import AveragePolicy

class FixtureAverage(AveragePolicy):
    def __init__(self,arm):
        self.arm=arm;self.zero_mass=set();self.abstraction=HU20_UNCAPPED_SCHEMA;self.raise_cap=None
    def distribution(self,view):
        menu=choices(view,raise_cap=None,free_fold=False)
        self.zero_mass.add(information_key(view,menu,schema=self.abstraction))
        p=[1/len(menu)]*len(menu)
        if self.arm.endswith('prime') and len(menu)>1:
            p=[0.8]+[0.2/(len(menu)-1)]*(len(menu)-1)
        return menu,p,True

@pytest.fixture
def evidence(tmp_path):
    from pathlib import Path
    panel=next(p for p in json.loads(Path('configs/diagnostics/b500-cfr-average-comparison.json').read_text())['panels'] if p['name']=='native-pressure')
    panel=dict(panel,blocks=32)
    models=[{'arm':arm,'name':f'{arm}-{seed}','seed':seed,'strategy':'average'} for arm in ARMS for seed in (1,2,3)]
    plan={'root':78391,'models':models,'panels':[panel],'expected_hands':768}
    for spec in models:
        rows=[];source=FixtureAverage(spec['arm'])
        for b in range(32):
            for r in (0,1):
                row=play(source,spec,panel,plan['root'],b,r);row['arm']=spec['arm'];rows.append(row)
        with gzip.open(tmp_path/(spec['name']+'.hands.jsonl.gz'),'xt') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        with gzip.open(tmp_path/(spec['name']+'.result.json.gz'),'xt') as f:json.dump({'status':'complete','hands':64,'plan_sha256':digest(plan)},f)
    analyze(plan,tmp_path,False)
    return plan,tmp_path

def test_pressure_audit_accepts_labeled_nonuniform_zero_mass_and_recomputes(evidence):
    plan,path=evidence
    result=analyze(plan,path,True)
    assert result['status']=='verified' and result['hands']==768 and result['decisions_checked']>0
    assert result['independent_arithmetic_matches'] and set(result['contrasts'])=={'Tprime-T','Oprime-O'}

def test_pressure_audit_rejects_corrupted_settlement(evidence):
    plan,path=evidence;p=path/'Tprime-1.hands.jsonl.gz'
    with gzip.open(p,'rt') as f:rows=[json.loads(line) for line in f]
    rows[0]['net_chips_by_seat'][0]+=1
    with gzip.open(p,'wt') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    with pytest.raises(AssertionError):analyze(plan,path,True)
