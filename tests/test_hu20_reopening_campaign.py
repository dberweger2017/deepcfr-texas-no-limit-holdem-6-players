"""Frozen A/B schedules and inference exercise the production diagnostic pipeline."""

import json
from pathlib import Path
from time import time

import pytest

from scripts.evaluate_hu20_reopening import run as evaluate
from scripts.freeze_hu20_reopening import choose
from scripts.hu20_reopening_common import write_cases
from scripts.report_hu20_reopening import paired, report
from scripts.train_hu20_reopening import run as train
from src.blueprint.windowed import _hash


def test_seed_contrasts_are_averaged_inside_blocks_and_units_are_correct():
    groups={}
    for s in (1,2,3):
        groups[(f'A-{s}-3','Pressure-native')]={0:10*s,1:20*s,2:30*s}
        groups[(f'B-{s}-3','Pressure-native')]={0:10*s+10,1:20*s+20,2:30*s+30}
    r=paired(groups,[1,2,3],'Pressure-native')['B_minus_A']
    assert r['blocks']==3
    assert r['bb100']==20 and r['bb_hand']==.2 and r['buyins20_per100']==1
    assert r['level']==.975 and r['interval'][0]<20<r['interval'][1]
    del groups[('B-3-3','Pressure-native')]
    assert paired(groups,[1,2,3],'Pressure-native')['status']=='unavailable'


def test_budget_is_chosen_only_from_completed_resource_measurements():
    row={'status':'complete','complete_outer_cost':{'sum_seconds':10},'completed_nodes':1000000,
         'entries':1000,'peak_rss_bytes':10000000,'timing_panels':[{'rule':['lbr'],'seconds':1,'hands':4}]}
    decision=choose({'runs':[row]*6},35000)
    assert decision['choice']['nodes']==20000000
    assert decision['choice']['lbr_blocks']==2048
    with pytest.raises(ValueError):choose({'runs':[row]*5},35000)
    assert choose({'runs':[row]*6},1)['choice'] is None


def test_actual_fresh_training_dual_menus_and_native_audit(tmp_path):
    plan=json.loads(Path('configs/blueprint/hu20-native-reopening-m4.json').read_text())
    plan.update(training_nodes=1000,training_nodes_options=[1000],training_seeds=[31,32,33],
        independent_blocks=1,independent_path=str(tmp_path/'independent.jsonl.gz'),
        cheap_blocks=2,lbr_blocks=2,secondary_panel_blocks=2,chance_samples=1,
        reference_policies=[],prior_roots=[])
    root=tmp_path;write_cases(plan,Path(plan['independent_path']))
    for seed in plan['training_seeds']:
        for arm in ('A','B'):
            r=train(plan,seed,arm,root/'training'/f'{arm}-{seed}',time()+120)
            assert r['status']=='complete' and len(r['milestones'])==4
            assert r['initial_entries']==0 and r['initial_iteration']==0
            assert all(m['independent']['observations']>0 for m in r['milestones'])
    ev=evaluate(plan,root,root/'evaluation',time()+120)
    assert ev['status']=='complete' and ev['hands']==648
    (root/'frozen-plan.json').write_text(json.dumps(plan))
    (root/'campaign.json').write_text(json.dumps({'started':time(),'deadline':time()+120,'swap_baseline':None}))
    (root/'preflight').mkdir()
    # Simulated supervisor measurement; actual native hands and checkpoints above.
    (root/'resources.jsonl').write_text(json.dumps({'rss_bytes':0,'free_disk_bytes':10*1024**3,'swap':None})+'\n')
    audited=report(root,root/'audit')
    assert audited['status']=='complete' and audited['native_replayed_hands']==648
    assert audited['primary']['Pressure-native']['B_minus_A']['blocks']==2
    from scripts.check_hu20_reopening_summary import check
    independent=check(root/'evaluation/hands.jsonl.gz',audited,plan['training_seeds'])
    assert independent['hands_sha256']==_hash(root/'evaluation/hands.jsonl.gz')
