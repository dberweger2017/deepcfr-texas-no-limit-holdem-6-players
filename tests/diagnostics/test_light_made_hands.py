"""Generated record checks: no policies, deals, or campaign artifacts required."""
import gzip
import json

import pytest

from scripts.report_hu20_light_made_hands import analyze, markdown, verified_inputs
from src.arena.schedule import digest, stream_seed
from src.diagnostics.saved_hu20 import file_hash
from src.diagnostics.stackoff_tails import hand_tails


def inputs(root, seeds=(1,2,3), missing=False, duplicate=False, wrong_tail=False):
    plan={'parents':[{'seed':s} for s in seeds], 'light_totals':[100],
          'light_panels':[{'name':'selective_stackoff','root':42,'blocks':1}]}
    p=root/'configs/blueprint/plan.json';p.parent.mkdir(parents=True);p.write_text(json.dumps(plan))
    manifest={'remote_plan':'/campaign/configs/blueprint/plan.json','plan_file_sha256':file_hash(p),
              'canonical_plan_sha256':digest(plan),'scientific_source':'fixture','inputs':[]}
    for seed in seeds:
        if missing and seed==3:continue
        spec={'name':f'B-{seed}-100','seed':seed,'milestone':100,'players':2,
              'format':'holdem-hu20-native-reopening-blueprint-v1'}
        folder=root/f'results/light-{seed}';folder.mkdir(parents=True)
        rows=[]
        for rotation in (0,1):
            # A prior street raise is excluded from the same-street re-raise group.
            actions=[{'index':0,'logical_player':1,'street':'preflop','kind':'raise','raise_to':200,
                      'observation':{'hole_cards':['Ks','Kh']}}]
            if rotation==0:actions.append({'index':1,'logical_player':1,'street':'flop','kind':'raise','raise_to':200,'observation':{'hole_cards':['Ks','Kh']}})
            actions.extend([{'index':len(actions),'logical_player':0,'street':'flop','kind':'raise','raise_to':1000,
                 'observation':{'hole_cards':['As','2c'],'board':['Ac','Kd','7h'],'street':'flop',
                     'trained':True,'large_raise_opportunity':True,'jam_opportunity':False,
                     'menu':[{'kind':'raise','raise_to':1000,'rival_call_amount':800,'jam':False}]}},
                {'index':len(actions)+1,'logical_player':1,'street':'flop','kind':'call','raise_to':None,
                 'observation':{'hole_cards':['Ks','Kh'],'call_amount':800}}])
            chips=-2000 if rotation==0 else 2000
            row={'status':'complete','policy':spec['name'],'panel':'selective_stackoff','seed':seed,
                 'milestone':100,'campaign_stage':'light','native_replay_verified':True,
                 'block':0,'rotation':rotation,'button':0,'root_seed':42,
                 'deal_seed':stream_seed(42,'test','deal',2,0),'target_chips':chips,
                 'net_chips_by_seat':[chips,-chips] if rotation==0 else [-chips,chips], 'actions':actions}
            row['tails']=hand_tails(row)
            if wrong_tail:row['tails']['counts']['full_stack_losses']=5
            rows.append(row)
        if duplicate:rows[1]=rows[0]
        with gzip.open(folder/'hands.jsonl.gz','wt') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        (folder/'result.json').write_text(json.dumps({'status':'complete','stage':'light','spec':spec,'hands':2}))
        manifest['inputs'].append({'task':f'light-{seed}-100','remote_dir':f'/campaign/results/light-{seed}',
             'hands_bytes':(folder/'hands.jsonl.gz').stat().st_size,'hands_sha256':file_hash(folder/'hands.jsonl.gz'),
             'result_sha256':file_hash(folder/'result.json'),'model':spec})
    (root/'transfer-manifest.json').write_text(json.dumps(manifest))
    return manifest


def test_counts_and_both_positions(tmp_path):
    inputs(tmp_path);result,events=analyze(tmp_path)
    assert result['selective_hands']==6 and len(events)==6 and not result['pending_tasks']
    aggregate=result['three_lineage'][0]
    assert aggregate['whole_hand_chips']==0
    assert aggregate['counts']['full_stack_wins']==aggregate['counts']['full_stack_losses']==3
    after,without=(s['counts'] for s in aggregate['situations'])
    assert after['continued_behind']==after['category_pair']==after['selected_full_stack_losses']==3
    assert without['selected_full_stack_wins']==3
    assert all(e['board']==['Ac','Kd','7h'] for e in events)
    assert '3/3' in markdown(result)


@pytest.mark.parametrize('option,message',[('duplicate','duplicate hand'),('wrong_tail','Tail arithmetic')])
def test_invalid_records_rejected(tmp_path,option,message):
    inputs(tmp_path,**{option:True})
    with pytest.raises(ValueError,match=message):analyze(tmp_path)


def test_missing_lineage_is_explicit_not_aggregated(tmp_path):
    inputs(tmp_path,missing=True);result,_=analyze(tmp_path)
    assert result['pending_tasks']==[[3,100]] and result['three_lineage']==[]


def test_bytes_and_identity_fail_closed(tmp_path):
    manifest=inputs(tmp_path)
    p=tmp_path/'results/light-1/result.json';p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='hash'):verified_inputs(tmp_path)
    manifest['inputs'][0]['result_sha256']=file_hash(p)
    manifest['inputs'][0]['model']['seed']=9
    (tmp_path/'transfer-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='identity'):verified_inputs(tmp_path)
