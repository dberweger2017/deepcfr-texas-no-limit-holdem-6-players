"""Turn-only fallback and public native replay controls."""
from src.diagnostics.flop_check import compile_tree, fixture_root
from src.game.types import Street
from src.game.observation import replay
from src.blueprint.abstraction import choices


def test_turn_cap_two_retains_native_lookup_and_legal_jams():
    request,histories=compile_tree(fixture_root('3-bet',street=Street.TURN),raise_cap=2)
    assert request['initial_street']=='turn' and len(request['board'])==4
    removed=0
    for node,history in zip(request['nodes'],histories,strict=True):
        if node['terminal']:continue
        view=replay(history,request['seat_map'][node['player']],())
        native=choices(view,raise_cap=None,free_fold=False)
        assert node['native_names']==[c.name for c in native]
        for c in native:
            jam=c.action.raise_to is not None and c.action.raise_to-view.players[view.seat].street_bet==view.players[view.seat].stack
            if jam:assert c.name in node['names']
        removed+=len(node['native_names'])-len(node['names'])
    assert removed>0


def test_turn_root_identity_survives_real_native_replay_and_rejects_tampering():
    from src.diagnostics.turn_check import root_record,replay_root
    import pytest
    for kind in ('limped','min-raised','pot-raised','3-bet'):
        for button in (0,1):
            record=root_record(fixture_root(kind,street=Street.TURN,button=button))
            assert root_record(replay_root(record))==record
            with pytest.raises(ValueError,match='replay mismatch'):
                replay_root(dict(record,spot='tampered'))


def test_turn_job_order_is_complete_balanced_and_reproducible():
    from src.diagnostics.turn_campaign import jobs
    from collections import Counter
    corpus={g:{'roots':[{'spot':g+str(i),'kind':('limped','min-raised','pot-raised','3-bet')[i%4],
                            'button':(i//4)%2} for i in range(24)]} for g in ('A','B')}
    schedule=jobs(corpus,[{}]*6,901)
    assert schedule==jobs(corpus,[{}]*6,901)
    assert len(schedule)==288 and len({r['job'] for r in schedule})==288
    for group in ('A','B'):
        assert Counter(r['policy_index'] for r in schedule if r['set']==group)=={i:24 for i in range(6)}
    assert Counter(r['policy_index'] for r in schedule[:48])=={i:8 for i in range(6)}


def test_turn_monitor_bridge_handles_partial_records_restart_and_policy_labels(tmp_path):
    import json
    from scripts.monitor_turn_check import Bridge
    from src.diagnostics.flop_check import atomic_json
    base=tmp_path/'spots/B-root-0';leaf=base/'attempt-01/solver';leaf.mkdir(parents=True)
    atomic_json(base/'job.json',{'job':{'root':{'spot':'root'},'set':'B'},'policy':{'name':'seed-average'}})
    source=leaf/'progress.jsonl';text=json.dumps({'event':'progress','iteration':25,'exploitability_pct_pot':.2})
    source.write_text(text[:12]);Bridge(tmp_path).poll()
    assert not (tmp_path/'progress.jsonl').exists()
    source.write_text(text+'\n');Bridge(tmp_path).poll()
    rows=(tmp_path/'progress.jsonl').read_text().splitlines()
    assert len(rows)==1 and json.loads(rows[0])['series']=='seed-average/root'
    Bridge(tmp_path).poll()
    assert (tmp_path/'progress.jsonl').read_text().splitlines()==rows


def test_turn_ratios_bootstrap_paired_roots_and_use_ratio_of_means():
    from src.diagnostics.turn_report import intervals,common_rows,COLUMNS
    import pytest
    policies=[{'name':str(p)} for p in range(6)];rows=[]
    for i,bp in enumerate((1.,100.,1.,100.)):
        for policy in policies:
            metrics={name:bp*.1 for name in COLUMNS}
            metrics.update(e_bp=bp,e_v1proj=.9 if bp==1 else 10.,alias_difference_bb=.1*bp)
            rows.append({'spot':str(i),'policy':policy['name'],'reach_weight':1.,'stratum':[i//2,0],
                         'decision_eligible':True,**metrics})
    summary=intervals(rows,resamples=100)
    assert summary['independent_roots']==4
    assert summary['ratios']['R']['point']==pytest.approx(5.45/50.5)
    assert summary['ratios']['alias_cost']['ci95']==pytest.approx([.1,.1])
    valid,excluded=common_rows(rows[1:],policies)
    assert len(valid)==18 and len(excluded)==1 and excluded[0]['spot']=='0'


def test_selected_fold_coverage_counts_excluded_frozen_roots():
    from scripts.report_turn_check import selected_folds
    rows=[{'spot':'eligible','stratum':['limped',0],'inclusion_probability':.5,
           'selected_lbr_nodes':[{'source_sha256':'source','source_line':1,'action_index':4,
                                  'fold_bp':.5,'fold_eq':.5}]} for _ in range(6)]
    result=selected_folds(rows,expected_total=4.)
    assert result['coverage']==.5 and result['covered_weight']==2.
    assert not result['H0']
