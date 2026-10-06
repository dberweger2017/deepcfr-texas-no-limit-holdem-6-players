"""Balanced schedules, worker approval and cumulative resource admission."""

from dataclasses import asdict
import gzip
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import evaluate_hu20_turn_search as campaign
from scripts.calibrate_hu20_turn_search import finalists
from scripts.freeze_hu20_turn_search import arena_plan
from scripts.hu20_search_runtime import RunBudget, PaidWorkerBudget, validate_admission
from src.blueprint.abstraction import choices, HU20_UNCAPPED_SCHEMA
from src.blueprint.hu20_turn_search import TurnSearchConfig


def admission():
    return {"145_main_complete":True,"145_final_report_pushed":True,"145_processes_empty":True,
        "followup_claim":"none","checked_at":1000,"reclaimable_bytes":20*1024**3,
        "rss_limit_bytes":5*1024**3,"145_report_sha256":"report","ownership_evidence":"verified"}


def test_resume_reads_every_complete_hand_of_a_truncated_stream(tmp_path):
    path = tmp_path / "hands.jsonl.gz"
    with gzip.open(path, "wt") as stream:
        for block in range(40):
            stream.write(json.dumps({"panel": "p", "block": block, "rotation": 0, "pad": "x" * 500}) + "\n")
            stream.flush()
    whole = path.read_bytes()
    # A worker killed mid-write leaves an unterminated stream whose tail may be cut anywhere.
    for cut in (len(whole) - 8, len(whole) - 300, len(whole) // 2):
        path.write_bytes(whole[:cut])
        rows = campaign.read_complete_rows(path)
        assert 1 <= len(rows) <= 40 and [r["block"] for r in rows] == list(range(len(rows)))
    assert campaign.read_complete_rows(tmp_path / "missing.gz") == []


@pytest.mark.parametrize("change",[{"145_main_complete":False},{"145_final_report_pushed":False},
    {"145_processes_empty":False},{"followup_claim":"pending"},{"checked_at":0},
    {"reclaimable_bytes":1024**3},{"ownership_evidence":None}])
def test_active_145_and_stale_or_insufficient_admission_refused(change):
    with pytest.raises(ValueError):validate_admission(dict(admission(),**change),now=1001)


def test_part_a_retry_clock_includes_pilot_failures_and_holds_lock(tmp_path,monkeypatch):
    import scripts.hu20_search_runtime as runtime
    monkeypatch.setattr(runtime,"time",lambda:1001)
    monkeypatch.setattr(runtime,"swap_bytes",lambda:0)
    path=tmp_path/"budget.json"
    path.write_text(json.dumps({"limit_seconds":86400,"used_seconds":21590,
        "attempts":[{"phase":"pilot","seconds":300},{"phase":"part-a","seconds":21290}]}))
    budget=RunBudget(path,tmp_path,"part-a",21600,admission())
    assert budget.deadline-budget.started==pytest.approx(10,abs=1e-6)
    with pytest.raises(BlockingIOError):RunBudget(path,tmp_path,"part-a",21600,admission())
    budget.close("failed","retained test")
    assert json.loads(path.read_text())["used_seconds"]>=21590


def test_paid_worker_requires_selected_settings_actual_pod_parity(tmp_path,monkeypatch):
    import scripts.hu20_search_runtime as runtime
    monkeypatch.setattr(runtime,"swap_bytes",lambda:0)
    approved={"owner_approved":True,"arena_plan_sha256":"plan","search_config_sha256":"config",
        "selected_settings_parity":"passed","quote_sha256":"quote","parity_sha256":"actual-pod",
        "worker_seconds":60,"rss_limit_bytes":1024**3}
    PaidWorkerBudget(tmp_path,approved,"plan","config")
    for key in ("owner_approved","selected_settings_parity","parity_sha256","search_config_sha256"):
        with pytest.raises(ValueError):PaidWorkerBudget(tmp_path,dict(approved,**{key:None}),"plan","config")


def test_runner_retains_all_paired_coordinates_and_native_replays(tmp_path,monkeypatch):
    class Uniform:
        description={"fixture":"uniform"};abstraction=HU20_UNCAPPED_SCHEMA
        def distribution(self,view):
            menu=choices(view,raise_cap=None,free_fold=False)
            return menu,(1/len(menu),)*len(menu),False
    class Budget:
        def check(self):pass
    models=[{"name":f"{seed}-{arm}","seed":seed,"strategy":arm}
            for seed in (1,2,3) for arm in ("current","average")]
    plan={"stage":"frozen-final","root":42,"models":models,"panels":[
        {"name":"uniform","rule":"uniform","contract":"native","blocks":2}],"expected_hands":24}
    monkeypatch.setattr(campaign,"load",lambda spec,inputs:Uniform())
    monkeypatch.setattr(campaign,"select_base",lambda summary:{"base":"average"})
    result=campaign.run(plan,tmp_path,tmp_path/"run",Budget())
    assert result["status"]=="complete" and result["hands"]==24
    from scripts.audit_hu20_turn_search import audit
    assert audit([tmp_path/"run"])["hands"]==24
    records=[]
    for path in (tmp_path/"run").glob("*.hands.jsonl.gz"):
        with gzip.open(path,"rt") as stream:records.extend(json.loads(s) for s in stream)
    assert all(r["native_replay_verified"] and sum(r["net_chips_by_seat"])==0 for r in records)
    assert len({(r["policy"],r["block"],r["rotation"]) for r in records})==24
    for block in (0,1):assert len({r["deal_seed"] for r in records if r["block"]==block})==1


def test_arena_freeze_preserves_original_counts_and_requires_completed_science():
    planned=json.loads(Path("configs/diagnostics/hu20-turn-search-part-a.json").read_text())
    part={"status":"complete","base_decision":{"base":"average"}}
    cal={"status":"qualified","selected":{"config":asdict(TurnSearchConfig())}}
    frozen=arena_plan(planned,part,cal)
    assert frozen["expected_hands"]==82944 and len(frozen["models"])==3
    assert [p["blocks"] for p in frozen["panels"]]==[p["blocks"] for p in planned["panels"]]
    assert frozen["root"]!=planned["root"]
    with pytest.raises(ValueError):arena_plan(planned,dict(part,status="incomplete"),cal)


def test_staged_finalists_use_speed_before_quality():
    rows=[]
    for floor in (0,.01):
        for threads in (1,2,4):
            config=asdict(TurnSearchConfig(threads=threads,opponent_likelihood_floor=floor))
            rows.extend({"configuration_id":f"{floor}/{threads}","config":config,
                         "cold_seconds":value,"residual_pct_pot":100/threads}
                        for value in (threads,threads+1))
    chosen=finalists(rows,{"opponent_likelihood_floor":[0,.01],"iterations":[25,100],
                           "staging":{"finalist_settings_per_floor":2}})
    assert len(chosen)==8 and {c.threads for c in chosen}=={1,2}


def test_river_sampling_slots_balance_strata_seats_and_lineages():
    from collections import Counter
    from scripts.validate_hu20_search_rivers import slots
    items=[{"root":{"kind":kind,"button":button,"spot":str(spot)},"policy":{"seed":seed}}
        for kind in ("limped","min-raised","pot-raised","3-bet") for button in (0,1)
        for spot in (0,1) for seed in (1,2,3)]
    sample=slots(items)
    assert len(sample)==32 and Counter(s["bot"] for s in sample)=={0:16,1:16}
    assert sorted(Counter(s["item"]["policy"]["seed"] for s in sample).values())==[10,11,11]
    assert set(Counter((s["item"]["root"]["kind"],s["item"]["root"]["button"]) for s in sample).values())=={4}


def test_quote_refuses_omitted_speculative_lbr_or_stale_offer():
    from src.arena.schedule import digest
    from scripts.quote_hu20_search_arena import quote
    calibration={"status":"qualified","selected":{"config":asdict(TurnSearchConfig())}}
    plan={"stage":"frozen-final","calibration_sha256":digest(calibration),"expected_hands":36,
          "panels":[{"name":"lbr","blocks":3}]}
    timing={"configuration_sha256":digest(calibration["selected"]["config"]),
        "includes_preparation":True,"includes_parsing":True,"includes_lbr_speculative_solves":True,
        "includes_base_and_search_arms":True,"panels":{"lbr":{"paired_blocks":1,"seconds_per_joint_block_p95":60}},
        "reserves_seconds_per_worker":{k:60 for k in ("setup_build","actual_pod_parity","replay_verification",
            "retrieval_hash_verification","shutdown")},"required_storage_gb_per_worker":20,"rss_limit_bytes_per_worker":5*1024**3}
    offer={"retrieved_at":1000,"source_url":"https://www.runpod.io/","architecture":"x86_64",
        "provider":"RunPod","gpu":False,"available_workers":2,"compute_hourly_usd":.2,
        "container_disk_hourly_usd":.01,"storage_gb":30}
    result=quote(plan,calibration,timing,offer,workers=2,now=1001)
    assert result["worker_seconds"]==630 and result["owner_approved"] is False
    with pytest.raises(ValueError):quote(plan,calibration,dict(timing,includes_lbr_speculative_solves=False),offer,workers=2,now=1001)
    with pytest.raises(ValueError):quote(plan,calibration,timing,offer,workers=2,now=10000)


def test_quote_prices_shared_cpu_workers_once_per_pod_and_refuses_contention():
    from src.arena.schedule import digest
    from scripts.quote_hu20_search_arena import quote
    calibration={"status":"qualified","selected":{"config":asdict(TurnSearchConfig())}}
    plan={"stage":"frozen-final","calibration_sha256":digest(calibration),"expected_hands":120,
          "panels":[{"name":"lbr","blocks":10}]}
    timing={"configuration_sha256":digest(calibration["selected"]["config"]),
        "includes_preparation":True,"includes_parsing":True,"includes_lbr_speculative_solves":True,
        "includes_base_and_search_arms":True,"panels":{"lbr":{"paired_blocks":8,
            "seconds_per_joint_block_p95":60,"seconds_per_joint_block_mean":30}},
        "reserves_seconds_per_worker":{k:60 for k in ("setup_build","actual_pod_parity","replay_verification",
            "retrieval_hash_verification","shutdown")},"required_storage_gb_per_worker":20,"rss_limit_bytes_per_worker":5*1024**3}
    offer={"retrieved_at":1000,"source_url":"https://mcp.getrunpod.io/","architecture":"x86_64",
        "provider":"RunPod","gpu":True,"compute_workload":"cpu","availability":"LOW",
        "compute_hourly_usd":.22,"container_disk_hourly_usd":.01,"storage_gb":60,
        "minimum_cpu_per_pod":23.8,"reserved_cpu_per_pod":4,"minimum_ram_bytes_per_pod":32*1024**3}
    result=quote(plan,calibration,timing,offer,workers=6,workers_per_pod=3,now=1001)
    assert result["worker_seconds"]==630 and result["pods"]==2
    assert result["maximum_cost_usd"]==.09 and result["owner_approved"] is False
    assert result["expected_cost_usd"]==.05 and result["expected_worker_hours"]==.1
    unavailable=quote(plan,calibration,timing,dict(offer,availability="NONE"),
        workers=6,workers_per_pod=3,price_only=True,now=1001)
    assert unavailable["stock_available"] is False and unavailable["owner_approved"] is False
    for changes, per_pod in (({},4),({"storage_gb":59},3),
                             ({"minimum_ram_bytes_per_pod":14*1024**3},3),
                             ({"availability":"NONE"},3),({"compute_workload":"gpu"},3)):
        with pytest.raises(ValueError):
            quote(plan,calibration,timing,dict(offer,**changes),workers=6,workers_per_pod=per_pod,now=1001)


def test_guard_stop_retains_incomplete_native_hand(tmp_path):
    class Uniform:
        description={"fixture":"uniform"};abstraction=HU20_UNCAPPED_SCHEMA
        def distribution(self,view):
            menu=choices(view,raise_cap=None,free_fold=False)
            return menu,(1/len(menu),)*len(menu),False
    count=0
    def guard():
        nonlocal count
        count+=1
        if count==2:raise TimeoutError("Frozen phase expired")
    spec={"name":"fixture","strategy":"average","seed":1}
    panel={"name":"uniform","rule":"uniform","contract":"native"}
    with pytest.raises(TimeoutError):campaign.play(Uniform(),spec,panel,42,0,0,guard,failure_dir=tmp_path)
    files=list(tmp_path.glob("*.json"));assert len(files)==1
    partial=json.loads(files[0].read_text())
    assert partial["status"]=="incomplete" and len(partial["actions"])==1
    assert "Frozen phase expired" in partial["failure"]


def test_arena_rows_keep_actual_base_strategy_and_explicit_arm():
    class Uniform:
        description={"fixture":"uniform"};abstraction=HU20_UNCAPPED_SCHEMA
        def distribution(self,view):
            menu=choices(view,raise_cap=None,free_fold=False)
            return menu,(1/len(menu),)*len(menu),False
    spec={"name":"fixture","strategy":"average","seed":1}
    panel={"name":"uniform","rule":"uniform","contract":"native"}
    rows=[campaign.play(Uniform(),spec,panel,42,0,rotation,lambda:None,arm=arm)
          for arm in ("base","search") for rotation in (0,1)]
    assert [r["arm"] for r in rows]==["base","base","search","search"]
    assert all(r["strategy"]=="average" and r["host"] and r["architecture"] for r in rows)
    assert rows[0]["deal_seed"]==rows[1]["deal_seed"]
    # Aliases belong only to the paired estimator, never the retained hand.
    summary=campaign.summarize_phase(rows,"arena")
    assert {p["arm"] for p in summary["panels"]}=={"base","search"}
    assert all(p["strategy"]=="average" for p in summary["panels"])
    assert all(r["strategy"]=="average" for r in rows)


def test_watchdog_interrupts_blocking_load_and_retains_guard_reason(tmp_path):
    code='''
from pathlib import Path
from time import monotonic,sleep
from types import SimpleNamespace
from scripts.hu20_search_runtime import install_stop_handlers,start_resource_watchdog,swap_bytes
install_stop_handlers()
b=SimpleNamespace(out=Path(__import__('sys').argv[1]),started=monotonic(),
    deadline=monotonic()+.1,admission={'rss_limit_bytes':10*1024**3},
    peak_rss=0,swap_baseline=swap_bytes())
stop=start_resource_watchdog(b)
try:
    sleep(20)
except RuntimeError:
    print('blocking load interrupted')
finally:stop()
'''
    result=subprocess.run([sys.executable,"-c",code,str(tmp_path)],capture_output=True,text=True,timeout=5)
    assert result.returncode==0 and "blocking load interrupted" in result.stdout
    failure=json.loads(next(tmp_path.glob("resource-guard-failure-*.json")).read_text())
    assert failure["cause"]=="TimeoutError" and failure["elapsed_seconds"]<5


def test_native_allocation_reserves_measured_family_rss_and_overhead(monkeypatch):
    from types import SimpleNamespace
    import scripts.hu20_search_runtime as runtime
    gib=1024**3
    checked=[]
    budget=SimpleNamespace(admission={'rss_limit_bytes':int(4.25*gib),
        'solver_allocation_reserve_bytes':256*1024**2},check=lambda:checked.append(True))
    monkeypatch.setattr(runtime,'owned_rss',lambda:2*gib+17)
    admitted=runtime.native_allocation_budget(budget,5*gib)
    assert admitted < 2*gib and admitted+2*gib+17+256*1024**2 <= int(4.25*gib)
    assert runtime.native_allocation_budget(budget,512*1024**2)==512*1024**2
    monkeypatch.setattr(runtime,'owned_rss',lambda:4*gib)
    assert runtime.native_allocation_budget(budget,5*gib)==0
    assert len(checked)==3


def test_cache_inclusive_readmission_records_components_and_sidecar():
    from scripts.hu20_search_runtime import macos_memory_admission
    vm="""Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 131072.
Pages inactive: 327680.
Pages speculative: 8192.
File-backed pages: 327680.
"""
    measured=macos_memory_admission(vm)
    assert measured["reclaimable_bytes"]==sum(measured["memory_components_bytes"].values())
    assert measured["rss_limit_bytes"]==8*1024**3
    a=dict(admission(),**measured)
    validate_admission(a,now=1001)
    a["sidecar_reserved_bytes"]=20*1024**3
    with pytest.raises(ValueError):validate_admission(a,now=1001)


def test_final_forecast_counts_quality_and_conditional_research_work_without_using_outcomes():
    from scripts.forecast_hu20_search_final import budget_forecast
    rows=[]
    for floor in (0,.01):
        for thread in (4,6):
            config=asdict(TurnSearchConfig(threads=thread,opponent_likelihood_floor=floor))
            for n in range(8):
                rows.append({'configuration_id':f'{floor}/{thread}','config':config,
                    'cold_seconds':20 if thread==6 else 25,'fallback':False,
                    'receipts':[{'seconds':19},{'seconds':22}], 'residual_pct_pot':999})
    result=budget_forecast(rows,used_seconds=12000)
    assert result['status']=='owner-decision-needed' and result['outcome_fields_used']==[]
    one=result['options'][1]
    assert one['phases'][0]['entries'][0]['iterations']==[25,50,100]
    assert one['phases'][1]['entries'][0]['iterations']==[25,50,100,200,400]
    assert one['phases'][0]['entries'][0]['unique_solves_per_iteration']==144
    assert one['phases'][0]['entries'][1]['unique_solves_per_iteration']==288
    assert one['phases'][0]['quality_seconds']>0 and one['river_reserve_seconds']==10800
    for r in rows:r['residual_pct_pot']=0
    assert budget_forecast(rows,used_seconds=12000)==result


def test_exact_unlocked_solver_reuses_full_matrices_but_not_failures_or_different_requests():
    from copy import deepcopy
    from time import monotonic
    from scripts.hu20_search_resume import SharedRootSolver
    from src.blueprint.hu20_turn_solver import SolveFailure
    from tests.test_hu20_turn_search import FakeSolver, Uniform, fixture
    from src.blueprint.hu20_turn_search import HU20TurnSearchPolicy
    import numpy as np
    solvers=[FakeSolver(),FakeSolver()]
    policies=[HU20TurnSearchPolicy(Uniform(),s,TurnSearchConfig(opponent_likelihood_floor=0)) for s in solvers]
    solutions=[p._resolve(fixture().events,bot,monotonic()+30) for bot,p in enumerate(policies)]
    class Native:
        expected_sha256='fixture'
        def __init__(self):self.records=[];self.calls=[];self.fail=False
        def solve(self,request,deadline,mode='play'):
            self.calls.append((deepcopy(request),mode))
            self.records.append({'seconds':.2,'status':'failure' if self.fail else 'completed','path':'fixture'})
            if self.fail:raise SolveFailure('timeout','retained timeout')
            return solutions[0].profiles
    native=Native();shared=SharedRootSolver(native)
    shared.start_coordinate();a=shared.solve(solutions[0].request,monotonic()+30)
    shared.start_coordinate();b=shared.solve(solutions[1].request,monotonic()+30)
    assert len(native.calls)==1 and shared.records[0]['shared'] and shared.records[0]['seconds']==0
    assert shared.reused_play_seconds==.2
    for key in a:
        assert a[key].holdings==b[key].holdings
        np.testing.assert_array_equal(a[key].probabilities,b[key].probabilities)
    shared.solve(dict(solutions[1].request,locks=[{'line':[]}]),monotonic()+30)
    shared.solve(dict(solutions[1].request,memory_budget_bytes=1),monotonic()+30)
    shared.solve(solutions[1].request,monotonic()+30,mode='quality')
    assert len(native.calls)==4
    native.fail=True
    other=SharedRootSolver(native)
    for _ in range(2):
        other.start_coordinate()
        with pytest.raises(SolveFailure):other.solve(solutions[0].request,monotonic()+30)
        assert other.records[-1]['status']=='failure'
    assert len(native.calls)==6 and not other.cache


def test_retained_screen_resume_preserves_every_coordinate_and_shares_only_epsilon_zero(tmp_path,monkeypatch):
    from collections import Counter
    from scripts import calibrate_hu20_turn_search as calibration
    from src.arena.schedule import digest
    from src.blueprint.hu20_turn_solver import file_hash
    from tests.test_hu20_turn_search import FakeSolver, Uniform, fixture
    native_calls=[]
    class Native(FakeSolver):
        def __init__(self,*a,**kw):super().__init__();self.records=[]
        def solve(self,request,deadline,mode='play'):
            native_calls.append(mode)
            profiles=super().solve(request,deadline)
            self.records.append({'seconds':.005,'status':'completed','path':'fixture',
                'quality':[{'law':'reference','exploitability_pct_pot':.01,'retained_mass':[1,1]}]})
            return profiles
    class Budget:
        def check(self):pass
    monkeypatch.setattr(calibration,'load',lambda spec,inputs:Uniform())
    monkeypatch.setattr(calibration,'replay_root',lambda root:fixture().events)
    monkeypatch.setattr(calibration,'ExternalTurnSolver',Native)
    items=[{'root':{'kind':kind,'button':button,'spot':f'{kind}/{button}'},
        'policy':{'name':'fixture','seed':1},'reference_ranges':[[{'weight':1}],[{'weight':1}]],
        'reference_native_verified':True,'e_bp_pct_pot':2}
        for kind in ('limped','min-raised','pot-raised','3-bet') for button in (0,1)]
    refs={'items':items,'exclusions':[],'145_final_report_pushed':True,'final_report_sha256':'report'}
    protocol={'stage':'frozen-final','reference_index_sha256':digest(refs),
        'menus':['native'],'iterations':[25,100],'threads':[6],'compress':[False],
        'opponent_likelihood_floor':[0,.01],'decision_seconds':30,'share_identical_unlocked_seats':True,
        'staging':{'screen_iterations':100,'finalist_settings_per_floor':1}}
    screen=[]
    for floor in (0,.01):
        config=asdict(TurnSearchConfig(iterations=100,threads=6,compress=False,opponent_likelihood_floor=floor))
        for item,bot in calibration.screen_items(items):
            screen.append({'stage':'screen','root':f"{item['root']['spot']}/1/{bot}",
                'configuration_id':digest(config),'config':config,'cold_seconds':2,'fallback':False})
    retained=tmp_path/'retained.jsonl';retained.write_text(''.join(json.dumps(r)+'\n' for r in screen))
    protocol['resume_screen']={'path':str(retained),'sha256':file_hash(retained),'rows':len(screen)}
    protocol['finalist_configuration_ids']=[digest(asdict(c)) for c in calibration.finalists(screen,protocol)]
    result=calibration.run(protocol,refs,tmp_path,tmp_path/'binary',tmp_path/'resumed',Budget())
    assert result['status']=='qualified' and result['roots']==64 and result['screen_curve']==screen
    rows=[json.loads(line) for line in (tmp_path/'resumed'/'curve.jsonl').read_text().splitlines()]
    assert rows[:len(screen)]==screen and all(r['stage']=='final' for r in rows[len(screen):])
    assert Counter(native_calls)=={'play':48,'quality':48}
    final=rows[len(screen):]
    for row in final:
        reused=any(r.get('shared') for r in row['receipts'])
        assert reused==(row['config']['opponent_likelihood_floor']==0 and row['bot_seat']==1)
        if reused:
            assert row['cold_seconds']>=row['actual_cold_seconds']+.005
            first=next(r for r in final if r['configuration_id']==row['configuration_id']
                and r['root']==row['root'].rsplit('/',1)[0]+'/0')
            assert row['cold_seconds']>=first['cold_seconds']
    before=len(native_calls)
    retained.write_text(retained.read_text()+'\n')
    with pytest.raises(ValueError,match='hash differs'):
        calibration.run(protocol,refs,tmp_path,tmp_path/'binary',tmp_path/'refused',Budget())
    assert len(native_calls)==before and not (tmp_path/'refused').exists()


def test_shared_cold_deadline_still_returns_measured_blueprint_fallback(tmp_path,monkeypatch):
    from scripts import calibrate_hu20_turn_search as calibration
    from scripts.hu20_search_resume import SharedRootSolver
    from tests.test_hu20_turn_search import FakeSolver, Uniform, fixture
    class Native(FakeSolver):
        def __init__(self):super().__init__();self.records=[]
        def solve(self,request,deadline,mode='play'):
            profiles=super().solve(request,deadline)
            self.records.append({'seconds':31,'status':'completed','path':'fixture',
                'quality':[{'law':'reference','exploitability_pct_pot':.01,'retained_mass':[1,1]}]})
            return profiles
    monkeypatch.setattr(calibration,'replay_root',lambda root:fixture().events)
    item={'root':{'spot':'fixture','button':0,'kind':'limped'},'policy':{'seed':1},
        'reference_ranges':[[{'weight':1}],[{'weight':1}]],'reference_native_verified':True,'e_bp_pct_pot':2}
    native=Native();shared=SharedRootSolver(native)
    config=TurnSearchConfig(opponent_likelihood_floor=0)
    first=calibration.quality_row(Uniform(),tmp_path/'binary',config,item,0,tmp_path,lambda:None,shared_solver=shared)
    second=calibration.quality_row(Uniform(),tmp_path/'binary',config,item,1,tmp_path,lambda:None,shared_solver=shared)
    assert not first['fallback'] and second['fallback'] and second['fallback_cause']=='timeout'
    assert second['cold_seconds']>=31 and second['residual_pct_pot']==2
    assert second['full_native_verified'] and second['played_strategy_verified']
    assert second['receipts'][0]['shared'] and len(native.requests)==2


@pytest.mark.parametrize("floor,expected_failure", [(None, False), (20*1024**3, True)])
def test_blocking_watchdog_keeps_paid_worker_approval_interface(tmp_path, monkeypatch, floor, expected_failure):
    from types import SimpleNamespace
    from time import monotonic, sleep
    import scripts.hu20_search_runtime as runtime
    approval = {"rss_limit_bytes": 8*1024**3}
    if floor is not None: approval["minimum_disk_free_bytes"] = floor
    budget = SimpleNamespace(approval=approval, out=tmp_path, started=monotonic(),
        deadline=monotonic()+30, peak_rss=0, swap_baseline=0)
    monkeypatch.setattr(runtime, "owned_rss", lambda: 100)
    monkeypatch.setattr(runtime, "swap_bytes", lambda: 0)
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda _: SimpleNamespace(free=12*1024**3))
    signals=[]
    monkeypatch.setattr(runtime.os, "kill", lambda *args: signals.append(args))
    close=runtime.start_resource_watchdog(budget)
    try: sleep(.6)
    finally: close()
    assert bool(signals) == expected_failure
    failures=list(tmp_path.glob("resource-guard-failure-*.json"))
    if expected_failure:
        assert json.loads(failures[0].read_text())["cause"] == "OSError"
    else: assert not failures


def test_timing_pilot_runs_both_arms_and_reports_only_costs(tmp_path,monkeypatch):
    from collections import Counter
    class Uniform:
        description={"fixture":"uniform"};abstraction=HU20_UNCAPPED_SCHEMA
        def distribution(self,view):
            menu=choices(view,raise_cap=None,free_fold=False)
            return menu,(1/len(menu),)*len(menu),False
    class Search(Uniform):
        def __init__(self,source,solver,config):self.records=[];self.stats=Counter()
        def distribution(self,view,query_kind="play"):
            self.records.append({"status":"completed","seconds":.25})
            return Uniform.distribution(self,view)
    class Budget:
        def check(self):pass
    config=TurnSearchConfig(iterations=50,threads=6,compress=False,opponent_likelihood_floor=0)
    models=[{"name":f"{seed}-average","seed":seed,"strategy":"average"} for seed in (1,2,3)]
    plan={"stage":"timing-pilot","root":43,"models":models,"selected_search_config":asdict(config),
          "panels":[{"name":"uniform","rule":"uniform","contract":"native","blocks":4}],"expected_hands":48}
    monkeypatch.setattr(campaign,"load",lambda spec,inputs:Uniform())
    class Solver:
        def __init__(self,*a,**k):self.records=[]
    monkeypatch.setattr(campaign,"ExternalTurnSolver",Solver)
    monkeypatch.setattr(campaign,"HU20TurnSearchPolicy",Search)
    result=campaign.run(plan,tmp_path,tmp_path/"run",Budget(),phase="timing",search_config=config,
                        worker_index=1,worker_count=2)
    # Worker 1 of 2 owns blocks 1 and 3: two arms, two positions, three lineages each.
    assert result["status"]=="complete" and result["hands"]==24
    timing=result["timing_panels"]["uniform"]
    assert timing["paired_blocks"]==2 and timing["seconds_per_joint_block_p95"]>0
    assert result["solves"]>0 and result["solve_seconds_mean"]==.25
    assert "panels" not in result and "three_lineage_changes" not in result  # no payoff summary
    with pytest.raises(ValueError,match="Timing pilot"):
        campaign.run(dict(plan,stage="frozen-final"),tmp_path,tmp_path/"bad",Budget(),phase="timing",search_config=config)
