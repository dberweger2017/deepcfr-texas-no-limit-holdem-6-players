"""Balanced schedules, worker approval and cumulative resource admission."""

from dataclasses import asdict
import gzip
import json
from pathlib import Path

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
    assert budget.deadline-budget.started==10
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
