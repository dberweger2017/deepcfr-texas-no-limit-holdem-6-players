"""Guard, cost-only storage admission and complete paired reporter integration."""
import gzip
import json
from pathlib import Path
import subprocess

import pytest

from scripts import run_native_hu100_growth_1b as campaign
from src.policies.files import file_hash


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_compact_capacity_and_real_resource_breaches():
    n=campaign.memory_capacity()
    assert 110*n+100000000 <= campaign.FAMILY_SOFT
    assert 110*(n+1)+100000000 > campaign.FAMILY_SOFT
    sample={"pressure_level":1,"free_percent":70,"swap_bytes":100,
            "ac":True,"disk_free_bytes":campaign.DISK_FLOOR+1}
    assert campaign.limits(sample,100,campaign.FAMILY_SOFT)==None
    assert campaign.limits(sample,100,campaign.FAMILY_HARD)=="hard whole-family RSS"
    assert campaign.limits({**sample,"pressure_level":2},100,0)=="system pressure/headroom"
    assert campaign.limits({**sample,"free_percent":14},100,0)=="system pressure/headroom"
    assert campaign.limits({**sample,"swap_bytes":101+campaign.SWAP_GROWTH},100,0)=="swap growth"
    assert campaign.limits({**sample,"disk_free_bytes":campaign.DISK_FLOOR},100,0)=="disk floor"
    assert campaign.limits({**sample,"ac":False},100,0)=="AC power"


def test_training_requires_owner_readiness_and_unchanged_quote(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,"OUT",tmp_path)
    monkeypatch.setattr(campaign,"clean_source",lambda:"source")
    put(tmp_path/"preflight-live.json",{"status":"admitted"})
    put(tmp_path/"training-ready.json",{"source":"source","owner_confirmed_ready":False})
    with pytest.raises(ValueError,match="Owner readiness"):
        campaign.train()
    put(tmp_path/"training-ready.json",{"source":"source","owner_confirmed_ready":True,
                                      "preflight_sha256":"changed"})
    with pytest.raises(ValueError,match="quote changed"):
        campaign.train()
    assert not (tmp_path/"training").exists()


def test_storage_projection_does_not_scale_model_snapshots_by_blocks(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,"OUT",tmp_path)
    monkeypatch.setattr(campaign,"host",lambda:{"disk_free_bytes":10**12})
    entries=7643261
    put(tmp_path/"pilot-complete.json",{"source":"source"})
    put(tmp_path/"gate/audit.json",{"entries":entries,"files":{
        "checkpoint.gz":{"bytes":37*entries},"current.gz":{"bytes":25*entries},"average.gz":{"bytes":25*entries}}})
    put(tmp_path/"gate/telemetry.jsonl",{"elapsed_seconds_including_writes":40,
        "write_seconds":22,"completed_nodes":39438279})
    for op in ("gate-export","gate-audit"):
        put(tmp_path/"operations"/op/"receipt.json",{"seconds":10})
    for root in ("pilot","pilot-reproduction"):
        for label in ("off","on"):
            put(tmp_path/root/label/"complete.json",{"model_load_seconds":30,"wall_seconds":31,"winnings":999})
            put(tmp_path/"pilot"/(label+"-audit.json"),{"seconds":1})
            models=tmp_path/root/label/"models";models.mkdir()
            (models/"snapshot.gz").write_bytes(b"x"*100000)
    measured=sum(p.stat().st_size for root in ("pilot","pilot-reproduction")
                 for p in (tmp_path/root).rglob("*") if p.is_file() and "models" not in p.parts)
    campaign.prepare()
    q=campaign.read(tmp_path/"preflight-corrected.json")
    assert q["pilot_raw_bytes"]==measured
    assert q["retained_final_raw_forecast_bytes"]==measured*128*3
    assert q["final_model_snapshot_forecast_bytes"]>0
    assert q["pilot_outcomes_inspected"] is False
    assert q["blocks_per_opponent"]==2048


def test_prior_roots_are_unique_and_both_campaign_roots_are_fresh(monkeypatch):
    monkeypatch.setattr(campaign,"gate_spec",lambda:{"name":"fixture","path":"fixture",
        "sha256":"0"*64,"format":"holdem-hu100-stored-cfr-average-research-v1"})
    receipt=campaign.freshness()
    roots=[r["root"] for r in receipt["roots"]]
    # Historical #200 roots also occur in #205's PRIOR_ROOTS.
    assert len(roots)==len(set(roots))
    assert campaign.PILOT_ROOT in roots and campaign.FINAL_ROOT in roots
    assert 2026100820512 in roots and 2026100820521 in roots
    assert receipt["all_pairwise_disjoint"]


def test_three_checkpoint_curve_has_only_two_formal_parent_contrasts_and_secondary(tmp_path,monkeypatch):
    from tests.test_native_hu100_baseline import model_fixture
    from tests.test_native_hu100_growth_50m import audit_fixture
    from scripts.native_hu100_model_metadata import audited_average_spec
    from scripts.evaluate_native_hu100_baseline import execute
    from scripts.audit_native_hu100_baseline import audit
    from scripts.report_native_hu100_learning_curves import frozen_schedule
    from src.arena import artifacts
    repo=tmp_path/"repo";repo.mkdir()
    subprocess.run(["git","init","-q",str(repo)],check=True)
    (repo/".gitignore").write_text("results/\n")
    (repo/"fixture.py").write_text("# committed fixture\n")
    subprocess.run(["git","-C",str(repo),"add","."],check=True)
    subprocess.run(["git","-C",str(repo),"-c","user.name=Fixture","-c","user.email=fixture@example.invalid",
                    "commit","-qm","fixture"],check=True)
    revision=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    out=repo/"results/campaign";run=out/"evaluation";run.mkdir(parents=True)
    specs=[]
    for nodes in (10000,15000,20000):
        directory=out/str(nodes);directory.mkdir()
        _,_,path=model_fixture(directory)
        with gzip.open(path,"rt") as f:
            rows=f.readlines()
        header=json.loads(rows[0]);header["checkpoint_header"]["native_state"]={"completed_nodes":nodes}
        with gzip.open(path,"wt") as f:
            f.write(json.dumps(header)+"\n"+"".join(rows[1:]))
        receipt=audit_fixture(path,nodes,checkpoint=header["source_checkpoint_sha256"])
        specs.append(audited_average_spec(path,receipt,checkpoint_sha256=header["source_checkpoint_sha256"],actual_nodes=nodes))
    settings=json.loads(Path("configs/arena/hu100-playing-baseline-v1.json").read_text())
    settings.pop("model");settings.update(models=specs,final_root=7654321)
    put(run/"settings.json",settings)
    put(run/"frozen-schedule.json",frozen_schedule(settings,2,settings["final_root"]))
    put(run/"frozen-final.json",{"source":revision,"blocks_per_opponent":2,"final_root":settings["final_root"],
                                "schedule_sha256":file_hash(run/"frozen-schedule.json")})
    monkeypatch.setattr(artifacts,"ROOT",repo);monkeypatch.chdir(repo)
    for index,spec in enumerate([*specs,specs[-1]]):
        translated=index==3
        label=("translated-" if translated else "")+str(spec["actual_nodes"])
        config={k:v for k,v in settings.items() if k!="models"}
        config["model"]=spec
        config["action_translation"]={"max_states":512,"max_events":128} if translated else None
        config_path=run/(label+"-config.json");put(config_path,config)
        target=run/"final"/label;reference=run/"final"/"10000"
        execute(config_path,target,2,settings["final_root"],revision,reference_run=reference if index else None)
        audit(target,run/("final-"+label+"-audit.json"))
        if translated:
            # The translated panel's arena audit is also retained by its folder.
            put(run/"final"/(label+"-audit.json"),json.loads((run/("final-"+label+"-audit.json")).read_text()))
        repeat=run/"final-reproduction"/label
        execute(config_path,repeat,2,settings["final_root"],revision,reproduce=target,
                reference_run=run/"final-reproduction"/"10000" if index else None)
    monkeypatch.setattr(campaign,"OUT",out);monkeypatch.setattr(campaign,"BLOCKS",2)
    campaign.report()
    result=json.loads((run/"campaign-result.json").read_text())
    formal=[c for c in result["final_minus_earlier"] if "primary_adjusted" in c]
    assert len(formal)==2
    assert {c["earlier_nodes"] for c in formal}=={10000}
    assert {c["opponent"] for c in formal}=={"tight_aggressive","loose_aggressive"}
    assert all(c["primary_adjusted"]["alpha"]==.025 for c in formal)
    descriptive=[c for c in result["final_minus_earlier"] if c["earlier_nodes"]==15000]
    assert all(c["formal_label"]=="descriptive" and c["descriptive_interval"]["alpha"]==.05 for c in descriptive)
    assert result["secondary"]["translated_minus_off"]["alpha"]==.05
    assert result["unique_final_hands_including_translated_arm"]==100

