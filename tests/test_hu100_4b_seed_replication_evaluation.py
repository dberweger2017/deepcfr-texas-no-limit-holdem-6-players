"""Prospective PR236 controls reject changed evidence and preserve archive inputs."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]


def load(name):
    path=ROOT/"scripts"/(name+".py")
    if not path.exists():
        path=ROOT/"planning"/(name+".py")
    spec=importlib.util.spec_from_file_location("scripts."+name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


archive=load("archive_hu100_4b_seed_replication")
evaluation=load("evaluate_hu100_4b_seed_replication")


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+"\n")


def test_calibration_does_not_snapshot_its_live_guard_and_keeps_closed_evidence(tmp_path,monkeypatch):
    out=tmp_path/"research";out.mkdir()
    monkeypatch.setattr(archive,"OUT",out)
    monkeypatch.setattr(archive,"ROOT",tmp_path)
    index=tmp_path/"index.json";write(index,{"assets":[]})
    monkeypatch.setattr(archive.campaign,"INDEX",index)
    binary=tmp_path/"trainer";binary.write_bytes(b"fixture")
    monkeypatch.setattr(archive.campaign,"BINARY",binary)
    monkeypatch.setattr(archive.subprocess,"check_output",lambda *a,**k:"")
    write(out/"guards/operations/archive-calibration-initial/resources.jsonl",{"live":True})
    write(out/"guards/operations/closed/resources.jsonl",{"complete":True})
    files,omitted,aliases=archive.inventory()
    names={m for p,m in files}
    assert "research/guards/operations/archive-calibration-initial/resources.jsonl" not in names
    assert "research/guards/operations/closed/resources.jsonl" in names
    write(out/"guards/operations/archive-calibration-initial/receipt.json",{"status":"complete"})
    files,_,_=archive.inventory()
    assert "research/guards/operations/archive-calibration-initial/resources.jsonl" in {m for p,m in files}


def test_indexed_models_are_references_and_changed_bytes_stop(tmp_path,monkeypatch):
    out=tmp_path/"research";out.mkdir()
    monkeypatch.setattr(archive,"OUT",out)
    monkeypatch.setattr(archive,"ROOT",tmp_path)
    model=out/"inputs/2026100601/4000000000/average.gz"
    model.parent.mkdir(parents=True);model.write_bytes(b"pinned original")
    sha=hashlib.sha256(model.read_bytes()).hexdigest()
    index=tmp_path/"index.json"
    write(index,{"assets":[{"sha256":sha,"bytes":model.stat().st_size}]})
    monkeypatch.setattr(archive.campaign,"INDEX",index)
    binary=tmp_path/"trainer";binary.write_bytes(b"fixture")
    monkeypatch.setattr(archive.campaign,"BINARY",binary)
    monkeypatch.setattr(archive.subprocess,"check_output",lambda *a,**k:"")
    files,omitted,_=archive.inventory()
    assert model not in {p for p,m in files}
    assert omitted[0]["asset"]["sha256"]==sha
    write(index,{"assets":[{"sha256":sha,"bytes":model.stat().st_size+1}]})
    with pytest.raises(ValueError,match="indexed model changed"):
        archive.inventory()


def test_readout_uses_two_primary_family_and_rejects_post_replay_changes(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluation,"OUT",tmp_path)
    values=[10,12,9,11,10,8,13,9]
    write(tmp_path/"frozen-final.json",{"source":"frozen","primary":["gain"],"descriptive":[],
                                       "blocks_per_contrast":{"gain":len(values)},
                                       "scripted_blocks_per_opponent":4096, "scripted_root":123})
    monkeypatch.setattr(evaluation,"verify_scripted",lambda *args:{})
    base=tmp_path/"final-direct/gain"
    raw=base/"play/hands.jsonl.gz";raw.parent.mkdir(parents=True)
    with gzip.open(raw,"wt") as stream:
        for b,v in enumerate(values):
            for r in (0,1):
                stream.write(json.dumps({"block":b,"rotation":r,"candidate_chips":v,"status":"completed"})+"\n")
    complete={"hands":16,"decisions":32,"files":{"hands.jsonl.gz":{
        "bytes":raw.stat().st_size,"sha256":evaluation.file_hash(raw)}}}
    write(base/"play/complete.json",complete)
    repeat=base/"reproduction/hands.jsonl.gz";repeat.parent.mkdir()
    repeat.write_bytes(raw.read_bytes())
    write(base/"reproduction/complete.json",{**complete,"reproduced_all_hands_and_decisions":True})
    write(base/"replay.json",{"status":"verified","blocks":8,"hands_replayed":16,"actions_replayed":32,
                             "coverage":{},"bonferroni_three_ci":[-999,999]})
    write(tmp_path/"scripted-summary.json",{"status":"verified","evidence_bindings":{}})
    evaluation.readout()
    result=evaluation.read(tmp_path/"summary.json")["results"]["gain"]
    assert result["confidence"]==.975
    assert result["bb_per_100"]==sum(values)/len(values)
    assert result["interval"][0]>0
    assert result["interval"]!=[-999,999]
    (tmp_path/"summary.json").unlink()
    raw.write_bytes(raw.read_bytes()+b"changed")
    with pytest.raises(ValueError,match="raw evidence changed"):
        evaluation.readout()


def test_archive_refresh_triggers_on_either_workload_axis():
    measured={"logical_member_bytes":100,"members":10}
    assert not archive.workload_requires_refresh(150,15,measured)
    assert archive.workload_requires_refresh(201,10,measured)
    assert archive.workload_requires_refresh(100,21,measured)


def scripted_fixture(base):
    for label in ("terminal","1b"):
        write(base/(label+"-replay.json"),{"status":"verified","hands_replayed":160,
              "actions_replayed":400,"all_settlements_replayed":True})
        for run in (label,label+"-reproduction"):
            folder=base/run
            complete={"status":"complete","hands":160,"decisions":400,
                      "blocks_per_opponent":8,"root":123,
                      "reproduced_all_hands_and_decisions":run.endswith("-reproduction")}
            write(folder/"complete.json",complete)
            write(folder/"hands.jsonl",{"fixture":"verified raw"})
            pins={name:{"bytes":(folder/name).stat().st_size,
                        "sha256":evaluation.file_hash(folder/name)}
                  for name in ("complete.json","hands.jsonl")}
            write(folder/"output-files.json",pins)


def test_scripted_binding_rejects_changed_raw_after_replay(tmp_path):
    scripted_fixture(tmp_path)
    binding=evaluation.verify_scripted(tmp_path,8,123)
    assert binding["terminal"]["counts"][0]["hands"]==160
    (tmp_path/"terminal/hands.jsonl").write_text("changed")
    with pytest.raises(ValueError,match="scripted evidence changed"):
        evaluation.verify_scripted(tmp_path,8,123)


def test_scripted_binding_rejects_inconsistent_full_replay_count(tmp_path):
    scripted_fixture(tmp_path)
    replay=evaluation.read(tmp_path/"1b-replay.json")
    replay["actions_replayed"]+=1
    (tmp_path/"1b-replay.json").write_text(json.dumps(replay))
    with pytest.raises(ValueError,match="counts differ"):
        evaluation.verify_scripted(tmp_path,8,123)


def test_planning_authorization_and_quote_are_archive_provenance(tmp_path,monkeypatch):
    out=tmp_path/"research";out.mkdir()
    monkeypatch.setattr(archive,"OUT",out);monkeypatch.setattr(archive,"ROOT",tmp_path)
    index=tmp_path/"index.json";write(index,{"assets":[]})
    monkeypatch.setattr(archive.campaign,"INDEX",index)
    binary=tmp_path/"trainer";binary.write_bytes(b"fixture")
    monkeypatch.setattr(archive.campaign,"BINARY",binary)
    monkeypatch.setattr(archive.subprocess,"check_output",lambda *a,**k:"")
    write(tmp_path/"planning/final-quote-posted.json",{"pr":236,"sha256":"fixture"})
    write(tmp_path/"planning/stagewise-start-authorization.json",{"authorized":True})
    files,_,_=archive.inventory()
    assert {"provenance/planning/final-quote-posted.json",
            "provenance/planning/stagewise-start-authorization.json"} <= {m for p,m in files}


def test_historical_full_audit_is_bound_to_every_indexed_asset(monkeypatch):
    pins={kind:{"bytes":i+100,"sha256":str(i)*64,"entries":500,"actual_nodes":2000}
          for i,kind in enumerate(("checkpoint.gz","current.gz","average.gz"),1)}
    monkeypatch.setattr(evaluation.campaign,"reference",lambda seed,target:pins)
    audit={"status":"verified","stack_bb":100,"entries":500,
           "native_state":{"completed_nodes":2000},
           "files":{"/old/"+kind:{"bytes":pin["bytes"],"sha256":pin["sha256"]} for kind,pin in pins.items()},
           "audit":{"status":"verified","all_nodes_verified":500,"counts":{"entries":500},
                    "checkpoint_sha256":pins["checkpoint.gz"]["sha256"],
                    "current_sha256":pins["current.gz"]["sha256"],
                    "average_sha256":pins["average.gz"]["sha256"]}}
    evaluation.bind_historical_audit(audit,2000)
    audit["files"]["/old/current.gz"]["sha256"]="changed"
    with pytest.raises(ValueError,match="indexed pin differs"):
        evaluation.bind_historical_audit(audit,2000)
    audit["files"]["/old/current.gz"]["sha256"]=pins["current.gz"]["sha256"]
    audit["audit"]["all_nodes_verified"]-=1
    with pytest.raises(ValueError,match="lineage/counts differ"):
        evaluation.bind_historical_audit(audit,2000)
