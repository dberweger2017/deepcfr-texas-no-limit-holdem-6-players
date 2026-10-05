"""Paid dispatch checkpoints and bounded, verifiable retained evidence."""

from collections import Counter
import json
from types import SimpleNamespace

import pytest

from scripts.hu20_search_arena_control import ArenaControl
from scripts.hu20_search_evidence import finalize_profiles, verify_retention, pack, verify_archives
from scripts.run_hu20_search_arena_guarded import guarded_policy
from src.blueprint.hu20_turn_solver import file_hash, sampled_profile


def controller(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"pods": [
        {"id": f"pod-{i}", "workers": list(range(i*3, i*3+3)), "created_at": 100,
         "production_stop_at": 10000, "hourly_usd": .228334, "parity_retention_passed": True}
        for i in range(3)]}))
    control = ArenaControl(tmp_path / "control.json", ledger, clock=lambda: 200)
    assert control.request({"op": "start"})["status"] == "running"
    return control


def decision(control, i, fallback, pod=0):
    body = {"event": str(i), "pod": f"pod-{pod}", "worker": pod*3}
    assert control.request({"op": "acquire", **body})["status"] == "running"
    return control.request({"op": "complete", **body, "fallback": fallback})


@pytest.mark.parametrize("bad,stopped", [(25, False), (26, True)])
def test_exact_first_500_global_and_per_pod_checkpoint(tmp_path, bad, stopped):
    control = controller(tmp_path)
    for i in range(499):
        decision(control, i, i < bad, pod=i%3)
    body = {"event": "499", "pod": "pod-1", "worker": 3}
    control.request({"op": "acquire", **body})
    assert control.request({"op": "acquire", "event": "500", "pod": "pod-2", "worker": 6})["status"] == "wait"
    result = control.request({"op": "complete", **body, "fallback": False})
    assert (result["status"] == "stopped") == stopped
    assert control.state["checkpoints"][0]["decisions"] == 500


def test_per_pod_gate_catches_concentrated_failures_hidden_by_global_rate(tmp_path):
    control = controller(tmp_path)
    for i in range(1498):
        result = decision(control, i, i%3 == 0 and (i//3)%19 == 0, pod=i%3)
    assert result["status"] == "stopped"
    assert "pod-0 checkpoint 500" in control.state["reason"]
    assert sum(r["fallback"] for r in control.state["events"])*20 < len(control.state["events"])


def test_trailing_500_stop_even_with_low_cumulative_rate(tmp_path):
    control = controller(tmp_path)
    for i in range(600):
        result = decision(control, i, i >= 574, pod=i%3)
    assert result["status"] == "stopped"
    assert control.state["checkpoints"][-1]["fallbacks"] == 26
    assert control.state["checkpoints"][-1]["decisions"] == 600


def test_identity_idempotency_and_spend_closeout_guards(tmp_path):
    control = controller(tmp_path)
    decision(control, 0, True)
    body = {"op": "complete", "event": "0", "pod": "pod-0", "worker": 0, "fallback": True}
    assert control.request(body)["decisions"] == 1
    with pytest.raises(ValueError):
        control.request({**body, "fallback": False})
    with pytest.raises(ValueError):
        control.request({"op": "acquire", "event": "alien", "pod": "old-pod", "worker": 0})
    with pytest.raises(ValueError):
        ArenaControl(control.journal, control.ledger)
    control.clock = lambda: 11000
    assert control.request({"op": "check"})["status"] == "stopped"
    assert "closeout" in control.state["reason"]
    control.clock = lambda: 200000
    assert control.request({"op": "check"})["charge_usd"] > 21


def test_worker_counts_cached_fallbacks_but_excludes_blueprint_and_probes():
    class Policy:
        def __init__(self):
            self.stats = Counter()
        def distribution(self, view, query_kind="probe"):
            self.stats[query_kind+":fallback:timeout"] += 1
            return "unchanged"
    class Client:
        calls = []
        def acquire(self, guard):
            guard()
            return "event"
        def request(self, op, **body):
            self.calls.append((op, body))
    client = Client()
    policy = guarded_policy(Policy, client, lambda: None)()
    view = SimpleNamespace(street=SimpleNamespace(value="turn"))
    assert policy.distribution(view, query_kind="probe") == "unchanged"
    assert not client.calls
    assert policy.distribution(view, query_kind="play") == "unchanged"
    assert client.calls == [("complete", {"event": "event", "fallback": True, "cause": "timeout"})]
    client.calls.clear()
    view.street.value = "flop"
    policy.distribution(view, query_kind="play")
    assert not client.calls


def test_hash_before_delete_and_lossless_retrieval_without_extraction(tmp_path, monkeypatch):
    owned = tmp_path / "owned"
    evidence = owned / "evidence"
    evidence.mkdir(parents=True)
    samples = {}
    i = 0
    while len(samples) < 2:
        data = json.dumps({"request": i}).encode()
        import hashlib
        samples.setdefault(sampled_profile(hashlib.sha256(data).hexdigest()), data)
        i += 1
    for selected, data in samples.items():
        solve = evidence / str(selected)
        solve.mkdir()
        (solve / "request.json").write_bytes(data)
        (solve / "profile.jsonl").write_bytes(b"large profile body\n" * 100)
        for name in ("receipt.json", "response.jsonl", "hand.jsonl.gz"):
            (solve / name).write_bytes(b"keep all bytes\n")
    before = {str(p.relative_to(evidence)): file_hash(p) for p in evidence.rglob("*") if p.is_file()}
    finalize_profiles(evidence, owned)
    counts = verify_retention(evidence)
    assert counts["requests"] == 2 and counts["profile_bodies"] == 1 and counts["deleted_profiles"] == 1
    for name, sha in before.items():
        if name != "False/profile.jsonl":
            assert file_hash(evidence / name) == sha
    out = tmp_path / "archives"
    pack(evidence, out, limit=10000)
    assert verify_archives(out)["verified"]
    assert not (out / "True").exists()
    archive = next(out.glob("*.gz"))
    archive.write_bytes(archive.read_bytes()+b"corruption")
    with pytest.raises(ValueError):
        verify_archives(out)
    with pytest.raises(ValueError):
        finalize_profiles(evidence, tmp_path / "unowned")
    # A durability failure never authorizes deletion of the full body.
    profile = evidence / "False" / "profile.jsonl"
    profile.write_bytes(b"retained on failure")
    def fail(*args):
        raise OSError("fsync failed")
    monkeypatch.setattr("scripts.hu20_search_evidence.durable_json", fail)
    with pytest.raises(OSError):
        finalize_profiles(evidence, owned)
    assert profile.exists()


@pytest.mark.parametrize("corrupt", [False, True])
def test_supervisor_only_terminates_attributed_pod_after_stream_hash_verification(tmp_path, monkeypatch, corrupt):
    import shutil
    import threading
    from scripts import monitor_hu20_search_arena as monitor
    root = tmp_path / "supervisor"
    root.mkdir()
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence/"hand.gz").write_bytes(b"every hand byte")
    archive = tmp_path / "remote-archive"
    pack(evidence, archive)
    pod = {"id": "owned", "creation_receipt": "created.json", "ssh_host": "host", "ssh_port": 22}
    ledger = {"pods": [pod]}
    (root/"created.json").write_text(json.dumps({"response": {"result": {"content": [
        {"text": json.dumps({"id": "owned"})}]}}}))
    monkeypatch.setattr(monitor, "ssh", lambda *args, **kwargs: "")
    def copy(args, **kwargs):
        source = args[-2].split("/workspace/archives/", 1)[1]
        target = Path(args[-1])/source
        shutil.copy2(archive/source, target)
        if corrupt and source.endswith(".gz"):
            target.write_bytes(b"changed archive")
    from pathlib import Path
    monkeypatch.setattr(monitor.subprocess, "run", copy)
    actions = []
    def mcp(method, params, ident):
        actions.append(params["name"])
        assert params["arguments"]["id"] == "owned"
        if params["name"] == "get-pod" and "delete-pod" in actions:
            return {"result": {"isError": True, "content": [{"text": "404 Pod not found"}]}}
        return {"result": {"isError": False, "content": [{"text": "{}"}]}}
    if corrupt:
        with pytest.raises(ValueError):
            monitor.closeout(pod, root, mcp, ledger, threading.Lock())
        assert "delete-pod" not in actions
    else:
        assert monitor.closeout(pod, root, mcp, ledger, threading.Lock()) == "owned"
        assert actions == ["get-pod", "delete-pod", "get-pod"]
        assert pod["retrieval_verified"] and pod["terminated_at"]


def test_mixed_handoff_start_and_fixed_sleep_exclusion(tmp_path):
    ledger=tmp_path/'ledger.json'
    pods=[{'id':f'mixed-{i}','workers':([0,1] if i==0 else [i+1]),'created_at':100,
           'hourly_usd':.2,'production_stop_at':10000,'parity_retention_passed':True} for i in range(4)]
    ledger.write_text(json.dumps({'pods':pods,'expected_active_pods':4,'owner_excluded_charge_usd':.4}))
    control=ArenaControl(tmp_path/'control.json',ledger,clock=lambda:1900)
    assert control.request({'op':'start'})['status']=='running'
    assert control.charge()==pytest.approx(1.5)
    pods[3]['parity_retention_passed']=False
    ledger.write_text(json.dumps({'pods':pods,'expected_active_pods':4}))
    other=ArenaControl(tmp_path/'other.json',ledger,clock=lambda:1900)
    with pytest.raises(ValueError,match='parity'):other.request({'op':'start'})


def test_public_controller_requires_authentication(tmp_path):
    from scripts.hu20_search_arena_control import serve
    control=controller(tmp_path)
    with pytest.raises(ValueError,match='authentication'):
        serve(control,0,bind='0.0.0.0')


def test_first_stop_reason_survives_worker_closeout(tmp_path):
    control=controller(tmp_path)
    control.request({'op':'stop','reason':'Fallback rate exceeds 5%'})
    control.request({'op':'stop','reason':'Worker exited incompletely'})
    assert control.state['reason']=='Fallback rate exceeds 5%'


def test_partial_start_never_admits_an_unchecked_joining_host(tmp_path):
    control=controller(tmp_path)
    ledger=json.loads(control.ledger.read_text())
    ledger['start_required_pods']=['pod-0','pod-1']
    ledger['pods'][2]['parity_retention_passed']=False
    control.ledger.write_text(json.dumps(ledger))
    fresh=ArenaControl(tmp_path/'partial.json',control.ledger,clock=lambda:200)
    assert fresh.request({'op':'start'})['status']=='running'
    with pytest.raises(ValueError,match='Unknown owned'):
        fresh.request({'op':'acquire','event':'6/1','pod':'pod-2','worker':6})
    ledger['pods'][2]['parity_retention_passed']=True
    control.ledger.write_text(json.dumps(ledger))
    assert fresh.request({'op':'acquire','event':'6/1','pod':'pod-2','worker':6})['status']=='running'


def test_compressed_native_audit_keeps_original_manifests_and_rejects_changed_seal(tmp_path):
    import gzip
    from pathlib import Path
    from scripts.finish_hu20_mixed_arena import materialize_hands,ArchivedFile
    evidence=tmp_path/'evidence';worker=evidence/'arena/worker-0';worker.mkdir(parents=True)
    (worker/'summary.json').write_text(json.dumps({'worker_index':0,'status':'complete'}))
    (worker/'request.json').write_text('all request bytes')
    with gzip.open(worker/'base.hands.jsonl.gz','wt') as stream:stream.write('{}\n')
    original={'request.json':{'bytes':17,'sha256':file_hash(worker/'request.json')}}
    (worker/'manifest.json').write_text(json.dumps(original))
    root=tmp_path/'supervisor';out=root/'retrieved/pod';out.parent.mkdir(parents=True)
    pack(evidence,out)
    (out/'retrieval-verified.json').write_text(json.dumps({**verify_archives(out),'archive_manifest_sha256':file_hash(out/'manifest.json')}))
    directories,summaries,proofs=materialize_hands(root)
    assert summaries[0]['worker_index']==0 and proofs[0]['pod_id']=='pod'
    virtual=directories[0]/'request.json'
    assert isinstance(virtual,ArchivedFile) and virtual.stat().st_size==len('all request bytes')
    assert virtual.spec==original['request.json']
    assert json.loads((root/'audit-hands/worker-0/manifest.json').read_text())==original
    (out/'manifest.json').write_text((out/'manifest.json').read_text()+' ')
    with pytest.raises(ValueError,match='manifest changed'):materialize_hands(root)


def test_report_imports_approved_science_from_separate_checkout(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    source=tmp_path/'scientific-source'
    (source/'scripts').mkdir(parents=True)
    (source/'scripts/evaluate_hu20_turn_search.py').write_text('IDENTITY = "approved-science"\n')
    (source/'scripts/audit_hu20_turn_search.py').write_text(
        'from scripts.evaluate_hu20_turn_search import IDENTITY\n')
    code=('from pathlib import Path; from scripts.finish_hu20_mixed_arena import load_approved_audit; '
          f'audit=load_approved_audit(Path({str(source)!r})); '
          'assert audit.IDENTITY == "approved-science"')
    subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],check=True)


def test_fixed_work_controller_stops_first_defect_without_latency_boundary(tmp_path):
    control=controller(tmp_path)
    ledger=json.loads(control.ledger.read_text())
    ledger["work_protocol"]="hu20-fixed50-no-fallback-v1"
    control.ledger.write_text(json.dumps(ledger))
    control.clock=lambda:11000 # Historical per-pod production clock no longer applies.
    assert control.request({"op":"check"})["status"]=="running"
    assert decision(control,0,True)["status"]=="stopped"
    assert control.state["reason"]=="Search fallback is a defect under fixed-work protocol"


def test_fixed_work_admission_uses_quota_and_memory_without_latency_headroom():
    from scripts.quote_hu20_fixed_work_arena import admitted_workers
    assert admitted_workers(18,32*10**9)==3
    assert admitted_workers(17.99,32*10**9)==2
    assert admitted_workers(18.7,24*10**9)==2
    assert admitted_workers(23.8,64*10**9)==3


def test_cpu_quote_refuses_missing_censored_stratum_and_wrong_thread_pool():
    from scripts.quote_hu20_fixed_work_arena import work_estimate
    stage4=[{"status":"completed","request_sha256":"a","threads":6},
            {"status":"failure","request_sha256":"b","cause":"timeout","threads":6}]
    sample=[{"status":"completed","request_sha256":"a","threads":6,"cpu_seconds":9,
             "wall_seconds":1,"completion":{"iterations":50,"status":"play_complete"}}]
    with pytest.raises(ValueError,match="thread checks"):work_estimate(stage4,sample)
    sample[0].update(cpu_seconds=1,solver_seconds=1)
    with pytest.raises(ValueError,match="hash ordering"):work_estimate(stage4,sample)


def test_fixed_replay_rejects_small_scientific_difference(tmp_path):
    import json
    from scripts.replay_hu20_search_requests import replay
    reference=tmp_path/'refs'/'slot'/'solve';reference.mkdir(parents=True)
    request={'work_protocol':'hu20-fixed50-no-fallback-v1','max_iterations':50,'threads':6,'seconds':None,'mode':'play'}
    completion={'event':'completion','work_protocol':request['work_protocol'],'iterations':50,'status':'play_complete','elapsed_seconds':1,'solver_peak_rss_bytes':12}
    (reference/'request.json').write_text(json.dumps(request));(reference/'response.jsonl').write_text(json.dumps(completion)+'\n')
    (reference/'profile.jsonl').write_text(json.dumps({'strategy':[.5,.5],'solver_peak_rss_bytes':12})+'\n')
    binary=tmp_path/'solver';binary.write_text('#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\nq=json.loads(Path(sys.argv[1]).read_text())\nPath(q["dump_path"]).write_text(json.dumps({"strategy":[.5000001,.4999999],"solver_peak_rss_bytes":99})+"\\n")\nPath(sys.argv[2]).write_text('+repr(json.dumps(completion)+'\n')+')\n');binary.chmod(0o755)
    row=replay(binary,reference,tmp_path/'out')
    assert row['status']=='mismatch' and not row['exact_scientific_outputs']
    assert (tmp_path/'out'/'slot'/'solve'/'request.json').exists()


def test_fixed_controller_counts_prior_spend(tmp_path):
    import json
    from scripts.hu20_search_arena_control import ArenaControl
    ledger=tmp_path/'ledger.json';ledger.write_text(json.dumps({'work_protocol':'hu20-fixed50-no-fallback-v1','historical_cap_charge_usd':20,'pilot_reserve_usd':.5,'storage_contingency_usd':1,'pods':[]}))
    controller=ArenaControl(tmp_path/'journal.json',ledger,clock=lambda:100)
    assert controller.request({'op':'check'})['status']=='stopped'
    assert controller.charge()==21.5


def test_lower_owner_ceiling_lowers_financial_dispatch_stop(tmp_path):
    import json
    from scripts.hu20_search_arena_control import ArenaControl
    ledger=tmp_path/'ledger.json';ledger.write_text(json.dumps({'work_protocol':'hu20-fixed50-no-fallback-v1','hard_ceiling_usd':20,'dispatch_stop_usd':16,'historical_cap_charge_usd':15,'pilot_reserve_usd':.5,'storage_contingency_usd':1,'pods':[]}))
    controller=ArenaControl(tmp_path/'journal.json',ledger,clock=lambda:100)
    state=controller.request({'op':'check'})
    assert state['status']=='stopped' and '$16' in state['reason']


def test_equivalent_layout_keeps_quote_pricing_rules():
    import json
    from pathlib import Path
    from scripts.quote_hu20_fixed_work_arena import forecast_layout
    original=json.loads((Path(__file__).parents[1]/'docs/reports/hu20-fixed-work-arena/fixed-work-quote.json').read_text())
    baseline=forecast_layout(original,compute_hourly_usd=.13,workers_per_pod=3,pods=2,hard_ceiling=25)
    assert (baseline['expected_cost_usd'],baseline['maximum_forecast_cost_usd'])==(5.48,20.43)
    equivalent=forecast_layout(original,compute_hourly_usd=.18,workers_per_pod=5)
    assert (equivalent['expected_cost_usd'],equivalent['maximum_forecast_cost_usd'])==(4.79,16.96)
    assert equivalent['minimum_quota_cpus']==30 and equivalent['minimum_ram_bytes']==51*10**9
    assert equivalent['hard_ceiling_usd']==20 and equivalent['dispatch_stop_usd']==16
