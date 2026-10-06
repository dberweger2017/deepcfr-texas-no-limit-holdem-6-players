"""Durable supervisor for pods attributed by creation or explicit owner handoff.

This never allocates or restarts work. Controller loss halts the remote workers;
the supervisor reserves closeout time, retrieves exact evidence, then terminates
only pods whose creation response and ledger agree.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import base64
import re
import shlex
import urllib.request
import secrets
from pathlib import Path
import shutil
import subprocess
import threading
from time import sleep, time

from scripts.hu20_search_arena_control import ArenaControl, durable_json, serve
from scripts.hu20_search_evidence import verify_archives, file_hash

PROTECTED = {"43z4itur3hwnyv", "cl0riravggku4r", "xu414eguzakxfr", "k9rdph2fwhym87"}


def ssh(pod, command, *, timeout=45):
    base=["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new",
          "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2"]
    if pod.get("ssh_host"):
        return subprocess.run([*base, "-p", str(pod["ssh_port"]), "root@"+pod["ssh_host"], command],
                              capture_output=True, text=True, timeout=timeout, check=True).stdout
    encoded=base64.b64encode(command.encode()).decode()
    script="stty -echo\nPS1=''\nPS2=''\nprintf %s "+encoded+" | base64 -d | bash\nprintf '\\nHU20_EXIT=%s\\n' \"$?\"\nexit\n"
    result=subprocess.run([*base,"-tt",pod["ssh_proxy"]+"@ssh.runpod.io"],input=script,
                          capture_output=True,text=True,timeout=timeout,check=True)
    statuses=re.findall(r"HU20_EXIT=(\d+)",result.stdout)
    if not statuses or statuses[-1]!="0":
        raise RuntimeError("Proxy command failed on "+pod["id"]+": "+result.stdout[-1500:])
    return result.stdout


def download(pod, remote_path, destination, root):
    if pod.get("ssh_host"):
        subprocess.run(["scp","-q","-o","BatchMode=yes","-P",str(pod["ssh_port"]),
                        "root@"+pod["ssh_host"]+":"+remote_path,str(destination.parent)],check=True,timeout=900)
        return
    code="910-"+secrets.token_hex(12)
    log="/workspace/transfer-"+secrets.token_hex(6)+".log"
    ssh(pod,"nohup $(test -x /workspace/runpodctl && echo /workspace/runpodctl || echo runpodctl) send "+shlex.quote(remote_path)+" --code "+code+
        " > "+log+" 2>&1 < /dev/null &")
    actual=None
    for _ in range(30):
        output=ssh(pod,"cat "+log)
        match=re.search(r"runpodctl receive (\S+)",output)
        if match: actual=match[1];break
        sleep(1)
    if actual is None: raise RuntimeError("Transfer sender did not become ready")
    result=subprocess.run([str(root/"bin/runpodctl"),"receive",actual],cwd=destination.parent,
                          input="y\n",text=True,capture_output=True,timeout=900,check=True)
    if not destination.exists():raise RuntimeError("Expected transfer file absent: "+result.stdout[-1000:])


class RemoteControl:
    def __init__(self,url,token,ledger):self.url,self.token,self.ledger=url,token,ledger
    def request(self,body):
        request=urllib.request.Request(self.url,json.dumps(body).encode(),
            {"Content-Type":"application/json","User-Agent":"Mozilla/5.0 (compatible; HU20Arena/1.0)","Authorization":"Bearer "+self.token})
        with urllib.request.urlopen(request,timeout=10) as response:return json.load(response)
    def charge(self):
        ledger=json.loads(self.ledger.read_text())
        return ledger.get("historical_cap_charge_usd",0)+ledger.get("pilot_reserve_usd",.5)+ledger.get("storage_contingency_usd",1)+sum(
            max(0,p.get("terminated_at",time())-p["created_at"])/3600*p["hourly_usd"]
            for p in ledger["pods"])-ledger.get("owner_excluded_charge_usd",0)


def pod_status(pod):
    output = ssh(pod, "python3 - <<'PY'\n"
        "import json\nfrom pathlib import Path\nr=Path('/workspace/evidence')\n"
        "workers=[]\n"
        "for p in sorted((r/'arena').glob('worker-*/status.json')):\n"
        " x=json.loads(p.read_text());x['worker']=p.parent.name;workers.append(x)\n"
        "print('HU20_STATUS='+json.dumps({'preflight_passed':(r/'PREFLIGHT_PASSED').exists(),"
        "'preflight_failed':(r/'PREFLIGHT_FAILED').exists(),'workers':workers,"
        "'resource_failures':[str(p) for p in r.rglob('resource-guard-failure-*.json')],"
        "'hang_attention':[str(p) for p in r.rglob('hang-attention.json')]}))\nPY")
    return json.loads(re.search(r"HU20_STATUS=(\{[^\n]+\})",output)[1])


def cancel_owned_workers(pod):
    command = "python3 - <<'PY'\nimport os,signal,time\nfrom pathlib import Path\n"
    command += "pids=[]\nfor p in Path('/workspace/evidence/arena').glob('worker-*.pid'):\n"
    command += " pid=int(p.read_text());pids.append(pid)\n"
    command += " try:os.killpg(pid,signal.SIGTERM)\n except ProcessLookupError:pass\n"
    # Stop only descendants of this arena's exact preflight script, so partial
    # setup/parity files stop changing before hashing and archive creation.
    command += "parents={};roots=[]\nfor p in Path('/proc').glob('[0-9]*/cmdline'):\n"
    command += " try:\n  args=p.read_bytes().split(b'\\0');pid=int(p.parent.name);stat=(p.parent/'stat').read_text();parents[pid]=int(stat[stat.rfind(')')+2:].split()[1])\n"
    command += " except (OSError,ValueError):continue\n"
    command += " if len(args)>1 and args[0].endswith(b'bash') and args[1]==b'/workspace/bundle/preflight.sh':roots.append(pid)\n"
    command += "owned=set(roots)\nwhile True:\n more={pid for pid,parent in parents.items() if parent in owned}-owned\n if not more:break\n owned.update(more)\n"
    command += "for pid in sorted(owned,reverse=True):\n try:os.kill(pid,signal.SIGTERM)\n except ProcessLookupError:pass\n"
    command += "time.sleep(10)\nfor pid in pids:\n"
    command += " try:os.killpg(pid,signal.SIGKILL)\n except ProcessLookupError:pass\n"
    command += "for pid in sorted(owned,reverse=True):\n try:os.kill(pid,signal.SIGKILL)\n except ProcessLookupError:pass\nPY"
    ssh(pod, command)


def closeout(pod, root, mcp_call, ledger, lock):
    if pod["id"] in PROTECTED:
        raise ValueError("Protected historical pod")
    if pod.get("owner_handoff"):
        proof=json.loads((root/pod["owner_handoff"]).read_text())
        if not proof.get("owner_authorized") or pod["id"] not in proof["named_pods"]:
            raise ValueError("Explicit owner handoff attribution differs")
    else:
        proof = json.loads((root/pod["creation_receipt"]).read_text())
        created = json.loads(proof["response"]["result"]["content"][0]["text"])
        if created["id"] != pod["id"]:
            raise ValueError("Creation attribution differs")
    cancel_owned_workers(pod)
    # The preflight outputs and every arena file, including interrupted hands and
    # partial solver outputs, stay inside the persistent /workspace evidence tree.
    command = "if test -f /workspace/repo/scripts/hu20_search_evidence.py && test -x /workspace/venv/bin/python; then "
    command += "cd /workspace/repo && . /workspace/venv/bin/activate && export PYTHONPATH=/workspace/repo && "
    command += "for worker in /workspace/evidence/arena/worker-*/solver; do "
    command += "test -d \"$worker\" || continue; name=$(basename \"$(dirname \"$worker\")\"); "
    command += "python -m scripts.replay_hu20_search_requests --binary /workspace/hu20-turn-search-tool/harness/target/release/hu20-exact-flop-tool "
    command += "--solves \"$worker\" --out \"/workspace/evidence/sampled-replay/$name\" > \"/workspace/evidence/sampled-replay-$name.log\" 2>&1 "
    command += "|| printf 'Numerical replay failed; retain evidence\\n' >> /workspace/evidence/replay-failures.log; done && "
    command += "python -m scripts.hu20_search_evidence finalize /workspace/evidence --owned-pod-root /workspace > /workspace/closeout-retention.json && "
    command += "mv /workspace/closeout-retention.json /workspace/evidence/closeout-retention.json && "
    command += "python -m scripts.hu20_search_evidence retention-check /workspace/evidence > /workspace/retention-check.json && "
    command += "mv /workspace/retention-check.json /workspace/evidence/closeout-retention-check.json && "
    command += "python -m scripts.hu20_search_evidence pack /workspace/evidence /workspace/archives > /workspace/pack.log; "
    command += "else test -z \"$(find /workspace/evidence -name profile.jsonl -print -quit)\" && "
    command += "PYTHONPATH=/workspace/bundle/ops python3 -m scripts.hu20_search_evidence pack /workspace/evidence /workspace/archives > /workspace/pack.log; fi"
    # This is a closeout operation, bounded by the explicitly reserved clock.
    ssh(pod, command, timeout=5400)
    destination = root/"retrieved"/pod["id"]
    destination.mkdir(parents=True, exist_ok=False)
    download(pod,"/workspace/archives/manifest.json",destination/"manifest.json",root)
    manifest = json.loads((destination/"manifest.json").read_text())
    size = sum(r["bytes"] for r in manifest["archives"])
    with lock:
        ledger["retrieval_reserved_bytes"] = ledger.get("retrieval_reserved_bytes", 0)+size
        if ledger["retrieval_reserved_bytes"] > 4*10**9 or shutil.disk_usage(root).free-size < 5*1024**3:
            raise OSError("Predeclared retrieval size/free-space guard")
        durable_json(root/"ledger.json", ledger)
    for row in manifest["archives"]:
        if row["bytes"] > 10**9 or Path(row["path"]).name != row["path"]:
            raise ValueError("Unexpected transfer chunk")
        download(pod,"/workspace/archives/"+row["path"],destination/row["path"],root)
    verified = verify_archives(destination)
    durable_json(destination/"retrieval-verified.json", {**verified, "pod_id": pod["id"], "archive_manifest_sha256":file_hash(destination/"manifest.json"), "at": time()})
    readback = mcp_call("tools/call", {"name": "get-pod", "arguments": {"id": pod["id"]}}, 400)
    if readback["result"].get("isError"):
        raise RuntimeError("Cannot verify owned pod before termination")
    result = mcp_call("tools/call", {"name": "delete-pod", "arguments": {"id": pod["id"]}}, 401)
    durable_json(destination/"termination.json", result)
    if result["result"].get("isError"):
        raise RuntimeError("Owned pod termination failed")
    check = mcp_call("tools/call", {"name": "get-pod", "arguments": {"id": pod["id"]}}, 402)
    durable_json(destination/"termination-readback.json", check)
    missing = check["result"].get("isError") and any(
        word in json.dumps(check).lower() for word in ("not found", "404", "not_found"))
    if not missing:
        raise RuntimeError("Cannot confirm termination by an actual not-found readback")
    with lock:
        pod["terminated_at"] = time()
        pod["retrieval_verified"] = True
        durable_json(root/"ledger.json", ledger)
    return pod["id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mcp-helper", type=Path, required=True)
    parser.add_argument("--remote-control")
    parser.add_argument("--token-file",type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    spec = importlib.util.spec_from_file_location("runpod_session", args.mcp_helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    if args.remote_control:
        control=RemoteControl(args.remote_control,args.token_file.read_text().strip(),root/"ledger.json")
    else:
        control = ArenaControl(root/"control.json", root/"ledger.json")
        threading.Thread(target=serve, args=(control, 18066), daemon=True).start()
    durable_json(root/"SUPERVISOR_READY.json", {"pid": __import__('os').getpid(), "at": time()})
    ever_started, finished = False, False
    failures = 0
    first_failure = None
    while not finished:
        sleep(5)
        ledger = json.loads((root/"ledger.json").read_text())
        active = [p for p in ledger["pods"] if not p.get("terminated_at")]
        try:
            state = control.request({"op": "check"})
            with ThreadPoolExecutor(max_workers=len(active)) as pool:
                statuses=list(pool.map(pod_status,active))
            durable_json(root/"live-status.json", {"control": state, "hosts": statuses, "at": time()})
            if any(s["preflight_failed"] or s["resource_failures"] or any(
                    w["status"] == "incomplete" for w in s["workers"]) for s in statuses):
                control.request({"op": "stop", "reason": "Actual-host preflight/worker/resource failure"})
            ever_started |= state["status"] == "running"
            state=control.request({"op":"check"})
            finished = (state["status"] == "stopped" or ever_started and len(statuses)==len(active) and
                        all(len(s["workers"])==len(p["workers"]) and
                            all(w["status"]=="complete" for w in s["workers"])
                            for p,s in zip(active,statuses)))
            failures = 0
            first_failure = None
        except Exception as exc:
            failures += 1
            first_failure = first_failure or time()
            durable_json(root/"supervisor-poll-failure.json", {"failures": failures, "reason": str(exc), "at": time()})
            # Forwards and SSH reconnect on their own; only a sustained outage ends the run.
            if failures >= 3 and time()-first_failure >= 1800:
                try:control.request({"op": "stop", "reason": "Supervision unavailable for thirty minutes"})
                except Exception:pass  # Cancel all owned groups during closeout even if the controller is unreachable.
                finished = True
    # Read the persisted stop reason without restarting the scientific run.
    try:state=control.request({"op":"check"})
    except Exception:state={"status":"stopped","reason":"Controller unavailable; cancel owned work and retain partials"}
    durable_json(root/"final-control.json",state)
    try:control.request({"op": "stop", "reason": state.get("reason") or "All frozen workers complete; closeout"})
    except Exception:pass
    ledger = json.loads((root/"ledger.json").read_text())
    lock = threading.Lock()
    try:
        active=[p for p in ledger['pods'] if not p.get('terminated_at')]
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(cancel_owned_workers,active))
        for p in active:
            if p['id']==ledger.get('controller_pod_id','m5pxmipuqyjtoo'):
                ssh(p,"python3 - <<'FREEZE'\nimport os,signal,shutil\nfrom pathlib import Path\np=Path('/workspace/controller-v2.pid')\nif p.exists():\n pid=int(p.read_text())\n try:\n  args=Path('/proc/'+str(pid)+'/cmdline').read_bytes().split(b'\\0')\n  assert b'scripts.hu20_search_arena_control' in args and b'/workspace/evidence/control-v2.json' in args\n  os.kill(pid,signal.SIGTERM)\n except FileNotFoundError:pass\nfor p in Path('/workspace').glob('controller*.log'):\n shutil.copy2(p,Path('/workspace/evidence')/p.name)\nFREEZE")
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = [pool.submit(closeout, pod, root, helper.call, ledger, lock)
                    for pod in ledger["pods"] if not pod.get("terminated_at")]
            completed = [job.result() for job in jobs]
        listed=helper.call("tools/call",{"name":"list-pods","arguments":{}},410)
        durable_json(root/"final-list-pods.json",listed)
        durable_json(root/"CLOSEOUT_COMPLETE.json", {"pods": completed, "charge_upper_usd": control.charge(), "at": time()})
    except Exception as exc:
        durable_json(root/"CLOSEOUT_FAILED.json", {"reason": str(exc), "at": time(),
            "instruction": "Preserve unretrieved evidence; owner intervention required; do not restart science"})
        raise


if __name__ == "__main__":
    main()
