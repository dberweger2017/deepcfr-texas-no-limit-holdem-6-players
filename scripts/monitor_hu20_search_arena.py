"""Local durable supervisor for the three pods attributed by creation receipts.

This never allocates or restarts work. Controller loss halts the remote workers;
the supervisor reserves closeout time, retrieves exact evidence, then terminates
only pods whose creation response and ledger agree.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import threading
from time import sleep, time

from scripts.hu20_search_arena_control import ArenaControl, durable_json, serve
from scripts.hu20_search_evidence import verify_archives

PROTECTED = {"43z4itur3hwnyv", "cl0riravggku4r", "xu414eguzakxfr", "k9rdph2fwhym87"}


def ssh(pod, command, *, timeout=45):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                           "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
                           "-p", str(pod["ssh_port"]), "root@"+pod["ssh_host"], command],
                          capture_output=True, text=True, timeout=timeout, check=True).stdout


def pod_status(pod):
    return json.loads(ssh(pod, "python3 - <<'PY'\n"
        "import json\nfrom pathlib import Path\nr=Path('/workspace/evidence')\n"
        "workers=[]\n"
        "for p in sorted((r/'arena').glob('worker-*/status.json')):\n"
        " x=json.loads(p.read_text());x['worker']=p.parent.name;workers.append(x)\n"
        "print(json.dumps({'preflight_passed':(r/'PREFLIGHT_PASSED').exists(),"
        "'preflight_failed':(r/'PREFLIGHT_FAILED').exists(),'workers':workers,"
        "'resource_failures':[str(p) for p in r.rglob('resource-guard-failure-*.json')]}))\nPY"))


def cancel_owned_workers(pod):
    command = "python3 - <<'PY'\nimport os,signal,time\nfrom pathlib import Path\n"
    command += "pids=[]\nfor p in Path('/workspace/evidence/arena').glob('worker-*.pid'):\n"
    command += " pid=int(p.read_text());pids.append(pid)\n"
    command += " try:os.killpg(pid,signal.SIGTERM)\n except ProcessLookupError:pass\n"
    command += "time.sleep(10)\nfor pid in pids:\n"
    command += " try:os.killpg(pid,signal.SIGKILL)\n except ProcessLookupError:pass\nPY"
    ssh(pod, command)


def closeout(pod, root, mcp_call, ledger, lock):
    if pod["id"] in PROTECTED:
        raise ValueError("Protected historical pod")
    proof = json.loads((root/pod["creation_receipt"]).read_text())
    created = json.loads(proof["response"]["result"]["content"][0]["text"])
    if created["id"] != pod["id"]:
        raise ValueError("Creation attribution differs")
    cancel_owned_workers(pod)
    # The preflight outputs and every arena file, including interrupted hands and
    # partial solver outputs, stay inside the persistent /workspace evidence tree.
    command = "if test -d /workspace/repo && test -x /workspace/venv/bin/python; then "
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
    base = ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-P", str(pod["ssh_port"])]
    remote = "root@"+pod["ssh_host"]+":/workspace/archives/"
    subprocess.run([*base, remote+"manifest.json", str(destination)], check=True, timeout=120)
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
        subprocess.run([*base, remote+row["path"], str(destination)], check=True, timeout=900)
    verified = verify_archives(destination)
    durable_json(destination/"retrieval-verified.json", {**verified, "pod_id": pod["id"], "at": time()})
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
    args = parser.parse_args()
    root = args.root.resolve()
    spec = importlib.util.spec_from_file_location("runpod_session", args.mcp_helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    control = ArenaControl(root/"control.json", root/"ledger.json")
    threading.Thread(target=serve, args=(control, 18066), daemon=True).start()
    durable_json(root/"SUPERVISOR_READY.json", {"pid": __import__('os').getpid(), "at": time()})
    ever_started, finished = False, False
    failures = 0
    while not finished:
        sleep(5)
        ledger = json.loads((root/"ledger.json").read_text())
        active = [p for p in ledger["pods"] if not p.get("terminated_at")]
        try:
            state = control.request({"op": "check"})
            statuses = [pod_status(p) for p in active if p.get("ssh_host")]
            durable_json(root/"live-status.json", {"control": state, "hosts": statuses, "at": time()})
            if any(s["preflight_failed"] or s["resource_failures"] or any(
                    w["status"] == "incomplete" for w in s["workers"]) for s in statuses):
                control.request({"op": "stop", "reason": "Actual-host preflight/worker/resource failure"})
            ever_started |= state["status"] == "running"
            finished = (control.state["status"] == "stopped" or ever_started and len(statuses) == 3 and
                        all(len(s["workers"]) == 3 and all(w["status"] == "complete" for w in s["workers"]) for s in statuses))
            failures = 0
        except Exception as exc:
            failures += 1
            durable_json(root/"supervisor-poll-failure.json", {"failures": failures, "reason": str(exc), "at": time()})
            if failures >= 3:
                control.request({"op": "stop", "reason": "Three consecutive supervision failures"})
                finished = True
    control.request({"op": "stop", "reason": control.state["reason"] or "All frozen workers complete; closeout"})
    ledger = json.loads((root/"ledger.json").read_text())
    lock = threading.Lock()
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = [pool.submit(closeout, pod, root, helper.call, ledger, lock)
                    for pod in ledger["pods"] if not pod.get("terminated_at")]
            completed = [job.result() for job in jobs]
        durable_json(root/"CLOSEOUT_COMPLETE.json", {"pods": completed, "charge_upper_usd": control.charge(), "at": time()})
    except Exception as exc:
        durable_json(root/"CLOSEOUT_FAILED.json", {"reason": str(exc), "at": time(),
            "instruction": "Preserve unretrieved evidence; owner intervention required; do not restart science"})
        raise


if __name__ == "__main__":
    main()
