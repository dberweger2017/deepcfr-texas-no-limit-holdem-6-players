"""Autonomous fixed-work arena: each pod runs its own controller and workers; the operator only checks in.

Block assignment is static (worker_index mod worker_count, fixed at dispatch), so a pod needs nothing
from any other pod or from the operator while it runs. Fallbacks and defects are recorded per decision
and never stop play; a crashed worker is restarted on its own partition from its last completed hand.
An operator check-in reads each pod's status, pulls compressed evidence deltas and lists new defects and
crashes for the PR. Operator connectivity loss never stops work.
"""
import argparse
import hashlib
import json
import re
import secrets
import shlex
import subprocess
from pathlib import Path
from time import time

from scripts.hu20_search_arena_control import durable_json
from scripts.monitor_hu20_search_arena import pod_status, ssh

PORT = 18066
# Files of the stopped first attempt are kept inside the evidence tree and never reused.
ATTEMPT_ONE = ("arena", "control-v2.json", "control-v2.events.jsonl", "closeout-retention.json",
               "closeout-retention-check.json", "replay-failures.log", "sampled-replay", "preflight-controller-journal")


def pod_ledger(pod, workers):
    return {"work_protocol": "hu20-fixed50-no-fallback-v1", "autonomous_pod": True, "expected_active_pods": 1,
            "pods": [{"id": pod["id"], "workers": workers, "created_at": pod["created_at"],
                      "hourly_usd": pod["hourly_usd"], "parity_retention_passed": True}]}


MAX_RESTARTS = 25


def worker_command(pod_id, worker, worker_count, *, arena="/workspace/evidence/arena", pause=30, command=None):
    """One loop per worker: run, and after a crash resume the same static partition (never the campaign)."""
    stop_file = str(Path(arena).parent / "STOP_REQUEST")
    run_worker = command or ("cd /workspace/repo; . /workspace/venv/bin/activate; export PYTHONPATH=/workspace/repo OMP_NUM_THREADS=1 "
                  "OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; "
                  f"export HU20_CONTROL_URL=http://127.0.0.1:{PORT} HU20_POD_ID={pod_id} HU20_WORKER_INDEX={worker} "
                  f"HU20_WORKER_EPOCH=$1 HU20_DEFECT_LOG={arena}/worker-{worker}.defects.jsonl; "
                  "export HU20_CONTROL_TOKEN=$(cat /workspace/control-token); "
                  "exec python -m scripts.run_hu20_search_arena_guarded --plan /workspace/bundle/arena-plan.json "
                  "--inputs /workspace/bundle/inputs --phase arena --binary "
                  "/workspace/hu20-turn-search-tool/harness/target/release/hu20-exact-flop-tool "
                  "--search-config /workspace/bundle/config.json --paid-approval /workspace/paid-approval.json "
                  f"--worker-index {worker} --worker-count {worker_count} --out {arena}/worker-{worker} $2")
    loop = (f"epoch=0\nwhile true; do\n  extra=''; if [ $epoch -gt 0 ]; then extra='--resume'; fi\n"
            f"  bash -c {shlex.quote(run_worker)} _ $epoch $extra >> {arena}/worker-{worker}.launch.log 2>&1 < /dev/null\n"
            f"  code=$?\n  if [ $code -eq 0 ]; then break; fi\n"
            f"  echo \"$epoch $code $(date +%s)\" >> {arena}/worker-{worker}.crashes.log\n"
            f"  if [ -e {stop_file} ]; then break; fi\n"
            f"  epoch=$((epoch+1))\n  if [ $epoch -gt {MAX_RESTARTS} ]; then touch {arena}/worker-{worker}.GAVE_UP; break; fi\n"
            f"  sleep {pause}\ndone\n")
    return (f"setsid bash -c {shlex.quote(loop)} > /dev/null 2>&1 < /dev/null &\n"
            f"echo $! > {arena}/worker-{worker}.pid\n")


def launch_script(pod_id, workers, worker_count):
    script = ("set -eu\nE=/workspace/evidence\ntest ! -e $E/ARENA_AUTONOMOUS_STARTED\nmkdir -p $E/arena\n"
              "touch $E/ARENA_AUTONOMOUS_STARTED\ncd /workspace/control-repo\n"
              f"setsid nohup python3 -m scripts.hu20_search_arena_control --journal $E/control-pod.json "
              f"--ledger /workspace/ledger-pod.json --bind 127.0.0.1 --port {PORT} --token-file /workspace/control-token "
              "> /workspace/controller-pod.log 2>&1 < /dev/null &\necho $! > /workspace/controller-pod.pid\nsleep 3\n"
              "curl -sf -m 5 -X POST -H \"Authorization: Bearer $(cat /workspace/control-token)\" "
              "-H 'User-Agent: Mozilla/5.0 (compatible; HU20Arena/1.0)' -d '{\"op\":\"start\"}' "
              f"http://127.0.0.1:{PORT}/ | grep -q '\"status\": \"running\"'\n")
    for worker in workers:
        script += worker_command(pod_id, worker, worker_count)
    return script + "echo AUTONOMOUS_LAUNCHED\n"


def archive_attempt_one_command():
    moves = " ".join(shlex.quote(name) for name in ATTEMPT_ONE)
    return ("set -eu\ncd /workspace/evidence\ntest ! -e attempt-1\nmkdir attempt-1\n"
            f"for n in {moves}; do if [ -e \"$n\" ]; then mv \"$n\" attempt-1/; fi; done\n"
            "if [ -e /workspace/archives ]; then mv /workspace/archives /workspace/attempt-1-archives; fi\n"
            "rm -f POD_STOPPED.json STOP_REQUEST\n"
            "pkill -f '[t]unnel.sh' || true; pkill -f '[s]sh -N' || true\n"
            "if [ -f /workspace/controller-v2.pid ]; then kill $(cat /workspace/controller-v2.pid) 2>/dev/null || true; fi\n"
            "echo ATTEMPT_ONE_ARCHIVED\n")


STATUS = """python3 - <<'HOST'
import json,os
from pathlib import Path
r=Path('/workspace/evidence')
def load(p):
    try:return json.loads(p.read_text())
    except Exception:return None
workers=[]
for p in sorted((r/'arena').glob('worker-*/status.json')):
    x=load(p) or {}
    x['worker']=p.parent.name;workers.append(x)
journal=load(r/'control-pod.json') or {}
def lines(p):
    try:return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    except Exception:return []
defects=[];crashes=[];gave_up=[]
for p in sorted((r/'arena').glob('worker-*.defects.jsonl')):
    defects+=lines(p)
for p in sorted((r/'arena').glob('worker-*.crashes.log')):
    for x in p.read_text().splitlines():
        f=x.split()
        if len(f)==3:crashes.append({'worker':p.name.split('.')[0],'epoch':int(f[0]),'exit':int(f[1]),'at':int(f[2])})
gave_up=[p.name.split('.')[0] for p in sorted((r/'arena').glob('worker-*.GAVE_UP'))]
pid=None
try:
    pid=int(Path('/workspace/controller-pod.pid').read_text());alive=os.path.exists('/proc/%d'%pid)
except Exception:alive=False
st=os.statvfs('/workspace')
print('HU20_STATUS='+json.dumps({'workers':workers,'journal':{k:journal.get(k) for k in ('status','reason','decisions','fallbacks','stop_kind')},
 'marker':load(r/'POD_STOPPED.json'),'defect_count':len(defects),'defects_tail':defects[-200:],
 'crashes':crashes,'gave_up':gave_up,'stop_request':(r/'STOP_REQUEST').exists(),'controller_alive':alive,
 'free_bytes':st.f_bavail*st.f_frsize,'resource_failures':[str(p) for p in r.rglob('resource-guard-failure-*.json')],
 'hang_attention':[str(p) for p in r.rglob('hang-attention.json')],'at':__import__('time').time()}))
HOST"""


def read_status(pod):
    out = ssh(pod, STATUS, timeout=90)
    return json.loads(re.search(r"HU20_STATUS=(\{[^\n]+\})", out)[1])


def pod_done(status, expected_workers):
    if status["marker"] or status["journal"].get("status") == "stopped":
        return "stopped"
    done = [w for w in status["workers"] if w.get("status") == "complete"]
    if len(done) == expected_workers:
        return "complete"
    # A worker that exhausted its restarts is finished but incomplete; wait for the rest of the pod first.
    finished = len(done) + len(status["gave_up"])
    return "incomplete" if finished == expected_workers else None


def spend_record(ledger, now):
    pods = [{"id": p["id"], "hours": (p.get("terminated_at", now) - p["created_at"]) / 3600,
             "usd": (p.get("terminated_at", now) - p["created_at"]) / 3600 * p["hourly_usd"]} for p in ledger["pods"]]
    return {"at": now, "fleet_pods": pods, "fleet_new_usd_since_allocation": sum(p["usd"] for p in pods),
            "basis": "provisioning wall clock x readback rates; record only, no spend limit applies"}


def run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, timeout=kwargs.pop("timeout", 1800), **kwargs)
    # rsync exit 24 means source files vanished mid-transfer (live evidence files being renamed); the next pass has them.
    if result.returncode != 0 and not (command[0] == "rsync" and result.returncode == 24):
        raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
    return result


def route(pod, ledger):
    """Return the (direct) pod that serves this pod's files: itself, or its relay host."""
    if pod.get("ssh_host") and pod["ssh_host"] != "127.0.0.1":
        return pod, f"/workspace/evidence"
    relay = next(p for p in ledger["pods"] if p["id"] == ledger["relay_pod_id"])
    return relay, f"/workspace/relay/{pod['id']}"


def pull(pod, ledger, root, last_epoch):
    """Compressed delta of new solver files plus a mirror of everything else; returns the new epoch."""
    host, remote = route(pod, ledger)
    now = int(time())
    delta = f"/workspace/deltas/{pod['id']}-{now}.tar.gz"
    pack = ("mkdir -p /workspace/deltas && cd /workspace/evidence && find arena -path '*/solver/*' -type f "
            f"-newermt @{max(0, last_epoch - 600)} -print0 | tar --null --no-recursion -czf {delta} -T - && gzip -t {delta} "
            "&& sha256sum " + delta)
    out = ssh(pod, pack, timeout=900)
    digest = re.search(r"([0-9a-f]{64})  " + re.escape(delta), out)[1]
    if host is not pod:  # proxy-only pod: push to the relay pod over the private pod-to-pod link
        ssh(host, f"mkdir -p /workspace/relay/{pod['id']}-deltas /workspace/relay/{pod['id']}")
        key = f"-e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p {host['ssh_port']}'"
        ssh(pod, f"mkdir -p /tmp/relay-ok; rsync -a --partial {key} {delta} root@{host['ssh_host']}:/workspace/relay/{pod['id']}-deltas/ "
                 f"&& rsync -a --partial {key} --exclude 'arena/*/solver/' --exclude 'replay/' --exclude 'checker/processes' "
                 f"/workspace/evidence/ root@{host['ssh_host']}:{remote}/", timeout=1500)
        delta_remote = f"/workspace/relay/{pod['id']}-deltas/{Path(delta).name}"
    else:
        delta_remote = delta
    base = root / "incremental" / pod["id"]
    (base / "mirror").mkdir(parents=True, exist_ok=True)
    (base / "deltas").mkdir(parents=True, exist_ok=True)
    ssh_opts = f"ssh -o BatchMode=yes -p {host['ssh_port']}"
    local = base / "deltas" / Path(delta).name
    run(["rsync", "-a", "--partial", "--timeout=300", "-e", ssh_opts, f"root@{host['ssh_host']}:{delta_remote}", str(local)])
    sha = hashlib.sha256(local.read_bytes()).hexdigest()
    if sha != digest:
        local.unlink()
        raise ValueError("Delta hash differs after transfer")
    with (base / "deltas" / "SHA256SUMS").open("a") as stream:
        stream.write(f"{sha}  {local.name}\n")
    run(["rsync", "-a", "--partial", "--timeout=300", "-e", ssh_opts, "--exclude", "arena/*/solver/", "--exclude", "replay/",
         "--exclude", "checker/processes", f"root@{host['ssh_host']}:{remote}/", str(base / "mirror") + "/"])
    ssh(pod, f"find /workspace/deltas -name '{pod['id']}-*.tar.gz' ! -name '{Path(delta).name}' -delete", timeout=60)
    return now


def checkin(root):
    ledger = json.loads((root / "ledger.json").read_text())
    pods = [p for p in ledger["pods"] if not p.get("terminated_at")]
    now = time()
    report = {"at": now, "pods": {}}
    state_path = root / "checkin-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"last_epoch": {}}
    for pod in pods:
        entry = report["pods"][pod["id"]] = {}
        try:
            status = read_status(pod)
            entry["status"] = status
            entry["done"] = pod_done(status, len(pod["workers"]))
        except Exception as exc:  # connectivity never stops the run; try again next check-in
            entry["status_error"] = str(exc)[-300:]
            continue
        try:
            state["last_epoch"][pod["id"]] = pull(pod, ledger, root, state["last_epoch"].get(pod["id"], 0))
        except Exception as exc:
            entry["pull_error"] = str(exc)[-300:]
    # Defects never stop play. Anything new since the last check-in is queued for a PR comment.
    seen = state.setdefault("reported", {"defects": {}, "crashes": {}})
    new = {"defects": [], "crashes": [], "stopped": []}
    for pod_id, entry in report["pods"].items():
        status = entry.get("status")
        if not status:
            continue
        count = seen["defects"].get(pod_id, 0)
        tail = status["defects_tail"]
        fresh = tail[-(status["defect_count"] - count):] if status["defect_count"] > count else []
        new["defects"] += [dict(row, pod=pod_id) for row in fresh]
        seen["defects"][pod_id] = status["defect_count"]
        known = seen["crashes"].get(pod_id, 0)
        new["crashes"] += [dict(row, pod=pod_id) for row in status["crashes"][known:]]
        seen["crashes"][pod_id] = len(status["crashes"])
        if status["marker"] and pod_id not in seen.setdefault("stopped", []):
            new["stopped"].append({"pod": pod_id, **status["marker"]})
            seen["stopped"].append(pod_id)
    report["new_since_last_checkin"] = new
    if any(new.values()):
        with (root / "NEW_ISSUES.jsonl").open("a") as stream:
            stream.write(json.dumps({"at": now, **new}, sort_keys=True) + "\n")
    report["spend"] = spend_record(ledger, now)
    report["all_done"] = all(report["pods"].get(p["id"], {}).get("done") for p in pods)
    durable_json(state_path, state)
    durable_json(root / "live-status.json", report)
    with (root / "checkins.jsonl").open("a") as stream:
        stream.write(json.dumps(report, sort_keys=True) + "\n")
    if report["all_done"] and not (root / "ALL_PODS_DONE.json").exists():
        durable_json(root / "ALL_PODS_DONE.json", {"at": now, "pods": {p["id"]: report["pods"][p["id"]]["done"] for p in pods}})
    return report


# Scientific/runner files overlaid on the approved-source checkout of every pod; hashes are recorded at launch.
OVERLAY = {
    "scripts/hu20_search_arena_control.py": ["/workspace/control-repo/scripts/", "/workspace/repo/scripts/"],
    "scripts/run_hu20_search_arena_guarded.py": ["/workspace/repo/scripts/"],
    "scripts/evaluate_hu20_turn_search.py": ["/workspace/repo/scripts/"],
    "src/blueprint/hu20_turn_search.py": ["/workspace/repo/src/blueprint/"],
}


def push(pod, relay, root, data, remote):
    """Copy bytes to a pod path. Proxy-only pods cannot take bulk stdin, so they pull from the relay pod."""
    name = hashlib.sha256(data).hexdigest()
    stage = root / "stage"
    stage.mkdir(exist_ok=True)
    (stage / name).write_bytes(data)
    direct = pod.get("ssh_host") and pod["ssh_host"] != "127.0.0.1"
    target = pod if direct else relay
    remote_stage = f"/workspace/stage/{name}"
    if not direct or pod is relay:
        ssh(relay, "mkdir -p /workspace/stage")
        run(["scp", "-q", "-o", "BatchMode=yes", "-P", str(relay["ssh_port"]), str(stage / name),
             f"root@{relay['ssh_host']}:{remote_stage}"])
    if direct:
        ssh(pod, f"mkdir -p {shlex.quote(str(Path(remote).parent))}")
        run(["scp", "-q", "-o", "BatchMode=yes", "-P", str(pod["ssh_port"]), str(stage / name), f"root@{pod['ssh_host']}:{remote}"])
    else:
        ssh(pod, f"mkdir -p {shlex.quote(str(Path(remote).parent))} && scp -q -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
                 f"-P {relay['ssh_port']} root@{relay['ssh_host']}:{remote_stage} {shlex.quote(remote)}", timeout=300)
    out = ssh(pod, f"sha256sum {shlex.quote(remote)}", timeout=60)
    if name not in out:
        raise ValueError("Uploaded file hash differs on " + pod["id"] + ": " + remote)
    return name


def launch(root, source_sha, owner_checks, repo):
    """Archive attempt one on every pod, install the autonomous bundle and start each pod's own arena."""
    ledger = json.loads((root / "ledger.json").read_text())
    pods = [p for p in ledger["pods"] if not p.get("terminated_at")]
    relay = next(p for p in pods if p["id"] == ledger["relay_pod_id"])
    worker_count = sum(len(p["workers"]) for p in pods)
    overlay = {rel: (repo / rel).read_bytes() for rel in OVERLAY}
    receipts = {}
    for pod in pods:
        ssh(pod, archive_attempt_one_command(), timeout=300)
        for rel, directories in OVERLAY.items():
            for directory in directories:
                push(pod, relay, root, overlay[rel], directory + Path(rel).name)
        ssh(pod, "touch /workspace/control-repo/scripts/__init__.py")
        push(pod, relay, root, secrets.token_hex(32).encode(), "/workspace/control-token")
        push(pod, relay, root, json.dumps(pod_ledger(pod, pod["workers"])).encode(), "/workspace/ledger-pod.json")
        push(pod, relay, root, json.dumps(pod["paid_approval"], sort_keys=True).encode(), "/workspace/paid-approval.json")
        push(pod, relay, root, launch_script(pod["id"], pod["workers"], worker_count).encode(), "/workspace/launch-autonomous.sh")
        ssh(pod, "chmod 600 /workspace/control-token")
        receipts[pod["id"]] = {"workers": pod["workers"]}
    ssh(relay, "rm -rf /workspace/stage")
    (root / "stage").exists() and [f.unlink() for f in (root / "stage").iterdir()]
    durable_json(root / "ARENA_LAUNCH.json", {"at": time(), "status": "dispatching", "source": source_sha,
        "autonomous": True, "worker_count": worker_count, "pods": receipts,
        "overlay_sha256": {rel: hashlib.sha256(data).hexdigest() for rel, data in overlay.items()}, **owner_checks})
    for pod in pods:
        out = ssh(pod, "bash /workspace/launch-autonomous.sh", timeout=120)
        assert "AUTONOMOUS_LAUNCHED" in out, out[-500:]
    launched = json.loads((root / "ARENA_LAUNCH.json").read_text())
    launched.update(status="launched", launched_at=time())
    durable_json(root / "ARENA_LAUNCH.json", launched)
    return worker_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="op", required=True)
    sub.add_parser("checkin").add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(checkin(args.root.resolve()), sort_keys=True)[:4000])


if __name__ == "__main__":
    main()
