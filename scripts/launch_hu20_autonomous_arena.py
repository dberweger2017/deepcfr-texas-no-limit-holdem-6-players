"""One-time launch of the held fleet as independent pods after every actual-host gate.

Same frozen science as launch_hu20_fixed_work_arena (approved source, plan, quote, exact parity,
50 complete iterations, zero fallback tolerance); only orchestration differs. The owner removed
spend limits and handles billing, so the measured quote is recorded but never gates dispatch.
"""
import argparse
from math import ceil
import hashlib
import json
import re
from pathlib import Path
from time import time

from src.arena.schedule import digest
from scripts.hu20_autonomous_arena import launch
from scripts.hu20_search_arena_control import durable_json
from scripts.launch_hu20_fixed_work_arena import PLAN, PROTOCOL, QUOTE, SOURCE, actual_quote, validate_gate
from scripts.monitor_hu20_search_arena import PROTECTED, ssh
from scripts.quote_hu20_held_fleet import READ
from scripts.quote_hu20_fixed_work_arena import admitted_workers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / "ARENA_LAUNCH.json").exists():
        raise ValueError("Preserve the earlier launch receipt (rename it) before a new fresh run")
    owner = json.loads((root / "owner-approval.json").read_text())
    ci = json.loads((root / "CI_GREEN.json").read_text())
    ops_ci = json.loads((root / "OPS_CI_GREEN.json").read_text())
    assert owner["owner_approved"] and owner["source"] == SOURCE and owner["protocol"] == PROTOCOL
    assert owner["quote_sha256"] == QUOTE and owner["plan_sha256"] == PLAN and owner.get("spend_caps_removed")
    assert ci["head_sha"] == SOURCE and ci["conclusion"] == "success"
    assert hashlib.sha256((root / "approved-quote-original.json").read_bytes()).hexdigest() == QUOTE
    assert digest(json.loads((root / "plan.json").read_text())) == PLAN
    ledger = json.loads((root / "ledger.json").read_text())
    assert ops_ci["conclusion"] == "success" and ops_ci["head_sha"] == ledger["operations_source"]
    fleet = json.loads((root / "approved-fleet-quote.json").read_text())
    assert hashlib.sha256((root / "approved-fleet-quote.json").read_bytes()).hexdigest() == owner["fleet_quote_sha256"]
    pods = [h for h in ledger["pods"] if not h.get("terminated_at")]
    assert len(pods) == fleet["pods"] and not any(h["id"] in PROTECTED for h in pods)
    assert all(h["gpu_id"] in fleet["gpu_ids"] and h["gpu_count"] == fleet["gpu_count"] for h in pods)
    first = 0
    for pod in pods:
        output = ssh(pod, READ, timeout=120)
        gate = json.loads(re.search(r"HU20_GATE=(\{[^\n]+\})", output)[1])
        validate_gate(gate, fleet["minimum_workers_per_pod"], fleet["minimum_ram_bytes"])
        durable_json(root / (pod["id"] + "-verified-gates.json"), gate)
        count = admitted_workers(gate["admission"]["quota_cpus"], gate["admission"]["admitted_ram_bytes"])
        pod.update(workers=list(range(first, first + count)), parity_sha256=digest(gate), replay=gate["replay"])
        first += count
        times = sorted(x["solver_seconds"] for x in gate["replay"]["rows"])
        pod["host_replay_p99_seconds"] = times[ceil(len(times) * .99) - 1]
        pod["paid_approval"] = {
            "owner_approved": True, "work_protocol": PROTOCOL, "arena_plan_sha256": PLAN,
            "search_config_sha256": digest(json.loads((root / "config.json").read_text())),
            "selected_settings_parity": "passed", "quote_sha256": QUOTE, "parity_sha256": pod["parity_sha256"],
            "rss_limit_bytes": 9 * 1024 ** 3, "host_replay_p99_seconds": pod["host_replay_p99_seconds"],
            "owner_approval": owner.get("owner_chat", "chat: go ahead for revision 3")}
    quote = actual_quote(json.loads((root / "approved-quote-original.json").read_text()), pods,
                         json.loads((root / "cost-only-results.json").read_text()), 0,
                         hard_ceiling=10 ** 9, dispatch_stop=10 ** 9)
    durable_json(root / "actual-host-quote-record.json", {**quote, "gate": "none: owner removed spend limits"})
    for h in ledger["pods"]:
        h.pop("replay", None)
    durable_json(root / "ledger.json", ledger)
    ledger["relay_pod_id"] = pods[0]["id"]
    durable_json(root / "ledger.json", ledger)
    workers = launch(root, SOURCE, {"operations_source": ledger["operations_source"], "plan_sha256": PLAN,
                                    "defect_policy": "record per decision and continue; crash restarts resume the same partition"},
                     Path(__file__).resolve().parent.parent)
    print(json.dumps({"launched_workers": workers, "at": time(), "expected_seconds_per_worker_estimate": quote["expected_seconds"]}))


if __name__ == "__main__":
    main()
