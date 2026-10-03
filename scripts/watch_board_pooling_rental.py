"""Independent provider cutoff for one exact, prospectively owned rental name."""

import argparse
import json
import os
from pathlib import Path
from time import sleep, time

from scripts.mature_cpu_rental_guard import api, check_quote, owned_pods
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.saved_hu20 import file_hash


def validate_lease(state, quote):
    if (not state.get("owner_resumed") or not state.get("owner_approved_quote")
            or not state.get("name") or not state.get("cpu_id")
            or not 0 < state["deadline"]-state["started"] <= quote["maximum_hours"]*3600
            or not 0 < state["upper_rate"] <= quote["rate_ceiling_usd_per_hour"]
            or state["upper_rate"]*(state["deadline"]-state["started"])/3600
                 > quote["total_cap_usd"]-state["fee_reserve_usd"]
            or state["fee_reserve_usd"] < .9):
        raise ValueError("Invalid resumed single-pod quote/immutable lease")


def watch(lease, quote_path, key_path):
    state = json.loads(lease.read_text()); quote = json.loads(quote_path.read_text())
    if file_hash(quote_path) != state["quote_sha256"]:
        raise ValueError("Lease quote fingerprint differs")
    validate_lease(state, quote)
    names = {state["name"]}; out = lease.with_name("provider-watchdog.json")
    if owned_pods(api(key_path, "/v2/pods")["pods"], names):
        raise ValueError("Owned name existed before arming; no takeover")
    config = {"cpu_id": state["cpu_id"], "vcpus": quote["vcpus"], "ram_gb": quote["ram_gb"]}
    observed = []; error = None
    atomic_json(out, {"status": "armed", "pid": os.getpid(), "deadline": state["deadline"], "name": state["name"]})
    while time() < state["deadline"]:
        try:
            pods = owned_pods(api(key_path, "/v2/pods")["pods"], names)
            if len(pods) > 1 or any(not check_quote(config, p, state["upper_rate"]) for p in pods):
                error = "Owned pod shape/rate differs; controller must retrieve and terminate"
                atomic_json(lease.with_name("provider-stop-request.json"), {"error": error, "timestamp": time()})
                break
            observed = [{"id": p["id"], "name": p["name"], "cost_per_hour": p.get("cost")} for p in pods]
            if lease.with_name("operator-finished.json").exists() and not pods:
                atomic_json(out, {"status": "operator-finished", "no_owned_pods": True, "heartbeat": time()})
                return
            if time() >= state["deadline"]-quote["retrieval_shutdown_reserve_seconds"]:
                atomic_json(lease.with_name("provider-stop-request.json"), {"stage": "retrieval-reserve", "timestamp": time()})
            atomic_json(out, {"status": "armed", "pid": os.getpid(), "heartbeat": time(),
                        "deadline": state["deadline"], "observed": observed})
        except Exception as exception:
            # Never serialize provider credentials or request URLs.
            atomic_json(out, {"status": "provider-poll-error", "heartbeat": time(), "error_type": type(exception).__name__})
        sleep(min(15, max(0, state["deadline"]-time())))
    for attempt in range(10):
        try:
            for pod in owned_pods(api(key_path, "/v2/pods")["pods"], names):
                api(key_path, "/v2/pods/"+pod["id"], "DELETE")
            if not owned_pods(api(key_path, "/v2/pods")["pods"], names):
                atomic_json(out, {"status": "cutoff-verified", "heartbeat": time(),
                            "error": error, "retrieval_verified": lease.with_name("retrieval-verified.json").exists()})
                return
        except Exception as exception:
            atomic_json(out, {"status": "cutoff-error", "attempt": attempt+1, "heartbeat": time(), "error_type": type(exception).__name__})
        sleep(10)
    raise RuntimeError("Owned rental cutoff could not be verified")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("lease", "quote", "key"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args(); watch(a.lease, a.quote, a.key)


if __name__ == "__main__":
    main()
