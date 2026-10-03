"""Guard every paid preparation/qualification stage under the rental clock."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
from time import sleep, time

from src.diagnostics.flop_check import atomic_json
from src.diagnostics.flop_check_runtime import append, rss_for_tree
from src.diagnostics.pooling_runtime import linux_snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--approval", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args(); budget = json.loads(a.approval.read_text())
    if not budget["owner_approved_quote"] or not budget.get("owner_resumed"):
        raise ValueError("Paid budget approval and owner resume are required after rental deferral")
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command:
        raise ValueError("A guarded stage command is required")
    a.out.mkdir(parents=True, exist_ok=False)
    initial = linux_snapshot(); atomic_json(a.out / "admission.json", {"machine": initial, "budget": budget, "command": command})
    process = subprocess.Popen(command, start_new_session=True); failure = None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("Owned stage guard stopped")))
    try:
        while process.poll() is None:
            state = linux_snapshot(); rss = rss_for_tree(os.getpid())
            append(a.out / "resources.jsonl", {"timestamp": time(), "rss_bytes": rss,
                   "swap_used_bytes": state["swap_used_bytes"], "cgroup_used_bytes": state["cgroup_used_bytes"]})
            if time() + budget["retrieval_shutdown_reserve_seconds"] >= budget["rental_deadline_epoch"]:
                failure = "Rental clock reached retrieval/shutdown reserve"
            elif rss > budget["aggregate_rss_bytes"]:
                failure = "Stage aggregate RSS budget exceeded"
            elif state["swap_used_bytes"] - budget["swap_baseline_bytes"] > 1024**3:
                failure = "Stage swap growth exceeds 1 GiB"
            elif state["memory_events"] != initial["memory_events"]:
                failure = "Stage cgroup memory event changed"
            if failure:
                raise RuntimeError(failure)
            sleep(1)
        if process.returncode:
            raise RuntimeError(f"Stage exit {process.returncode}")
    except BaseException as error:
        atomic_json(a.out / "failure.json", {"error": str(error), "timestamp": time(), "automatic_restart": False})
        raise
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
    atomic_json(a.out / "completion.json", {"exit_code": process.returncode, "timestamp": time()})


if __name__ == "__main__":
    main()
