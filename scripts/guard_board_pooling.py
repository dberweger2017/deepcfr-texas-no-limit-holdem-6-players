"""Reuse #148's cumulative macOS guard for every board-pooling stage."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
from time import sleep, time

from scripts.hu20_search_runtime import RunBudget, install_stop_handlers
from src.diagnostics.flop_check import atomic_json
from src.diagnostics.pooling_runtime import resource_snapshot, admit_m4


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--approval", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args(); approval = json.loads(a.approval.read_text())
    if not approval.get("149_owner_authorized") or not approval.get("owner_resumed"):
        raise ValueError("Owner authorization and resume are required")
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command: raise ValueError("A guarded command is required")
    a.out.mkdir(parents=True, exist_ok=False)
    admission = admit_m4(approval, resource_snapshot())
    atomic_json(a.out / "admission.json", {"machine": admission, "budget": approval, "command": command})
    budget = RunBudget(Path(approval["clock_path"]), a.out, "board-pooling",
                       approval["experiment_deadline_epoch"]-time(), admission)
    install_stop_handlers(); process = None; failure = None
    try:
        budget.check()
        process = subprocess.Popen(command, start_new_session=True)
        while process.poll() is None:
            budget.check(); sleep(.5)
        if process.returncode: raise RuntimeError(f"Stage exit {process.returncode}")
        budget.check()
    except BaseException as error:
        failure = str(error)
        atomic_json(a.out / "failure.json", {"error": failure, "timestamp": time(), "automatic_restart": False})
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
        budget.close("failed" if failure else "completed", failure)
    atomic_json(a.out / "completion.json", {"exit_code": process.returncode, "timestamp": time()})


if __name__ == "__main__": main()
