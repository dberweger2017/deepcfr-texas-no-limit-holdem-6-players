"""Board-pooling adapter to the existing macOS experiment guards."""

import os
import sys

from scripts.hu20_search_runtime import macos_memory_admission, validate_admission
from src.diagnostics.flop_check_runtime import machine_snapshot, run_tool


def resource_snapshot():
    state = machine_snapshot()
    admission = macos_memory_admission(state["vm_stat"])
    return dict(state, **admission, available_bytes=admission["reclaimable_bytes"],
                effective_cores=state["cores"])


def worker_rss_limit(budget):
    limit = budget["worker_rss_bytes"]
    if limit == 5 * 1024**3:
        return limit
    if (limit != 7 * 1024**3 or budget.get("host") != "m4"
            or budget.get("worker_rss_amendment") != "owner-approved-m4-7gib-before-main"
            or budget.get("aggregate_rss_bytes") != 8 * 1024**3
            or budget.get("arena_bytes") != 4 * 1024**3
            or budget.get("workers") != 1 or budget.get("threads_per_worker") != 6):
        raise ValueError("Worker RSS change needs the fixed owner-approved M4 amendment")
    return limit


def admit_m4(budget, snapshot):
    worker_rss_limit(budget)
    admission = dict(snapshot, experiment="hu20-board-pooling", checked_at=__import__("time").time(),
                     **{key: budget[key] for key in ("148_merged", "148_processes_empty", "149_owner_authorized",
                          "ownership_evidence", "followup_claim", "minimum_disk_free_bytes")})
    admission["rss_limit_bytes"] = min(snapshot["rss_limit_bytes"], budget["aggregate_rss_bytes"])
    validate_admission(admission)
    if (budget["workers"] != 1 or budget["threads_per_worker"] != 6
            or budget["threads_per_worker"] > snapshot["effective_cores"]
            or budget["worker_rss_bytes"] > admission["rss_limit_bytes"]):
        raise ValueError("Frozen M4 worker shape exceeds measured memory/CPU")
    return admission


def run_portable_tool(*args, **kwargs):
    if sys.platform != "darwin":
        raise ValueError("Revision 3 uses the owner-approved M4; Linux rental is superseded")
    return run_tool(*args, **kwargs)


run_owned_tool = run_portable_tool
