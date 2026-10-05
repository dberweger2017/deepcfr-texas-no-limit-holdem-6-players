"""Cost a frozen arena from measured end-to-end work and a timestamped live offer."""

import argparse
import json
from math import ceil
from pathlib import Path
from time import time

from scripts.hu20_search_runtime import atomic_json
from src.arena.schedule import digest


def quote(plan, calibration, timing, offer, *, workers, workers_per_pod=1, now=None):
    now=time() if now is None else now
    if plan["stage"]!="frozen-final" or calibration["status"]!="qualified":
        raise ValueError("Publish Part A and qualified calibration first")
    if plan["calibration_sha256"]!=digest(calibration):raise ValueError("Calibration identity differs")
    if not 0<now-offer.get("retrieved_at",0)<3600 or not offer.get("source_url"):
        raise ValueError("A fresh, attributed RunPod offer is required")
    if (offer.get("architecture")!="x86_64" or offer.get("provider")!="RunPod"
            or (offer.get("gpu") and offer.get("compute_workload")!="cpu")):
        raise ValueError("Quote independent native x86_64 CPU workers")
    if not isinstance(workers,int) or workers<1 or not isinstance(workers_per_pod,int) or workers_per_pod<1:
        raise ValueError("Positive worker and per-pod counts required")
    pods=ceil(workers/workers_per_pod)
    if workers_per_pod==1:
        if workers>offer.get("available_workers",0):
            raise ValueError("Worker count exceeds the live offer")
    else:
        # Stock labels do not promise a pod count. Price this requested layout;
        # actual host resources and availability remain pre-production admission.
        if offer.get("availability") not in ("LOW","MEDIUM","HIGH"):
            raise ValueError("A live in-stock pod offer is required")
        if (workers_per_pod*calibration["selected"]["config"]["threads"]
                +offer["reserved_cpu_per_pod"]>offer["minimum_cpu_per_pod"]):
            raise ValueError("Solver threads leave insufficient pod CPU headroom")
        if workers_per_pod*timing["rss_limit_bytes_per_worker"]>offer["minimum_ram_bytes_per_pod"]:
            raise ValueError("Worker family limits exceed pod memory")
    if timing.get("configuration_sha256")!=digest(calibration["selected"]["config"]):
        raise ValueError("Timing must include the selected settings")
    required=("includes_preparation","includes_parsing","includes_lbr_speculative_solves",
              "includes_base_and_search_arms")
    if not all(timing.get(k) is True for k in required):
        raise ValueError("End-to-end evidence must include both arms and all LBR work")
    cells=timing["panels"]
    forecast=[];largest_worker=0
    for panel in plan["panels"]:
        cost=cells[panel["name"]]
        if cost["paired_blocks"]<1 or cost["seconds_per_joint_block_p95"]<=0:
            raise ValueError("Each panel needs measured balanced end-to-end timing")
        seconds=panel["blocks"]*cost["seconds_per_joint_block_p95"]
        largest_worker+=ceil(panel["blocks"]/workers)*cost["seconds_per_joint_block_p95"]
        forecast.append({"panel":panel["name"],"blocks":panel["blocks"],
            "hands":12*panel["blocks"],"worker_seconds":seconds,"timing":cost})
    reserves=timing["reserves_seconds_per_worker"]
    if any(reserves.get(k,0)<=0 for k in ("setup_build","actual_pod_parity","replay_verification",
                                          "retrieval_hash_verification","shutdown")):
        raise ValueError("Quote every setup/build/parity/replay/verification/shutdown reserve")
    # Balanced coordinate partitioning; price the largest worker, with 50% reserve.
    worker_seconds=ceil(1.5*(largest_worker+sum(reserves.values())))
    hours=worker_seconds/3600
    hourly=offer["compute_hourly_usd"]+offer["container_disk_hourly_usd"]
    if hourly<=0 or offer.get("storage_gb",0)<workers_per_pod*timing["required_storage_gb_per_worker"]:
        raise ValueError("Positive total price and sufficient evidence storage required")
    cost=pods*hours*hourly+offer.get("retained_storage_reserve_usd",0)
    return {"status":"quote-awaiting-owner-approval","owner_approved":False,
        "arena_plan_sha256":digest(plan),"calibration_sha256":digest(calibration),
        "timing_sha256":digest(timing),"offer":offer,"workers":workers,
        "workers_per_pod":workers_per_pod,"pods":pods,"worker_seconds":worker_seconds,
        "rss_limit_bytes":timing["rss_limit_bytes_per_worker"],"expected_hands":plan["expected_hands"],
        "panel_forecasts":forecast,"reserves_seconds_per_worker":reserves,"headroom_multiplier":1.5,
        "maximum_cost_usd":ceil(cost*100)/100,"maximum_worker_hours":hours,
        "actual_pod_parity":"required before production; build/setup priced above",
        "no_paid_allocation_performed":True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("plan","calibration","timing","offer","out"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--workers",type=int,required=True)
    p.add_argument("--workers-per-pod",type=int,default=1)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("Preserve earlier quote")
    result=quote(*(json.loads(getattr(a,n).read_text()) for n in ("plan","calibration","timing","offer")),
                 workers=a.workers,workers_per_pod=a.workers_per_pod)
    atomic_json(a.out,result)
    print(json.dumps({k:result[k] for k in ("status","workers","maximum_cost_usd","maximum_worker_hours")}))


if __name__ == "__main__":main()
