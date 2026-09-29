"""Choose a common lifetime budget using timing and full-path memory alone."""

from math import ceil


def choose(parity, memory, remaining):
    if parity["status"] != "exact" or len(parity["host_results"]) != 2:
        raise ValueError("Exact resumed/evaluator cross-host parity required")
    if set(memory) != {"m1", "m4"} or any(r["status"] != "complete" or r["synthetic_entries"] != 3000000 for r in memory.values()):
        raise ValueError("Missing full-path memory measurements")
    if any(r["peak_rss_bytes"] >= 9*1024**3 for r in memory.values()):
        raise ValueError("Insufficient memory headroom below 10.5 GiB")
    measurements = dict(zip(("m1", "m4"), parity["host_results"]))
    shares = {"m1": .4, "m4": .6}
    proposals = []
    # Preserve the larger confirmation count by reducing work first.
    for lbr_blocks in (2048, 1024):
        for nodes in (180000000, 140000000, 100000000):
            additional = nodes-20000000
            host_training = {}; host_eval = {}
            for host, r in measurements.items():
                runs = 1 if host == "m1" else 2
                seconds_per_node = r["complete_outer_seconds"]/r["completed_additional_nodes"]
                recovery = ceil(additional/10000000)*memory[host]["checkpoint_seconds"]
                milestone = 3*(memory[host]["checkpoint_seconds"]+memory[host]["export_seconds"]+3)
                host_training[host] = runs*(additional*seconds_per_node*1.2+recovery+milestone)
                lbr = next(p["seconds"]/p["hands"] for p in r["timing_panels"] if p["rule"] == "lbr")
                # Full five-rule curves and three A references; six-target secondary.
                cheap_hands = (12*5+3*3)*4096*2+6*6*256*2
                host_eval[host] = shares[host]*(6*lbr_blocks*2*lbr*1.25+cheap_hands*.003)
                host_eval[host] += 2*15*memory[host]["verified_reload_seconds"]+900
            training = max(host_training.values()); evaluation = max(host_eval.values())
            forecast = training+evaluation+1800
            proposals.append({"nodes": nodes, "lbr_blocks": lbr_blocks,
                              "host_training_seconds": host_training, "host_evaluation_audit_seconds": host_eval,
                              "training_seconds": training, "evaluation_audit_seconds": evaluation,
                              "report_reserve_seconds": 1800, "forecast_seconds": forecast,
                              "feasible": training <= 21600 and forecast <= remaining})
            if proposals[-1]["feasible"]:
                return {"choice": proposals[-1], "proposals": proposals, "remaining_seconds": remaining,
                        "node_time_allowance": 1.2, "lbr_allowance": 1.25,
                        "max_entries": 3000000, "entry_bound_basis": "Measured complete artifact path on both hosts, not a fitted growth bound",
                        "qualification": "Time projections are conservative estimates; original deadline and resource guards remain authoritative"}
    return {"choice": None, "proposals": proposals, "remaining_seconds": remaining}
