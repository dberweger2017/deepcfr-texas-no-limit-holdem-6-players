"""Prospective power, budget and strict-first native search selection."""

from collections import Counter
from copy import deepcopy
from math import ceil, isfinite, sqrt

import numpy as np
from scipy.stats import nct, norm, t


def power(blocks, *, sd=317.8, effect=20, alpha=.05):
    if blocks < 2 or sd <= 0 or effect <= 0 or not 0 < alpha < 1:
        raise ValueError("Invalid paired power inputs")
    critical = t.ppf(1-alpha/2, blocks-1)
    shift = effect * sqrt(blocks) / sd
    return float(nct.sf(critical, blocks-1, shift) + nct.cdf(-critical, blocks-1, shift))


def freeze_part_a(plan, timings, *, load_seconds, pilot_hash, budget_seconds=21600):
    if plan["stage"] != "planned-not-admitted":
        raise ValueError("Only a planned protocol can be frozen")
    if not 0 < budget_seconds <= 21600 or load_seconds < 0 or not pilot_hash:
        raise ValueError("Invalid timing admission")
    costs = {}
    for panel in plan["panels"]:
        values = timings[panel["name"]]
        if not values or any(not isfinite(v) or v <= 0 for v in values):
            raise ValueError("Every panel needs positive timing-only observations")
        costs[panel["name"]] = max(values) * 12
    secondary = sum(p["blocks"] * costs[p["name"]] for p in plan["panels"] if p["name"] != "lbr")
    maximum = int(((budget_seconds / 1.5) - load_seconds - secondary) / costs["lbr"])
    primary = min(next(p["blocks"] for p in plan["panels"] if p["name"] == "lbr"), maximum)
    if primary < 30:
        raise ValueError("Six-hour budget cannot admit a useful primary sample")
    frozen = deepcopy(plan)
    frozen["stage"] = "frozen-final"
    for panel in frozen["panels"]:
        if panel["name"] == "lbr": panel["blocks"] = primary
    frozen["expected_hands"] = 12 * sum(p["blocks"] for p in frozen["panels"])
    frozen["timing_admission"] = {"pilot_sha256": pilot_hash, "headroom_multiplier": 1.5,
        "seconds_per_joint_block": costs, "load_seconds": load_seconds,
        "forecast_seconds": 1.5*(load_seconds + secondary + primary*costs["lbr"])}
    frozen["power"]["achieved_design_power"] = power(primary, sd=plan["power"]["prior_paired_sd"])
    frozen["power"]["normal_approximation_blocks"] = ceil(
        ((norm.ppf(.975)+norm.ppf(.8))*plan["power"]["prior_paired_sd"]/20)**2)
    return frozen


def select_base(summary, margins=None):
    margins = margins or {"lbr": -10, "native-pressure": -20, "selective-stackoff": -10}
    rows = {r["panel"]: r["average_minus_current"] for r in summary["three_lineage_changes"]}
    if any(p not in rows or rows[p]["ci95"] is None for p in margins):
        raise ValueError("Incomplete paired primary/secondary result cannot select a base")
    regressions = [p for p, margin in margins.items() if rows[p]["ci95"][1] < margin]
    return {"base": "current" if regressions else "average", "regressions": regressions,
            "margins": margins, "rule": "average-default-material-regression-upper-bound"}


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    cumulative = np.cumsum(np.asarray(weights)[order])
    index = np.searchsorted(cumulative, q*cumulative[-1], side="left")
    return float(np.asarray(values)[order][min(index, len(order)-1)])


def qualify_curve(rows, expected_roots):
    """Failures require measured fallback quality; missing rows block qualification."""
    expected = set(expected_roots)
    if not expected: raise ValueError("Need a frozen nonempty corpus")
    groups = {}
    for row in rows: groups.setdefault(row["configuration_id"], []).append(row)
    summaries = []
    for identity, items in sorted(groups.items()):
        config = items[0]["config"]
        eligible = (len(items) == len(expected) and {r["root"] for r in items} == expected
                    and all(r["config"] == config and r.get("full_native_verified")
                        and r.get("reference_supported") and r.get("played_strategy_verified")
                        and all(isinstance(r.get(k), (int, float)) and isfinite(r[k])
                            for k in ("residual_pct_pot", "weight", "cold_seconds", "blueprint_pct_pot"))
                        and r["weight"] > 0 and r["cold_seconds"] >= 0
                        and r["residual_pct_pot"] >= 0 and r["blueprint_pct_pot"] >= 0
                        for r in items))
        row = {"configuration_id": identity, "config": config, "complete": eligible,
               "strict": False, "relaxed": False, "roots": len(items)}
        row["turn_conditioning_gap_count"]=sum(v for item in items
            for k,v in item.get("range_coverage",{}).items() if k.startswith("turn_conditioning_fallback:"))
        row["turn_conditioning_gap_tolerance"]=0
        hosts={}
        for item in items:
            hosts.setdefault(item.get("host","unrecorded"),[]).append(item)
        row["latency_by_host"]={host:{"roots":len(values),
            "timeout_fallback_rate":sum(v.get("fallback",False) and any(
                f.get("phase")=="play" and f.get("cause")=="timeout" for f in v.get("failures",[]))
                for v in values)/len(values),
            "fallback_causes":dict(Counter(f["cause"] for v in values for f in v.get("failures",[])
                if f.get("phase")=="play")),
            "cold_p99_seconds":float(np.percentile([v["cold_seconds"] for v in values],99))
                if all(isinstance(v.get("cold_seconds"),(int,float)) and isfinite(v["cold_seconds"]) for v in values) else None,
            "cold_max_seconds":max((v["cold_seconds"] for v in values),default=None)
                if all(isinstance(v.get("cold_seconds"),(int,float)) and isfinite(v["cold_seconds"]) for v in values) else None}
            for host,values in hosts.items()}
        if eligible:
            values = [r["residual_pct_pot"] for r in items]
            weights = [r["weight"] for r in items]
            mean = float(np.average(values, weights=weights))
            p95 = weighted_quantile(values, weights, .95)
            latency = float(np.percentile([r["cold_seconds"] for r in items], 95))
            median = float(np.median([r["cold_seconds"] for r in items]))
            native = config["menu"] == "native"
            row.update(mean_pct_pot=mean, p95_pct_pot=p95, cold_p95_seconds=latency,
                       cold_median_seconds=median,
                       cold_p99_seconds=float(np.percentile([r["cold_seconds"] for r in items],99)),
                       cold_max_seconds=max(r["cold_seconds"] for r in items), fallbacks=sum(r.get("fallback", False) for r in items))
            row["strict"] = native and latency <= 30 and mean <= .5 and p95 <= .5
            row["relaxed"] = (native and latency <= 30 and mean <= 1 and p95 <= 2
                and all(r["residual_pct_pot"] < .1*r["blueprint_pct_pot"] for r in items))
        summaries.append(row)
    tier = "strict" if any(r["strict"] for r in summaries) else "relaxed"
    candidates = [r for r in summaries if r[tier]]
    chosen = min(candidates, key=lambda r: (r["cold_p95_seconds"], r["cold_median_seconds"],
        -r["config"]["opponent_likelihood_floor"], r["configuration_id"])) if candidates else None
    return {"status": "qualified" if chosen else "owner-decision-needed", "tier": tier if chosen else None,
            "selected": chosen, "full_curve": summaries}
