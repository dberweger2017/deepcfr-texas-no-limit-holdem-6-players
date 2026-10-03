"""Fail-closed admission and deterministic replay checks for pooled losses."""

import numpy as np


def completion(rows):
    if not rows or rows[-1].get("status") != "solved":
        raise ValueError("No completed equilibrium solve")
    final = rows[-1]
    if not any(r.get("gate") == "V1" and r["passed"] for r in rows):
        raise ValueError("Missing passing native-tree gate")
    if not np.isfinite(final["exploitability_pct_pot"]) or final["exploitability_pct_pot"] > .2:
        raise ValueError("Convergence target not achieved")
    return final


def statistics(rows):
    found = [r["groups"] for r in rows if r["event"] == "pooling_statistics"]
    if len(found) != 1:
        raise ValueError("Missing or duplicate sufficient statistics")
    return found[0]


def check_replay(first, second, pot):
    left, right = completion(first), completion(second)
    if left["iterations"] != right["iterations"] or left["compressed"] != right["compressed"]:
        raise ValueError("Replay solve recipe differs")
    for field in ("current_ev_chips", "mes_ev_chips"):
        if not np.allclose(left[field], right[field], atol=1e-5 * pot, rtol=0):
            raise ValueError("Replay equilibrium values differ")
    if abs(left["exploitability_pct_pot"] - right["exploitability_pct_pot"]) > .001:
        raise ValueError("Replay residual differs")
    a = {(r["metric"], r["key"]): r for r in statistics(first)}
    b = {(r["metric"], r["key"]): r for r in statistics(second)}
    if len(a) != len(statistics(first)) or len(b) != len(statistics(second)) or set(a) != set(b):
        raise ValueError("Replay projection keys differ")
    for key, row in a.items():
        other = b[key]
        if row["names"] != other["names"]:
            raise ValueError("Replay menu order differs")
        scale = max(1., row["mass"])
        if not np.allclose([row["mass"], *row["action_mass"]],
                           [other["mass"], *other["action_mass"]], atol=2e-5 * scale, rtol=0):
            raise ValueError("Replay projection statistics differ")
    return {"gate": "deterministic-pooling-replay", "passed": True,
            "groups": len(a), "iterations": left["iterations"]}


def common_mask(manifest, results, policies):
    """Support/convergence only; no losses enter the admission mask."""
    by_root = {}
    for job in manifest["jobs"]:
        by_root.setdefault(job["spot"], set()).add(job["lineage"])
    required = {p["seed"] for p in policies}
    admitted = []; excluded = []
    for spot, lineages in sorted(by_root.items()):
        jobs = [j for j in manifest["jobs"] if j["spot"] == spot]
        eligible = lineages == required and all(results.get(j["job"], {}).get("eligible") for j in jobs)
        (admitted if eligible else excluded).append(spot)
    for row in manifest["support_exclusions"]:
        if row["spot"] not in excluded:
            excluded.append(row["spot"])
        if row["spot"] in admitted:
            admitted.remove(row["spot"])
    return {"admitted": admitted, "excluded": sorted(excluded)}
