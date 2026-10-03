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


def check_lock_only(first, actual, pot, reference_hash):
    reference = completion(first)
    final = actual[-1]
    if (final.get("status") != "locked-evaluated" or final.get("iterations") != 0
            or final.get("reference_response_sha256") != reference_hash
            or not any(r.get("gate") == "V1" and r["passed"] for r in actual)
            or not np.allclose(final["reference_equilibrium_ev_chips"], reference["current_ev_chips"], atol=1e-5*pot, rtol=0)):
        raise ValueError("Lock-only result lost its validated equilibrium reference")
    for row in (r for r in actual if r["event"] == "pooling_metric"):
        seat = row["target_solver_seat"] ^ 1
        value = row["reference_responder_value_chips"]
        if (not np.isfinite([value, row["responder_br_chips"], row["gain_bb"], row["gain_pct_pot"]]).all()
                or abs(value-reference["current_ev_chips"][seat]) > 1e-5*pot
                or abs(row["gain_bb"]*100 - (row["responder_br_chips"]-value)) > 1e-5*pot):
            raise ValueError("Invalid locked best-response value/reference")
    return {"gate": "lock-only-reference", "passed": True, "equilibrium_iterations": reference["iterations"],
            "equilibrium_residual_pct_pot": reference["exploitability_pct_pot"], "reference_response_sha256": reference_hash}


def check_locked_br_parity(fresh, solved, pot):
    key = lambda r: (r["metric"], r["target_solver_seat"])
    a = {key(r): r for r in fresh if r["event"] == "pooling_metric"}
    b = {key(r): r for r in solved if r["event"] == "pooling_metric"}
    if not a or set(a) != set(b):
        raise ValueError("Lock-only/replay measurement identities differ")
    for identity in a:
        for field in ("gain_bb", "gain_pct_pot", "responder_br_chips", "reference_responder_value_chips"):
            tolerance = 1e-5 * (pot / 100 if field == "gain_bb" else 100 if field == "gain_pct_pot" else pot)
            if not np.isclose(a[identity][field], b[identity][field], atol=tolerance, rtol=0):
                raise ValueError("Lock-only BR differs from solved-tree BR")
    return {"gate": "lock-only-vs-solved-BR", "passed": True, "measurements": len(a)}
