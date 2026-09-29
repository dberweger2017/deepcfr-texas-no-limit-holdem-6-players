"""Native per-host replay followed by raw-block, lineage-paired statistics."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from statistics import mean
from time import time

from scripts.evaluate_hu20_scaling import shard_owner, tasks
from scripts.evaluate_hu20_reopening import Target
from scripts.hu20_scaling_common import acquire, inventory
from scripts.report_hu20_reopening import estimate, verify_hand, verify_phase
from scripts.train_hu20 import write_json, rss
from src.arena.schedule import stream_seed


def audit(plan, evaluation, out):
    acquire(out); started = time()
    result = json.loads((evaluation / "result.json").read_text())
    host = result["host"]; specs = json.loads((evaluation / "models.json").read_text())
    verified = verify_phase(evaluation); records = []; total = 0; failures = []
    source = None; current = None
    for spec, label, rule, contract, count, root, phase in tasks(plan, specs):
        path = evaluation / f'{spec["name"]}--{label}.jsonl.gz'
        if not path.exists():
            failures.append({"policy": spec["name"], "attacker": label, "reason": "Missing panel"}); continue
        if current != spec["name"]:
            source = None
            import gc
            gc.collect(); source = Target(spec); current = spec["name"]
        blocks = {}; roles = defaultdict(dict); telemetry = Counter(); visits = Counter(); lbr = Counter()
        with gzip.open(path, "rt") as saved:
            for line in saved:
                row = json.loads(line); total += 1
                b = row["block"]; rot = row["rotation"]
                if (row["host"] != host or shard_owner(b, plan) != host or not 0 <= b < count
                        or row["root_seed"] != root or row["policy"] != spec["name"]
                        or row["attacker"] != label or row["contract"] != contract
                        or row["deal_seed"] != stream_seed(root, "test", "deal", 2, b)
                        or row["button"] != b % 2):
                    raise ValueError("Changed frozen schedule/host shard")
                verify_hand(row, spec, source)
                if row["status"] != "complete":
                    failures.append({k: v for k, v in row.items() if k != "actions"}); continue
                block = blocks.setdefault(str(b), {})
                if str(rot) in block: raise ValueError("Duplicate rotation")
                block[str(rot)] = row["target_chips"]
                role = "button_small_blind" if rot == row["button"] else "big_blind"
                roles[role][str(b)] = row["target_chips"]
                for action in row["actions"]:
                    if action["logical_player"] == 0:
                        coords = (action["street"], action["kind"], action["target_trained"],
                                  action["preceding_original_off_menu"], action["preceding_target_off_menu"])
                        telemetry[coords] += 1
                        visits[(action["street"], action["target_visits"] or 0)] += 1
                    if "lbr" in action:
                        d = action["lbr"]; lbr["decisions"] += 1
                        lbr["completed"] += d["completed"]; lbr["over_soft_budget"] += d["over_soft_budget"]
                        lbr["samples"] += d["samples"]
                        lbr["zero_likelihood_events_cumulative"] += d["zero_likelihood_events"]
                        lbr["max_seconds"] = max(lbr["max_seconds"], d["seconds"])
        expected = {str(b) for b in range(count) if shard_owner(b, plan) == host}
        complete = set(blocks) == expected and all(set(v) == {"0", "1"} for v in blocks.values())
        if not complete: failures.append({"policy": spec["name"], "attacker": label, "reason": "Missing block/rotation"})
        records.append({"policy": spec["name"], "seed": spec["seed"], "milestone": spec["milestone"],
                        "arm": spec["arm"], "attacker": label, "host": host, "complete": complete,
                        "blocks": {b: mean(v.values()) for b, v in blocks.items() if len(v) == 2},
                        "roles": dict(roles), "lbr": dict(lbr),
                        "telemetry": [{"coordinates": list(k), "count": v} for k, v in sorted(telemetry.items())],
                        "visit_histograms": [{"street": s, "visits": n, "decisions": c} for (s, n), c in sorted(visits.items())]})
    report = {"status": "complete" if result["status"] == "complete" and not failures else "incomplete",
              "host": host, "native_replayed_hands": total, "verified_phase_files": verified,
              "panels": records, "failures": failures, "started": started, "finished": time(),
              "peak_rss_bytes": rss(),
              "audit_scope": "Every native action/settlement/event digest, target RNG/visit/menu replay; LBR legality and work telemetry, not recomputation of every LBR decision"}
    write_json(out / "results.json", report)
    write_json(out.with_name(out.name+"-inventory.json"), inventory(out))
    return report


def merge_panels(reports):
    merged = {}
    for report in reports:
        for p in report["panels"]:
            key = (p["policy"], p["attacker"])
            if key not in merged:
                merged[key] = {**p, "blocks": {}, "roles": {"big_blind": {}, "button_small_blind": {}}}
            m = merged[key]
            if set(m["blocks"]) & set(p["blocks"]): raise ValueError("Duplicated host blocks")
            m["blocks"].update(p["blocks"])
            for role, blocks in p["roles"].items(): m["roles"][role].update(blocks)
    return merged


def contrast(panels, seeds, attacker, before, after, role=None, level=.975):
    pairs = [(panels.get((f"B-{s}-{before}", attacker)), panels.get((f"B-{s}-{after}", attacker))) for s in seeds]
    if any(a is None or b is None for a, b in pairs): return {"status": "unavailable"}
    def values(p): return p["blocks"] if role is None else p["roles"][role]
    sets = [set(values(p)) for pair in pairs for p in pair]
    if not sets[0] or any(s != sets[0] for s in sets[1:]): return {"status": "unavailable", "reason": "Incomplete pairing"}
    keys = sorted(sets[0], key=int)
    effects = {str(seed): [values(b)[k]-values(a)[k] for k in keys] for seed, (a,b) in zip(seeds,pairs)}
    return {"status": "available", "long_minus_20M": estimate([mean([e[i] for e in effects.values()]) for i in range(len(keys))],level),
            "early_absolute": estimate([mean([values(a)[k] for a,b in pairs]) for k in keys],level),
            "long_absolute": estimate([mean([values(b)[k] for a,b in pairs]) for k in keys],level),
            "per_lineage": {s: estimate(v,level) for s,v in effects.items()}}


def combine(plan, paths, out):
    reports = [json.loads(p.read_text()) for p in paths]
    if {r["host"] for r in reports} != set(plan["block_host_cycle"]): raise ValueError("Missing host audit")
    panels = merge_panels(reports); seeds = plan["training_seeds"]; final = plan["training_total_nodes"]
    expected = {label: count for _,label,_,_,count,_,_ in tasks(plan, json.loads(Path(plan["coordinator_models"]).read_text()))}
    pending = []
    for spec, attack, _, _, count, _, _ in tasks(plan, json.loads(Path(plan["coordinator_models"]).read_text())):
        panel = panels.get((spec["name"], attack))
        present = set(panel["blocks"]) if panel else set()
        wanted = {str(b) for b in range(count)}
        if present - wanted:
            raise ValueError("Unexpected global blocks")
        if present != wanted:
            pending.append({"policy": spec["name"], "attacker": attack,
                            "pending_blocks": sorted(wanted-present, key=int)})
    primary = {a: contrast(panels,seeds,a,20000000,final) for a in ("LBR-original-cap2","Pressure-native")}
    for attack in primary:
        count = plan["lbr_blocks"] if attack == "LBR-original-cap2" else plan["cheap_blocks"]
        if primary[attack]["status"] == "available" and primary[attack]["long_minus_20M"]["blocks"] != count:
            primary[attack] = {"status": "unavailable", "reason": "Prespecified block count incomplete"}
    roles = {role: {a: contrast(panels,seeds,a,20000000,final,role) for a in primary} for role in ("button_small_blind","big_blind")}
    effects = [{"milestone": milestone, "attacker": attacker,
                **contrast(panels,seeds,attacker,20000000,milestone,level=.95)}
               for milestone in plan["milestones"] for attacker in sorted(expected)]
    result = {"status": "complete" if not pending and all(r["status"] == "complete" for r in reports) else "incomplete",
              "primary": primary, "primary_roles": roles, "exploratory_curves": effects,
              "native_replayed_hands": sum(r["native_replayed_hands"] for r in reports),
              "per_policy": [{"policy": p["policy"], "attacker": p["attacker"], "target": estimate(list(p["blocks"].values())),
                              "attacker_profit": estimate([-v for v in p["blocks"].values()]),
                              "roles": {r: estimate(list(b.values())) for r,b in p["roles"].items()}}
                             for p in panels.values()],
              "pending_panels": pending,
              "quality_gate": "unavailable", "pressure_safeguard": "unavailable", "finished": time()}
    for attack, field, threshold in [("LBR-original-cap2","quality_gate",0),("Pressure-native","pressure_safeguard",-10)]:
        r = primary[attack]
        if r["status"] == "available" and r["long_minus_20M"]["interval"] is not None:
            result[field] = "pass" if r["long_minus_20M"]["interval"][0] > threshold else "not established"
    write_json(out, result); return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--evaluation",type=Path); p.add_argument("--audits",type=Path,nargs="+")
    p.add_argument("--out",type=Path,required=True); a=p.parse_args(); plan=json.loads(a.plan.read_text())
    result=audit(plan,a.evaluation,a.out) if a.evaluation else combine(plan,a.audits,a.out)
    print(json.dumps({k:v for k,v in result.items() if k not in ("panels","per_policy","exploratory_curves")}))
    return result["status"] != "complete"


if __name__ == "__main__": raise SystemExit(main())
