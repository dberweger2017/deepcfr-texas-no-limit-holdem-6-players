"""Audit every TP20 attempt and report stratified, rotation-block paired effects."""

import argparse
import gzip
import json
from math import sqrt
from pathlib import Path
from statistics import mean, variance
from types import SimpleNamespace

from scipy.stats import t

from scripts.report_postflop_replication import estimate
from scripts.tp20_common import density, schedule, validate, write_json
from scripts.run_tp20_campaign import swap_bytes
from src.arena.schedule import digest
from src.blueprint.abstraction import TP20_SCHEMA
from src.blueprint.artifact import _checked_schema
from src.blueprint.solver import BlueprintTrainer, PilotConfig, TP20_GAME
from src.blueprint.windowed import _hash
from src.game.hand import Table


def stratified(groups, confidence=.95):
    """Fixed equal-weight lineups; sampling variation is within deal blocks."""
    if not groups or any(len(g) < 2 for g in groups):
        return {"blocks":0,"bb_per_100":None,"interval":None}
    center = mean(mean(g) for g in groups)
    terms = [variance(g)/len(g)/len(groups)**2 for g in groups]
    var = sum(terms)
    denominator = sum(v*v/(len(g)-1) for v,g in zip(terms,groups))
    df = var*var/denominator if denominator else None
    radius = float(t.ppf((1+confidence)/2,df))*sqrt(var) if df else 0
    return {"blocks":sum(map(len,groups)),"lineups":len(groups),"bb_per_100":center,
        "confidence":confidence,"interval":[center-radius,center+radius],"degrees_of_freedom":df,
        "estimator":"equal fixed lineups; three seeds averaged within three-rotation block"}


def read_json(path, failures):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        failures.append(f"Missing or invalid {path}: {exc}")
        return None


def lines(path, failures):
    if not path.exists():
        failures.append(f"Missing {path}")
        return []
    rows = []
    for i,line in enumerate(path.read_text().splitlines()):
        try:
            rows.append(json.loads(line))
        except ValueError:
            failures.append(f"Partial JSONL row {path}:{i+1}")
    return rows


def verify(path, failures):
    checked = read_json(path/"checksums.json",failures)
    if checked is None:
        return {"checksums":0,"mismatches":["Missing checksums"]}
    mismatches = [name for name,sha in checked.items()
                  if not (path/name).is_file() or _hash(path/name) != sha]
    extras = [str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()
              and p.name != "checksums.json" and str(p.relative_to(path)) not in checked]
    if mismatches or extras:
        failures.append(f"Artifact inventory mismatch {path}: {mismatches+extras}")
    return {"checksums":len(checked),"mismatches":mismatches+extras,
            "checksums_sha256":digest(checked)}


def read_run(path, plan, phase, lineup, arm, failures):
    local = []
    verification = verify(path,local)
    result = read_json(path/"result.json",local)
    manifest = read_json(path/"manifest.json",local)
    saved_schedule = read_json(path/"schedule.json",local)
    n = plan[phase]["blocks_per_lineup"]
    _, expected, document = schedule(plan,phase,lineup)
    if (not result or result.get("status") != "complete"
            or result.get("completed_blocks") != n
            or result.get("attempts") != 3*n
            or result.get("peak_process_rss_bytes",float("inf")) >= plan["limits"]["max_rss_gib"]*1024**3):
        local.append(f"Incomplete or resource violation {path}")
    if (not manifest or manifest.get("plan_sha256") != digest(plan)
            or manifest.get("game") != TP20_GAME or manifest.get("arm") != arm
            or manifest.get("phase") != phase or manifest.get("lineup") != lineup
            or manifest.get("resource_only") is not False
            or manifest.get("schedule_sha256") != digest(document) or digest(saved_schedule) != digest(document)):
        local.append(f"Frozen identity or schedule mismatch {path}")
    rows = lines(path/"hands.jsonl",local)
    indexed = {(r["block"],r["rotation"]):r for r in rows}
    if len(rows) != 3*n or len(indexed) != len(rows):
        local.append(f"Missing or duplicate hand attempts {path}")
    for b in expected:
        for rotation in range(3):
            row = indexed.get((b.index,rotation))
            if (not row or row.get("status") != "completed" or row.get("candidate_chips") is None
                    or row.get("deal_seed") != b.deal_seeds[0] or row.get("button") != b.button
                    or tuple(row.get("opponents",())) != b.opponents
                    or tuple(row.get("action_seeds",())) != b.action_seeds
                    or sum(row.get("net_chips",[1])) != 0
                    or row.get("candidate_chips") != row.get("net_chips",[None]*3)[rotation]):
                local.append(f"Illegal, unpaired or missing row {path}/{b.index}/{rotation}")
                break
        if len(local) > 10:
            break
    failures.extend(local)
    values = ([mean(indexed[(b,r)]["candidate_chips"] for r in range(3)) for b in range(n)]
              if not local else None)
    return {"values":values,"result":result,"manifest":manifest,"verification":verification,
            "hand_attempts":len(rows),"failures":local}


def checkpoint_visits(path):
    """Stream regrets away; only integer visits are needed for density auditing."""
    with gzip.open(path,"rt") as saved:
        header = json.loads(next(saved))
        _checked_schema(header)
        if header["abstraction"] != TP20_SCHEMA or header["kind"] != "training":
            raise ValueError("Not a TP20 training checkpoint")
        d = header["table"]
        BlueprintTrainer(Table(tuple(d["player_ids"]),tuple(d["stacks"]),d["button"],
            d["small_blind"],d["big_blind"],d["chip_unit"]),PilotConfig(**header["config"]))
        visits = {}
        for line in saved:
            key,names,regrets,average,count = json.loads(line)
            if key in visits or not isinstance(count,int) or count < 0:
                raise ValueError("Duplicate or invalid checkpoint visits")
            visits[key] = SimpleNamespace(visits=count)
    return header,visits


def analyze(plan,root):
    validate(plan,frozen=True)
    failures = []
    campaign = read_json(root/"campaign.json",failures) or {}
    report = {"schema":"tp20-final-report-v1","game":TP20_GAME,
        "plan_sha256":digest(plan),"primary_extraction":"C","campaign":campaign,
        "training":{},"suites":{},"crossplay":{},"failures":failures,
        "interpretation":"Restricted three-player benchmark; no model promotion or convergence claim."}
    observation_path = root/"independent"/"reached.json"
    observations = read_json(observation_path,failures) or []
    independent_manifest = read_json(root/"independent"/"manifest.json",failures) or {}
    independent_hash = _hash(observation_path) if observation_path.exists() else None
    source_observation = root/"independent"/"uniform-observations-uniform"/"reached.json"
    if (independent_hash != independent_manifest.get("observations_sha256")
            or not source_observation.exists() or _hash(source_observation) != independent_hash):
        failures.append("Independent observation set provenance mismatch")
    verify(root/"independent"/"uniform-observations-uniform",failures)
    checkpoints = {}
    policy_hashes = {}
    for seed in plan["training_seeds"]:
        path = root/"training"/str(seed)
        local = []
        verification = verify(path,local)
        result = read_json(path/"result.json",local)
        manifest = read_json(path/"manifest.json",local)
        rows = lines(path/"checkpoints.jsonl",local)
        iterations = lines(path/"iterations.jsonl",local)
        nodes = sum(r["nodes"] for r in iterations)
        if (not result or result.get("status") != "complete"
                or result.get("completed_nodes") != nodes or nodes < plan["training_nodes"]
                or nodes > plan["training_nodes"]+plan["limits"]["max_nodes_per_iteration"]
                or result.get("discarded_nodes") != 0
                or result.get("iterations") != len(iterations)
                or result.get("peak_process_rss_bytes",float("inf")) >= plan["limits"]["max_rss_gib"]*1024**3
                or len(rows) != len(plan["checkpoints"])
                or [r["iteration"] for r in iterations] != list(range(1,len(iterations)+1))):
            local.append(f"Training {seed} incomplete or work/resource accounting mismatch")
        if (not manifest or manifest.get("plan_sha256") != digest(plan)
                or manifest.get("seed") != seed or manifest.get("preflight") is not False
                or manifest.get("independent_observations_sha256") != independent_hash
                or manifest.get("initialization") != "zero regrets; no parent checkpoint"):
            local.append(f"Training {seed} identity/initialization mismatch")
        for i,row in enumerate(rows):
            checkpoint,policy = path/f"checkpoint-{i}.json.gz",path/f"policy-{i}.json.gz"
            if (i >= len(plan["checkpoints"]) or row["requested_nodes"] != plan["checkpoints"][i]
                    or row["completed_nodes"] < row["requested_nodes"]
                    or row["overshoot_nodes"] != row["completed_nodes"]-row["requested_nodes"]
                    or row["overshoot_nodes"] > plan["limits"]["max_nodes_per_iteration"]
                    or not checkpoint.exists() or _hash(checkpoint) != row["checkpoint_sha256"]
                    or not policy.exists() or _hash(policy) != row["policy_sha256"]):
                local.append(f"Checkpoint lineage mismatch seed {seed}, milestone {i}")
            checkpoints[(seed,i)] = row
            policy_hashes[f"E{seed}-{i}"] = row["policy_sha256"]
        final = path/"current.json.gz"
        if not result or not rows or not final.exists() or (
                _hash(final) != rows[-1]["policy_sha256"]
                or result.get("current_export_sha256") != rows[-1]["policy_sha256"]
                or result.get("final_checkpoint_sha256") != rows[-1]["checkpoint_sha256"]):
            local.append(f"Final inference lineage mismatch seed {seed}")
        elif rows:
            policy_hashes[f"C{seed}"] = rows[-1]["policy_sha256"]
        report["training"][str(seed)] = {"result":result,"manifest":manifest,
            "verification":verification,"checkpoints":rows,"failures":local}
        failures.extend(local)

    all_runs = {}
    for phase,lineups in (("development",plan["primary_lineups"]),
                          ("confirmation",plan["primary_lineups"]),
                          ("secondary",plan["secondary_lineups"])):
        local = []
        if phase == "development":
            arms = ["uniform"]+[f"E{s}-{i}" for s in plan["training_seeds"]
                                    for i in range(len(plan["checkpoints"]))]
        else:
            arms = ["uniform"]+[f"C{s}" for s in plan["training_seeds"]]
        runs = {}
        for lineup in lineups:
            for arm in arms:
                item = read_run(root/phase/f"{lineup}-{arm}",plan,phase,lineup,arm,local)
                runs[(lineup,arm)] = item
                all_runs[(phase,lineup,arm)] = item
                identity = (item["manifest"] or {}).get("source_identity",{})
                if arm != "uniform" and identity.get("policy_sha256") != policy_hashes.get(arm):
                    local.append(f"Candidate export mismatch {phase}/{lineup}/{arm}")
        effects = {}
        n = plan[phase]["blocks_per_lineup"]
        indexes = range(len(plan["checkpoints"])) if phase == "development" else [None]
        for index in indexes:
            candidates = [f"E{s}-{index}" if index is not None else f"C{s}" for s in plan["training_seeds"]]
            groups, individual_groups = [], {str(s):[] for s in plan["training_seeds"]}
            label = f"checkpoint-{index}" if index is not None else "final"
            for lineup in lineups:
                control = runs[(lineup,"uniform")]["values"]
                values = [runs[(lineup,arm)]["values"] for arm in candidates]
                if control is None or any(v is None for v in values):
                    continue
                paired = [mean(v[b]-control[b] for v in values) for b in range(n)]
                effects[f"{label}/{lineup}/aggregate-minus-uniform"] = estimate(paired)
                groups.append(paired)
                for seed, candidate in zip(plan["training_seeds"],values):
                    diff = [x-y for x,y in zip(candidate,control)]
                    effects[f"{label}/{lineup}/{seed}-minus-uniform"] = estimate(diff)
                    individual_groups[str(seed)].append(diff)
            if len(groups) == len(lineups) and not local:
                effects[f"{label}/aggregate-minus-uniform"] = stratified(groups,plan["primary_confidence"])
                for seed,g in individual_groups.items():
                    effects[f"{label}/{seed}/aggregate-minus-uniform"] = stratified(g)
        compact = {}
        for (lineup,arm),item in runs.items():
            compact[f"{lineup}-{arm}"] = {k:v for k,v in item.items() if k != "values"}
            compact[f"{lineup}-{arm}"]["absolute"] = estimate(item["values"] or [])
        report["suites"][phase] = {"effects":effects,"runs":compact,"failures":local}
        failures.extend(local)

    for i,seed in enumerate(plan["training_seeds"]):
        lineup = f"hero-{i}"
        arms = (f"E{seed}-0",f"C{seed}")
        items = []
        local = []
        for arm in arms:
            item = read_run(root/"crossplay"/f"{lineup}-{arm}",plan,"crossplay",lineup,arm,local)
            identities = (item["manifest"] or {}).get("opponent_identities",{})
            opponents = [f"C{plan['training_seeds'][(i+j)%3]}" for j in (1,2)]
            if set(identities) != set(opponents) or any(
                    identities.get(opp,{}).get("policy_sha256") != policy_hashes.get(opp)
                    for opp in opponents):
                local.append(f"Frozen independent crossplay opponent mismatch {arm}")
            if (item["manifest"] or {}).get("source_identity",{}).get("policy_sha256") != policy_hashes.get(arm):
                local.append(f"Crossplay candidate export mismatch {arm}")
            items.append(item)
        diff = (estimate([x-y for x,y in zip(items[1]["values"],items[0]["values"])])
                if not local and all(v["values"] is not None for v in items) else None)
        report["crossplay"][str(seed)] = {"early":{k:v for k,v in items[0].items() if k != "values"},
            "final":{k:v for k,v in items[1].items() if k != "values"},
            "final_minus_early":diff,"failures":local}
        failures.extend(local)

    # Checkpoint-specific density: compare independent and own reached decisions.
    report["density"] = {}
    for seed in plan["training_seeds"]:
        report["density"][str(seed)] = {}
        for index in range(len(plan["checkpoints"])):
            checkpoint = root/"training"/str(seed)/f"checkpoint-{index}.json.gz"
            if not checkpoint.exists():
                continue
            try:
                header,nodes = checkpoint_visits(checkpoint)
                if header["config"]["seed"] != seed:
                    raise ValueError("Wrong checkpoint seed")
                independent = density(nodes,observations)
                saved = checkpoints.get((seed,index),{}).get("independent_density_by_street")
                if independent != saved:
                    failures.append(f"Independent density mismatch seed {seed} checkpoint {index}")
                own = []
                for lineup in plan["primary_lineups"]:
                    arm = f"E{seed}-{index}"
                    own.extend(read_json(root/"development"/f"{lineup}-{arm}"/"reached.json",failures) or [])
                entry = {"independent":independent,"development_own":density(nodes,own)}
                if index == len(plan["checkpoints"])-1:
                    own = []
                    for lineup in plan["primary_lineups"]:
                        own.extend(read_json(root/"confirmation"/f"{lineup}-C{seed}"/"reached.json",failures) or [])
                    entry["confirmation_own"] = density(nodes,own)
                report["density"][str(seed)][str(index)] = entry
                del nodes
            except Exception as exc:
                failures.append(f"Checkpoint density {seed}/{index}: {type(exc).__name__}: {exc}")
    attempts = campaign.get("attempts",[])
    if (campaign.get("status") not in ("measurements_complete","complete")
            or campaign.get("frozen_plan_sha256") != digest(plan)
            or any(a["status"] != "complete" for a in attempts if a["stage"] != "report")
            or any(a.get("finished_unix_seconds",0) > a["deadline_unix_seconds"] for a in attempts)
            or campaign.get("updated_unix_seconds",float("inf")) > campaign.get("deadline_unix_seconds",0)):
        failures.append("Campaign has incomplete/failed attempts or a deadline/plan violation")
    report["resources"] = lines(root/"resources.jsonl",failures)
    baseline = swap_bytes(campaign.get("swap_baseline"))
    for sample in report["resources"]:
        used = swap_bytes(sample.get("swap"))
        if (sample["rss_bytes"] >= plan["limits"]["max_rss_gib"]*1024**3
                or sample["free_disk_bytes"] < plan["limits"]["min_free_gib"]*1024**3
                or (baseline is not None and used is not None and
                    used-baseline > plan["limits"]["max_swap_growth_gib"]*1024**3)):
            failures.append(f"Resource ceiling in retained sample: {sample['stage']}/{sample['name']}")
    report["status"] = "complete" if not failures else "incomplete"
    return report


def readable(report,plan,root):
    rows = ["# Three-player 20BB M4 pilot", "", f"Audit status: **{report['status']}**.", "",
        "Three independent from-zero K1 seeds. Final-current C was fixed before outcomes.",
        "This restricted three-player panel does not establish six-seat strength or equilibrium quality.", "",
        f"Budget: {plan['training_nodes']:,} completed traversal nodes per seed; checkpoints {plan['checkpoints']}.",
        "", "## Confirmation", "",
        "| Effect (BB/100) | Estimate | 95% interval |", "| --- | ---: | --- |"]
    effects = report["suites"]["confirmation"]["effects"]
    for name,e in effects.items():
        if "aggregate" in name:
            interval = e.get("interval")
            rows.append(f"| {name} | {e['bb_per_100']:+.2f} | {interval} |")
    rows += ["", "## Development learning curve", "",
        "| Requested nodes per seed | Paired trained minus uniform BB/100 | 95% interval |",
        "| ---: | ---: | --- |"]
    curve = report["suites"]["development"]["effects"]
    for i,nodes in enumerate(plan["checkpoints"]):
        e = curve.get(f"checkpoint-{i}/aggregate-minus-uniform",{})
        rows.append(f"| {nodes:,} | {e.get('bb_per_100')} | {e.get('interval')} |")
    rows += ["", "## Coverage and useful update density", "",
        "Final checkpoint; decision-weighted street coverage and median raw traverser updates.",
        "Independent observations were frozen before training; own trajectories depend on the policy.", "",
        "| Seed | Street | Independent coverage / median updates | Confirmation-own coverage / median updates |",
        "| --- | --- | --- | --- |"]
    for seed in plan["training_seeds"]:
        d = report["density"].get(str(seed),{}).get(str(len(plan["checkpoints"])-1),{})
        for street in ("preflop","flop","turn","river"):
            def cell(source):
                v = d.get(source,{}).get(street,{})
                coverage = v.get("coverage")
                return (f"{100*coverage:.1f}% / {v['visit_quantiles']['p50']}"
                        if coverage is not None else "unavailable")
            rows.append(f"| {seed} | {street} | {cell('independent')} | {cell('confirmation_own')} |")
    rows += ["", "## Crossplay", "",
        "Same two independently trained frozen opponents for early and final hero; three-position blocks.", "",
        "| Hero seed | Final minus early BB/100 | 95% interval |",
        "| --- | ---: | --- |"]
    for seed,item in report["crossplay"].items():
        e = item.get("final_minus_early") or {}
        rows.append(f"| {seed} | {e.get('bb_per_100')} | {e.get('interval')} |")
    rows += ["", "All per-seed/per-lineup effects, absolute returns, development curves, mixed-lineup results,",
        "crossplay, suite telemetry and both independent/own street densities are in `final-report.json`.", "",
        "## Checkpoints and resources", "", "| Seed | Nodes | Iterations | Entries | Peak RSS GiB | Final checkpoint SHA-256 | Current policy SHA-256 |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for seed,item in report["training"].items():
        r = item["result"] or {}
        rows.append(f"| {seed} | {r.get('completed_nodes')} | {r.get('iterations')} | {r.get('entries')} | "
            f"{r.get('peak_process_rss_bytes',0)/1024**3:.3f} | {r.get('final_checkpoint_sha256')} | {r.get('current_export_sha256')} |")
    rows += ["", "## Failures and incomplete phases", ""]
    rows += [f"- {f}" for f in report["failures"]] or ["None found in the completed artifact audit."]
    rows += ["", "## Retrieval and play", "", "Large artifacts remain on SSH host `m4` at:",
        f"`/Users/dberweger/Local/tp20-pr113/{root}`", "",
        "Demo bots use the first two training seeds, by fixed order, not score.", "", "```sh",
        "mkdir -p results/tp20-demo"]
    seeds = plan["training_seeds"][:2]
    for seed in seeds:
        rows.append(f"scp m4:/Users/dberweger/Local/tp20-pr113/{root}/training/{seed}/current.json.gz results/tp20-demo/{seed}.json.gz")
    hashes = [(report["training"].get(str(s),{}).get("result") or {}).get("current_export_sha256","UNAVAILABLE") for s in seeds]
    rows += ["python -m scripts.play_tp20 \\",
        "  --policies "+" ".join(f"results/tp20-demo/{s}.json.gz" for s in seeds)+" \\",
        "  --hashes "+" ".join(hashes)+" \\",
        "  --history results/tp20-human.jsonl", "python -m scripts.play_tp20 --replay results/tp20-human.jsonl", "```", "",
        "No default model is promoted. The PR remains draft; no automatic next campaign."]
    return "\n".join(rows)+"\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan","root","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    report = analyze(plan,a.root)
    write_json(a.out,report)
    a.out.with_suffix(".md").write_text(readable(report,plan,a.root))
    print(json.dumps({"status":report["status"],"failures":report["failures"]}),flush=True)
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
