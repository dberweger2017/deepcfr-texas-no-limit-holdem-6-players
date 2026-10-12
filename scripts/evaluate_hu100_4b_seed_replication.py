"""PR236 fixed-model duplicate evaluation; pilots inspect costs only."""
import argparse
from dataclasses import asdict
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
from statistics import mean, stdev
import subprocess
import sys
from time import perf_counter, time

from scipy.stats import t
from scripts import run_hu100_4b_seed_replication as campaign
from scripts import evaluate_hu100_4b_seed_ladder as previous
from scripts import run_hu100_3b_ladder as scripted
from scripts.evaluate_hu100_direct import make_plan, put
from scripts.native_hu100_model_metadata import audited_average_spec
from scripts.report_native_hu100_learning_curves import frozen_schedule
from src.arena.schedule import build_schedule
from src.policies.files import file_hash

ROOT, OUT, BINARY = campaign.ROOT, campaign.OUT, campaign.BINARY
MODULE = "scripts.evaluate_hu100_4b_seed_replication"
PILOT_ROOT, FINAL_ROOT = 202610114501, 202610114502
SCRIPT_PILOT_ROOT, SCRIPT_FINAL_ROOT = 202610114503, 202610114504
GIB = 1024**3
read = campaign.read


def specs():
    models = {}
    for seed in campaign.SEEDS:
        complete = read(OUT / f"seed-{seed}-complete.json")
        if complete["status"] != "verified":
            raise ValueError("Both training lineages must be fully audited")
        terminal = Path(complete["terminal"]).parent
        for target, dest in ((2_000_000_000, OUT / "training" / str(seed) / "2000000000"),
                             (4_000_000_000, terminal)):
            gate, checked = read(dest / "gate.json"), read(dest / "audit.json")
            model = audited_average_spec(dest / "average.gz", checked,
                        checkpoint_sha256=gate["sha256"], actual_nodes=gate["actual_nodes"])
            model.update(name=f"HU100-seed-{seed}-average-{gate['actual_nodes']}", seed=seed)
            models[seed, target] = model
            put(dest / "spec.json", model)
    for target in (2_000_000_000, 4_000_000_000):
        dest = OUT / "inputs/2026100601" / str(target)
        checked = read(dest / "audit.json")
        pin = next(p for p in read(campaign.INDEX)["assets"]
                   if p["seed"] == 2026100601 and p["requested_nodes"] == target and p["kind"] == "checkpoint.gz")
        model = audited_average_spec(dest / "average.gz", checked,
                                     checkpoint_sha256=pin["sha256"], actual_nodes=pin["actual_nodes"])
        model.update(name=f"HU100-seed-2026100601-average-{pin['actual_nodes']}", seed=2026100601)
        models[2026100601, target] = model
        put(dest / "spec.json", model)
    pairs = {f"seed-{s}-terminal-vs-2b": [models[s, 4_000_000_000], models[s, 2_000_000_000]]
             for s in campaign.SEEDS}
    primary = list(pairs)
    descriptive = []
    for a, b in ((2026100601, 2026100901), (2026100601, 2026100902), (2026100901, 2026100902)):
        name = f"terminal-seed-{a}-vs-{b}"
        pairs[name] = [models[a, 4_000_000_000], models[b, 4_000_000_000]]
        descriptive.append(name)
    put(OUT / "pairs.json", {"pairs": pairs, "primary": primary, "descriptive": descriptive,
        "scripted": [models[2026100601, 4_000_000_000], models[2026100601, 2_000_000_000]]})
    for name, pair in pairs.items():
        put(OUT / "specs" / (name + ".json"), pair)


def bind_historical_audit(audit, target):
    pins = campaign.reference(2026100601, target)
    checkpoint, average = pins["checkpoint.gz"], pins["average.gz"]
    recorded = {Path(path).name: pin for path, pin in audit["files"].items()}
    if set(recorded) != set(pins):
        raise ValueError("STOP historical audit file inventory differs")
    for kind, pin in pins.items():
        if recorded[kind] != {"bytes": pin["bytes"], "sha256": pin["sha256"]}:
            raise ValueError("STOP historical full audit indexed pin differs")
    if (audit["status"] != "verified" or audit["audit"]["status"] != "verified"
            or audit["stack_bb"] != 100
            or audit["entries"] != checkpoint["entries"]
            or audit["audit"]["all_nodes_verified"] != checkpoint["entries"]
            or audit["audit"]["counts"]["entries"] != checkpoint["entries"]
            or audit["native_state"]["completed_nodes"] != checkpoint["actual_nodes"]
            or audit["audit"]["checkpoint_sha256"] != checkpoint["sha256"]
            or audit["audit"]["average_sha256"] != average["sha256"]
            or audit["audit"]["current_sha256"] != pins["current.gz"]["sha256"]):
        raise ValueError("STOP historical full audit lineage/counts differ")


def restore():
    # Only the merged owning PR's retained original bytes are linked. If an
    # original is absent, stop for indexed archive restoration; do not retrain.
    state = json.loads(subprocess.check_output(["gh", "pr", "view", "226", "--json", "state,mergedAt,url"], text=True))
    if state["state"] != "MERGED":
        raise ValueError("PR226 must be merged before using retained originals")
    pins = [p for p in read(campaign.INDEX)["assets"]
            if p["seed"] == 2026100601 and p["requested_nodes"] in (2_000_000_000, 4_000_000_000)
            and p["kind"] == "average.gz"]
    receipts = []
    owner_states = {}
    for owner in sorted({pin["owning_pr"] for pin in pins} | {226}):
        owner_states[owner] = json.loads(subprocess.check_output(
            ["gh","pr","view",str(owner),"--json","state,mergedAt,url"],text=True))
        if owner_states[owner]["state"] != "MERGED":
            raise ValueError("Owning PR must be merged before model restoration")
    old = Path.home() / "Local/hu100-4b-seed-ladder-20261010/results/hu100-4b-seed-ladder/training/2026100601"
    for pin in pins:
        path = old / str(pin["requested_nodes"]) / pin["kind"]
        actual = file_hash(path)
        if path.stat().st_size != pin["bytes"] or actual != pin["sha256"]:
            raise ValueError("STOP historical model size/hash mismatch")
        dest = OUT / "inputs/2026100601" / str(pin["requested_nodes"]) / pin["kind"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.hardlink_to(path)
        # The owner retained these two averages during merged #237 cleanup.
        # Their unchanged hashes preserve #226's full stream-export/audit proof;
        # do not restore/re-audit removed checkpoint/current bytes unnecessarily.
        audit_path = path.parent / "audit.json"
        audit = read(audit_path)
        bind_historical_audit(audit, pin["requested_nodes"])
        put(dest.parent / "audit.json", audit)
        audited_average_spec(dest, audit,
            checkpoint_sha256=audit["audit"]["checkpoint_sha256"], actual_nodes=pin["actual_nodes"])
        receipts.append({**pin, "source_path": str(path), "restored_path": str(dest),
                         "historical_full_audit_source_path": str(audit_path),
                         "historical_full_audit_sha256": file_hash(audit_path),
                         "historical_full_audit_reused": True,
                         "historical_streamed_export_reused": True,
                         "actual_sha256": actual, "source_mtime_ns": path.stat().st_mtime_ns,
                         "method": "hash-verified hardlink; shared bytes remain read-only",
                         "command": f"ln '{path}' '{dest}'; shasum -a 256 '{dest}'"})
    put(OUT / "restoration.json", {"status": "verified", "live_owning_pr": state,
                                  "assets": receipts, "owning_pr_live_states": owner_states, "remote_bytes_downloaded": False})


def prepare():
    revision = campaign.source()
    campaign.guarded("restore-candidate-inputs", [sys.executable, "-m", MODULE, "restore-worker"])
    campaign.guarded("prepare-model-specs", [sys.executable, "-m", MODULE, "specs-worker"])
    put(OUT / "evaluation-prepare.json", {"source": revision, "all_evaluated_checkpoints_fully_audited": True,
        "restored_exports_are_pr226_streamed_exact_bytes": True,
        "new_checkpoint_full_audits_fresh": True,
        "unchanged_candidate_input_full_audits_reused_from_pr226": True})


def checked_specs(plan, frozen=None):
    previous.OUT = OUT
    pins = previous.checked_specs(plan["pairs"], frozen)
    return pins


def schedule_seeds(specs, root, rung, blocks):
    schedule = build_schedule(make_plan(specs, blocks, root, rung))
    values = {b.deal_seeds[0] for b in schedule}
    if len(values) != blocks:
        raise ValueError("STOP duplicate physical deals")
    return values


def live_root_inventory(label):
    """Bind every recent/open PR to a dated manual schedule classification."""
    declaration = read(ROOT / "planning/pr-root-classification.json")
    recent = json.loads(subprocess.check_output(
        ["gh", "pr", "list", "--state", "all", "--limit", "100", "--json",
         "number,title,state,mergedAt,updatedAt,headRefName,url"], text=True))
    opened = json.loads(subprocess.check_output(
        ["gh", "pr", "list", "--state", "open", "--limit", "100", "--json",
         "number,title,state,updatedAt,headRefName,url"], text=True))
    relevant = {p["number"]: p for p in [*recent, *opened]
                if p["number"] >= 215 or p["state"] == "OPEN"}
    classes = declaration["prs"]
    unknown = sorted(n for n in relevant if n != 236 and str(n) not in classes)
    if unknown:
        raise ValueError("Preventive unknown PR/root inventory: " + str(unknown))
    for n, item in relevant.items():
        if n == 236:
            continue
        pin = classes[str(n)]
        if pin["head"] != json.loads(subprocess.check_output(
                ["gh", "pr", "view", str(n), "--json", "headRefOid"], text=True))["headRefOid"]:
            raise ValueError("Preventive changed PR/root classification: " + str(n))
        if pin["hu100_schedule"] not in ("included-pr215", "included-pr223", "included-pr226", "none"):
            raise ValueError("Explicit earlier HU100 root expansion required")
    path = OUT / (label + "-live-root-inventory.json")
    put(path, {"at": time(), "launch_or_freeze": label, "source": campaign.source(),
               "all_current_prs_classified": True, "live_recent": recent, "live_open": opened,
               "classification": declaration,
               "classification_sha256": file_hash(ROOT / "planning/pr-root-classification.json")})
    return path


def freshness(blocks, label):
    plan = read(OUT / "pairs.json")
    first = next(iter(plan["pairs"].values()))
    seen, prior = previous.prior_deals(first)
    old = read(ROOT / "docs/reports/hu100-4b-seed-ladder-artifacts/frozen-final.json")
    for root in (old["pilot_root"], old["root"]):
        for rung, count in old["blocks_per_contrast"].items():
            seen.update(schedule_seeds(first, root, rung, 32 if root == old["pilot_root"] else count))
    prior["pr226_freeze"] = "docs/reports/hu100-4b-seed-ladder-artifacts/frozen-final.json"
    inventory_path = live_root_inventory(label)
    prior["live_root_inventory"] = {"path": str(inventory_path),
                                    "sha256": file_hash(inventory_path)}
    initial = len(seen)
    hashes = {}
    def add(name, values):
        if seen.intersection(values):
            raise ValueError("STOP physical deal collision before play")
        seen.update(values)
        hashes[name] = hashlib.sha256(b"".join(n.to_bytes(16, "big") for n in sorted(values))).hexdigest()
    for count in (32, 512):
        for rung, pair in plan["pairs"].items():
            add(f"pilot/{rung}/{count}", schedule_seeds(pair, PILOT_ROOT, rung + f"-timing-{count}", count))
    for rung, pair in plan["pairs"].items():
        add("final/" + rung, schedule_seeds(pair, FINAL_ROOT, rung, blocks[rung]))
    # The two scripted arms intentionally share the same physical schedule.
    for root, count in ((SCRIPT_PILOT_ROOT, 32), (SCRIPT_FINAL_ROOT, 4096)):
        doc = frozen_schedule({"models": [plan["scripted"][0]]}, count, root)
        values = {b["deal_seeds"][0] for panel in doc["panels"].values() for b in panel["blocks"]}
        add(f"scripted/{root}/{count}", values)
    put(OUT / (label + "-freshness.json"), {"all_pairwise_disjoint": True, "prior": prior,
        "prior_physical_deals": initial, "new_physical_deals": len(seen)-initial,
        "new_deal_seed_hashes": hashes, "scripted_arms_intentionally_share_deals": True,
        "blocks": blocks})


def panel(pilot=False):
    pair = read(OUT / "pairs.json")["scripted"]
    # #223's checked arena panel is reused unchanged; isolate all of its output
    # and configuration roots here, replacing only fixed model inputs/roots.
    scripted.ROOT = ROOT
    scripted.OUT = OUT / "scripted"
    scripted.SCRIPT_PILOT_ROOT = SCRIPT_PILOT_ROOT
    scripted.SCRIPT_FINAL_ROOT = SCRIPT_FINAL_ROOT
    from src.blueprint.action_translation import TranslationOptions
    if asdict(TranslationOptions()) != {"max_states":512,"max_events":128}:
        raise ValueError("Exact predeclared translation settings required")
    if not (scripted.OUT / "pairs.json").exists():
        put(scripted.OUT / "pairs.json", {"terminal-vs-1b": pair})
    if read(scripted.OUT / "pairs.json") != {"terminal-vs-1b": pair}:
        raise ValueError("STOP scripted executable pair differs from frozen candidate/comparator")
    scripted.secondary(pilot=pilot)
    if not pilot:
        bindings = verify_scripted(scripted.OUT / "secondary", 4096, SCRIPT_FINAL_ROOT)
        summary = read(scripted.OUT / "secondary/summary.json")
        summary["evidence_bindings"] = bindings
        summary["comparison"] = "candidate4B-minus-v051-2B"
        summary["descriptive"] = False
        summary["predeclared_safeguard"] = True
        summary["severe_regression_rule"] = "difference ordinary95 upper bound below -20BB/100"
        summary["severe_regression"] = {o: value["interval"][1] < -20
                                       for o, value in summary["contrasts"].items()}
        summary["no_severe_regression_flag"] = not any(summary["severe_regression"].values())
        put(OUT / "scripted-summary.json", summary)


def calibrate():
    revision = campaign.source()
    plan = read(OUT / "pairs.json")
    checked_specs(plan)
    campaign.guarded("pilot-freshness", [sys.executable, "-m", MODULE, "freshness-worker",
                                       "--label", "pilot", "--blocks", 524288])
    for rung, pair in plan["pairs"].items():
        # PR226's largest fresh dual-policy pilot used <47.4 family B/key.
        # 52 B/key plus200MB supplies headroom without rejecting every4B pair.
        bound = 52 * sum(p["entries"] for p in pair) + 200_000_000
        if bound >= 7 * GIB:
            raise ValueError("Preventive dual-model memory admission; no pilot launched")
        for blocks in (32, 512):
            campaign.guarded(f"pilot-{rung}-{blocks}", [sys.executable, "-m", "scripts.evaluate_hu100_direct",
                "--specs", OUT / "specs" / (rung+".json"), "--blocks", blocks,
                "--root", PILOT_ROOT, "--rung", rung+f"-timing-{blocks}",
                "--out", OUT / "direct-pilot" / rung / str(blocks),
                "--source", revision, "--pilot"])
    campaign.guarded("scripted-pilot", [sys.executable, "-m", MODULE, "panel-pilot-worker"])


def raw_values(path, blocks):
    values = {}
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            coordinate = row["block"], row["rotation"]
            if coordinate in values or row["status"] != "completed":
                raise ValueError("STOP invalid direct block coordinate")
            values[coordinate] = row["candidate_chips"]
    if set(values) != {(b,r) for b in range(blocks) for r in (0,1)}:
        raise ValueError("STOP incomplete direct paired coordinates")
    return [(values[b,0]+values[b,1])/2 for b in range(blocks)]


def adjusted(series, confidence):
    center = mean(series)
    margin = float(t.ppf((1+confidence)/2, len(series)-1))*stdev(series)/math.sqrt(len(series))
    return {"bb_per_100": center, "interval": [center-margin, center+margin],
            "half_width": margin, "confidence": confidence}


def exclusions():
    pinned = {p["sha256"]: p for p in read(campaign.INDEX)["assets"]}
    found = []
    for base in (OUT / "training", OUT / "inputs"):
        for path in base.rglob("*.gz"):
            sha = file_hash(path)
            if sha in pinned:
                if path.stat().st_size != pinned[sha]["bytes"]:
                    raise ValueError("STOP indexed model size differs")
                found.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size,
                              "sha256": sha, "reference_index": str(campaign.INDEX.relative_to(ROOT)),
                              "asset": pinned[sha]})
    return found


def freeze():
    revision = campaign.source()
    plan = read(OUT / "pairs.json")
    rates = {}
    for rung in plan["pairs"]:
        small, large = (read(OUT / "direct-pilot" / rung / str(n) / "costs.json") for n in (32,512))
        if any(c["outcomes_inspected_for_quote"] for c in (small, large)):
            raise ValueError("Cost-only pilots required")
        fixed = max(sum(c[a]["load_or_validation_seconds"] for a in ("primary","repeat")) for c in (small,large))
        variable = max(max(0,c["seconds"]-sum(c[a]["load_or_validation_seconds"] for a in ("primary","repeat")))/c["blocks"]
                       for c in (small,large))
        raw = max(sum(p["bytes"] for a in ("primary","repeat") for p in c[a]["files"].values())/c["blocks"]
                  for c in (small,large))
        peak = max(read(OUT / "guards/operations" / f"pilot-{rung}-{n}/receipt.json")["peak_family_rss_bytes"]
                   for n in (32,512))
        rates[rung] = {"fixed_seconds": fixed, "seconds_per_block": variable,
                      "raw_bytes_per_block": raw, "pilot_peak_family_bytes": peak}
    blocks, choices = {}, (524288,262144,131072)
    for rung, rate in rates.items():
        selected = next((n for n in choices if rate["pilot_peak_family_bytes"] + 5200*n + 100_000_000 < 7*GIB), None)
        if selected is None:
            raise ValueError("Preventive final memory admission; no final hand launched")
        blocks[rung] = selected
    omitted = exclusions()
    from scripts.archive_hu100_4b_seed_replication import inventory, calibration_receipt
    archive_files, archive_omitted, archive_aliases = inventory()
    base = sum(path.stat().st_size for path, member in archive_files)
    panel_cost = read(OUT / "scripted/secondary-pilot/costs.json")
    fixed_panel = sum(m["load_seconds"] for m in panel_cost["measurements"])
    # Existing panel exposes phase costs; loading is fixed, remaining pilot
    # costs conservatively scale to4096 without inspecting any outcomes.
    variable_panel = sum(m["variable_and_fixed_seconds"] for m in panel_cost["measurements"])/32
    script_quote = 2 * (fixed_panel + 4096 * variable_panel)
    script_pilot_root = OUT / "scripted/secondary-pilot"
    seen_inodes, script_raw = set(), 0
    for path in script_pilot_root.rglob("*"):
        if not path.is_file() or path.is_symlink() or "models" in path.relative_to(script_pilot_root).parts:
            continue
        stat = path.stat()
        inode = stat.st_dev, stat.st_ino
        if inode not in seen_inodes:
            seen_inodes.add(inode)
            script_raw += stat.st_size
    script_raw *= 4096/32
    def need():
        direct = sum(blocks[r]*rates[r]["raw_bytes_per_block"] for r in rates)
        return 16*GIB + base + 2*math.ceil(direct+script_raw) + 2*GIB
    free = shutil.disk_usage(OUT).free
    reductions = []
    while free < need():
        options = [r for r in plan["descriptive"] if blocks[r] > choices[-1]]
        if not options:
            raise ValueError("Preventive final storage admission; required primaries/safeguard preserved")
        rung = max(options, key=lambda r:blocks[r]*rates[r]["raw_bytes_per_block"])
        before=blocks[rung]; blocks[rung]//=2
        reductions.append({"rung":rung,"from":before,"to":blocks[rung],"reason":"disk"})
    quotes = {r:2*(rates[r]["fixed_seconds"]+blocks[r]*rates[r]["seconds_per_block"]) for r in rates}
    archive = calibration_receipt()
    archive_bytes = base + math.ceil(sum(blocks[r]*rates[r]["raw_bytes_per_block"] for r in rates)+script_raw)
    archive_quote = 3*archive["seconds_including_source_hash_pack_readback_whole_hash"]*max(1,archive_bytes/archive["logical_member_bytes"])
    result = {"source":revision,"pairs_sha256":file_hash(OUT/"pairs.json"),
        "executable_spec_sha256":checked_specs(plan),"blocks_per_contrast":blocks,
        "primary":plan["primary"],"descriptive":plan["descriptive"],
        "primary_confidence":.975,"scripted_blocks_per_opponent":4096,
        "scripted_translation":{"max_states":512,"max_events":128},
        "direct_root":FINAL_ROOT,"scripted_root":SCRIPT_FINAL_ROOT,"pilot_root":PILOT_ROOT,
        "quotes_seconds":quotes,"scripted_quote_seconds":script_quote,
        "archive_quote_seconds":archive_quote,"archive_projected_bytes":archive_bytes,
        "archive_quote_basis":"3x fresh multiGB source hash/write/all-member readback/whole hash",
        "outcomes_used_for_budget":False,"timing_rates":rates,
        "storage":{"available_bytes":free,"required_additional_free_bytes":need()},
        "indexed_model_exclusions":omitted,"descriptive_disk_reductions":reductions,
        "planning_sd_bb_per_100":1113,
        "projected_primary_half_width":{r:float(t.ppf(.9875,blocks[r]-1))*1113/math.sqrt(blocks[r]) for r in plan["primary"]}}
    completed = [read(path) for path in (OUT/"guards/operations").glob("*/receipt.json")]
    result["completed_operations_seconds"] = sum(item["seconds"] for item in completed)
    result["completed_training_export_audit_seconds"] = sum(
        item["seconds"] for item in completed
        if item["name"].startswith(("train-", "export-", "audit-")))
    # Reporting is bounded from the pilot's full replay/reproduction and raw
    # rehash costs, with the same measured fixed+scaled allowance as final play.
    result["reporting_quote_seconds"] = max(quotes.values()) + script_quote
    result["archive_calibration"] = archive
    freshness(blocks,"final")
    result["freshness_sha256"] = file_hash(OUT/"final-freshness.json")
    put(OUT / "frozen-final.json",result)


def final():
    revision=campaign.source()
    freeze=read(OUT/"frozen-final.json")
    posting=read(ROOT/"planning/final-quote-posted.json")
    review=read(ROOT/"planning/source-review.json")
    if (posting.get("quote_sha256")!=file_hash(OUT/"frozen-final.json")
        or posting.get("pr")!=236 or not posting.get("url")):
        raise ValueError("Exact quote must be posted before final play")
    if review.get("status")!="clear" or review.get("source")!=revision or review.get("open_findings"):
        raise ValueError("Independent source clearance required")
    if freeze["source"]!=revision or freeze["pairs_sha256"]!=file_hash(OUT/"pairs.json"):
        raise ValueError("Frozen source/models differ")
    checked_specs(read(OUT/"pairs.json"),freeze["executable_spec_sha256"])
    if (file_hash(OUT/"final-freshness.json") != freeze["freshness_sha256"]
            or not read(OUT/"final-freshness.json")["all_pairwise_disjoint"]):
        raise ValueError("Freshness proof required")
    if shutil.disk_usage(OUT).free<freeze["storage"]["required_additional_free_bytes"]:
        raise ValueError("Preventive final storage admission")
    # Required evidence runs first; all descriptive samples are already fixed.
    order=[*freeze["primary"],"scripted",*freeze["descriptive"]]
    for name in order:
        if name=="scripted":
            campaign.guarded("final-scripted",[sys.executable,"-m",MODULE,"panel-final-worker"],
                             freeze["scripted_quote_seconds"])
        else:
            campaign.guarded("final-direct-"+name,[sys.executable,"-m","scripts.evaluate_hu100_direct",
                "--specs",OUT/"specs"/(name+".json"),"--blocks",freeze["blocks_per_contrast"][name],
                "--root",FINAL_ROOT,"--rung",name,"--out",OUT/"final-direct"/name,"--source",revision],
                freeze["quotes_seconds"][name])
    campaign.guarded("final-readout",[sys.executable,"-m",MODULE,"readout-worker"],
                     freeze["reporting_quote_seconds"])


def verify_scripted(base, blocks, root):
    """Reconcile full replay counts and rehash both immutable raw arms."""
    bindings = {}
    for label in ("terminal", "1b"):
        replay_path = base / (label + "-replay.json")
        replay = read(replay_path)
        counts = []
        pins = {}
        for suffix in (label, label + "-reproduction"):
            folder = base / suffix
            complete = read(folder / "complete.json")
            if (complete["status"] != "complete" or complete["blocks_per_opponent"] != blocks
                    or complete["root"] != root
                    or complete["hands"] != replay["hands_replayed"]
                    or complete["decisions"] != replay["actions_replayed"]
                    or complete["hands"] != blocks * 2 * 2 * 5):
                raise ValueError("STOP scripted replay/reproduction counts differ")
            if suffix.endswith("-reproduction") and not complete["reproduced_all_hands_and_decisions"]:
                raise ValueError("STOP scripted reproduction incomplete")
            outputs = read(folder / "output-files.json")
            if "complete.json" not in outputs:
                raise ValueError("STOP scripted complete receipt missing from output pins")
            for filename, pin in outputs.items():
                path = folder / filename
                if path.stat().st_size != pin["bytes"] or file_hash(path) != pin["sha256"]:
                    raise ValueError("STOP scripted evidence changed after verification")
            counts.append({"run": suffix, "hands": complete["hands"], "decisions": complete["decisions"]})
            pins[suffix] = file_hash(folder / "output-files.json")
        if replay["status"] != "verified" or not replay["all_settlements_replayed"]:
            raise ValueError("STOP scripted replay incomplete")
        bindings[label] = {"replay_sha256": file_hash(replay_path), "output_pin_sha256": pins,
                           "counts": counts}
    return bindings


def readout():
    freeze=read(OUT/"frozen-final.json")
    results={}
    for name in [*freeze["primary"],*freeze["descriptive"]]:
        base=OUT/"final-direct"/name
        replay,repeat=read(base/"replay.json"),read(base/"reproduction/complete.json")
        n=freeze["blocks_per_contrast"][name]
        if (replay["status"]!="verified" or replay["blocks"]!=n
            or not repeat["reproduced_all_hands_and_decisions"]
            or repeat["hands"]!=replay["hands_replayed"] or repeat["decisions"]!=replay["actions_replayed"]):
            raise ValueError("STOP complete replay/reproduction required")
        primary=name in freeze["primary"]
        done=read(base/"play/complete.json")
        for folder,complete in ((base/"play",done),(base/"reproduction",repeat)):
            for filename,pin in complete["files"].items():
                path=folder/filename
                if path.stat().st_size!=pin["bytes"] or file_hash(path)!=pin["sha256"]:
                    raise ValueError("STOP final raw evidence changed after verification")
        results[name]={**adjusted(raw_values(base/"play/hands.jsonl.gz",n),.975 if primary else .95),
            "primary":primary,"blocks":n,"hands_replayed":replay["hands_replayed"],
            "actions_replayed":replay["actions_replayed"],"coverage":replay["coverage"],
            "all_hands_reproduced":True}
    panel=read(OUT/"scripted-summary.json")
    bindings = verify_scripted(OUT/"scripted/secondary", freeze["scripted_blocks_per_opponent"],
                               freeze["scripted_root"])
    if panel["evidence_bindings"] != bindings:
        raise ValueError("STOP scripted replay/output bindings changed")
    put(OUT/"summary.json",{"status":"verified","results":results,"scripted":panel,
        "source":freeze["source"],"freeze_sha256":file_hash(OUT/"frozen-final.json"),
        "all_final_hands_replayed_and_reproduced":True,"training_seed_population_inference":False,
        "both_primary_lower_bounds_positive":all(results[r]["interval"][0]>0 for r in freeze["primary"])})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation",choices=("prepare","restore-worker","specs-worker","calibrate",
        "freshness-worker","panel-pilot-worker","panel-final-worker","freeze","freeze-worker","final","readout-worker"))
    p.add_argument("--label",choices=("pilot","final"))
    p.add_argument("--blocks",type=int)
    a=p.parse_args()
    actions={"prepare":prepare,"restore-worker":restore,"specs-worker":specs,"calibrate":calibrate,
             "panel-pilot-worker":lambda:panel(True),"panel-final-worker":lambda:panel(False),
             "freeze-worker":freeze,"final":final,"readout-worker":readout}
    if a.operation=="freeze":
        campaign.guarded("freeze", [sys.executable, "-m", MODULE, "freeze-worker"])
    elif a.operation=="freshness-worker":
        freshness({r:a.blocks for r in read(OUT/"pairs.json")["pairs"]},a.label)
    else:
        actions[a.operation]()


if __name__=="__main__":
    main()
