"""Cost-only sample freeze and final comparisons for PR226."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from time import time

from scripts import run_hu100_4b_seed_ladder as campaign
from scripts.evaluate_hu100_direct import make_plan, put
from scripts.report_native_hu100_learning_curves import frozen_schedule
from src.arena.schedule import build_schedule
from src.policies.files import file_hash

ROOT, OUT, MODULE = campaign.ROOT, campaign.OUT, "scripts.evaluate_hu100_4b_seed_ladder"
GIB = 1024**3
read = campaign.read


def prior_deals(specs):
    roots = read(ROOT/"docs/reports/hu100-independent-stages-artifacts/freshness.json")["roots"]
    settings = {"models": [specs[0]]}
    seen = set()
    for root, blocks in roots.items():
        doc = frozen_schedule(settings, blocks, int(root))
        seen.update(b["deal_seeds"][0] for panel in doc["panels"].values() for b in panel["blocks"])
        del doc
    # #223's two completed direct rungs, plus the prospectively reserved third,
    # are excluded conservatively; its scripted secondary uses five scenarios.
    dummy = [dict(specs[0], name="prior-candidate"), dict(specs[0], name="prior-baseline")]
    direct_counts = {2026100922201: 32, 2026100922202: 524288}
    for root, blocks in direct_counts.items():
        for rung in ("terminal-vs-1b", "1b-vs-500m", "2b-vs-1b"):
            schedule = build_schedule(make_plan(dummy, blocks, root, rung))
            seen.update(b.deal_seeds[0] for b in schedule)
            del schedule
    for root, blocks in ((2026100922203, 32), (2026100922204, 4096)):
        doc = frozen_schedule(settings, blocks, root)
        seen.update(b["deal_seeds"][0] for panel in doc["panels"].values() for b in panel["blocks"])
        del doc
    return seen, {"scripted_prior_roots": roots, "pr223_direct_roots": direct_counts,
        "pr223_direct_rungs": ["terminal-vs-1b", "1b-vs-500m", "2b-vs-1b"],
        "pr223_scripted_roots": {2026100922203: 32, 2026100922204: 4096}}


def freshness(blocks, label):
    pairs = read(OUT/"pairs.json")["pairs"]
    seen, prior = prior_deals(next(iter(pairs.values())))
    old_count = len(seen)
    hashes = {}
    for root, count in ((campaign.PILOT_ROOT, 32), (campaign.FINAL_ROOT, blocks)):
        for rung, specs in pairs.items():
            schedule = build_schedule(make_plan(specs, count, root, rung))
            seeds = {b.deal_seeds[0] for b in schedule}
            if len(seeds) != count or seen.intersection(seeds):
                raise ValueError("STOP physical deal collision before play")
            hashes[f"{root}/{rung}"] = hashlib.sha256(
                b"".join(n.to_bytes(16, "big") for n in sorted(seeds))).hexdigest()
            seen.update(seeds)
            del schedule, seeds
    put(OUT/(label+"-freshness.json"), {"all_pairwise_disjoint": True,
        "prior": prior, "prior_physical_deals": old_count,
        "new_blocks": blocks, "deal_seed_hashes": hashes,
        "new_physical_deals": len(seen)-old_count})


def calibrate():
    revision = campaign.source()
    # Review receives all campaign sources, including these cost gates, before
    # any final hand. Pilots never produce a payoff report.
    campaign.guarded("pilot-freshness", [sys.executable, "-m", MODULE, "freshness",
        "--blocks", 32, "--label", "pilot"])
    for rung, specs in read(OUT/"pairs.json")["pairs"].items():
        # #223's largest two-policy pilot measured 41.9 RSS B/stored key.
        # 55 B/key plus 200 MB provides >30% headroom before attempting
        # a guarded pilot; actual peaks then admit the final schedule.
        bound = 55*sum(spec["entries"] for spec in specs)+200_000_000
        if bound >= 7*GIB:
            raise ValueError("Preventive dual-model pilot memory admission")
        campaign.guarded("direct-pilot-"+rung, [sys.executable, "-m", "scripts.evaluate_hu100_direct",
            "--specs", OUT/"specs"/(rung+".json"), "--blocks", 32, "--root", campaign.PILOT_ROOT,
            "--rung", rung, "--out", OUT/"direct-pilot"/rung, "--source", revision, "--pilot"])


def unique_bytes(root):
    seen, total = set(), 0
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        stat = path.stat()
        inode = stat.st_dev, stat.st_ino
        if inode not in seen:
            seen.add(inode)
            total += stat.st_size
    return total


def freeze():
    revision = campaign.source()
    plan = read(OUT/"pairs.json")
    costs = {r: read(OUT/"direct-pilot"/r/"costs.json") for r in plan["pairs"]}
    if any(c["outcomes_inspected_for_quote"] or c["blocks"] != 32 for c in costs.values()):
        raise ValueError("Cost-only fixed pilot required")
    rates = {}
    for rung, cost in costs.items():
        fixed = sum(cost[arm]["load_or_validation_seconds"] for arm in ("primary", "repeat"))
        rates[rung] = {"fixed_seconds": fixed,
            "seconds_per_block": max(0, cost["seconds"]-fixed)/32,
            "raw_bytes_per_block": sum(pin["bytes"] for arm in ("primary", "repeat")
                for pin in cost[arm]["files"].values())/32,
            "pilot_family_peak_bytes": read(OUT/"guards/operations"/("direct-pilot-"+rung)/"receipt.json")["peak_family_rss_bytes"]}
    history = read(ROOT/"docs/reports/hu100-3b-ladder-artifacts/model-input-index.json")
    existing_pins = {pin["sha256"]: pin["bytes"] for model in history["models"] for pin in model["files"].values()}
    other = read(ROOT/"docs/reports/hu100-independent-stages-artifacts/model-index.json")
    existing_pins.update({asset["sha256"]: asset["bytes"] for asset in other["assets"]})
    indexed_bytes = 0
    exclusions = []
    for path in (OUT/"training").glob("*/*/*.gz"):
        actual = file_hash(path)
        if actual in existing_pins:
            if path.stat().st_size != existing_pins[actual]:
                raise ValueError("STOP indexed model size mismatch")
            indexed_bytes += path.stat().st_size
            exclusions.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": actual})
    archive_base = unique_bytes(OUT)-indexed_bytes
    free = shutil.disk_usage(OUT).free
    candidates = []
    for blocks in (1_048_576, 524_288, 262_144, 131_072):
        memory = {r: rates[r]["pilot_family_peak_bytes"]+2048*blocks+100_000_000 for r in rates}
        raw = math.ceil(blocks*sum(r["raw_bytes_per_block"] for r in rates.values()))
        required = 16*GIB+archive_base+2*raw+2*GIB
        candidates.append({"blocks": blocks, "forecast_family_bytes": memory,
            "required_additional_free_bytes": required,
            "memory_admitted": max(memory.values()) < 7*GIB,
            "disk_admitted": free >= required})
    admitted = next((c for c in candidates if c["memory_admitted"] and c["disk_admitted"]), None)
    if admitted is None:
        put(OUT/"final-storage-memory-inadmission.json", {"candidates": candidates, "available_bytes": free})
        raise ValueError("Preventive final storage/memory admission")
    blocks = admitted["blocks"]
    operations = [read(p) for p in (OUT/"guards/operations").glob("*/receipt.json")]
    elapsed_operations = sum(o["seconds"] for o in operations)
    quotes = {r: 2*(rate["fixed_seconds"]+blocks*rate["seconds_per_block"]) for r, rate in rates.items()}
    archive_pilot = read(OUT/"archive-pilot-cost.json")
    archive_bytes = archive_base+math.ceil(blocks*sum(r["raw_bytes_per_block"] for r in rates.values()))+2*GIB
    archive_quote = 2*archive_pilot["seconds_including_source_hash_pack_readback_whole_hash"]*archive_bytes/archive_pilot["logical_member_bytes"]
    timestamps = subprocess.check_output(["git", "reflog", "--format=%ct"], text=True).splitlines()
    elapsed_checkout_wall = time()-min(map(int, timestamps))
    total = elapsed_checkout_wall+sum(quotes.values())+archive_quote
    # The ten-hour threshold changes order only: all authorized descriptive
    # comparisons follow primaries. This fixed workflow omits optional scripts.
    put(OUT/"frozen-final.json", {"source": revision, "blocks_per_contrast": blocks,
        "root": campaign.FINAL_ROOT, "pilot_root": campaign.PILOT_ROOT,
        "primary": plan["primary"], "descriptive": plan["descriptive"],
        "primary_confidence": 1-.05/3, "decision": "each adjusted paired Student-t lower bound >0",
        "quotes_seconds": quotes, "completed_operations_seconds": elapsed_operations,
        "full_scope_seconds": total, "archive_quote_seconds": archive_quote,
        "archive_quote_basis": "2x measured timing ZIP source hash/write/member readback/whole hash per logical byte",
        "elapsed_checkout_wall_seconds_including_preparation_and_storage_wait": elapsed_checkout_wall,
        "primary_first_due_to_ten_hour_scope": total > 10*3600,
        "optional_scripted_panel": "omitted; required primary and descriptive matches take priority",
        "planning_sd_bb_per_100": 1113,
        "projected_adjusted_half_width": 2.394*1113/math.sqrt(blocks),
        "outcomes_used_for_budget": False, "pilot_costs": rates,
        "storage": {"available_bytes": free, "archive_base_bytes": archive_base,
            "indexed_model_bytes_excluded_from_zip": indexed_bytes,
            "required_additional_free_bytes": admitted["required_additional_free_bytes"],
            "raw_and_archive_copies": 2, "floor_bytes": 16*GIB},
        "admission_candidates": candidates, "indexed_model_exclusions": exclusions,
        "pairs_sha256": file_hash(OUT/"pairs.json")})
    freshness(blocks, "final")


def final():
    revision = campaign.source()
    campaign.posted("frozen-final")
    freeze = read(OUT/"frozen-final.json")
    review = read(ROOT/"planning/source-review.json")
    if review.get("status") != "clear" or review.get("source") != revision or review.get("open_findings"):
        raise ValueError("Independent clear exact-source review required before final play")
    if freeze["source"] != revision or freeze["pairs_sha256"] != file_hash(OUT/"pairs.json"):
        raise ValueError("Frozen source/models changed")
    if not read(OUT/"final-freshness.json")["all_pairwise_disjoint"]:
        raise ValueError("Final physical freshness receipt required")
    if shutil.disk_usage(OUT).free < freeze["storage"]["required_additional_free_bytes"]:
        raise ValueError("Preventive final disk admission")
    deadline = time()+sum(freeze["quotes_seconds"].values())
    for rung in [*freeze["primary"], *freeze["descriptive"]]:
        remaining = min(freeze["quotes_seconds"][rung], deadline-time())
        if remaining <= 0:
            raise ValueError("Frozen campaign budget exhausted")
        campaign.guarded("final-direct-"+rung, [sys.executable, "-m", "scripts.evaluate_hu100_direct",
            "--specs", OUT/"specs"/(rung+".json"), "--blocks", freeze["blocks_per_contrast"],
            "--root", freeze["root"], "--rung", rung, "--out", OUT/"final-direct"/rung,
            "--source", revision], remaining)
    campaign.guarded("readout", [sys.executable, "-m", MODULE, "readout"])


def readout():
    freeze = read(OUT/"frozen-final.json")
    results = {}
    for rung in [*freeze["primary"], *freeze["descriptive"]]:
        base = OUT/"final-direct"/rung
        report = read(base/"replay.json")
        repeat = read(base/"reproduction/complete.json")
        if report["status"] != "verified" or not repeat["reproduced_all_hands_and_decisions"]:
            raise ValueError("Every final hand must replay and reproduce")
        if report["blocks"] != freeze["blocks_per_contrast"]:
            raise ValueError("Frozen blocks differ")
        primary = rung in freeze["primary"]
        interval = report["bonferroni_three_ci"] if primary else report["ci95"]
        results[rung] = {"primary": primary, "bb_per_100": report["bb_per_100"],
            "confidence": 1-.05/3 if primary else .95, "interval": interval,
            "decision": "improves" if interval[0] > 0 else "declines" if interval[1] < 0 else "inconclusive",
            "blocks": report["blocks"], "half_width": (interval[1]-interval[0])/2,
            "hands_replayed": report["hands_replayed"], "actions_replayed": report["actions_replayed"],
            "coverage": report["coverage"], "all_hands_reproduced": True}
    put(OUT/"summary.json", {"status": "verified", "results": results,
        "all_final_hands_replayed_and_reproduced": True,
        "historical_seed_2026100601_2b_minus_1b": {"bb_per_100": 29.51, "confidence": .95,
            "interval": [26.82, 32.20], "owning_pr": 223, "same_schedule": False},
        "source": freeze["source"], "freeze_sha256": file_hash(OUT/"frozen-final.json"),
        "training_seed_population_inference": False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("calibrate", "freeze", "freshness", "final", "readout"))
    p.add_argument("--blocks", type=int)
    p.add_argument("--label")
    args = p.parse_args()
    if args.command == "freshness":
        if args.blocks is None or args.label not in ("pilot", "final"):
            p.error("Freshness requires positive blocks and pilot/final label")
        freshness(args.blocks, args.label)
    else:
        globals()[args.command]()


if __name__ == "__main__":
    main()
