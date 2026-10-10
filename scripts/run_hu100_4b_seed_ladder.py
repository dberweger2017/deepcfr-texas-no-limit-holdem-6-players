"""One-use PR226 M4 campaign; exact gates precede every dependent operation."""
import argparse
import gzip
import json
import math
import os
import psutil
from pathlib import Path
import shutil
import subprocess
import sys
from time import time

from scripts.evaluate_hu100_direct import put, make_plan
from scripts.native_hu100_model_metadata import audited_average_spec
from src.policies.files import file_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/hu100-4b-seed-ladder"
BINARY = ROOT / "build/native-trainer/release/hu20-trainer"
MODULE = "scripts.run_hu100_4b_seed_ladder"
GIB = 1024**3
PR = 226
SEEDS = (2026100601, 2026100901, 2026100902)
GATE_CAP = 57_658_644
REFERENCE_2B_CAP = 67_419_934
CAP = (7*GIB-200_000_000)//101
PILOT_ROOT, FINAL_ROOT = 202610104401, 202610104402
PINS = {
    (2026100601, 1_000_000_000): "cca0b54a609f47c60b29e9fe5a920475a91ec641df2ff0e3ea48fdac615147ec",
    (2026100601, 2_000_000_000): "e84039c7a934a966a2c01f237748a23951124a8b665edc2585ef4809e5681676",
    (2026100901, 1_000_000_000): "5f4898d839011d62fda20e3df7e1546150ca1d72dd666654a6d4bfcaaf63a6e7",
    (2026100902, 1_000_000_000): "c21e6dd51694b379b28cb5dcc2ab595fa154092f1b69a4cfd1a9149fc85b23e4",
}


def read(path):
    return json.loads(Path(path).read_text())


def source():
    if Path.cwd() != ROOT:
        raise ValueError("Campaign checkout required")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("Clean committed source required")
    for key, expected in (("machdep.cpu.brand_string", "Apple M4"), ("hw.ncpu", "10"), ("hw.memsize", str(16*GIB))):
        if subprocess.check_output(["sysctl", "-n", key], text=True).strip() != expected:
            raise ValueError("Only the free 10-core/16-GiB M4 is authorized")
    if subprocess.check_output(["git", "diff", "origin/main", "--", "native/hu20-trainer"], text=True).strip():
        raise ValueError("Native trainer source must remain unchanged")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def guard_main():
    # Keep the reviewed ancestry/limits/cleanup implementation, with this
    # checkout's binary outside the protected native directory.
    from scripts import overnight_research_guard as guard
    from scripts import run_native_hu100_growth_1b as resources
    resources.BINARY = BINARY
    original_host = resources.host
    original_family = guard.owned_processes
    parent_pid = os.environ.get("HU100_PR226_CONTROLLER_PID")
    parent_created = os.environ.get("HU100_PR226_CONTROLLER_CREATED")
    def controller():
        if parent_pid is None:
            return None
        try:
            process = psutil.Process(int(parent_pid))
            return process if process.create_time() == float(parent_created) and process.status() != psutil.STATUS_ZOMBIE else None
        except psutil.NoSuchProcess:
            return None
    def family(known):
        members = original_family(known)
        parent = controller()
        if parent is not None and parent.pid not in {p.pid for p in members}:
            members.append(parent)
        # The parent contributes to RSS, but never enters the descendant
        # cleanup set: successful child cleanup must not kill its controller.
        return members
    guard.owned_processes = family
    def host():
        if parent_pid is not None and controller() is None:
            raise RuntimeError("Campaign controller identity disappeared")
        sample = original_host()
        sample["disk_free_bytes"] = min(sample["disk_free_bytes"], shutil.disk_usage(ROOT).free)
        return sample
    resources.host = host
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    guard.main()


def guarded(name, command, seconds=None, stop=None):
    args = [sys.executable, "-m", MODULE, "guard", "--out", OUT/"guards", "--name", name]
    if seconds is not None:
        args += ["--seconds", str(seconds)]
    if stop is not None:
        args += ["--stop-file", stop, "--accept-capacity-stop"]
    env = {**os.environ, "HU100_PR226_CONTROLLER_PID": str(os.getpid()),
        "HU100_PR226_CONTROLLER_CREATED": str(psutil.Process(os.getpid()).create_time())}
    subprocess.run([*map(str, args), "--", *map(str, command)], check=True, cwd=ROOT, env=env)


def cap_for(seed, target):
    if target <= 1_000_000_000:
        return GATE_CAP
    return REFERENCE_2B_CAP if (seed, target) == (SEEDS[0], 2_000_000_000) else CAP


def folder(seed, target):
    return OUT/"training"/str(seed)/str(target)


def train_command(seed, target, dest, resume=None):
    command = [BINARY, "train", "--stack-bb", "100", "--seed", seed,
        "--roots-per-seat", "1", "--average-rule", "opponent-sampled", "--nodes", target,
        "--max-entries", cap_for(seed, target), "--out", dest/"checkpoint.gz",
        "--telemetry", dest/"telemetry.jsonl", "--stop-file", dest/"stop.json"]
    if resume is not None:
        command += ["--resume", resume, "--resume-sha256", file_hash(resume)]
    return command


def telemetry(dest):
    rows = [json.loads(line) for line in (dest/"telemetry.jsonl").read_text().splitlines()]
    if len(rows) != 1:
        raise ValueError("One terminal telemetry row required")
    return rows[0]


def check_gate(seed, target):
    dest = folder(seed, target)
    row = telemetry(dest)
    actual = file_hash(dest/"checkpoint.gz")
    if row["checkpoint_sha256"] != actual or row["binary_sha256"] != file_hash(BINARY):
        raise ValueError("STOP checkpoint telemetry/binary mismatch")
    expected = PINS.get((seed, target))
    if expected is not None and actual != expected:
        put(dest/"exactness-mismatch.json", {"expected": expected, "actual": actual})
        raise ValueError("STOP exact checkpoint gate mismatch; no science retry")
    entries = row["diagnostics"]["entries"]
    stopped = row["completed_nodes"] < target
    if stopped and seed != SEEDS[0]:
        raise ValueError("STOP independent-seed 2B target not reached; retain checkpoint without substituting an endpoint")
    if stopped and not (entries >= cap_for(seed, target) or row["stop_requested"]):
        raise ValueError("STOP unexplained incomplete native target")
    if row["completed_nodes"] >= target and row["status"] != "saved":
        raise ValueError("STOP target status differs")
    put(dest/"gate.json", {"status": "passed", "sha256": actual, "expected": expected,
        "actual_nodes": row["completed_nodes"], "entries": entries,
        "terminal_capacity_stop": stopped, "entry_cap": cap_for(seed, target),
        "stop_requested": row["stop_requested"]})


def check_reference_exports(seed, target):
    if (seed, target) not in PINS:
        return
    if seed == SEEDS[0]:
        index = read(ROOT/"docs/reports/hu100-3b-ladder-artifacts/model-input-index.json")
        model = next(m for m in index["models"] if m["requested_nodes"] == target)
        pins = model["files"]
        reference = "docs/reports/hu100-3b-ladder-artifacts/model-input-index.json"
    else:
        index = read(ROOT/"docs/reports/hu100-independent-stages-artifacts/model-index.json")
        pins = {a["kind"]: a for a in index["assets"]
            if a.get("seed") == seed and a.get("endpoint") == "terminal"}
        reference = "docs/reports/hu100-independent-stages-artifacts/model-index.json"
    if set(pins) != {"checkpoint.gz", "current.gz", "average.gz"}:
        raise ValueError("Reference model set incomplete")
    verified = {}
    for name, pin in pins.items():
        path = folder(seed, target)/name
        actual = file_hash(path)
        if actual != pin["sha256"] or path.stat().st_size != pin["bytes"]:
            put(folder(seed, target)/"export-exactness-mismatch.json",
                {"file": name, "expected": pin["sha256"], "actual": actual, "reference": reference})
            raise ValueError("STOP exact indexed checkpoint/current/average mismatch")
        verified[name] = {"sha256": actual, "bytes": pin["bytes"]}
    put(folder(seed, target)/"indexed-reference.json",
        {"status": "byte-exact", "index": reference, "files": verified})


def model_spec(seed, target):
    check_reference_exports(seed, target)
    dest = folder(seed, target)
    gate = read(dest/"gate.json")
    spec = audited_average_spec(dest/"average.gz", read(dest/"audit.json"),
        checkpoint_sha256=gate["sha256"], actual_nodes=gate["actual_nodes"])
    spec["name"] = f"HU100-seed-{seed}-average-{gate['actual_nodes']}"
    spec["seed"] = seed
    put(dest/"spec.json", spec)


def pilot():
    revision = source()
    OUT.mkdir(parents=True, exist_ok=True)
    admission = read(ROOT/"docs/reports/hu100-4b-seed-ladder-artifacts/storage-admission.json")
    forecast = admission["forecast"]
    extra_entries = max(0, CAP-forecast["four_b_entries"])
    extra_model_copies = math.ceil(2*extra_entries*admission["historical_inputs"]["measured_2b_model_set_bytes"]
        /admission["historical_inputs"]["measured_2b_entries"])
    required = forecast["required_initial_free_bytes"]+extra_model_copies
    if shutil.disk_usage(OUT).free < required:
        raise ValueError("Preventive disk admission: provision campaign storage before pilot")
    put(OUT/"launch-admission.json", {"source": revision, "free_bytes": shutil.disk_usage(OUT).free,
        "required_bytes": required, "historical_forecast": admission["forecast"],
        "binary_sha256": file_hash(BINARY)})
    dest = OUT/"timing-training"
    dest.mkdir(exist_ok=False)
    guarded("pilot-training", train_command(SEEDS[0], 1_000_000, dest), stop=dest/"stop.json")
    row = telemetry(dest)
    rate = row["completed_nodes"]/max(.001, row["elapsed_seconds_including_writes"]-row["write_seconds"])
    guarded("pilot-archive", [sys.executable, "-m", "scripts.archive_hu100_4b_seed_ladder", "pilot"])
    history = read(ROOT/"docs/reports/hu100-3b-ladder-artifacts/training-quote.json")
    entries = {}
    for seed in SEEDS:
        for target in ((1_000_000_000, 2_000_000_000, 4_000_000_000) if seed == SEEDS[0] else (1_000_000_000, 2_000_000_000)):
            entries[f"{seed}/{target}"] = min(cap_for(seed, target), 41_100_000 if target == 1_000_000_000 else 55_500_000 if target == 2_000_000_000 else CAP)
    train_seconds = 2*8_000_000_000/min(rate, 1_700_000)
    save_seconds = history["save_seconds_per_entry"]*sum(entries.values())
    tools_seconds = history["tool_seconds_per_entry"]*sum(entries.values())
    # This conservative historical full-audit proxy is labelled, not claimed
    # to measure large-checkpoint native loading.
    loads_seconds = 2*923*(3*41_100_000+55_500_000)/41_010_014
    put(OUT/"training-quote.json", {"source": revision, "fresh_pilot_nodes_per_second": rate,
        "entries": entries, "training_seconds": train_seconds, "save_seconds": save_seconds,
        "export_audit_seconds": tools_seconds, "resume_load_proxy_seconds": loads_seconds,
        "total_seconds": train_seconds+save_seconds+tools_seconds+loads_seconds,
        "save_seconds_per_entry": history["save_seconds_per_entry"],
        "tool_seconds_per_entry": history["tool_seconds_per_entry"],
        "storage_admission_bytes": required,
        "basis": "Fresh 1M timing and historical #223/#207 rates with 2x headroom; explicitly conservative resume proxy",
        "outcomes_used_for_budget": False})


def posted(name):
    quote = OUT/(name+".json")
    receipt = read(ROOT/"planning"/(name+"-posted.json"))
    if receipt.get("pr") != PR or receipt.get("quote_sha256") != file_hash(quote) or not receipt.get("url"):
        raise ValueError("Exact quote must appear on the owning PR before dependent science")


def train():
    source()
    posted("training-quote")
    quote = read(OUT/"training-quote.json")
    if shutil.disk_usage(OUT).free < quote["storage_admission_bytes"]:
        raise ValueError("Preventive training disk admission")
    for seed in SEEDS:
        resume = None
        previous = 0
        targets = (1_000_000_000, 2_000_000_000, 4_000_000_000) if seed == SEEDS[0] else (1_000_000_000, 2_000_000_000)
        for target in targets:
            dest = folder(seed, target)
            dest.mkdir(parents=True, exist_ok=False)
            key = f"{seed}/{target}"
            budget = (quote["training_seconds"]*(target-previous)/8_000_000_000
                +quote["save_seconds_per_entry"]*quote["entries"][key])
            if resume is not None:
                budget += 2*923*telemetry(resume.parent)["diagnostics"]["entries"]/41_010_014
            name = f"{seed}-{target}"
            guarded("train-"+name, train_command(seed, target, dest, resume), budget, dest/"stop.json")
            guarded("gate-"+name, [sys.executable, "-m", MODULE, "gate", "--seed", seed, "--target", target])
            gate = read(dest/"gate.json")
            tool_budget = quote["tool_seconds_per_entry"]*max(quote["entries"][key], gate["entries"])
            guarded("export-"+name, [BINARY, "export", dest/"checkpoint.gz", "--current", dest/"current.gz",
                "--average", dest/"average.gz", "--zero-mass", "uniform"], tool_budget)
            guarded("audit-"+name, [sys.executable, "-m", "scripts.audit_native_hu_checkpoint",
                "--checkpoint", dest/"checkpoint.gz", "--current", dest/"current.gz",
                "--average", dest/"average.gz", "--stack-bb", "100",
                "--target-nodes", gate["actual_nodes"], "--out", dest/"audit.json"], tool_budget)
            guarded("spec-"+name, [sys.executable, "-m", MODULE, "spec", "--seed", seed, "--target", target])
            previous, resume = gate["actual_nodes"], dest/"checkpoint.gz"
            if gate["terminal_capacity_stop"]:
                break
    guarded("pairs", [sys.executable, "-m", MODULE, "pairs"])


def pairs():
    saved = {(int(p.parent.parent.name), int(p.parent.name)): read(p)
        for p in (OUT/"training").glob("*/*/spec.json")}
    for seed, target in PINS:
        if (seed, target) not in saved:
            raise ValueError("All exact reference gates required")
    terminal = max(target for seed, target in saved if seed == SEEDS[0])
    items = {
        "seed-2026100601-terminal-vs-2b": [saved[SEEDS[0], terminal], saved[SEEDS[0], 2_000_000_000]],
        "seed-2026100901-2b-vs-1b": [saved[SEEDS[1], 2_000_000_000], saved[SEEDS[1], 1_000_000_000]],
        "seed-2026100902-2b-vs-1b": [saved[SEEDS[2], 2_000_000_000], saved[SEEDS[2], 1_000_000_000]],
        "2b-seed-2026100601-vs-2026100901": [saved[SEEDS[0], 2_000_000_000], saved[SEEDS[1], 2_000_000_000]],
        "2b-seed-2026100601-vs-2026100902": [saved[SEEDS[0], 2_000_000_000], saved[SEEDS[2], 2_000_000_000]],
    }
    if terminal == 2_000_000_000:
        raise ValueError("No post-2B endpoint; cannot claim independent roles for an identical policy")
    for rung, specs in items.items():
        if specs[0]["name"] == specs[1]["name"]:
            raise ValueError("Distinct policies required")
        put(OUT/"specs"/(rung+".json"), specs)
    put(OUT/"pairs.json", {"primary": list(items)[:3], "descriptive": list(items)[3:], "pairs": items})


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "guard":
        guard_main()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pilot", "train", "gate", "spec", "pairs"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--target", type=int)
    args = parser.parse_args()
    if args.command in ("gate", "spec"):
        if args.seed is None or args.target is None:
            parser.error("Gate/spec requires seed and target")
        (check_gate if args.command == "gate" else model_spec)(args.seed, args.target)
    else:
        globals()[args.command]()


if __name__ == "__main__":
    main()
