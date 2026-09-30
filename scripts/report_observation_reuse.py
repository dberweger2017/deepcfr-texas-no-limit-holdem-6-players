"""Independent retained-artifact checks and compact M4 benchmark summary."""

import argparse
import ast
import gzip
from hashlib import sha256
import json
from pathlib import Path
import statistics
import subprocess


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    value = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def report(root, output):
    output.mkdir(parents=True, exist_ok=False)
    result = json.loads((root / "result.json").read_text())
    supervisor = json.loads((root / "supervisor/campaign.json").read_text())
    manifest_path = root.with_name(root.name + "-manifest.json")
    manifest = json.loads(manifest_path.read_text())
    checks = {"campaign_complete": result["status"] == supervisor["status"] == "complete",
              "all_attempts_complete": all(a["status"] == "complete" and a["exit_code"] == 0
                                            and a["guard_failure"] is None for a in supervisor["attempts"]),
              "inside_deadline": supervisor["finished"] < supervisor["deadline"],
              "all_global_hashes": all(digest(root / name) == item["sha256"]
                       and (root / name).stat().st_size == item["bytes"] for name, item in manifest.items())}
    expected_files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    checks["no_uninventoried_files"] = expected_files == set(manifest)
    checks["all_equivalence_gates"] = all(json.loads(p.read_text())["passed"]
                                            for p in root.glob("*-equivalence.json"))

    # Confirm the extracted uncached body and apply path are literally unchanged.
    base = ast.parse(subprocess.check_output(["git", "show", "7d74b6c:src/game/hand.py"], text=True))
    runtime = ast.parse(subprocess.check_output(["git", "show",
                     supervisor["identity"]["revision"] + ":src/game/hand.py"], text=True))
    def method(tree, name):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Hand")
        return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    checks["literal_original_replay_body"] = ast.dump(ast.Module(body=method(base, "observe").body,
        type_ignores=[])) == ast.dump(ast.Module(body=method(runtime, "_observe_uncached").body, type_ignores=[]))
    checks["unchanged_full_action_validation"] = ast.dump(method(base, "apply")) == ast.dump(method(runtime, "apply"))

    summary = {"runtime_source": supervisor["identity"]["revision"], "clock": result,
               "cases": {}, "trace": {}, "global_file_count": len(manifest),
               "manifest_sha256": digest(manifest_path), "attempt_count": len(supervisor["attempts"])}
    all_rows = {}
    all_results = {}
    for p in sorted(root.glob("*/result.json")):
        if p.parent.name == "supervisor":
            continue
        d = json.loads(p.read_text())
        rows = [json.loads(line) for line in (p.parent / "iterations.jsonl").read_text().splitlines()]
        all_rows[p.parent.name] = rows
        all_results[p.parent.name] = d
        checks[p.parent.name + ":complete_work"] = (sum(row["nodes"] for row in rows)
            == d["added_nodes"]-d.get("resume_added_nodes", 0)
            and rows[-1]["iteration"] == d["iteration"]
            and sha256(b"".join(encoded(row)+b"\n" for row in rows)).hexdigest() == d["work_sha256"])
        checks[p.parent.name + ":all_output_hashes"] = all(
            digest(p.parent / (name + ".json.gz")) == d[name]["sha256"] for name in ("final", "current", "next"))
        with gzip.open(p.parent / "final.json.gz", "rt") as source:
            header = json.loads(source.readline())
        checks[p.parent.name + ":checkpoint_identity"] = (header["iteration"] == d["iteration"]
                        and header["config"]["seed"] == d["config"]["seed"]
                        and header["identity"]["game"] == d["config"]["game"])

    for case in ("small", "mature"):
        values = []
        for pair in range(1, 4):
            a = all_results[f"perf-{case}-{pair}-original"]
            b = all_results[f"perf-{case}-{pair}-candidate"]
            values.append({"pair": pair, "original": a, "candidate": b,
                           "training_speedup": b["nodes_per_second"]/a["nodes_per_second"],
                           "startup_inclusive_speedup": a["elapsed_seconds"]/b["elapsed_seconds"]})
        reference = all_results[f"perf-{case}-1-original"]
        checks[case + ":all_repeat_state_identical"] = all(
            d[name] == reference[name] for row in values for d in (row["original"], row["candidate"])
            for name in ("final", "current", "next", "work_sha256", "next_streams", "added_nodes"))
        for variant in ("original", "candidate"):
            direct = all_results[f"perf-{case}-1-{variant}"]
            resumed = all_results[f"resume-{case}-{variant}"]
            checks[f"{case}:{variant}:independent_suffix"] = ([row for row in all_rows[f"perf-{case}-1-{variant}"]
                if row["added_nodes"] > resumed["resume_added_nodes"]] == all_rows[f"resume-{case}-{variant}"])
            checks[f"{case}:{variant}:independent_resume"] = all(
                direct[name] == resumed[name] for name in ("final", "current", "next", "next_streams", "added_nodes"))
        summary["cases"][case] = {"pairs": values,
                "median_training_speedup": statistics.median(row["training_speedup"] for row in values),
                "median_startup_inclusive_speedup": statistics.median(row["startup_inclusive_speedup"] for row in values)}
        a = all_results[f"trace-{case}-original"]["trace_result"]
        b = all_results[f"trace-{case}-candidate"]["trace_result"]
        checks[case + ":independent_trace_hashes"] = a["sha256"] == b["sha256"]
        # The literal-control method has a copied globals namespace, so its
        # observe-local replay calls bypassed the instrumentation monkeypatch.
        # Every successful original observe invokes replay exactly once. Keep
        # raw counts and derive the complete count without rerunning gameplay.
        original_replays = a["counts"]["replays"] + a["counts"]["observations"]
        summary["trace"][case] = {"original_raw": a, "candidate_raw": b,
            "original_replays_derived": original_replays,
            "candidate_replays": b["counts"]["replays"],
            "replays_saved": original_replays-b["counts"]["replays"],
            "counting_note": "Literal control's copied globals bypassed observe-local replay hook; "
                             "original total = counted transition replays + all original observe calls. "
                             "Returned observation/key/menu/RNG hashes were directly instrumented on both paths."}
    resources = [json.loads(line) for line in (root / "supervisor/resources.jsonl").read_text().splitlines()]
    summary["resources"] = {"peak_aggregate_rss_bytes": max(row["aggregate_job_rss_bytes"] for row in resources),
            "peak_process_rss_bytes": max(a.get("peak_process_rss_bytes", 0) for a in supervisor["attempts"]),
            "maximum_swap_growth_bytes": max(row["swap_growth_bytes"] for row in resources),
            "minimum_free_disk_bytes": min(row["free_disk_bytes"] for row in resources),
            "all_ac": all("AC Power" in row["power"] for row in resources)}
    checks["all_resource_guards"] = (summary["resources"]["peak_aggregate_rss_bytes"] < 10.5*2**30
            and summary["resources"]["maximum_swap_growth_bytes"] <= .5*2**30
            and summary["resources"]["minimum_free_disk_bytes"] >= 8*2**30
            and summary["resources"]["all_ac"])
    from scripts.run_observation_reuse_benchmark import PARENT, PARENT_HASH
    checks["original_parent_untouched"] = digest(PARENT) == PARENT_HASH
    verification = {"passed": all(checks.values()), "checks": checks}
    (output / "summary.json").write_bytes(encoded(summary)+b"\n")
    (output / "verification.json").write_bytes(encoded(verification)+b"\n")
    if not verification["passed"]:
        raise ValueError("Retained benchmark verification failed: " + str(checks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report(args.root, args.out)
