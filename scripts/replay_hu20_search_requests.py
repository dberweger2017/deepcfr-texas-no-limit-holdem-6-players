"""Replay retained native search requests on another host: strategy parity and solver speed.

Each retained solve directory holds the request, the solver's event stream (`response.jsonl`) and
the dumped profile (`profile.jsonl`). The same request runs with the local binary; every number in
the profile must agree within the checker's tolerances (strategies 1e-5, chip values 0.001), and the
completion events give both hosts' solver seconds for the identical work.
"""

import argparse
import json
from pathlib import Path
import subprocess
import os
import resource
import hashlib
from time import perf_counter

from scripts.hu20_search_runtime import atomic_json

IGNORED = {"elapsed_seconds", "solver_peak_rss_bytes", "dump_path", "seconds"}


def largest_difference(a, b, path=""):
    """Largest absolute numeric difference, or None when the structures differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) - IGNORED != set(b) - IGNORED:
            return None
        worst = 0.0
        for key in set(a) - IGNORED:
            d = largest_difference(a[key], b[key], path + "/" + key)
            if d is None:
                return None
            worst = max(worst, d)
        return worst
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return None
        return worst_of(largest_difference(x, y, path) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        return 0.0 if a == b else None
    return abs(float(a) - float(b))


def worst_of(differences):
    differences = list(differences)
    return None if any(d is None for d in differences) else max(differences, default=0.0)


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def completion_seconds(stream):
    done = [e for e in stream if e.get("event") == "completion"]
    return done[-1]["elapsed_seconds"] if done else None


def replay(binary, solve, out):
    request = json.loads((solve / "request.json").read_text())
    work = out / solve.parent.name / solve.name
    work.mkdir(parents=True, exist_ok=False)
    request["dump_path"] = str(work / "profile.jsonl")
    (work / "request.json").write_text(json.dumps(request))
    fixed = request.get("work_protocol") == "hu20-fixed50-no-fallback-v1"
    if fixed and (request.get("seconds") is not None or request.get("threads") != 6 or request.get("max_iterations") != 50):
        raise ValueError("Fixed reference violates approved work settings")
    started = perf_counter()
    cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = subprocess.run([str(binary), str(work / "request.json"), str(work / "response.jsonl")],
                            capture_output=True, text=True, timeout=None if fixed else 600,
                            env=dict(os.environ, RAYON_NUM_THREADS=str(request["threads"])))
    cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    (work / "stdout.log").write_text(result.stdout)
    (work / "stderr.log").write_text(result.stderr)
    wall = perf_counter() - started
    row = {"solve": f"{solve.parent.name}/{solve.name}", "street": request.get("initial_street"),
           "mode": request.get("mode"), "threads": request.get("threads"), "returncode": result.returncode,
           "wall_seconds": wall, "cpu_seconds": cpu_after.ru_utime + cpu_after.ru_stime - cpu_before.ru_utime - cpu_before.ru_stime, "reference_solver_seconds": completion_seconds(events(solve / "response.jsonl"))}
    if result.returncode != 0:
        row.update(status="failed", stderr=result.stderr[-2000:])
        return row
    row["solver_seconds"] = completion_seconds(events(work / "response.jsonl"))
    mine, theirs = events(work / "profile.jsonl"), events(solve / "profile.jsonl")
    difference = None if len(mine) != len(theirs) else worst_of(largest_difference(x, y) for x, y in zip(mine, theirs))
    if fixed:
        completions = [e for e in events(work / "response.jsonl") if e.get("event") == "completion"]
        reference_completions = [e for e in events(solve / "response.jsonl") if e.get("event") == "completion"]
        for completion in completions[-1:]:
            if completion.get("work_protocol") != request["work_protocol"] or completion.get("iterations") != 50:
                raise ValueError("Native solver did not acknowledge fixed work")
        if not completions:
            raise ValueError("Native fixed solve lacks completion")
        descriptive = {"solver_peak_rss_bytes", "elapsed_seconds"}
        canonical = lambda rows: [{k:v for k,v in r.items() if k not in descriptive} for r in rows]
        mine_quality = [e for e in events(work / "response.jsonl") if e.get("event") == "quality"]
        reference_quality = [e for e in events(solve / "response.jsonl") if e.get("event") == "quality"]
        encoded = lambda rows: json.dumps(canonical(rows),sort_keys=True,separators=(",", ":"),allow_nan=False)
        exact = (encoded(mine) == encoded(theirs) and encoded(completions) == encoded(reference_completions)
                 and encoded(mine_quality) == encoded(reference_quality))
        difference = 0.0 if exact else None
        row["exact_scientific_outputs"] = exact
        row["scientific_profile_sha256"] = hashlib.sha256(json.dumps(canonical(mine),sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    row["profile_lines"] = len(theirs)
    row["max_profile_difference"] = difference
    row["status"] = "passed" if difference is not None and difference <= 1e-5 else "mismatch"
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--solves", type=Path, required=True, help="Retained solves/<slot>/<request> directories")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for solve in sorted(d for d in a.solves.glob("*/*") if (d / "profile.jsonl").exists()):
        rows.append(replay(a.binary, solve, a.out))
        atomic_json(a.out / "status.json", {"replayed": len(rows)})
    timed = [r for r in rows if r.get("solver_seconds") and r.get("reference_solver_seconds")]
    summary = {"requests": len(rows), "passed": sum(r["status"] == "passed" for r in rows),
               "mismatched": sum(r["status"] == "mismatch" for r in rows),
               "failed": sum(r["status"] == "failed" for r in rows),
               "max_profile_difference": max((r["max_profile_difference"] for r in rows
                                              if r.get("max_profile_difference") is not None), default=None),
               "solver_seconds_total": sum(r["solver_seconds"] for r in timed),
               "reference_solver_seconds_total": sum(r["reference_solver_seconds"] for r in timed),
               "rows": rows}
    summary["exact_scientific_outputs"] = all(r.get("exact_scientific_outputs", False) for r in rows) and bool(rows)
    summary["speed_ratio_this_host_over_reference"] = (summary["solver_seconds_total"]
        / summary["reference_solver_seconds_total"]) if timed else None
    atomic_json(a.out / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    return 0 if summary["passed"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
