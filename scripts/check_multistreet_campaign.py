"""Calibrate, collect, and fit a resumable multi-street poker campaign."""

import argparse
import json
from math import isfinite
import os
from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from scripts import check_multistreet_representation as pilot
from src.arena.artifacts import git, source_fingerprint
from src.arena.schedule import digest
from src.holdem.multistreet_campaign import (
    ReferenceCache,
    _atomic_json,
    build_cached_rows,
    calibrate_world_counts,
    campaign_plan,
)
from src.holdem.multistreet_families import excluded_flops
from src.holdem.multistreet_reference import flop_key


def _calibration_names(plan):
    count = int(plan["calibration_contexts_per_stratum"])
    families = {}
    for index, row in enumerate(plan["contexts"]):
        if row["split"] != "train":
            continue
        family = row["family"]
        stratum = f"{row['street']}:{'facing' if row['facing'] else 'open'}"
        families.setdefault(family, {}).setdefault(stratum, []).append(
            f"{row['street']}-{index}"
        )
    if count < 1 or count > len(families):
        raise ValueError("Calibration needs distinct training families")
    selected = {}
    for ordinal, rows in enumerate(list(families.values())[:count]):
        for stratum, names in rows.items():
            if len(names) != 4:
                raise ValueError(
                    "Calibration family must contain four holdings per stratum"
                )
            selected.setdefault(stratum, []).append(names[ordinal % 4])
    if len(selected) != 6:
        raise ValueError("Calibration must cover all six strata")
    return selected


def _validate_contexts(plan):
    forbidden = excluded_flops(plan.get("forbidden_flop_plans", ()))
    groups = {}
    for row in plan["contexts"]:
        key = flop_key(row["board"])
        if key in forbidden:
            raise ValueError("Campaign repeats a prior flop family")
        previous = groups.setdefault(key, (row["family"], row["split"]))
        if previous != (row["family"], row["split"]):
            raise ValueError(
                "Canonical flop family occurs in multiple groups or splits"
            )
    counts = dict(Counter(row["split"] for row in plan["contexts"]))
    if counts != {"train": 576, "tuning": 192, "validation": 192, "test": 192}:
        raise ValueError("Campaign split counts changed")
    return counts


def _cache_manifest(rows, out):
    entries = []
    for row in rows:
        for path_text in row["cache_paths"]:
            path = Path(path_text)
            value = json.loads(path.read_text())
            entries.append(
                {
                    "path": os.path.relpath(path, out),
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                    "context": row["name"],
                    "profile": value["profile"],
                    "stratum": f"{row['street']}:{row['situation']}",
                }
            )
    return entries


def _stream_plan(plan, namespace, counts=None):
    result = dict(
        plan, stream_namespace=namespace, context_seed=plan[f"{namespace}_seed"]
    )
    if counts is None:
        result["world_samples"] = plan["calibration_max_worlds"]
    else:
        result["world_samples_by_stratum"] = counts
    return result


def _cache(root, source, plan):
    return ReferenceCache(
        root,
        source_sha256=source,
        plan_sha256=digest(plan),
        stream_namespace=plan["stream_namespace"],
        stream_seed=plan["context_seed"],
        selected_n=plan.get("world_samples_by_stratum", plan["world_samples"]),
    )


def run(
    plan,
    out,
    cache_root,
    *,
    deadline_seconds=None,
    reference_workers=None,
    fit_workers=None,
    resume=False,
    calibrate_only=False,
):
    started = perf_counter()
    out, cache_root = Path(out).resolve(), Path(cache_root).resolve()
    expanded = campaign_plan(plan)
    reference_seconds = (
        expanded["max_reference_seconds"]
        if deadline_seconds is None
        else deadline_seconds
    )
    if not isfinite(reference_seconds) or reference_seconds <= 0:
        raise ValueError("Reference timeout must be positive and finite")
    counts = _validate_contexts(expanded)
    source = source_fingerprint()
    identity = {"source_sha256": source, "plan_sha256": digest(expanded)}
    reference_workers = (
        int(plan.get("reference_workers", 1))
        if reference_workers is None
        else reference_workers
    )
    fit_workers = (
        int(plan.get("fit_workers", 1)) if fit_workers is None else fit_workers
    )
    if min(reference_workers, fit_workers) < 1:
        raise ValueError("Worker counts must be positive")
    manifest_path = out / "campaign-manifest.json"
    if manifest_path.exists():
        if not resume:
            raise ValueError("Output already exists; use --resume")
        manifest = json.loads(manifest_path.read_text())
        if any(manifest[key] != value for key, value in identity.items()):
            raise ValueError("Cannot resume a changed source or plan")
        if manifest["status"] == "completed":
            verify(out)
            return manifest
    else:
        out.mkdir(parents=True, exist_ok=resume)
        manifest = dict(
            identity,
            format="holdem-multistreet-campaign-v1",
            revision=git("rev-parse", "HEAD"),
            expanded_plan=expanded,
            counts=counts,
            status="prepared",
            calibration_contexts=_calibration_names(expanded),
        )
        _atomic_json(manifest_path, manifest)
    cache_root.mkdir(parents=True, exist_ok=True)
    deadline = perf_counter() + reference_seconds

    def status(phase, completed=0, total=0, row=None):
        record = {
            "phase": phase,
            "completed": completed,
            "total": total,
            "elapsed_seconds": perf_counter() - started,
        }
        if row:
            record.update(
                last_context=row["name"], last_context_worlds=row["world_count"]
            )
        _atomic_json(out / "status.json", record)

    try:
        calibration_plan = _stream_plan(expanded, "calibration")
        freeze_path = out / "calibration.json"
        if freeze_path.exists() and manifest.get("calibration_sha256") is not None:
            if (
                manifest.get("calibration_sha256")
                != sha256(freeze_path.read_bytes()).hexdigest()
            ):
                raise ValueError(
                    "Frozen calibration changed or was not fully published"
                )
            calibration = json.loads(freeze_path.read_text())
        else:
            status("calibrating", total=6 * plan["calibration_contexts_per_stratum"])
            names = {
                name
                for group in manifest["calibration_contexts"].values()
                for name in group
            }
            calibration_started = perf_counter()
            rows = build_cached_rows(
                calibration_plan,
                _cache(cache_root, source, calibration_plan),
                deadline=deadline,
                context_filter=lambda spec: spec["name"] in names,
                reference_workers=reference_workers,
                progress=lambda done, total, row: status(
                    "calibrating", done, total, row
                ),
            )
            strata = {key: [] for key in manifest["calibration_contexts"]}
            for row in rows:
                strata[f"{row['street']}:{row['situation']}"].append(
                    row["world_action_values_bb"]
                )
            calibration = calibrate_world_counts(
                strata,
                precision_bb=expanded["calibration_precision_bb"],
                minimum_n=expanded["calibration_min_worlds"],
                maximum_n=expanded["calibration_max_worlds"],
                multiplier=expanded["calibration_multiplier"],
            )
            _atomic_json(freeze_path, calibration)
            manifest["calibration_sha256"] = sha256(
                freeze_path.read_bytes()
            ).hexdigest()
            manifest["calibration_cache"] = _cache_manifest(rows, out)
            manifest["calibration_wall_seconds"] = perf_counter() - calibration_started
            manifest["calibration_worker_seconds"] = sum(
                row["reference_worker_seconds"] for row in rows
            )
            per_world = {
                key: sum(
                    row["reference_worker_seconds"]
                    for row in rows
                    if f"{row['street']}:{row['situation']}" == key
                )
                / sum(
                    row["world_count"]
                    for row in rows
                    if f"{row['street']}:{row['situation']}" == key
                )
                for key in strata
            }
            manifest["projected_reference_worker_seconds"] = sum(
                192 * calibration["decisions"][key]["n"] * seconds
                for key, seconds in per_world.items()
            )
        manifest["calibration"] = calibration
        manifest["status"] = "calibrated"
        frozen_counts = {key: row["n"] for key, row in calibration["decisions"].items()}
        manifest["production_worlds_by_stratum"] = frozen_counts
        _atomic_json(manifest_path, manifest)
        if calibrate_only:
            status("calibrated")
            return manifest
        production_plan = _stream_plan(expanded, "production", frozen_counts)
        status("collecting", total=len(expanded["contexts"]))
        rows = build_cached_rows(
            production_plan,
            _cache(cache_root, source, production_plan),
            deadline=deadline,
            reference_workers=reference_workers,
            progress=lambda done, total, row: status("collecting", done, total, row),
        )
        manifest["production_cache"] = _cache_manifest(rows, out)
        manifest["production_plan_sha256"] = digest(production_plan)
        manifest["status"] = "fitting"
        _atomic_json(manifest_path, manifest)
        status("fitting", total=len(plan["seeds"]) * len(plan["variants"]))
        report = pilot.run(
            production_plan,
            out / "fit",
            reference_rows=rows,
            fit_workers=fit_workers,
            resume=resume,
            progress=lambda done, total: status("fitting", done, total),
        )
        manifest["status"] = report["status"]
        manifest["seconds_this_attempt"] = perf_counter() - started
        _atomic_json(manifest_path, manifest)
        status(report["status"])
        return manifest
    except BaseException as error:
        manifest["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        manifest["error"] = repr(error)
        _atomic_json(manifest_path, manifest)
        status(manifest["status"])
        raise


def verify(out):
    out = Path(out)
    manifest = json.loads((out / "campaign-manifest.json").read_text())
    if (
        manifest["status"] != "completed"
        or manifest["source_sha256"] != source_fingerprint()
    ):
        raise ValueError(
            "Only a completed campaign with matching source can be verified"
        )
    if digest(manifest["expanded_plan"]) != manifest["plan_sha256"]:
        raise ValueError("Campaign plan changed")
    if (
        sha256((out / "calibration.json").read_bytes()).hexdigest()
        != manifest["calibration_sha256"]
    ):
        raise ValueError("Calibration changed")
    for kind, expected in (
        (
            "calibration_cache",
            12 * manifest["expanded_plan"]["calibration_contexts_per_stratum"],
        ),
        ("production_cache", 2304),
    ):
        entries = manifest[kind]
        if (
            len(entries) != expected
            or len({(row["context"], row["profile"]) for row in entries}) != expected
        ):
            raise ValueError("Reference cache roster changed")
        for row in entries:
            path = out / row["path"]
            if sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Cached reference changed")
            value = json.loads(path.read_text())
            if value["source_sha256"] != manifest["source_sha256"]:
                raise ValueError("Cached source changed")
    return {
        "status": "verified",
        "fit": pilot.verify(out / "fit"),
        "calibration_unresolved": manifest["calibration"]["unresolved"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path("configs/holdem/multistreet-campaign.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path, default=Path("results/multistreet-campaign-cache")
    )
    parser.add_argument("--max-reference-seconds", type=float)
    parser.add_argument("--reference-workers", type=int)
    parser.add_argument("--fit-workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = (
        verify(args.out)
        if args.verify
        else run(
            json.loads(args.plan.read_text()),
            args.out,
            args.cache,
            deadline_seconds=args.max_reference_seconds,
            reference_workers=args.reference_workers,
            fit_workers=args.fit_workers,
            resume=args.resume,
            calibrate_only=args.calibrate_only,
        )
    )
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "calibration_unresolved",
                    "production_worlds_by_stratum",
                    "projected_reference_worker_seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
