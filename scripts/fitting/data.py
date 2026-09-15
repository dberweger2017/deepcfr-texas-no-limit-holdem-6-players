"""Import pinned training artifacts for diagnosis, never for training recovery."""

import json
import tarfile
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import torch

from scripts.pilot_replay import FORMAT as REPLAY_FORMAT
from scripts.pilot_replay import load_replay
from src.solver.experiment import ROOT, canonical, write_json
from src.solver.neural.checkpoint import FORMAT, atomic_write, catalog_hash, restore
from src.solver.neural.experiment import provenance

PLAN = ROOT / "configs/solver/strategy-fitting-v1.json"
PLAN_SHA256 = "43ff1c80ebcf8d9b99dca4c1f57072ef700a9148512bdec95574cf137e6f0899"


def digest(path):
    with Path(path).open("rb") as handle:
        return sha256_file(handle)


def sha256_file(handle):
    result = sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        result.update(block)
    return result.hexdigest()


def load_plan(path=PLAN):
    if digest(path) != PLAN_SHA256:
        raise ValueError("Expected the frozen strategy-fitting-v1 plan")
    plan = json.loads(Path(path).read_text())
    source = plan["source"]
    report_path = ROOT / source["report"]
    if digest(report_path) != source["report_sha256"]:
        raise ValueError("Historical report hash differs")
    report = json.loads(report_path.read_text())
    original = report["exploration"]["manifest"]
    if original["source_sha256"] != source["training_source_sha256"]:
        raise ValueError("Historical source fingerprint differs")
    for name, expected in original["source_files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Historical source interpretation changed: {name}")
    return plan


def manifest(plan):
    current = provenance(plan)
    files = sorted((ROOT / "scripts/fitting").glob("*.py")) + [
        ROOT / "scripts/pilot_replay.py"
    ]
    sources = {str(p.relative_to(ROOT)): digest(p) for p in files}
    return {
        **current,
        "diagnostic_sources": sources,
        "diagnostic_source_sha256": sha256(canonical(sources).encode()).hexdigest(),
    }


def import_checkpoint(raw, spec, original):
    if sha256(raw).hexdigest() != spec["checkpoint_sha256"]:
        raise ValueError("Pinned checkpoint hash differs")
    payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError("Unexpected checkpoint format")
    if canonical(payload["manifest"]) != canonical(original["baseline"]["manifest"]):
        raise ValueError("Checkpoint provenance differs from the pinned report")
    solver = restore(payload["solver"])
    plan = payload["manifest"]["plan"]
    if (solver.tree.game, solver.iterations, solver.config.seed) != (
        plan["game"],
        plan["iterations"],
        spec["seed"],
    ) or canonical(asdict(solver.config)) != canonical(plan["training"]):
        raise ValueError("Checkpoint configuration differs")
    evaluation = payload["progress"]["report"]["evaluations"][-1]
    if canonical(evaluation) != canonical(original["baseline"]["evaluations"][-1]):
        raise ValueError("Checkpoint evaluation differs")
    return solver, payload["manifest"], evaluation


def export_inputs(plan, archive, output):
    if digest(archive) != plan["source"]["archive_sha256"]:
        raise ValueError("Pinned archive hash differs")
    report = json.loads((ROOT / plan["source"]["report"]).read_text())
    originals = {r["seed"]: r for r in report["exploration"]["runs"]}
    output.mkdir(parents=True, exist_ok=False)
    index = {"manifest": manifest(plan), "inputs": {}}
    with tarfile.open(archive, "r:gz") as source:
        for spec in plan["source"]["inputs"]:
            member = source.getmember(spec["checkpoint"])
            if not member.isfile():
                raise ValueError("Checkpoint archive member must be a file")
            with source.extractfile(member) as handle:
                solver, original, evaluation = import_checkpoint(
                    handle.read(), spec, originals[spec["seed"]]
                )
            path = output / str(spec["seed"])
            path.mkdir()
            memory = solver.strategy_memory
            buffer = BytesIO()
            np.savez_compressed(
                buffer,
                **{
                    key: getattr(memory, key)[: memory.size]
                    for key in ("infos", "iterations", "targets")
                },
            )
            replay_hash = atomic_write(path / "replay.npz", buffer.getvalue())
            extra = BytesIO()
            torch.save(
                {
                    "baseline": solver.strategy.state_dict(),
                    "played_strategy_sum": torch.from_numpy(solver.played_strategy_sum),
                },
                extra,
            )
            diagnostic_hash = atomic_write(path / "diagnostics.pt", extra.getvalue())
            write_json(
                path / "manifest.json",
                {
                    "format": REPLAY_FORMAT,
                    "game": solver.tree.game,
                    "seed": spec["seed"],
                    "iteration": solver.iterations,
                    "size": memory.size,
                    "seen": memory.seen,
                    "capacity": memory.capacity,
                    "catalog_sha256": catalog_hash(solver),
                    "replay_sha256": replay_hash,
                    "checkpoint_sha256": spec["checkpoint_sha256"],
                    "diagnostics_sha256": diagnostic_hash,
                    "training_manifest": original,
                    "original_evaluation": evaluation,
                    "purpose": "read-only diagnosis; not recovery or promotion",
                },
            )
            index["inputs"][str(spec["seed"])] = {
                "manifest_sha256": digest(path / "manifest.json"),
                "replay_sha256": replay_hash,
                "diagnostics_sha256": diagnostic_hash,
                "checkpoint_sha256": spec["checkpoint_sha256"],
            }
    write_json(output / "index.json", index)
    return index


def load_input(plan, root, seed, index_hash):
    specs = {s["seed"]: s for s in plan["source"]["inputs"]}
    if seed not in specs or digest(root / "index.json") != index_hash:
        raise ValueError("Unexpected seed or changed input index")
    index = json.loads((root / "index.json").read_text())
    expected = index["inputs"][str(seed)]
    path = root / str(seed)
    if digest(path / "manifest.json") != expected["manifest_sha256"]:
        raise ValueError("Input manifest hash differs")
    solver, memory, metadata = load_replay(path)
    for field in ("replay_sha256", "diagnostics_sha256", "checkpoint_sha256"):
        if metadata[field] != expected[field]:
            raise ValueError(f"Input lineage differs: {field}")
    if metadata["checkpoint_sha256"] != specs[seed]["checkpoint_sha256"]:
        raise ValueError("Input checkpoint was not declared")
    if digest(path / "diagnostics.pt") != metadata["diagnostics_sha256"]:
        raise ValueError("Diagnostic weights hash differs")
    if metadata["seed"] != seed or metadata["iteration"] != plan["source"]["iteration"]:
        raise ValueError("Input seed or iteration differs")
    for key in ("infos", "iterations", "targets"):
        getattr(memory, key).flags.writeable = False
    return solver, memory, metadata
