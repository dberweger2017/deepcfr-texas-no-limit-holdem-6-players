import json
import tarfile
from copy import deepcopy
from io import BytesIO

import pytest
import torch

from scripts.fitting.data import (
    PLAN,
    digest,
    export_inputs,
    import_checkpoint,
    load_input,
    load_plan,
)
from scripts.fitting.run import verify_artifacts, worker
from src.solver.neural.experiment import Plan, run
from src.solver.neural.solver import Config


@pytest.fixture
def bundle(tmp_path):
    training = tmp_path / "training"
    result = run(
        Plan(
            "kuhn",
            2,
            Config(
                seed=11,
                hidden=8,
                traversals=8,
                advantage_steps=2,
                strategy_steps=3,
                capacity=32,
                batch_size=8,
            ),
            evaluation_interval=1,
        ),
        training,
    )
    original = {
        "seed": 11,
        "baseline": {
            **result,
            "manifest": json.loads((training / "manifest.json").read_text()),
        },
    }
    descriptor = json.loads((training / "checkpoint.json").read_text())
    checkpoint = training / descriptor["file"]
    spec = {
        "seed": 11,
        "checkpoint": "original.pt",
        "checkpoint_sha256": digest(checkpoint),
    }
    report = tmp_path / "original.json"
    report.write_text(json.dumps({"exploration": {"runs": [original]}}))
    archive = tmp_path / "input.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(checkpoint, arcname="original.pt")
    plan = json.loads(PLAN.read_text())
    plan["source"].update(
        report=str(report), archive_sha256=digest(archive), inputs=[spec], iteration=2
    )
    plan["resources"]["local_smoke_steps_per_recipe"] = 2
    plan["fixed"]["strategy_hidden"] = 8
    return plan, archive, checkpoint, spec, original


def test_import_checks_checkpoint_contents_as_well_as_hash(bundle):
    _, _, path, spec, original = bundle
    raw = path.read_bytes()
    solver, _, _ = import_checkpoint(raw, spec, original)
    assert solver.iterations == 2
    with pytest.raises(ValueError, match="hash"):
        import_checkpoint(raw + b"changed", spec, original)
    changed = deepcopy(original)
    changed["baseline"]["manifest"]["revision"] = "different"
    with pytest.raises(ValueError, match="provenance"):
        import_checkpoint(raw, spec, changed)
    payload = torch.load(BytesIO(raw), weights_only=True)
    payload["solver"]["catalog_sha256"] = "changed"
    output = BytesIO()
    torch.save(payload, output)
    from hashlib import sha256

    with pytest.raises(ValueError):
        import_checkpoint(
            output.getvalue(),
            {**spec, "checkpoint_sha256": sha256(output.getvalue()).hexdigest()},
            original,
        )


def test_export_is_readonly_and_changed_inputs_fail_before_fitting(bundle, tmp_path):
    plan, archive, _, _, _ = bundle
    target = tmp_path / "inputs"
    export_inputs(plan, archive, target)
    index_hash = digest(target / "index.json")
    _, memory, _ = load_input(plan, target, 11, index_hash)
    assert not memory.targets.flags.writeable
    with pytest.raises(ValueError, match="seed"):
        load_input(plan, target, 401, index_hash)
    with pytest.raises(ValueError, match="index"):
        load_input(plan, target, 11, "changed")
    job = {
        "inputs": str(target),
        "seed": 11,
        "replicate": 0,
        "mode": "smoke",
        "recipe": "minibatch-fixed",
        "input_index_sha256": index_hash,
    }
    result = worker(plan, job, tmp_path / "fit")
    assert result["status"] == "completed" and result["scored"] is False
    verify_artifacts(tmp_path / "fit", result)
    (tmp_path / "fit/model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="model hash"):
        verify_artifacts(tmp_path / "fit", result)
    (target / "11/replay.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        worker(plan, job, tmp_path / "failed")
    failed = json.loads((tmp_path / "failed/report.json").read_text())
    assert failed["status"] == "error" and failed["evaluations"] == []
    assert not (tmp_path / "failed/model.pt").exists()


def test_historical_study_requires_its_original_solver_sources():
    with pytest.raises(ValueError, match="Historical source interpretation changed"):
        load_plan()
