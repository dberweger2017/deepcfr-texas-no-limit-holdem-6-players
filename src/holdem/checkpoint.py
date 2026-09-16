"""Hash-pinned inference exports and complete iteration-boundary recovery."""

import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import torch

from src.arena.schedule import canonical
from src.game.observation import RULES_PROFILE, SCHEMA_VERSION
from src.holdem.actions import SCHEMA as ACTION_SCHEMA
from src.holdem.average import AveragePolicy
from src.holdem.betting import BettingNetwork
from src.holdem.encoding import SCHEMA as DECISION_SCHEMA
from src.holdem.policy import FrozenProfile
from src.holdem.records import pack, unpack
from src.holdem.replay import RoleReservoir
from src.holdem.training import HoldemTrainer, _State
from src.solver.neural.checkpoint import atomic_write

TRAINING = "holdem-training-v1"
INFERENCE = "holdem-average-v1"
CONTRACT = {
    "rules": RULES_PROFILE,
    "observation": SCHEMA_VERSION,
    "decision": DECISION_SCHEMA,
    "actions": ACTION_SCHEMA,
    "average": "linear-own-reach-collection-profiles-v1",
    "resume": "completed-iterations-fresh-adam-per-role-v1",
}


def _profile_state(profile):
    profile.assert_unchanged()
    return {
        "fingerprint": profile.fingerprint,
        "models": [
            None
            if model is None
            else {
                "width": model.encoder.width,
                "weights": model.state_dict(),
            }
            for model in profile._models
        ],
    }


def _restore_profile(data):
    models = []
    for item in data["models"]:
        if item is None:
            models.append(None)
            continue
        width, weights = item["width"], item["weights"]
        if type(width) is not int or not 1 <= width <= 4096:
            raise ValueError("Invalid model width")
        if not isinstance(weights, dict) or any(
            not isinstance(t, torch.Tensor)
            or t.dtype != torch.float32
            or t.device.type != "cpu"
            or not torch.isfinite(t).all()
            for t in weights.values()
        ):
            raise ValueError("Invalid model weights")
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            model = BettingNetwork(width).float()
        model.load_state_dict(weights, strict=True)
        models.append(model)
    profile = FrozenProfile(models)
    if profile.fingerprint != data["fingerprint"]:
        raise ValueError("Profile fingerprint mismatch")
    return profile


def _save(path, payload):
    payload = {**payload, "manifest": json.loads(canonical(payload["manifest"]))}
    tensors = {}

    def encode(value):
        if isinstance(value, torch.Tensor):
            key = str(len(tensors))
            tensors[key] = value.detach().cpu().contiguous().clone()
            return {"tensor": key}
        if isinstance(value, tuple):
            return {"tuple": [encode(v) for v in value]}
        if isinstance(value, list):
            return [encode(v) for v in value]
        if isinstance(value, dict):
            return {k: encode(value[k]) for k in sorted(value)}
        return value

    header = canonical(encode({"contract": CONTRACT, **payload})).encode()
    weights = BytesIO()
    torch.save(tensors, weights)
    data = BytesIO()
    with ZipFile(data, "w") as archive:
        # Fixed metadata and canonical records make bytes independent of Python aliases.
        archive.writestr(ZipInfo("records.json"), header)
        archive.writestr(ZipInfo("weights.pt"), weights.getvalue())
    return atomic_write(Path(path), data.getvalue())


def _load(path, digest, kind):
    data = Path(path).read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError("Artifact hash mismatch")
    with ZipFile(BytesIO(data)) as archive:
        if sorted(archive.namelist()) != ["records.json", "weights.pt"]:
            raise ValueError("Invalid artifact members")
        record = json.loads(archive.read("records.json"))
        tensors = torch.load(
            BytesIO(archive.read("weights.pt")), map_location="cpu", weights_only=True
        )

    def decode(value):
        if isinstance(value, list):
            return [decode(v) for v in value]
        if isinstance(value, dict):
            if set(value) == {"tensor"}:
                return tensors[value["tensor"]]
            if set(value) == {"tuple"}:
                return tuple(decode(v) for v in value["tuple"])
            return {k: decode(v) for k, v in value.items()}
        return value

    payload = decode(record)
    if payload.get("format") != kind or payload.get("contract") != CONTRACT:
        raise ValueError("Unsupported artifact format or contract")
    return payload


def save_policy(trainer: HoldemTrainer, path: Path, *, manifest: dict) -> str:
    trainer.average_policy()
    return _save(
        path,
        {
            "format": INFERENCE,
            "manifest": manifest,
            "training_seed": trainer.config.seed,
            "table": pack(trainer.table),
            "archive": [_profile_state(p) for p in trainer._state.archive],
        },
    )


def load_policy(path: Path, digest: str) -> tuple[AveragePolicy, dict]:
    data = _load(path, digest, INFERENCE)
    table = unpack(data["table"])
    profiles = tuple(_restore_profile(p) for p in data["archive"])
    policy = AveragePolicy(profiles)
    if any(p.capacity != table.capacity for p in profiles):
        raise ValueError("Archive capacity differs from its training table")
    return policy, {
        "manifest": data["manifest"],
        "training_seed": data["training_seed"],
        "table": data["table"],
        "iterations": len(profiles),
    }


def save_training(trainer: HoldemTrainer, path: Path, *, manifest: dict) -> str:
    state = trainer._state
    return _save(
        path,
        {
            "format": TRAINING,
            "manifest": manifest,
            "table": pack(trainer.table),
            "config": pack(trainer.config),
            "iteration": state.iteration,
            "reports": pack(state.reports),
            "current": _profile_state(trainer.current_profile()),
            "archive": [_profile_state(p) for p in state.archive],
            "memories": [
                {
                    "role": m.role,
                    "capacity": m.capacity,
                    "seen": m.seen,
                    "random": m._random.getstate(),
                    "items": pack(m.items),
                    "fingerprint": m.fingerprint(),
                }
                for m in state.memories
            ],
            # Fits intentionally start with fresh Adam; no optimizer survives a commit.
            "optimizer": None,
        },
    )


def load_training(path: Path, digest: str, *, manifest: dict) -> HoldemTrainer:
    data = _load(path, digest, TRAINING)
    if canonical(data["manifest"]) != canonical(manifest):
        raise ValueError("Training provenance differs from the checkpoint")
    trainer = HoldemTrainer(unpack(data["table"]), unpack(data["config"]))
    iteration, reports = data["iteration"], unpack(data["reports"])
    if type(iteration) is not int or iteration < 0 or data["optimizer"] is not None:
        raise ValueError("Invalid iteration boundary")
    current = _restore_profile(data["current"])
    archive = tuple(_restore_profile(p) for p in data["archive"])
    if (
        len(archive) != iteration
        or len(reports) != iteration
        or current.capacity != trainer.table.capacity
        or any(p.capacity != current.capacity for p in archive)
        or len(data["memories"]) != current.capacity
    ):
        raise ValueError("Incomplete training generation")
    initial = FrozenProfile([None] * current.capacity).fingerprint
    if (archive[0].fingerprint if archive else current.fingerprint) != initial:
        raise ValueError("Training archive is missing uniform bootstrap")
    if any(
        model is not None and model.encoder.width != trainer.config.fit.width
        for profile in (*archive, current)
        for model in profile._models
    ):
        raise ValueError("Network width differs from the training configuration")
    for index, report in enumerate(reports):
        following = archive[index + 1] if index + 1 < iteration else current
        if (
            report.iteration != index + 1
            or report.collection_profile != archive[index].fingerprint
            or report.fitted_profile != following.fingerprint
            or len(report.roles) != current.capacity
        ):
            raise ValueError("Reports disagree with collection and fitted profiles")
        for role, update in enumerate(report.roles):
            previous_seen = reports[index - 1].roles[role].seen if index else 0
            if (
                update.role != role
                or type(update.new_samples) is not int
                or update.new_samples < 0
                or update.seen != previous_seen + update.new_samples
                or update.stored != min(update.seen, trainer.config.capacity)
            ):
                raise ValueError("Invalid role counters in iteration report")
    memories = []
    for role, saved in enumerate(data["memories"]):
        if saved["role"] != role or saved["capacity"] != trainer.config.capacity:
            raise ValueError("Replay role or capacity mismatch")
        memory = RoleReservoir(role, saved["capacity"], 0)
        items = unpack(saved["items"])
        seen = saved["seen"]
        if (
            type(seen) is not int
            or seen < 0
            or len(items) != min(seen, memory.capacity)
        ):
            raise ValueError("Replay size differs from reservoir counters")
        memory.extend(items)
        memory.seen = seen
        memory._random.setstate(saved["random"])
        if any(
            s.iteration > iteration or s.profile != archive[s.iteration - 1].fingerprint
            for s in items
        ):
            raise ValueError("Replay sample disagrees with its collection generation")
        if memory.fingerprint() != saved["fingerprint"]:
            raise ValueError("Replay fingerprint mismatch")
        if seen != sum(r.roles[role].new_samples for r in reports):
            raise ValueError("Replay count disagrees with iteration reports")
        memories.append(memory)
    trainer._state = _State(
        iteration, current._models, tuple(memories), reports, archive
    )
    return trainer
