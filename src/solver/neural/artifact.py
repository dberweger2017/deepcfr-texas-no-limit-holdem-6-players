"""Hash-pinned inference exports; training memories and optimizer state stay separate."""

from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import torch

from src.solver.neural.network import new_network
from src.solver.neural.solver import Config, DeepCFR, Policy

FORMAT = "small-game-deep-cfr-v1"


def save_policy(solver: DeepCFR, path: Path) -> str:
    if solver.failed or solver.strategy is None:
        raise ValueError(
            "Only a fitted policy from valid completed iterations can be exported"
        )
    payload = {
        "format": FORMAT,
        "encoding": "public-small-game-48-v1",
        "game": solver.tree.game,
        "iterations": solver.iterations,
        "config": asdict(solver.config),
        "strategy": solver.strategy.state_dict(),
    }
    data = BytesIO()
    torch.save(payload, data)
    with path.open("xb") as output:
        output.write(data.getvalue())
    return sha256(data.getvalue()).hexdigest()


def load_policy(path: Path, digest: str, *, player: int, seed: int) -> Policy:
    data = path.read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError("Neural policy artifact hash mismatch")
    payload = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != FORMAT
        or payload.get("encoding") != "public-small-game-48-v1"
    ):
        raise ValueError("Unsupported small-game inference artifact")
    if (
        payload.get("game") not in {"kuhn", "leduc"}
        or type(payload.get("iterations")) is not int
        or payload["iterations"] < 1
    ):
        raise ValueError("Invalid inference artifact metadata")
    config = Config(**payload["config"])
    weights = payload["strategy"]
    if not isinstance(weights, dict) or any(
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or not torch.isfinite(value).all()
        for value in weights.values()
    ):
        raise ValueError("Policy weights must be finite float32 tensors")
    model = new_network(config.hidden, 0)
    model.load_state_dict(weights, strict=True)
    model.eval().requires_grad_(False)
    return Policy(payload["game"], player, model, seed)
