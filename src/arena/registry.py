"""Resolve and pin every policy before an experiment starts dealing."""

from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path

from src.arena.heuristics import STYLES
from src.arena.policies import POLICIES, make_policy
from src.arena.schedule import Plan

ROOT = Path(__file__).resolve().parents[2]


class PolicyRegistry:
    def __init__(self, plan: Plan, *, artifact_dir: Path | None = None):
        self.plan = plan
        self.models = {}
        used = {plan.candidate, plan.baseline, *plan.opponents}
        builtins = set(POLICIES) | set(STYLES)
        for spec in plan.models:
            if spec.name in builtins or spec.name not in used:
                raise ValueError(
                    "Checkpoint names must be used and cannot shadow built-in policies"
                )
            from src.arena.frozen import FrozenNetwork

            path = (
                artifact_dir / f"{spec.sha256}.pt" if artifact_dir else ROOT / spec.path
            )
            model = FrozenNetwork(spec, path)
            for scenario in plan.scenarios:
                if scenario.mode != "fixed" or len(scenario.stacks) != model.players:
                    raise ValueError(
                        f"{spec.name} requires fixed {model.players}-player hands"
                    )
            self.models[spec.name] = model
        if used - builtins - self.models.keys():
            raise ValueError(
                f"Unknown policies: {sorted(used - builtins - self.models.keys())}"
            )

    def make_policy(self, name: str, seed: int):
        if name in self.models:
            return self.models[name].policy(seed)
        return make_policy(name, seed)

    def fingerprints(self) -> dict:
        paths = (
            "src/arena/policies.py",
            "src/arena/heuristics.py",
            "src/game/play.py",
            "src/arena/frozen.py",
            "src/arena/historical.py",
        )
        implementation = sha256(
            b"".join((ROOT / path).read_bytes() for path in paths)
        ).hexdigest()
        return {
            name: {
                "implementation_sha256": implementation,
                **(
                    self.models[name].description
                    if name in self.models
                    else {"kind": "builtin", "weights_sha256": None}
                ),
            }
            for name in sorted(
                {self.plan.candidate, self.plan.baseline, *self.plan.opponents}
            )
        }

    @property
    def inference_config(self):
        return (
            {"device": "cpu", "threads": 1, "deterministic_algorithms": True}
            if self.models
            else None
        )

    def runtime(self):
        if not self.models:
            return nullcontext()
        from src.arena.frozen import inference_runtime

        return inference_runtime()

    def snapshot(self, output: Path):
        if self.models:
            target = output / "models"
            target.mkdir()
            for model in self.models.values():
                (target / f"{model.spec.sha256}.pt").write_bytes(model.data)
