"""Resolve and pin every policy before an experiment starts dealing."""

from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from src.arena.heuristics import STYLES
from src.arena.policies import POLICIES, make_policy
from src.arena.schedule import Plan

ROOT = Path(__file__).resolve().parents[2]


AVERAGE_FORMATS = {"holdem-hu20-stored-cfr-average-diagnostic-v1", "holdem-hu100-stored-cfr-average-research-v1", "holdem-hu200-stored-cfr-average-research-v1"}


def artifact_suffix(format):
    return ".json.gz" if "blueprint" in format or format in AVERAGE_FORMATS else ".pt"


def load_frozen(spec, path):
    if spec.format in AVERAGE_FORMATS:
        from src.blueprint.average import AveragePolicy
        from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA, HU100_SCHEMA, HU200_SCHEMA
        model = AveragePolicy(path, spec.sha256, expected_schema=HU200_SCHEMA if "hu200" in spec.format else
                              HU100_SCHEMA if "hu100" in spec.format else HU20_UNCAPPED_SCHEMA)
        model.spec = spec
        model.source_path = path
        return model
    if spec.format in {"holdem-blueprint-v1", "holdem-hu20-blueprint-v2",
                        "holdem-hu20-native-reopening-blueprint-v1", "holdem-tp20-blueprint-v1",
                        "holdem-hu100-native-reopening-blueprint-v1", "holdem-hu200-native-reopening-blueprint-v1"}:
        from src.blueprint.artifact import FrozenBlueprint

        return FrozenBlueprint(spec, path)
    if spec.format == "holdem-average-v1":
        from src.arena.snapshots import FrozenAverage

        return FrozenAverage(spec, path)
    from src.arena.frozen import FrozenNetwork

    return FrozenNetwork(spec, path)


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
            suffix = artifact_suffix(spec.format)
            path = (
                artifact_dir / f"{spec.sha256}{suffix}"
                if artifact_dir
                else ROOT / spec.path
            )
            model = load_frozen(spec, path)
            for scenario in plan.scenarios:
                if scenario.mode != "fixed" or len(scenario.stacks) != model.players:
                    raise ValueError(
                        f"{spec.name} requires fixed {model.players}-player hands"
                    )
                identity = getattr(model, "identity", {})
                if identity and (list(scenario.stacks) != identity["stacks"]
                                 or scenario.small_blind != identity["small_blind"]
                                 or scenario.big_blind != identity["big_blind"]
                                 or scenario.chip_unit != "0.01"):
                    raise ValueError(f"{spec.name} game differs from the evaluation scenario")
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
            "src/arena/snapshots.py",
            "src/arena/registry.py",
            *(
                str(p.relative_to(ROOT))
                for p in sorted((ROOT / "src/blueprint").glob("*.py"))
            ),
            *(
                str(p.relative_to(ROOT))
                for p in sorted((ROOT / "src/holdem").glob("*.py"))
            ),
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
                suffix = artifact_suffix(model.spec.format)
                destination = target / f"{model.spec.sha256}{suffix}"
                if model.spec.format in AVERAGE_FORMATS:
                    from src.policies.files import file_hash
                    if file_hash(model.source_path) != model.spec.sha256:
                        raise ValueError("Average changed before snapshot")
                    copyfile(model.source_path, destination)
                    if file_hash(destination) != model.spec.sha256:
                        raise ValueError("Average snapshot differs from pinned bytes")
                else:
                    destination.write_bytes(model.data)
