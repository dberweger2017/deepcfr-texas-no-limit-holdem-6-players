"""A schedule is fixed before any policies play or results are inspected."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from random import Random

from src.arena.catalog import Checkpoint, OpponentPool
from src.game.hand import Table

SCHEDULE_VERSION = 2
SPLITS = {"train": 0, "validation": 1, "test": 2}
STREAMS = {"deal", "action", "opponent", "training"}


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return sha256(canonical(value).encode()).hexdigest()


def stream_seed(root: int, split: str, stream: str, *coordinates) -> int:
    if type(root) is not int or not 0 <= root < 2**64:
        raise ValueError("Root seed must be an unsigned 64-bit integer")
    if split not in SPLITS or stream not in STREAMS:
        raise ValueError("Unknown split or random stream")
    payload = (SCHEDULE_VERSION, root, stream, coordinates)
    # Split prefixes make the engine seed spaces disjoint, not just probably distinct.
    return (SPLITS[split] << 62) | (int(digest(payload)[:16], 16) & (2**62 - 1))


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    stacks: tuple[int, ...] = (10000,) * 6
    mode: str = "fixed"
    hands_per_rotation: int = 1
    small_blind: int = 50
    big_blind: int = 100
    chip_unit: str = "0.01"

    def __post_init__(self):
        object.__setattr__(self, "stacks", tuple(self.stacks))
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Scenario needs a name")
        if self.mode not in {"fixed", "session"}:
            raise ValueError("Mode must be fixed or session")
        if type(self.hands_per_rotation) is not int or self.hands_per_rotation < 1:
            raise ValueError("A rotation must contain at least one hand")
        if self.mode == "fixed" and self.hands_per_rotation != 1:
            raise ValueError("Fixed-stack blocks contain one deal per seat rotation")
        Table(
            tuple(f"player-{i}" for i in range(len(self.stacks))),
            self.stacks,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            chip_unit=self.chip_unit,
        )
        if self.mode == "session" and min(self.stacks) < self.big_blind:
            raise ValueError("Session reloads must cover a full big blind")


@dataclass(frozen=True, slots=True)
class Plan:
    scenarios: tuple[Scenario, ...]
    candidate: str = "check_call"
    baseline: str = "fold"
    opponents: tuple[str, ...] | None = None
    blocks: int = 100
    root_seed: int = 0
    split: str = "validation"
    max_decisions: int = 1000
    models: tuple[Checkpoint, ...] = ()
    pool: OpponentPool | None = None

    def __post_init__(self):
        object.__setattr__(self, "scenarios", tuple(self.scenarios))
        members = (
            self.opponents
            if self.opponents is not None
            else (self.pool.members if self.pool else ("check_call",))
        )
        object.__setattr__(self, "opponents", tuple(members))
        object.__setattr__(self, "models", tuple(self.models))
        if len({m.name for m in self.models}) != len(self.models):
            raise ValueError("Checkpoint policy names must be distinct")
        if self.pool:
            purpose = "training" if self.split == "train" else "evaluation"
            if self.pool.purpose != purpose or self.opponents != self.pool.members:
                raise ValueError(
                    "The declared pool must match the split purpose and opponent list"
                )
        if not self.scenarios or len({s.name for s in self.scenarios}) != len(
            self.scenarios
        ):
            raise ValueError("Provide distinct named scenarios")
        if not self.opponents or any(
            not isinstance(p, str) or not p
            for p in (self.candidate, self.baseline, *self.opponents)
        ):
            raise ValueError("Provide candidate, baseline, and opponent policy names")
        if type(self.blocks) is not int or self.blocks < 1:
            raise ValueError("Block count must be positive")
        if type(self.max_decisions) is not int or self.max_decisions < 1:
            raise ValueError("Decision limit must be positive")
        stream_seed(self.root_seed, self.split, "deal")

    @classmethod
    def from_dict(cls, value: dict) -> "Plan":
        return cls(
            **{
                **value,
                "scenarios": tuple(Scenario(**s) for s in value["scenarios"]),
                "models": tuple(Checkpoint(**m) for m in value.get("models", ())),
                "pool": OpponentPool(**value["pool"]) if value.get("pool") else None,
            }
        )


@dataclass(frozen=True, slots=True)
class Block:
    scenario: str
    index: int
    button: int
    deal_seeds: tuple[int, ...]
    action_seeds: tuple[int, ...]
    opponent_seed: int
    opponents: tuple[str, ...]


def build_schedule(plan: Plan) -> tuple[Block, ...]:
    result = []
    for scenario in plan.scenarios:
        n = len(scenario.stacks)
        for index in range(plan.blocks):

            def seed(stream, *extra, name=scenario.name, block_index=index):
                return stream_seed(
                    plan.root_seed, plan.split, stream, name, block_index, *extra
                )

            opponent_seed = seed("opponent")
            selection = Random(opponent_seed)
            result.append(
                Block(
                    scenario.name,
                    index,
                    index % n,
                    tuple(
                        seed("deal", hand)
                        for hand in range(scenario.hands_per_rotation)
                    ),
                    tuple(seed("action", player) for player in range(n)),
                    opponent_seed,
                    tuple(selection.choice(plan.opponents) for _ in range(n - 1)),
                )
            )
    return tuple(result)


def schedule_document(plan: Plan) -> dict:
    return {
        "version": SCHEDULE_VERSION,
        "plan": asdict(plan),
        "blocks": [asdict(block) for block in build_schedule(plan)],
    }
