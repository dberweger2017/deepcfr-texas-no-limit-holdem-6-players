"""Named opponent pools and immutable checkpoint references used by saved plans."""

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Checkpoint:
    name: str
    path: str
    sha256: str
    format: str = "legacy-standard-v1"

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("A checkpoint needs a policy name")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("A checkpoint needs a path")
        if not isinstance(self.sha256, str) or not re.fullmatch(
            r"[a-f0-9]{64}", self.sha256
        ):
            raise ValueError("Pin a checkpoint with its lowercase SHA-256 digest")
        if self.format != "legacy-standard-v1":
            raise ValueError("Unsupported checkpoint adapter")


@dataclass(frozen=True, slots=True)
class OpponentPool:
    name: str
    purpose: str
    members: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "members", tuple(self.members))
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("An opponent pool needs a versioned name")
        if self.purpose not in {"evaluation", "training"}:
            raise ValueError("Pool purpose must be evaluation or training")
        if (
            not self.members
            or any(not isinstance(m, str) or not m for m in self.members)
            or len(set(self.members)) != len(self.members)
        ):
            raise ValueError("Pool members must be distinct policy names")
