"""Named opponent pools and immutable checkpoint references used by saved plans."""

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Checkpoint:
    name: str
    path: str
    sha256: str
    format: str = "legacy-standard-v1"
    abstraction: str | None = None
    card_buckets: str | None = None

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("A checkpoint needs a policy name")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("A checkpoint needs a path")
        if not isinstance(self.sha256, str) or not re.fullmatch(
            r"[a-f0-9]{64}", self.sha256
        ):
            raise ValueError("Pin a checkpoint with its lowercase SHA-256 digest")
        from src.blueprint.abstraction import EQUITY_SCHEMAS, SUPPORTED_SCHEMAS
        if self.abstraction is not None and self.abstraction not in SUPPORTED_SCHEMAS:
            raise ValueError("Unsupported checkpoint abstraction")
        if self.abstraction in EQUITY_SCHEMAS:
            if not isinstance(self.card_buckets, str) or not self.card_buckets:
                raise ValueError("K50 arena models require an explicit card_buckets directory")
        elif self.card_buckets is not None:
            raise ValueError("Card bucket tables require a K50 schema")
        if self.format not in {
            "legacy-standard-v1",
            "holdem-average-v1",
            "holdem-blueprint-v1",
            "holdem-hu20-blueprint-v2",
            "holdem-hu20-native-reopening-blueprint-v1",
            "holdem-tp20-blueprint-v1",
            "holdem-hu100-native-reopening-blueprint-v1",
            "holdem-hu200-native-reopening-blueprint-v1",
            "holdem-hu20-stored-cfr-average-diagnostic-v1",
            "holdem-hu100-stored-cfr-average-research-v1",
            "holdem-hu200-stored-cfr-average-research-v1",
        }:
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
