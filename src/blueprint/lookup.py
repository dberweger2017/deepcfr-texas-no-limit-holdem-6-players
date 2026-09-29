"""Explicit lookup modes for immutable button-zero blueprint checkpoints."""

from collections import Counter
from math import isfinite
from string import hexdigits

from src.blueprint.abstraction import (
    BUTTON_ZERO_COMPAT_LOOKUP, LEGACY_LOOKUP, SCHEMA, choices, information_key,
)
from src.blueprint.solver import regret_match
from src.game.types import ActionKind


# The source M4 slice and its two checkpoint-0.4 descendants used one fixed
# six-seat, 100-BB, button-zero training table. The hashes are pinned in the
# repository's campaign manifests and reports. Other artifacts need their own
# provenance decision before using this compatibility lookup.
BUTTON_ZERO_CHECKPOINTS = {
    "94d41c737d0e8f41a548d58c64dd219dfb9593459c80d7a7382569bf6963e33a": 8733,
    "c6649a3e8ab061c70c5f6192d91f1f4090f15d68f4821cf3b50f2746f816c845": 18455,
    "1b7d9ef0a6f111ac82d99f802cf7ac7bc5c8685ff6a2f214cb918f7ef9361bc2": 99646,
}
REPLICATION_PARENT = "94d41c737d0e8f41a548d58c64dd219dfb9593459c80d7a7382569bf6963e33a"
DESCENDANT_LINEAGE_SCHEMA = "button-zero-blueprint-descendant-v1"
SAMPLERS = {1: "legacy-external-sampling-v1",
            4: "postflop-continuation-replication-v1"}


def validate_lookup(trainer, mode: str, checkpoint_sha256: str | None,
                    lineage: dict | None = None) -> None:
    if mode == LEGACY_LOOKUP:
        return
    if mode != BUTTON_ZERO_COMPAT_LOOKUP:
        raise ValueError("Unknown blueprint lookup mode")
    known = (checkpoint_sha256 in BUTTON_ZERO_CHECKPOINTS and
             trainer.iteration == BUTTON_ZERO_CHECKPOINTS[checkpoint_sha256])
    table = trainer.table
    descendant = (
        isinstance(lineage, dict)
        and lineage.get("schema") == DESCENDANT_LINEAGE_SCHEMA
        and lineage.get("parent_checkpoint_sha256") == REPLICATION_PARENT
        and checkpoint_sha256 not in BUTTON_ZERO_CHECKPOINTS
        and lineage.get("output_checkpoint_sha256") == checkpoint_sha256
        and isinstance(checkpoint_sha256, str) and len(checkpoint_sha256) == 64
        and all(c in hexdigits for c in checkpoint_sha256)
        and isinstance(lineage.get("source_revision"), str)
        and len(lineage["source_revision"]) == 40
        and all(c in hexdigits for c in lineage["source_revision"])
        and lineage.get("source_dirty") is False
        and lineage.get("key_schema") == SCHEMA
        and lineage.get("action_menu_raise_cap") == trainer.config.raise_cap
        and lineage.get("sampler_version") == SAMPLERS.get(trainer.config.postflop_replicates)
        and lineage.get("continuation_seed") == trainer.config.seed
        and lineage.get("parent_iteration") == BUTTON_ZERO_CHECKPOINTS[REPLICATION_PARENT]
        and lineage.get("output_iteration") == trainer.iteration
        and type(lineage.get("completed_nodes")) is int
        and lineage["completed_nodes"] > 0
        and type(lineage.get("completed_outer_iterations")) is int
        and lineage["completed_outer_iterations"] ==
            trainer.iteration - BUTTON_ZERO_CHECKPOINTS[REPLICATION_PARENT]
        and lineage.get("training_table") == {
            "player_ids": list(table.player_ids), "stacks": list(table.stacks),
            "button": table.button, "small_blind": table.small_blind,
            "big_blind": table.big_blind, "chip_unit": table.chip_unit,
        }
    )
    if (not (known or descendant)
            or trainer.config.abstraction != SCHEMA
            or trainer.table.button != 0
            or trainer.table.capacity != 6
            or trainer.table.stacks != (10_000,) * 6):
        raise ValueError("Canonical compatibility requires a verified button-zero checkpoint")


class TableDistribution:
    """Read one loaded checkpoint; callers may opt into the compatible lookup."""

    def __init__(self, trainer, *, lookup_mode: str = LEGACY_LOOKUP,
                 checkpoint_sha256: str | None = None, uniform: bool = False,
                 lineage: dict | None = None):
        validate_lookup(trainer, lookup_mode, checkpoint_sha256, lineage)
        self.trainer = trainer
        self.lookup_mode = lookup_mode
        self.checkpoint_sha256 = checkpoint_sha256
        self.uniform = uniform
        self.lineage = lineage

    def key(self, view, menu):
        return information_key(view, menu, schema=self.trainer.config.abstraction,
                               lookup_mode=self.lookup_mode)

    def distribution(self, view):
        trainer = self.trainer
        if view.capacity != trainer.table.capacity:
            raise ValueError("Blueprint table size differs from the search table")
        menu = choices(view, raise_cap=trainer.config.raise_cap)
        node = trainer.nodes.get(self.key(view, menu))
        if node is not None and node.names != tuple(item.name for item in menu):
            raise ValueError("Blueprint action labels differ from the observation")
        trained = node is not None
        probabilities = (
            regret_match(tuple(node.regrets))
            if trained and not self.uniform else (1 / len(menu),) * len(menu)
        )
        return menu, probabilities, trained


class NoFreeFoldDistribution:
    """Postprocess a distribution without changing its menu or lookup key."""

    def __init__(self, source):
        self.source = source
        self.interventions = Counter()

    def distribution(self, view):
        menu, probabilities, trained = self.source.distribution(view)
        if (len(menu) != len(probabilities) or not probabilities
                or any(not isfinite(value) or value < 0 for value in probabilities)
                or abs(sum(probabilities) - 1) > 1e-8):
            raise ValueError("Invalid source blueprint distribution")
        if ActionKind.CHECK not in view.legal_actions.kinds:
            return menu, probabilities, trained
        fold = [index for index, item in enumerate(menu)
                if item.action.kind == ActionKind.FOLD]
        if not fold:
            return menu, probabilities, trained
        removed = sum(probabilities[index] for index in fold)
        self.interventions["eligible"] += 1
        self.interventions["removed_probability"] += removed
        if removed > 0:
            self.interventions["changed"] += 1
        remaining = [0.0 if index in fold else value
                     for index, value in enumerate(probabilities)]
        total = sum(remaining)
        if total <= 0:
            self.interventions["all_fold_mass"] += 1
            check = next(index for index, item in enumerate(menu)
                         if item.action.kind == ActionKind.CHECK)
            remaining = [float(index == check) for index in range(len(menu))]
        else:
            remaining = [value / total for value in remaining]
        return menu, tuple(remaining), trained
