"""Portable, hash-pinned blueprint checkpoints and observation-only play."""

from dataclasses import asdict
from gzip import GzipFile, compress, decompress
from gzip import open as gzip_open
from hashlib import sha256
from io import TextIOWrapper
from json import dumps, loads
from math import isfinite
from os import fsync, replace
from pathlib import Path
from random import Random

from src.blueprint.abstraction import (
    SCHEMA,
    HU20_SCHEMA,
    TP20_SCHEMA,
    TP20_MENU_VERSION,
    SHORTSTACK_SEATS,
    HU20_MENU_VERSION,
    HU20_CARD_VERSION,
    SUPPORTED_SCHEMAS,
    choices,
    information_key,
)
from src.blueprint.solver import (
    FORMAT,
    HU20_GAME,
    TP20_GAME,
    SHORTSTACK_GAMES,
    LEGACY_GAME,
    BlueprintTrainer,
    Node,
    PilotConfig,
    regret_match,
)
from src.game.hand import Table

HU20_FORMAT = "holdem-hu20-blueprint-v2"


TP20_FORMAT = "holdem-tp20-blueprint-v1"
SHORTSTACK_FORMATS = {HU20_GAME: HU20_FORMAT, TP20_GAME: TP20_FORMAT}


def _format(config: PilotConfig) -> str:
    return SHORTSTACK_FORMATS.get(config.game, FORMAT)


def _identity(config: PilotConfig) -> dict:
    seats = SHORTSTACK_SEATS.get(config.abstraction)
    return ({"game": config.game, "players": seats, "stacks": [2000] * seats,
             "small_blind": 50, "big_blind": 100,
             "action_menu": HU20_MENU_VERSION if seats == 2 else TP20_MENU_VERSION,
             "card_descriptor": HU20_CARD_VERSION} if seats else {})


def _encoded(document: dict) -> bytes:
    return compress(
        dumps(
            document, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode(),
        mtime=0,
    )


def _write(path: Path, document: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _encoded(document)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    replace(temporary, path)
    return sha256(data).hexdigest()


def _read(path: Path) -> dict:
    document = loads(decompress(path.read_bytes()))
    if (
        not isinstance(document, dict)
        or document.get("format") not in (FORMAT, *SHORTSTACK_FORMATS.values())
    ):
        raise ValueError("Unknown blueprint artifact format")
    _checked_schema(document)
    return document


def _checked_schema(document: dict) -> str:
    schema = document.get("abstraction")
    config = document.get("config")
    if (
        schema not in SUPPORTED_SCHEMAS
        or not isinstance(config, dict)
        or config.get("abstraction", SCHEMA) != schema
        or document.get("format") != SHORTSTACK_FORMATS.get(
            SHORTSTACK_GAMES.get(schema), FORMAT)
        or config.get("game", LEGACY_GAME) !=
            SHORTSTACK_GAMES.get(schema, LEGACY_GAME)
        or (schema in SHORTSTACK_SEATS and document.get("identity") !=
            _identity(PilotConfig(**config)))
    ):
        raise ValueError("Unknown blueprint abstraction schema")
    return schema


def _config(config: PilotConfig) -> dict:
    document = asdict(config)
    if config.abstraction == SCHEMA:
        # Keep existing v1 checkpoint and export bytes reproducible.
        del document["abstraction"]
    if config.postflop_replicates == 1:
        # Older checkpoints did not record this optional sampling mode.
        del document["postflop_replicates"]
    if config.game == LEGACY_GAME:
        del document["game"]
    return document


def _table(table: Table) -> dict:
    return {
        "player_ids": table.player_ids,
        "stacks": table.stacks,
        "button": table.button,
        "small_blind": table.small_blind,
        "big_blind": table.big_blind,
        "chip_unit": table.chip_unit,
    }


def save_training(trainer: BlueprintTrainer, path: Path) -> str:
    header = {
        "format": _format(trainer.config),
        "abstraction": trainer.config.abstraction,
        "kind": "training",
        "checkpoint_format": "jsonl-v2",
        "table": _table(trainer.table),
        "config": _config(trainer.config),
        "iteration": trainer.iteration,
    }
    if trainer.config.game in SHORTSTACK_FORMATS:
        header["identity"] = _identity(trainer.config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw:
        with (
            GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped,
            TextIOWrapper(zipped, encoding="utf-8") as output,
        ):
            output.write(dumps(header, sort_keys=True, separators=(",", ":")) + "\n")
            for key in sorted(trainer.nodes):
                node = trainer.nodes[key]
                output.write(
                    dumps(
                        [key, node.names, node.regrets, node.average, node.visits],
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
        raw.flush()
        fsync(raw.fileno())
    replace(temporary, path)
    digest = sha256()
    with path.open("rb") as saved:
        for chunk in iter(lambda: saved.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training(path: Path) -> BlueprintTrainer:
    with gzip_open(path, "rt", encoding="utf-8") as source:
        document = loads(source.readline())
        if (
            not isinstance(document, dict)
            or document.get("format") not in (FORMAT, *SHORTSTACK_FORMATS.values())
        ):
            raise ValueError("Unknown blueprint artifact format")
        _checked_schema(document)
        if document.get("checkpoint_format") == "jsonl-v2":
            rows = (loads(line) for line in source)
        elif "checkpoint_format" not in document:
            rows = (
                ([key, *row] for key, row in document["nodes"].items())
                if document.get("kind") == "training"
                else ()
            )
        else:
            raise ValueError("Unknown blueprint checkpoint format")
        return _load_training_document(document, rows)


def _load_training_document(document: dict, rows) -> BlueprintTrainer:
    if document.get("kind") != "training":
        raise ValueError("An inference export cannot resume training")
    table_data = document["table"]
    table = Table(
        tuple(table_data["player_ids"]),
        tuple(table_data["stacks"]),
        table_data["button"],
        table_data["small_blind"],
        table_data["big_blind"],
        table_data["chip_unit"],
    )
    trainer = BlueprintTrainer(table, PilotConfig(**document["config"]))
    iteration = document["iteration"]
    if type(iteration) is not int or iteration < 0:
        raise ValueError("Invalid blueprint iteration")
    trainer.iteration = iteration
    for key, names, regrets, average, visits in rows:
        if (
            not isinstance(key, str)
            or len(key) != 32
            or len(names) == 0
            or len(names) != len(regrets)
            or len(names) != len(average)
            or len(set(names)) != len(names)
            or not all(isfinite(x) for x in (*regrets, *average))
            or type(visits) is not int
            or visits < 0
        ):
            raise ValueError("Invalid blueprint node")
        trainer.nodes[key] = Node(tuple(names), list(regrets), list(average), visits)
    if len(trainer.nodes) > trainer.config.max_entries:
        raise ValueError("Blueprint checkpoint exceeds its entry cap")
    return trainer


def export_policy(
    trainer: BlueprintTrainer, path: Path, *, strategy: str = "current"
) -> str:
    if strategy not in {"current", "average"}:
        raise ValueError("Export current or average strategy")
    if trainer.config.game in SHORTSTACK_FORMATS and strategy != "current":
        raise ValueError("Short-stack games require current export or a separately collected windowed extraction")
    entries = {}
    for key, node in trainer.nodes.items():
        if strategy == "current":
            probabilities = regret_match(tuple(node.regrets))
        else:
            total = sum(node.average)
            probabilities = (
                tuple(value / total for value in node.average)
                if total > 0
                else (1 / len(node.names),) * len(node.names)
            )
        entries[key] = [node.names, probabilities]
    document = {
            "format": _format(trainer.config),
            "abstraction": trainer.config.abstraction,
            "kind": "inference",
            "table": _table(trainer.table),
            "config": _config(trainer.config),
            "iteration": trainer.iteration,
            "strategy": strategy,
            "entries": entries,
        }
    if trainer.config.game in SHORTSTACK_FORMATS:
        document["identity"] = _identity(trainer.config)
    return _write(
        path,
        document,
    )


class FrozenBlueprint:
    def __init__(self, spec, path: Path):
        self.spec = spec
        self.data = path.read_bytes()
        if sha256(self.data).hexdigest() != spec.sha256:
            raise ValueError("Blueprint export hash mismatch")
        document = _read(path)
        if document.get("kind") != "inference":
            raise ValueError("Training checkpoints are not arena policies")
        if spec.format != document["format"]:
            raise ValueError("Checkpoint adapter format differs from inference artifact")
        if document["format"] in SHORTSTACK_FORMATS.values():
            table_data = document["table"]
            table = Table(tuple(table_data["player_ids"]), tuple(table_data["stacks"]),
                          table_data["button"], table_data["small_blind"],
                          table_data["big_blind"], table_data["chip_unit"])
            BlueprintTrainer(table, PilotConfig(**document["config"]))
        self.players = len(document["table"]["stacks"])
        self.raise_cap = document["config"]["raise_cap"]
        self.abstraction = document["abstraction"]
        self.entries = {}
        for key, row in document["entries"].items():
            names, probabilities = row
            if (
                not isinstance(key, str)
                or len(key) != 32
                or len(names) != len(probabilities)
                or len(names) == 0
                or len(set(names)) != len(names)
                or not all(isfinite(p) and p >= 0 for p in probabilities)
                or abs(sum(probabilities) - 1) > 1e-8
            ):
                raise ValueError("Invalid blueprint policy node")
            self.entries[key] = (tuple(names), tuple(probabilities))
        self.description = {
            "kind": document["format"],
            "weights_sha256": spec.sha256,
            "num_players": self.players,
            "iteration": document["iteration"],
            "training_seed": document["config"]["seed"],
            "strategy": document["strategy"],
            "abstraction": self.abstraction,
            "entries": len(self.entries),
        }

    def policy(self, seed: int):
        return _Player(self, seed)

    def distribution(self, view):
        if view.capacity != self.players:
            raise ValueError("Blueprint table size differs from the evaluation table")
        menu = choices(view, raise_cap=self.raise_cap,
                       free_fold=self.abstraction not in SHORTSTACK_SEATS)
        key = information_key(view, menu, schema=self.abstraction)
        saved = self.entries.get(key)
        if saved is None:
            return menu, (1 / len(menu),) * len(menu), False
        names, probabilities = saved
        if names != tuple(item.name for item in menu):
            raise ValueError("Blueprint action labels differ from the observation")
        return menu, probabilities, True


class _Player:
    def __init__(self, blueprint: FrozenBlueprint, seed: int):
        self.blueprint = blueprint
        self.random = Random(seed)

    def choose_action(self, view):
        menu, probabilities, _ = self.blueprint.distribution(view)
        return self.random.choices(menu, weights=probabilities, k=1)[0].action
