"""Closed, primitive-only checkpoint records; no import-by-name deserialization."""

from dataclasses import fields
from enum import Enum

from src.game.hand import Table
from src.game.observation import (
    ActionTaken,
    BlindPosted,
    BoardDealt,
    CardsMucked,
    CardsShown,
    Decision,
    HandFinished,
    HandStarted,
    Observation,
    ObservedHand,
)
from src.game.types import (
    Action,
    ActionKind,
    LegalActions,
    Player,
    Pot,
    Street,
    TableSeat,
)
from src.holdem.actions import BetCandidates
from src.holdem.encoding import DecisionInput
from src.holdem.fitting import FitConfig, FitMetrics
from src.holdem.replay import ReplaySample
from src.holdem.targets import CandidateTargets
from src.holdem.training import IterationReport, RoleUpdate, TrainConfig

TYPES = {
    t.__name__: t
    for t in (
        Table,
        ActionTaken,
        BlindPosted,
        BoardDealt,
        CardsMucked,
        CardsShown,
        Decision,
        HandFinished,
        HandStarted,
        Observation,
        ObservedHand,
        Action,
        LegalActions,
        Player,
        Pot,
        TableSeat,
        BetCandidates,
        DecisionInput,
        FitConfig,
        FitMetrics,
        ReplaySample,
        CandidateTargets,
        IterationReport,
        RoleUpdate,
        TrainConfig,
    )
}
ENUMS = {t.__name__: t for t in (ActionKind, Street)}


def pack(value):
    if isinstance(value, Enum) and type(value).__name__ in ENUMS:
        return {"enum": type(value).__name__, "value": value.value}
    if type(value).__name__ in TYPES and type(value) is TYPES[type(value).__name__]:
        return {
            "record": type(value).__name__,
            "fields": {f.name: pack(getattr(value, f.name)) for f in fields(value)},
        }
    if type(value) is tuple:
        return tuple(pack(v) for v in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"Unsupported checkpoint record: {type(value).__name__}")


def unpack(value):
    if type(value) is tuple:
        return tuple(unpack(v) for v in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    if type(value) is dict:
        if set(value) == {"enum", "value"} and value["enum"] in ENUMS:
            return ENUMS[value["enum"]](value["value"])
        if set(value) == {"record", "fields"} and value["record"] in TYPES:
            cls = TYPES[value["record"]]
            if set(value["fields"]) == {f.name for f in fields(cls)}:
                return cls(**{k: unpack(v) for k, v in value["fields"].items()})
    raise ValueError("Unknown or malformed checkpoint record")
