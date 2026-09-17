"""Diagnostic architectures; production policies and checkpoints are unchanged."""

from dataclasses import replace
from math import asinh

import torch
from torch import nn

from src.holdem.actions import AMOUNT_FIELDS
from src.holdem.betting import ActionScores, BettingNetwork
from src.holdem.encoding import (
    ACTIONS,
    CONTEXT_FIELDS,
    CONTEXT_SIZE,
    EVENT_AMOUNTS,
    EVENTS,
    SEATS,
    SEAT_FIELDS,
    STREETS,
)

VARIANTS = ("original", "scaled", "wide", "deep", "cards")


def scale_fields(values, names):
    return tuple(
        asinh(x) if name.endswith("_bb") else x
        for x, name in zip(values, names, strict=True)
    )


def scaled_candidates(candidates):
    d = candidates.decision
    numeric = list(scale_fields(d.context[: len(CONTEXT_FIELDS)], CONTEXT_FIELDS))
    numeric[8] /= SEATS
    numeric[9] /= SEATS
    offset = len(EVENTS) + SEATS + len(STREETS) + len(ACTIONS)
    events = tuple(
        row[:offset]
        + scale_fields(row[offset : offset + len(EVENT_AMOUNTS)], EVENT_AMOUNTS)
        + row[offset + len(EVENT_AMOUNTS) :]
        for row in d.events
    )
    decision = replace(
        d,
        context=tuple(numeric) + d.context[len(CONTEXT_FIELDS) :],
        seats=tuple(scale_fields(row, SEAT_FIELDS) for row in d.seats),
        pots=tuple((row[0], asinh(row[1])) + row[2:] for row in d.pots),
        events=events,
    )
    features = tuple(
        row[: len(ACTIONS)] + scale_fields(row[len(ACTIONS) :], AMOUNT_FIELDS)
        for row in candidates.features
    )
    return replace(candidates, decision=decision, features=features)


class Residual(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, width)
        )

    def forward(self, x):
        return torch.relu(x + self.layers(x))


class CardContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.cards = nn.Sequential(
            nn.Linear(208, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.numerical = nn.Sequential(nn.Linear(198, 32), nn.ReLU())
        self.combine = nn.Sequential(nn.Linear(96, 64), nn.ReLU())

    def forward(self, static):
        start, end = CONTEXT_SIZE, CONTEXT_SIZE + 208
        card = self.cards(static[:, start:end])
        context = self.numerical(torch.cat((static[:, :start], static[:, end:]), dim=1))
        return self.combine(torch.cat((card, context), dim=1))


class RepresentationNetwork(BettingNetwork):
    def __init__(self, variant):
        if variant not in VARIANTS:
            raise ValueError("Unknown diagnostic architecture")
        width = {"original": 32, "scaled": 32, "wide": 64, "deep": 48, "cards": 64}[
            variant
        ]
        super().__init__(width)
        self.variant = variant
        if variant == "deep":
            self.encoder.combine.extend([Residual(width, 96), Residual(width, 96)])
            self.action.append(Residual(width, width))
        if variant == "cards":
            self.encoder.context = CardContext()

    def forward(self, batch):
        transformed = (
            batch
            if self.variant == "original"
            else [scaled_candidates(c) for c in batch]
        )
        scores = super().forward(transformed)
        # Scaling is a model implementation detail, not a new target/action contract.
        return tuple(
            ActionScores(c, s.regrets, s.values)
            for c, s in zip(batch, scores, strict=True)
        )


def make_model(variant, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return RepresentationNetwork(variant)
