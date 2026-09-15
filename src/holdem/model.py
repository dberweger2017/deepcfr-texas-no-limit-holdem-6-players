"""A small history encoder for the replacement learner's decision representation."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from src.holdem.encoding import (
    CONTEXT_SIZE,
    EVENT_SIZE,
    POT_SIZE,
    SCHEMA,
    SEAT_FIELDS,
    SEATS,
    DecisionInput,
)


class DecisionEncoder(nn.Module):
    def __init__(self, width: int = 128):
        super().__init__()
        if type(width) is not int or width < 1:
            raise ValueError("Encoder width must be a positive integer")
        self.width = width
        static_size = CONTEXT_SIZE + 4 * 52 + SEATS * (len(SEAT_FIELDS) + POT_SIZE)
        self.context = nn.Sequential(nn.Linear(static_size, width), nn.ReLU())
        self.event = nn.Sequential(nn.Linear(EVENT_SIZE, width), nn.ReLU())
        self.history = nn.GRU(width, width, batch_first=True)
        self.combine = nn.Sequential(nn.Linear(2 * width, width), nn.ReLU())

    def forward(self, decisions: Sequence[DecisionInput]) -> torch.Tensor:
        if not decisions or any(not isinstance(d, DecisionInput) for d in decisions):
            raise TypeError("Provide a nonempty batch of encoded decisions")
        if any(d.schema != SCHEMA or not d.events for d in decisions):
            raise ValueError("Decision schema or event history is invalid")
        parameter = next(self.parameters())

        def tensor(values):
            return torch.tensor(values, device=parameter.device, dtype=parameter.dtype)

        static = tensor(
            [
                d.context
                + tuple(
                    value
                    for group in (d.cards, d.seats, d.pots)
                    for row in group
                    for value in row
                )
                for d in decisions
            ]
        )
        sequences = [tensor(d.events) for d in decisions]
        lengths = [len(d.events) for d in decisions]
        padded = self.event(pad_sequence(sequences, batch_first=True))
        # Packing keeps both padding and the batch's length order out of the state.
        packed = pack_padded_sequence(
            padded, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.history(packed)
        return self.combine(torch.cat((self.context(static), hidden[-1]), dim=1))
