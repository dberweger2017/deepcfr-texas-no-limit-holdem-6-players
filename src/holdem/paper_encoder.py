"""Paper-inspired card pooling with a public-history GRU for no-limit play."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from src.holdem.encoding import CONTEXT_SIZE, EVENT_SIZE, POT_SIZE, SCHEMA, SEAT_FIELDS, SEATS


class ResidualLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x):
        return torch.relu(x + self.linear(x))


class PaperEncoder(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        if width != 64:
            raise ValueError('The paper-inspired architecture requires width 64')
        self.width = width
        self.rank = nn.Embedding(13, 64)
        self.suit = nn.Embedding(4, 64)
        self.card = nn.Embedding(52, 64)
        self.cards = nn.Sequential(nn.Linear(256, 192), nn.ReLU(),
                                   ResidualLayer(192), nn.Linear(192, 64), nn.ReLU())
        static_size = CONTEXT_SIZE + SEATS * (len(SEAT_FIELDS) + POT_SIZE)
        self.context = nn.Sequential(nn.Linear(static_size, 64), nn.ReLU())
        # The final 52 event features are board indicators, already in the card tower.
        self.event = nn.Sequential(nn.Linear(EVENT_SIZE - 52, 64), nn.ReLU())
        self.history = nn.GRU(64, 64, batch_first=True)
        self.betting = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), ResidualLayer(64))
        self.combine = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                     ResidualLayer(64), nn.LayerNorm(64))

    def forward(self, decisions):
        if not decisions or any(d.schema != SCHEMA or not d.events for d in decisions):
            raise ValueError('Expected encoded public decisions with event history')
        parameter = next(self.parameters())
        def tensor(values):
            return torch.tensor(values, device=parameter.device, dtype=parameter.dtype)
        ids = torch.arange(52, device=parameter.device)
        embeddings = self.rank(ids // 4) + self.suit(ids % 4) + self.card(ids)
        # Existing encoding canonicalizes suits and represents unordered groups.
        groups = tensor([d.cards for d in decisions]) @ embeddings
        card_state = self.cards(groups.flatten(1))
        static = tensor([d.context + tuple(v for group in (d.seats, d.pots)
                                           for row in group for v in row) for d in decisions])
        sequences = [tensor([row[:-52] for row in d.events]) for d in decisions]
        packed = pack_padded_sequence(self.event(pad_sequence(sequences, batch_first=True)),
                                     [len(d.events) for d in decisions], batch_first=True,
                                     enforce_sorted=False)
        _, hidden = self.history(packed)
        betting = self.betting(torch.cat((self.context(static), hidden[-1]), dim=1))
        return self.combine(torch.cat((card_state, betting), dim=1))
