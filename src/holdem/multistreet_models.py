"""Three representation arms used only by the multi-street diagnostic."""

from dataclasses import replace

import torch
from torch import nn

from src.holdem.betting import BettingNetwork
from src.holdem.representation_models import RepresentationNetwork, scaled_candidates
from src.holdem.visible_features import PARTIAL_FEATURE_SIZE, partial_visible_features


VARIANTS = (
    "scaled_baseline",
    "learned_separate_card_branch",
    "explicit_visible_features",
)


class PartialVisibleDecisionEncoder(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.visible = nn.Sequential(
            nn.Linear(PARTIAL_FEATURE_SIZE, base.width), nn.ReLU()
        )

    def forward(self, decisions):
        contexts = self.base(decisions)
        features = contexts.new_tensor(
            [
                partial_visible_features(
                    decision.source.hole_cards, decision.source.board
                )
                for decision in decisions
            ]
        )
        return torch.relu(contexts + self.visible(features))


class MultiStreetNetwork(BettingNetwork):
    def __init__(self, variant):
        if variant not in VARIANTS:
            raise ValueError("Unknown multi-street architecture")
        if variant == "learned_separate_card_branch":
            prototype = RepresentationNetwork("cards")
            super().__init__(64)
            self.encoder = prototype.encoder
            self.action = prototype.action
            self.regret = prototype.regret
            self.value = prototype.value
        else:
            super().__init__(32)
            if variant == "explicit_visible_features":
                self.encoder = PartialVisibleDecisionEncoder(self.encoder)
        self.variant = variant

    def forward(self, batch):
        transformed = [scaled_candidates(c) for c in batch]
        scores = super().forward(transformed)
        # Preserve the original candidate objects in ActionScores, so targets
        # and evaluation use the unchanged interface contract.
        return tuple(
            type(score)(c, score.regrets, score.values)
            for c, score in zip(batch, scores, strict=True)
        )


def make_model(variant, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return MultiStreetNetwork(variant)
