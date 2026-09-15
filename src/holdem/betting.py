"""Action-conditioned value/regret heads and explicit per-candidate supervision."""

from collections.abc import Sequence
from dataclasses import dataclass
from random import Random

import torch
from torch import nn

from src.game.types import Action
from src.holdem.actions import FEATURES, SCHEMA, BetCandidates
from src.holdem.model import DecisionEncoder
from src.holdem.targets import CandidateTargets


@dataclass(frozen=True)
class ActionScores:
    candidates: BetCandidates
    regrets: torch.Tensor
    values: torch.Tensor

    def probabilities(self) -> torch.Tensor:
        values = self.regrets.detach()
        if (
            values.shape != (len(self.candidates.actions),)
            or not torch.isfinite(values).all()
        ):
            raise ValueError("Expected one finite regret per candidate")
        positive = values.clamp_min(0).double()
        largest = positive.max()
        if largest > 0:
            scaled = positive / largest
            return scaled / scaled.sum()
        # Match the validated small-game baseline's largest-regret fallback.
        return torch.nn.functional.one_hot(values.argmax(), len(values)).double()

    def choose(self, random: Random) -> Action:
        index = random.choices(
            range(len(self.candidates.actions)),
            weights=self.probabilities().cpu().tolist(),
            k=1,
        )[0]
        return self.candidates.actions[index]


class BettingNetwork(nn.Module):
    def __init__(self, width: int = 128):
        super().__init__()
        self.encoder = DecisionEncoder(width)
        self.action = nn.Sequential(nn.Linear(width + FEATURES, width), nn.ReLU())
        self.regret = nn.Linear(width, 1)
        self.value = nn.Linear(width, 1)

    def forward(self, batch: Sequence[BetCandidates]) -> tuple[ActionScores, ...]:
        if not batch or any(not isinstance(c, BetCandidates) for c in batch):
            raise TypeError("Provide a nonempty batch of bet candidates")
        if any(
            c.schema != SCHEMA or not c.actions or len(c.features) != len(c.actions)
            for c in batch
        ):
            raise ValueError("Invalid candidate schema or shape")
        contexts = self.encoder([c.decision for c in batch])
        lengths = [len(c.actions) for c in batch]
        features = contexts.new_tensor([row for c in batch for row in c.features])
        repeated = torch.repeat_interleave(
            contexts, torch.tensor(lengths, device=contexts.device), dim=0
        )
        hidden = self.action(torch.cat((repeated, features), dim=1))
        regrets = self.regret(hidden).squeeze(-1).split(lengths)
        values = self.value(hidden).squeeze(-1).split(lengths)
        return tuple(ActionScores(c, r, v) for c, r, v in zip(batch, regrets, values))


def betting_loss(
    scores: Sequence[ActionScores], targets: Sequence[CandidateTargets]
) -> torch.Tensor:
    if not scores or len(scores) != len(targets):
        raise ValueError("Match each prediction with its candidate targets")
    losses = []
    for score, target in zip(scores, targets):
        # DecisionInput equality intentionally excludes exact source records.
        if (
            score.candidates != target.candidates
            or score.candidates.decision.source != target.candidates.decision.source
        ):
            raise ValueError("Targets belong to different candidates or observations")
        count = len(score.candidates.actions)
        if any(len(v) != count for v in (target.values_bb, target.regrets_bb)) or any(
            t.shape != (count,) for t in (score.regrets, score.values)
        ):
            raise ValueError("Expected one value and regret per candidate")
        regrets = score.regrets.new_tensor(target.regrets_bb)
        values = score.values.new_tensor(target.values_bb)
        # Sum over actions as in the reference solver; average over sampled decisions.
        losses.append(
            (score.regrets - regrets).square().sum()
            + (score.values - values).square().sum()
        )
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite betting loss")
    return loss
