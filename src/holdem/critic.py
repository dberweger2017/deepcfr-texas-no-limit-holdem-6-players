"""A persistent own-information Monte Carlo critic for frozen-policy experiments."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
from math import isfinite
from pathlib import Path
from random import Random
from time import perf_counter

import torch
from torch import nn

from src.game.hand import Hand
from src.holdem.actions import FEATURES, BetCandidates, bet_candidates
from src.holdem.collection import CollectionLimitExceeded
from src.holdem.model import DecisionEncoder
from src.holdem.policy import FrozenProfile
from src.holdem.records import pack, unpack
from src.solver.neural.checkpoint import atomic_write

FORMAT = "holdem-monte-carlo-critic-v1"


def check_limit(deadline: float) -> None:
    if perf_counter() >= deadline:
        raise CollectionLimitExceeded("Critic experiment exceeded its time budget")


def model_digest(model: nn.Module) -> str:
    digest = sha256(FORMAT.encode())
    for name, value in model.state_dict().items():
        digest.update(f"{name}/{value.shape}/{value.dtype}".encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class ValueNetwork(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.encoder = DecisionEncoder(width)
        self.head = nn.Sequential(
            nn.Linear(width + FEATURES + 6, width), nn.ReLU(), nn.Linear(width, 1)
        )

    def forward(self, batch: list[BetCandidates]) -> tuple[torch.Tensor, ...]:
        encoded = self.encoder([c.decision for c in batch])
        result = []
        for candidates, context in zip(batch, encoded):
            view = candidates.decision.source
            role = view.seat_numbers[view.seat]
            features = context.new_tensor(candidates.features)
            roles = context.new_zeros((len(features), 6))
            roles[:, role] = 1
            result.append(
                self.head(
                    torch.cat((context.expand(len(features), -1), features, roles), 1)
                ).squeeze(1)
            )
        return tuple(result)


@dataclass(frozen=True)
class CriticConfig:
    seed: int
    width: int = 32
    learning_rate: float = 0.001
    gradient_clip: float = 1.0
    replay_capacity: int = 4096
    batch_size: int = 32

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Invalid critic seed")
        if any(
            type(x) is not int or x < 1
            for x in (self.width, self.replay_capacity, self.batch_size)
        ):
            raise ValueError("Critic dimensions must be positive integers")
        if any(
            not isfinite(x) or x <= 0 for x in (self.learning_rate, self.gradient_clip)
        ):
            raise ValueError("Invalid critic fitting settings")


@dataclass(frozen=True)
class ValueRecord:
    candidates: BetCandidates
    action: int
    value_bb: float

    def __post_init__(self):
        if (
            not isinstance(self.candidates, BetCandidates)
            or type(self.action) is not int
            or not 0 <= self.action < len(self.candidates.actions)
            or not isfinite(self.value_bb)
        ):
            raise ValueError("Invalid critic record")


def monte_carlo_records(
    root: Hand,
    profile: FrozenProfile,
    random: Random,
    *,
    deadline: float,
    max_nodes: int,
) -> tuple[ValueRecord, ...]:
    """Explore at the root only; every recorded action has on-policy continuation."""
    if root.finished or root.table.capacity != profile.capacity:
        raise ValueError("Critic collection requires a live matching table")
    profile.assert_unchanged()
    hand, pending = root, []
    while not hand.finished:
        check_limit(deadline)
        if len(pending) >= max_nodes:
            raise CollectionLimitExceeded("Critic rollout exceeded its node budget")
        candidates = bet_candidates(hand.observe(hand.actor))
        if pending:
            probabilities = profile.distribution(candidates)
            action = random.choices(range(len(candidates.actions)), probabilities)[0]
        else:
            action = random.randrange(len(candidates.actions))
        pending.append((candidates, action))
        hand = hand.apply(candidates.actions[action])
    final = hand.events[-1].stacks
    if sum(final) != sum(root.table.stacks):
        raise ValueError("Critic rollout settlement is not zero sum")
    profile.assert_unchanged()
    return tuple(
        ValueRecord(
            c,
            a,
            (final[c.decision.source.seat] - root.table.stacks[c.decision.source.seat])
            / root.table.big_blind,
        )
        for c, a in pending
    )


class PersistentCritic:
    def __init__(self, config: CriticConfig):
        self.config = config
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.model = ValueNetwork(config.width)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.random = Random(config.seed)
        self.identity: tuple[str, str] | None = None
        self.replay: list[ValueRecord] = []
        self.cursor = self.steps = self.rollouts = self.phases = 0
        self.failed = False

    def begin_phase(self, profile: FrozenProfile, distribution: str) -> None:
        self._ready()
        profile.assert_unchanged()
        if not distribution:
            raise ValueError("Name the starting distribution")
        identity = (profile.fingerprint, distribution)
        if identity != self.identity:
            self.replay.clear()
            self.cursor = 0
        self.identity = identity
        self.phases += 1

    def _ready(self) -> None:
        if self.failed:
            raise RuntimeError("Cannot reuse a failed critic; restore a checkpoint")

    def collect(self, root, profile, *, deadline, max_nodes) -> int:
        self._ready()
        if self.identity is None or self.identity[0] != profile.fingerprint:
            raise ValueError("Start a phase for this continuation profile")
        try:
            records = monte_carlo_records(
                root, profile, self.random, deadline=deadline, max_nodes=max_nodes
            )
            for record in records:
                if len(self.replay) < self.config.replay_capacity:
                    self.replay.append(record)
                else:
                    self.replay[self.cursor] = record
                self.cursor = (self.cursor + 1) % self.config.replay_capacity
            self.rollouts += 1
        except Exception:
            self.failed = True
            raise
        return len(records)

    def fit(self, steps: int, *, deadline: float) -> list[float]:
        self._ready()
        if not self.replay or type(steps) is not int or steps < 1:
            raise ValueError("Fit needs replay and a positive step count")
        losses = []
        self.model.train()
        try:
            for _ in range(steps):
                check_limit(deadline)
                batch = self.random.choices(self.replay, k=self.config.batch_size)
                predictions = self.model([r.candidates for r in batch])
                chosen = torch.stack([p[r.action] for p, r in zip(predictions, batch)])
                target = chosen.new_tensor([r.value_bb for r in batch])
                loss = torch.mean((chosen - target) ** 2)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
                if not torch.isfinite(loss) or any(
                    not torch.isfinite(p).all() for p in self.model.parameters()
                ):
                    raise FloatingPointError("Nonfinite critic fit")
                losses.append(float(loss.detach()))
                self.steps += 1
        except Exception:
            self.failed = True
            raise
        return losses

    def save(self, path: Path) -> str:
        self._ready()
        if self.identity is None or not self.steps:
            raise ValueError("Save only a fitted critic phase")
        payload = {
            "format": FORMAT,
            "config": asdict(self.config),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "random": self.random.getstate(),
            "identity": self.identity,
            "replay": tuple(
                (pack(r.candidates), r.action, r.value_bb) for r in self.replay
            ),
            "cursor": self.cursor,
            "steps": self.steps,
            "rollouts": self.rollouts,
            "phases": self.phases,
        }
        buffer = BytesIO()
        torch.save(payload, buffer)
        return atomic_write(path, buffer.getvalue())

    @classmethod
    def load(cls, path: Path, digest: str) -> "PersistentCritic":
        data = path.read_bytes()
        if sha256(data).hexdigest() != digest:
            raise ValueError("Critic checkpoint hash mismatch")
        state = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
        if state["format"] != FORMAT:
            raise ValueError("Unknown critic checkpoint format")
        critic = cls(CriticConfig(**state["config"]))
        critic.model.load_state_dict(state["model"], strict=True)
        critic.optimizer.load_state_dict(state["optimizer"])
        critic.random.setstate(state["random"])
        critic.identity = state["identity"]
        critic.replay = [ValueRecord(unpack(c), a, v) for c, a, v in state["replay"]]
        for name in ("cursor", "steps", "rollouts", "phases"):
            value = state[name]
            if type(value) is not int or value < 0:
                raise ValueError("Invalid critic checkpoint counter")
            setattr(critic, name, value)
        if (
            not 0 < len(critic.replay) <= critic.config.replay_capacity
            or critic.cursor >= critic.config.replay_capacity
            or (
                len(critic.replay) < critic.config.replay_capacity
                and critic.cursor != len(critic.replay)
            )
            or not critic.steps
            or not critic.phases
            or not critic.rollouts
            or not isinstance(critic.identity, tuple)
            or len(critic.identity) != 2
            or any(not isinstance(s, str) or not s for s in critic.identity)
        ):
            raise ValueError("Invalid critic checkpoint state")
        for tensor in list(critic.model.state_dict().values()) + [
            v
            for item in critic.optimizer.state.values()
            for v in item.values()
            if isinstance(v, torch.Tensor)
        ]:
            if not torch.isfinite(tensor).all():
                raise ValueError("Nonfinite critic checkpoint")
        return critic


class BaselineProfile(FrozenProfile):
    """Change only the control variate, preserving the playing profile exactly."""

    def __init__(self, policy, *, kind, critic=None, historical=None):
        if kind not in ("zero", "accounting", "historical", "learned"):
            raise ValueError("Unknown baseline")
        if (kind == "learned") != (critic is not None) or (kind == "historical") != (
            historical is not None
        ):
            raise ValueError("Supply exactly the selected baseline")
        policy.assert_unchanged()
        if critic is not None:
            critic._ready()
            if critic.identity is None or critic.identity[0] != policy.fingerprint:
                raise ValueError("Critic was not fitted for this policy profile")
        if historical is not None and historical.capacity != policy.capacity:
            raise ValueError("Historical head belongs to a different table capacity")
        super().__init__([None] * policy.capacity)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "kind", kind)
        model = (
            None
            if critic is None
            else deepcopy(critic.model).eval().requires_grad_(False)
        )
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "historical", historical)
        snapshot = (
            policy.fingerprint,
            kind,
            None if model is None else model_digest(model),
            None if historical is None else historical.fingerprint,
        )
        object.__setattr__(self, "snapshot", snapshot)
        object.__setattr__(
            self, "fingerprint", sha256(repr(snapshot).encode()).hexdigest()
        )

    def assert_unchanged(self):
        self.policy.assert_unchanged()
        if self.historical is not None:
            self.historical.assert_unchanged()
        current = (
            self.policy.fingerprint,
            self.kind,
            None if self.model is None else model_digest(self.model),
            None if self.historical is None else self.historical.fingerprint,
        )
        if (
            current != self.snapshot
            or self.fingerprint != sha256(repr(current).encode()).hexdigest()
            or (
                self.model is not None and any(m.training for m in self.model.modules())
            )
        ):
            raise RuntimeError("Frozen critic baseline changed")

    def distribution(self, candidates):
        return self.policy.distribution(candidates)

    def action_values(self, candidates):
        if self.kind == "zero":
            return (0.0,) * len(candidates.actions)
        if self.kind == "accounting":
            view = candidates.decision.source
            player = view.players[view.seat]
            return ((player.stack - player.starting_stack) / view.big_blind,) * len(
                candidates.actions
            )
        if self.kind == "historical":
            return self.historical.action_values(candidates)
        with torch.inference_mode():
            values = self.model([candidates])[0]
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite critic prediction")
        return tuple(values.tolist())
