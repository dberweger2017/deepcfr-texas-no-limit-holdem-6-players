"""In-memory poker table sessions for the web client."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pokers as pkrs

from src.agents.random_agent import RandomAgent
from src.core.deep_cfr import DeepCFRAgent
from src.opponent_modeling.deep_cfr_with_opponent_modeling import DeepCFRAgentWithOpponentModeling
from src.utils.actions import build_raise_action, preset_raise_action, raise_bounds, sanitize_action
from src.utils.checkpoints import load_play_agent, select_random_checkpoints_in_dir
from src.utils.logging import apply_action_with_logging
from src.utils.cards import STAGE_LABELS, card_display_payload

MAX_AI_STEPS = 200


@dataclass
class ActionLogEntry:
    seat: int
    action: str
    amount: float = 0.0
    label: str = ""

    def to_dict(self) -> dict:
        return {"seat": self.seat, "action": self.action, "amount": self.amount, "label": self.label}


@dataclass
class PokerWebSession:
    session_id: str
    models_dir: Optional[str]
    hero_seat: int = 0
    stake: float = 200.0
    sb: float = 1.0
    bb: float = 2.0
    device: str = "cpu"
    agents: List[Optional[Any]] = field(default_factory=list)
    state: Optional[pkrs.State] = None
    hand_number: int = 0
    session_profit: float = 0.0
    action_log: List[ActionLogEntry] = field(default_factory=list)
    reveal_all: bool = False
    load_warnings: List[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.agents:
            self.agents = [None] * 6
            self._load_agents()

    def _load_agents(self) -> None:
        self.load_warnings.clear()
        selected = (
            select_random_checkpoints_in_dir(self.models_dir, num_models=5)
            if self.models_dir
            else []
        )

        model_idx = 0
        for seat in range(6):
            if seat == self.hero_seat:
                self.agents[seat] = None
                continue
            if model_idx < len(selected):
                path = selected[model_idx]
                agent, warning = load_play_agent(path, seat, self.device)
                self.agents[seat] = agent
                if warning is not None:
                    self.load_warnings.append(warning)
                model_idx += 1
            else:
                agent = RandomAgent(seat)
                agent.model_name = "Random"
                self.agents[seat] = agent

    def list_opponents(self) -> List[dict]:
        opponents = []
        for seat, agent in enumerate(self.agents):
            if seat == self.hero_seat or agent is None:
                continue
            opponents.append(
                {
                    "seat": seat,
                    "name": getattr(agent, "model_name", f"AI {seat}"),
                    "type": agent.__class__.__name__,
                }
            )
        return opponents

    def start_new_hand(self) -> dict:
        self.hand_number += 1
        self.action_log.clear()
        self.reveal_all = False
        seed = random.randint(0, 1_000_000)
        self.state = pkrs.State.from_seed(
            n_players=6,
            button=(self.hand_number - 1) % 6,
            sb=self.sb,
            bb=self.bb,
            stake=self.stake,
            seed=seed,
        )
        self._advance_ai()
        return self.serialize()

    def apply_hero_action(self, action_name: str, amount: Optional[float] = None) -> dict:
        if not self.state or self.state.final_state:
            raise ValueError("No active hand")
        if self.state.current_player != self.hero_seat:
            raise ValueError("Not your turn")

        action = self._build_action(action_name, amount)
        self._apply_action(action, self.hero_seat)
        if not self.state.final_state:
            self._advance_ai()
        else:
            self._finalize_hand()
        return self.serialize()

    def toggle_reveal(self) -> dict:
        self.reveal_all = not self.reveal_all
        return self.serialize()

    def _build_action(self, action_name: str, amount: Optional[float]) -> pkrs.Action:
        legal = self.state.legal_actions
        name = action_name.lower().strip()

        if name == "fold":
            if pkrs.ActionEnum.Fold not in legal:
                raise ValueError("Fold is not legal")
            return pkrs.Action(pkrs.ActionEnum.Fold)
        if name in {"check", "call"}:
            if pkrs.ActionEnum.Check in legal:
                return pkrs.Action(pkrs.ActionEnum.Check)
            if pkrs.ActionEnum.Call in legal:
                return pkrs.Action(pkrs.ActionEnum.Call)
            raise ValueError("Check/call is not legal")
        if name == "min":
            return preset_raise_action(self.state, "min")
        if name == "half_pot":
            return preset_raise_action(self.state, "half_pot")
        if name == "pot":
            return preset_raise_action(self.state, "pot")
        if name == "all_in":
            return preset_raise_action(self.state, "all_in")
        if name == "raise":
            if pkrs.ActionEnum.Raise not in legal:
                raise ValueError("Raise is not legal")
            if amount is None:
                bounds = raise_bounds(self.state)
                amount = bounds.min_raise
            return build_raise_action(self.state, amount)

        raise ValueError(f"Unknown action: {action_name}")

    def _apply_action(self, action: pkrs.Action, seat: int) -> None:
        action = sanitize_action(self.state, action)
        new_state, _, status = apply_action_with_logging(
            self.state,
            action,
            strict=False,
            error_prefix="Web poker action failed",
        )
        if new_state is None:
            raise ValueError(f"Illegal action ({status})")

        self.action_log.append(
            ActionLogEntry(
                seat=seat,
                action=self._action_name(action),
                amount=float(action.amount),
                label=self._action_label(seat, action),
            )
        )
        self.state = new_state

    def _advance_ai(self) -> None:
        steps = 0
        while self.state and not self.state.final_state and steps < MAX_AI_STEPS:
            seat = self.state.current_player
            if seat == self.hero_seat:
                return
            agent = self.agents[seat]
            if agent is None:
                agent = RandomAgent(seat)
            if isinstance(agent, DeepCFRAgentWithOpponentModeling):
                action = agent.choose_action(self.state, opponent_id=seat)
            else:
                action = agent.choose_action(self.state)
            self._apply_action(action, seat)
            steps += 1

        if self.state and self.state.final_state:
            self._finalize_hand()

    def _finalize_hand(self) -> None:
        if not self.state:
            return
        reward = float(self.state.players_state[self.hero_seat].reward)
        self.session_profit += reward
        self.reveal_all = True

    @staticmethod
    def _action_name(action: pkrs.Action) -> str:
        if action.action == pkrs.ActionEnum.Fold:
            return "fold"
        if action.action == pkrs.ActionEnum.Check:
            return "check"
        if action.action == pkrs.ActionEnum.Call:
            return "call"
        if action.action == pkrs.ActionEnum.Raise:
            return "raise"
        return "unknown"

    def _action_label(self, seat: int, action: pkrs.Action) -> str:
        who = "You" if seat == self.hero_seat else f"AI {seat}"
        name = self._action_name(action)
        if name == "raise":
            return f"{who} raises to ${action.amount:.2f}"
        return f"{who} {name}"

    def serialize(self) -> dict:
        if not self.state:
            return {
                "session_id": self.session_id,
                "hand_number": self.hand_number,
                "hero_seat": self.hero_seat,
                "session_profit": self.session_profit,
                "opponents": self.list_opponents(),
                "load_warnings": self.load_warnings,
                "has_active_hand": False,
            }

        legal = []
        if pkrs.ActionEnum.Fold in self.state.legal_actions:
            legal.append("fold")
        if pkrs.ActionEnum.Check in self.state.legal_actions:
            legal.append("check")
        if pkrs.ActionEnum.Call in self.state.legal_actions:
            legal.append("call")
        if pkrs.ActionEnum.Raise in self.state.legal_actions:
            legal.extend(["raise", "min", "half_pot", "pot", "all_in"])

        bounds_payload = None
        if pkrs.ActionEnum.Raise in self.state.legal_actions:
            bounds = raise_bounds(self.state)
            if bounds.can_raise:
                bounds_payload = {
                    "min_raise": bounds.min_raise,
                    "max_raise": bounds.max_raise,
                    "call_amount": bounds.call_amount,
                }

        players = []
        for seat, player in enumerate(self.state.players_state):
            show_cards = (
                self.reveal_all
                or seat == self.hero_seat
                or (self.state.final_state and player.active)
            )
            cards = []
            if show_cards and hasattr(player, "hand") and player.hand:
                cards = [card_display_payload(card) for card in player.hand]

            agent = self.agents[seat]
            players.append(
                {
                    "seat": seat,
                    "name": "You" if seat == self.hero_seat else getattr(agent, "model_name", f"AI {seat}"),
                    "stack": float(player.stake),
                    "bet": float(player.bet_chips),
                    "pot_won": float(player.pot_chips),
                    "active": bool(player.active),
                    "is_hero": seat == self.hero_seat,
                    "is_button": seat == self.state.button,
                    "is_current": seat == self.state.current_player,
                    "cards": cards,
                }
            )

        hand_result = None
        if self.state.final_state:
            reward = float(self.state.players_state[self.hero_seat].reward)
            hand_result = {
                "hero_reward": reward,
                "won": reward > 0,
            }

        return {
            "session_id": self.session_id,
            "hand_number": self.hand_number,
            "hero_seat": self.hero_seat,
            "session_profit": self.session_profit,
            "has_active_hand": True,
            "stage": STAGE_LABELS.get(int(self.state.stage), str(self.state.stage)),
            "pot": float(self.state.pot),
            "button": int(self.state.button),
            "current_player": int(self.state.current_player),
            "is_hero_turn": self.state.current_player == self.hero_seat and not self.state.final_state,
            "final_state": bool(self.state.final_state),
            "community_cards": [card_display_payload(card) for card in self.state.public_cards],
            "players": players,
            "legal_actions": legal,
            "raise_bounds": bounds_payload,
            "action_log": [entry.to_dict() for entry in self.action_log[-20:]],
            "hand_result": hand_result,
            "opponents": self.list_opponents(),
            "load_warnings": self.load_warnings,
        }


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, PokerWebSession] = {}

    def create(
        self,
        *,
        models_dir: Optional[str],
        hero_seat: int = 0,
        stake: float = 200.0,
        sb: float = 1.0,
        bb: float = 2.0,
        device: str = "cpu",
    ) -> PokerWebSession:
        session = PokerWebSession(
            session_id=str(uuid.uuid4()),
            models_dir=models_dir,
            hero_seat=hero_seat,
            stake=stake,
            sb=sb,
            bb=bb,
            device=device,
        )
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> PokerWebSession:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown session: {session_id}")
        return self._sessions[session_id]
