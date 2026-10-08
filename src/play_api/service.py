"""Durable heads-up turns around the existing hand and blueprint interfaces."""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from random import Random

from src.arena.runner import public_events
from src.arena.schedule import digest
from src.blueprint.abstraction import choices
from src.blueprint.artifact import HU20_UNCAPPED_FORMAT
from src.game.hand import Hand, Table
from src.game.observation import ActionTaken, BlindPosted, BoardDealt, CardsMucked, CardsShown
from src.game.types import Action, ActionKind
from src.play_api.configuration import PlayTable, policy_table, recorded_table, inference_record, distribution

MODEL_NAME = "B100M · seed 2026093001"
API_VERSION = "hu20-play-api-v1"
ADAPTER_ID = "direct-v1"
BENCHMARK_PROTOCOL = "hu20-human-benchmark-v1"
BUTTON_SCHEDULE = "alternating-seat-0-first-v1"
PRESETS = (("⅓ pot", 1, 3), ("½ pot", 1, 2), ("⅔ pot", 2, 3),
           ("¾ pot", 3, 4), ("Pot", 1, 1), ("1.5× pot", 3, 2))


class PlayError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tuple(value):
    return tuple(_tuple(item) if isinstance(item, list) else item for item in value)


def _rng(state):
    generator = Random()
    generator.setstate(_tuple(state))
    return generator


def _table(button):
    return PlayTable().table(("human", "trained"), button)


def _action(row):
    return Action(ActionKind(row["kind"]), row.get("raiseTo"))


def _hand(row):
    hand = Hand.start(recorded_table(row).table(("human", "trained"), row["button"]), hand_id=row["handId"], seed=row["dealSeed"])
    for record in row["actions"]:
        if hand.actor != record["seat"]:
            raise RuntimeError("Private journal actor mismatch")
        hand = hand.apply(_action(record))
    return hand


def _public_event(event):
    if isinstance(event, BlindPosted):
        return {"event": "blind", "seat": event.seat, "amount": event.amount}
    if isinstance(event, ActionTaken):
        return {"event": "action", "seat": event.seat, "street": event.street.value,
                "kind": event.action.kind.value, "raiseTo": event.action.raise_to, "paid": event.paid}
    if isinstance(event, BoardDealt):
        return {"event": "board", "street": event.street.value, "cards": list(event.cards)}
    if isinstance(event, CardsShown):
        return {"event": "shown", "seat": event.seat, "cards": list(event.cards)}
    if isinstance(event, CardsMucked):
        return {"event": "mucked", "seat": event.seat}
    return None


def _events(hand):
    return [item for event in hand.events if (item := _public_event(event)) is not None]


def _model_info(policy):
    return {"name": getattr(policy, "name", MODEL_NAME), "sha256": policy.spec.sha256,
            "game": policy.game, "schema": policy.abstraction,
            "format": getattr(policy, "format_id", HU20_UNCAPPED_FORMAT), "strategy": policy.description["strategy"],
            "adapter": getattr(policy, "adapter_id", ADAPTER_ID),
            "benchmarkOnly": getattr(policy, "benchmark_only", False),
            "research": getattr(policy, "research", False),
            "table": policy_table(policy).record(), "inference": inference_record(policy)}


class PlayService:
    def __init__(self, db_path: Path, policy, *, source_version: str = "unknown", table: PlayTable | None = None):
        self.policy = policy
        self.table = table or policy_table(policy)
        self.table.validate_policy(policy)
        self.model = _model_info(policy)
        self.db_path = db_path
        self.source_version = source_version
        self.lock = threading.RLock()
        db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(db_path.parent, 0o700)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        os.chmod(db_path, 0o600)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, state TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS operations (key TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, response TEXT NOT NULL)")
        self.db.commit()

    def close(self):
        self.db.close()

    def _load(self, session_id):
        row = self.db.execute("SELECT state FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise PlayError("Unknown session", 404)
        state = json.loads(row[0])
        if state["modelSha256"] != self.policy.spec.sha256 or state["modelGame"] != self.policy.game:
            raise PlayError("Session model differs from the loaded policy", 409)
        if (recorded_table(state) != self.table
                or state.get('modelIdentity', self.model if self.table.stack == 2000 else None) != self.model):
            raise PlayError("Session table or inference identity differs", 409)
        for record in [*state['history'], *([state['current']] if state['current'] else [])]:
            if recorded_table(record) != self.table:
                raise PlayError("Hand table configuration differs", 409)
        return state

    def _ack_matches(self, response):
        recorded = response['model']
        if self.table.stack == 10000:
            return recorded == self.model
        # Pre-configuration HU20 acknowledgments retain their original shape.
        return all(self.model.get(key) == value for key, value in recorded.items())

    def _mutate(self, key, fingerprint, session_id, operation):
        if not isinstance(key, str) or not 16 <= len(key) <= 128 or not key.isascii():
            raise PlayError("Invalid idempotency key")
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                prior = self.db.execute("SELECT fingerprint,response FROM operations WHERE key=?", (key,)).fetchone()
                if prior is not None:
                    if prior[0] != fingerprint:
                        raise PlayError("Idempotency key conflicts with an earlier request", 409)
                    if not self._ack_matches(json.loads(prior[1])):
                        raise PlayError("Acknowledgment belongs to another model", 409)
                    self.db.commit()
                    return json.loads(prior[1])
                state = self._load(session_id) if session_id else None
                state, response = operation(state)
                self.db.execute("INSERT OR REPLACE INTO sessions (id,state) VALUES (?,?)",
                                (state["sessionId"], _json(state)))
                self.db.execute("INSERT INTO operations (key,fingerprint,response) VALUES (?,?,?)",
                                (key, fingerprint, _json(response)))
                self.db.commit()
                return response
            except Exception:
                self.db.rollback()
                raise

    def create(self, key, body):
        if (getattr(self.policy, "benchmark_only", False)
                and (body.get("sessionType") != "benchmark" or body.get("playMode") != "restricted")):
            raise PlayError("This control supports restricted benchmark sessions only")
        if body.get("playMode") not in ("restricted", "free"):
            raise PlayError("Choose a valid play mode")
        if body.get("sessionType") == "benchmark":
            if (set(body) != {"sessionType", "playMode", "targetHands"}
                    or type(body["targetHands"]) is not int
                    or not 1 <= body["targetHands"] <= 5000):
                raise PlayError("Benchmark needs a target of 1–5000 hands")
            visibility = "benchmark"
            session_type = "benchmark"
        elif (set(body) in ({"playMode", "visibility"}, {"sessionType", "playMode", "visibility"})
              and body.get("sessionType", "casual") == "casual"
              and body["visibility"] in ("developer", "benchmark")):
            visibility = body["visibility"]
            session_type = "casual"
        else:
            raise PlayError("Choose a valid session type and visibility")

        def operation(_):
            deals, bot = Random(secrets.randbits(256)), Random(secrets.randbits(256))
            state = {"sessionId": secrets.token_urlsafe(18), "playMode": body["playMode"],
                     "visibility": visibility, "sessionType": session_type,
                     "revision": 0, "handsPlayed": 0,
                     "totalChips": 0, "dealRng": deals.getstate(), "botRng": bot.getstate(),
                     "current": None, "history": [], "sourceVersion": self.source_version,
                     "modelSha256": self.policy.spec.sha256, "modelGame": self.policy.game,
                     "table": self.table.record(), "modelIdentity": self.model}
            if session_type == "benchmark":
                state["benchmark"] = {
                    "id": secrets.token_urlsafe(18), "protocolVersion": BENCHMARK_PROTOCOL,
                    "targetHands": body["targetHands"], "status": "ACTIVE",
                    "buttonSchedule": BUTTON_SCHEDULE, "startedAt": _utc_now(),
                    "endedAt": None, "abortedHandId": None,
                    "modelName": _model_info(self.policy)["name"], "modelSha256": self.policy.spec.sha256,
                    "game": self.policy.game, "schema": self.policy.abstraction,
                    "playMode": body["playMode"], "adapter": _model_info(self.policy)["adapter"],
                    "sourceVersion": self.source_version, "interfaceVersion": API_VERSION,
                    "visibility": "benchmark", "table": self.table.record(),
                    "inference": inference_record(self.policy)}
            return state, self._view(state)

        return self._mutate(key, _json(["create", body]), None, operation)

    def state(self, session_id):
        with self.lock:
            return self._view(self._load(session_id))

    def model_info(self):
        return _model_info(self.policy)

    def _check_revision(self, state, body, keys, *, hand=True):
        if set(body) != keys or type(body.get("revision")) is not int:
            raise PlayError("Invalid request fields")
        if body["revision"] != state["revision"]:
            raise PlayError("Stale state revision", 409)
        if hand and (state["current"] is None or body.get("handId") != state["current"]["handId"]):
            raise PlayError("Stale or unknown hand", 409)

    def new_hand(self, session_id, key, body):
        def operation(state):
            self._check_revision(state, body, {"revision"}, hand=False)
            self._require_active(state)
            if state["current"] is not None and not _hand(state["current"]).finished:
                raise PlayError("Finish the current hand first", 409)
            deals = _rng(state["dealRng"])
            state["current"] = {"handId": secrets.token_urlsafe(18),
                                "button": state["handsPlayed"] % 2,
                                "dealSeed": deals.randrange(2**63), "actions": [], "lookup": [],
                                "table": self.table.record(), "game": self.policy.game,
                                "modelIdentity": self.model, "samplingStart": state["botRng"],
                                "botDecisions": []}
            state["dealRng"] = deals.getstate()
            state["revision"] += 1
            return state, self._view(state)

        return self._mutate(key, _json(["new-hand", session_id, body]), session_id, operation)

    def act(self, session_id, key, body):
        def operation(state):
            expected = {"handId", "revision", "kind", "raiseTo"}
            self._check_revision(state, body, expected)
            self._require_active(state)
            hand = _hand(state["current"])
            if hand.finished or hand.actor != 0:
                raise PlayError("It is not your turn", 409)
            if body["kind"] not in tuple(k.value for k in ActionKind):
                raise PlayError("Invalid action")
            amount = body["raiseTo"]
            if body["kind"] == "raise":
                if type(amount) is not int:
                    raise PlayError("Raise-to must be an exact integer chip amount")
            elif amount is not None:
                raise PlayError("Only raises have a raise-to amount")
            try:
                action = Action(ActionKind(body["kind"]), amount)
            except (TypeError, ValueError) as exc:
                raise PlayError("Invalid action amount") from exc
            view = hand.observe(0)
            try:
                view.legal_actions.validate(action)
                if state["playMode"] == "restricted" and action not in tuple(
                    item.action for item in choices(view, raise_cap=None, free_fold=False)
                ):
                    raise PlayError("Action is outside the restricted menu")
            except ValueError as exc:
                raise PlayError("Action is not legal at this decision") from exc
            state["current"]["actions"].append(self._record_action(hand, action, view))
            state["revision"] += 1
            self._complete(state)
            return state, self._view(state)

        return self._mutate(key, _json(["action", session_id, body]), session_id, operation)

    def advance(self, session_id, key, body):
        def operation(state):
            self._check_revision(state, body, {"handId", "revision"})
            self._require_active(state)
            hand = _hand(state["current"])
            if hand.finished or hand.actor != 1:
                raise PlayError("The bot is not acting", 409)
            bot = _rng(state["botRng"])
            for _ in range(1000):
                if hand.finished or hand.actor != 1:
                    break
                view = hand.observe(1)
                menu, probabilities, trained, telemetry = distribution(self.policy, view)
                selected = bot.choices(range(len(menu)), weights=probabilities, k=1)[0]
                action = menu[selected].action
                from src.play_api.spectator import observation_record
                if 'botDecisions' in state['current']:
                    state['current']['botDecisions'].append({
                        'actionIndex': len(state['current']['actions']),
                        'observation': observation_record(view), 'telemetry': telemetry,
                        'menu': [{'label': item.name, 'kind': item.action.kind.value,
                                  'raiseTo': item.action.raise_to, 'probability': probability}
                                 for item, probability in zip(menu, probabilities)],
                        'selectedIndex': selected})
                view.legal_actions.validate(action)
                if trained is not None:
                    state["current"]["lookup"].append({"street": view.street.value, "trained": bool(trained),
                                                        **({"telemetry": telemetry} if telemetry is not None else {})})
                state["current"]["actions"].append(self._record_action(hand, action, view))
                hand = hand.apply(action)
            else:
                raise RuntimeError("Bot exceeded the decision limit")
            state["botRng"] = bot.getstate()
            state["revision"] += 1
            self._complete(state)
            return state, self._view(state)

        return self._mutate(key, _json(["advance", session_id, body]), session_id, operation)

    @staticmethod
    def _record_action(hand, action, view):
        legal = view.legal_actions
        return {"seat": hand.actor, "kind": action.kind.value, "raiseTo": action.raise_to,
                "legal": {"kinds": [kind.value for kind in legal.kinds],
                          "call": legal.call_amount, "minRaiseTo": legal.min_raise_to,
                          "maxRaiseTo": legal.max_raise_to}}

    @staticmethod
    def _require_active(state):
        if state.get("sessionType", "casual") == "benchmark" and state["benchmark"]["status"] != "ACTIVE":
            raise PlayError("Benchmark has ended", 409)

    def end_benchmark(self, session_id, key, body):
        def operation(state):
            if state.get("sessionType", "casual") != "benchmark":
                raise PlayError("This is not a benchmark session", 409)
            self._check_revision(state, body, {"revision", "handId", "confirm"}, hand=False)
            if body["confirm"] is not True:
                raise PlayError("Confirm early benchmark end")
            current_id = state["current"]["handId"] if state["current"] else None
            if body["handId"] != current_id:
                raise PlayError("Stale or unknown hand", 409)
            self._require_active(state)
            benchmark = state["benchmark"]
            benchmark["status"] = "ABORTED"
            benchmark["endedAt"] = _utc_now()
            if state["current"] is not None and not _hand(state["current"]).finished:
                benchmark["abortedHandId"] = current_id
            state["revision"] += 1
            return state, self._view(state)

        return self._mutate(key, _json(["end-benchmark", session_id, body]), session_id, operation)

    def _complete(self, state):
        row = state["current"]
        hand = _hand(row)
        if not hand.finished or row.get("completed"):
            return
        final = hand.observe(0)
        net = final.players[0].stack - recorded_table(row).stack
        if final.players[1].stack - recorded_table(row).stack != -net:
            raise RuntimeError("Native settlement did not conserve chips")
        row["completed"] = True
        row["humanChips"] = net
        row["publicEventsSha256"] = digest(public_events(hand.events))
        row["modelSha256"] = self.policy.spec.sha256
        row["game"] = self.policy.game
        row["schema"] = self.policy.abstraction
        row["adapter"] = _model_info(self.policy)["adapter"]
        row["apiVersion"] = API_VERSION
        row["sourceVersion"] = state["sourceVersion"]
        row["playMode"] = state["playMode"]
        row["visibility"] = state["visibility"]
        state["history"].append(row)
        state["handsPlayed"] += 1
        state["totalChips"] += net
        if (state.get("sessionType", "casual") == "benchmark"
                and state["handsPlayed"] == state["benchmark"]["targetHands"]):
            state["benchmark"]["status"] = "COMPLETE"
            state["benchmark"]["endedAt"] = _utc_now()

    def _presets(self, view, restricted):
        if restricted:
            return []
        legal = view.legal_actions
        if ActionKind.RAISE not in legal.kinds:
            return []
        player = view.players[0]
        matched = player.street_bet + legal.call_amount
        after_call = view.pot + legal.call_amount
        result = []
        for label, num, den in PRESETS:
            # A pot fraction adds to the matched street wager, rounded half-up to one chip.
            target = matched + (after_call * num * 2 + den) // (2 * den)
            available = legal.min_raise_to <= target <= legal.max_raise_to
            result.append({"label": label, "raiseTo": target, "available": available})
        result.append({"label": "All-in", "raiseTo": legal.max_raise_to,
                       "available": True})
        return result

    def _view(self, state):
        response = {"sessionId": state["sessionId"], "revision": state["revision"],
                    "playMode": state["playMode"], "visibility": state["visibility"],
                    "model": _model_info(self.policy), "handsPlayed": state["handsPlayed"],
                    "sessionType": state.get("sessionType", "casual"),
                    "table": self.table.record(), "phase": "ready" if state["current"] is None else "playing", "hand": None}
        benchmark = state.get("benchmark")
        if benchmark:
            response["benchmark"] = {"id": benchmark["id"],
                                     "protocolVersion": benchmark["protocolVersion"],
                                     "targetHands": benchmark["targetHands"],
                                     "completedHands": state["handsPlayed"],
                                     "status": benchmark["status"],
                                     "buttonSchedule": benchmark["buttonSchedule"]}
            if benchmark["status"] != "ACTIVE":
                response["benchmarkResult"] = self._benchmark_report(state)
                response["phase"] = benchmark["status"].lower()
        if benchmark is None or benchmark["status"] != "ACTIVE":
            response["sessionChips"] = state["totalChips"]
            response["sessionBB"] = state["totalChips"] / 100
        if state["current"] is None:
            return response
        row = state["current"]
        hand = _hand(row)
        view = hand.observe(0)
        legal = view.legal_actions
        own_turn = not hand.finished and hand.actor == 0 and (benchmark is None or benchmark["status"] == "ACTIVE")
        menu = choices(view, raise_cap=None, free_fold=False) if own_turn and state["playMode"] == "restricted" else ()
        if benchmark is None or benchmark["status"] == "ACTIVE":
            response["phase"] = "finished" if hand.finished else "playing"
        response["hand"] = {
            "id": row["handId"], "number": state["handsPlayed"] if not hand.finished else state["handsPlayed"] - 1,
            "street": view.street.value, "board": list(view.board), "humanCards": list(view.hole_cards),
            "button": view.button, "smallBlind": view.small_blind, "bigBlind": view.big_blind,
            "pot": sum(pot.amount for pot in hand.events[-1].pots) if hand.finished else view.pot,
            "actor": hand.actor,
            "players": [{"seat": p.seat, "name": p.player_id, "stack": p.stack,
                         "streetBet": p.street_bet, "contributed": p.contributed,
                         "folded": p.folded, "allIn": p.all_in,
                         "shownCards": list(p.shown_cards)} for p in view.players],
            "legal": {"kinds": [kind.value for kind in legal.kinds], "call": legal.call_amount,
                      "minRaiseTo": legal.min_raise_to, "maxRaiseTo": legal.max_raise_to} if own_turn else None,
            "menu": [{"label": item.name, "kind": item.action.kind.value,
                      "raiseTo": item.action.raise_to} for item in menu],
            "presets": self._presets(view, state["playMode"] == "restricted") if own_turn else [],
            "events": _events(hand),
            "result": {"humanChips": row["humanChips"], "humanBB": row["humanChips"] / 100}
                      if hand.finished else None,
        }
        return response

    def history(self, session_id):
        with self.lock:
            state = self._load(session_id)
            result = []
            hide_results = (state.get("sessionType", "casual") == "benchmark"
                            and state["benchmark"]["status"] == "ACTIVE")
            for row in state["history"]:
                hand = _hand(row)
                view = hand.observe(0)
                item = {"handId": row["handId"], "button": row["button"],
                        "humanCards": list(view.hole_cards), "board": list(view.board),
                        "events": _events(hand),
                        "publicEventsSha256": row["publicEventsSha256"],
                        "table": recorded_table(row).record(), "model": self.model,
                        "shownBotCards": list(view.players[1].shown_cards)}
                if not hide_results:
                    item["humanChips"] = row["humanChips"]
                result.append(item)
            return {"sessionId": session_id, "hands": result}

    def _benchmark_report(self, state):
        benchmark = state["benchmark"]
        if benchmark["status"] == "ACTIVE":
            raise PlayError("Benchmark result is available after it ends", 409)
        rows = state["history"]
        if len(rows) != state["handsPlayed"]:
            raise RuntimeError("Benchmark hand count mismatch")
        by_button = {0: {"hands": 0, "netChips": 0}, 1: {"hands": 0, "netChips": 0}}
        wins = losses = ties = trained = fallback = fallback_hands = net = total_pot = 0
        references = []
        for index, row in enumerate(rows):
            hand = _hand(row)
            payoff = hand.observe(0).players[0].stack - recorded_table(row).stack
            if (not hand.finished or row["button"] != index % 2
                    or row["humanChips"] != payoff
                    or digest(public_events(hand.events)) != row["publicEventsSha256"]):
                raise RuntimeError("Benchmark replay mismatch")
            net += payoff
            total_pot += sum(pot.amount for pot in hand.events[-1].pots)
            by_button[row["button"]]["hands"] += 1
            by_button[row["button"]]["netChips"] += payoff
            wins += payoff > 0
            losses += payoff < 0
            ties += payoff == 0
            references.append({"handId": row["handId"],
                               "publicEventsSha256": row["publicEventsSha256"]})
            if benchmark["playMode"] == "free":
                misses = sum(not item["trained"] for item in row["lookup"])
                trained += len(row["lookup"]) - misses
                fallback += misses
                fallback_hands += misses > 0
        if (net != state["totalChips"] or (benchmark["status"] == "COMPLETE")
                != (len(rows) == benchmark["targetHands"])):
            raise RuntimeError("Benchmark total mismatch")
        report = {"benchmarkId": benchmark["id"], "protocolVersion": benchmark["protocolVersion"],
                  "model": {"name": benchmark["modelName"], "sha256": benchmark["modelSha256"]},
                  "game": benchmark["game"], "schema": benchmark["schema"], "table": self.table.record(),
                  "inference": inference_record(self.policy),
                  "playMode": benchmark["playMode"], "adapter": benchmark["adapter"],
                  "targetHands": benchmark["targetHands"], "completedHands": len(rows),
                  "status": benchmark["status"], "netChips": net, "netBB": net / 100,
                  "bbPer100": net / len(rows) if rows else None,
                  "averagePotChips": total_pot / len(rows) if rows else None,
                  "averagePotBB": total_pot / (100 * len(rows)) if rows else None,
                  "buttonSB": {"hands": by_button[0]["hands"], "netChips": by_button[0]["netChips"],
                               "netBB": by_button[0]["netChips"] / 100},
                  "bigBlind": {"hands": by_button[1]["hands"], "netChips": by_button[1]["netChips"],
                               "netBB": by_button[1]["netChips"] / 100},
                  "wins": wins, "losses": losses, "ties": ties,
                  "startedAt": benchmark["startedAt"], "endedAt": benchmark["endedAt"],
                  "sourceVersion": benchmark["sourceVersion"],
                  "interfaceVersion": benchmark["interfaceVersion"],
                  "buttonSchedule": benchmark["buttonSchedule"],
                  "handRecords": references}
        if benchmark["playMode"] == "free":
            report["fallbackSummary"] = {
                "trainedLookups": trained, "fallbackLookups": fallback,
                "fallbackPercent": 100 * fallback / (trained + fallback) if trained + fallback else None,
                "handsWithFallback": fallback_hands,
                "handsWithFallbackFraction": fallback_hands / len(rows) if rows else None}
        return report

    def benchmark_result(self, session_id):
        with self.lock:
            state = self._load(session_id)
            if state.get("sessionType", "casual") != "benchmark":
                raise PlayError("This is not a benchmark session", 404)
            return self._benchmark_report(state)

    def diagnostics(self, session_id, hand_id):
        with self.lock:
            state = self._load(session_id)
            if state["visibility"] != "developer":
                raise PlayError("Diagnostics unavailable for this session", 403)
            row = next((r for r in state["history"] if r["handId"] == hand_id), None)
            if row is None:
                raise PlayError("Diagnostics available after a completed hand", 404)
            return {"handId": hand_id, "adapter": _model_info(self.policy)["adapter"],
                    "lookups": list(row["lookup"]),
                    "trained": sum(r["trained"] for r in row["lookup"]),
                    "fallback": sum(not r["trained"] for r in row["lookup"])}

    def verify_replay(self, session_id):
        with self.lock:
            state = self._load(session_id)
            for row in state["history"]:
                hand = _hand(row)
                if (not hand.finished or digest(public_events(hand.events)) != row["publicEventsSha256"]
                        or hand.observe(0).players[0].stack - recorded_table(row).stack != row["humanChips"]):
                    raise ValueError("Private hand replay mismatch")
            from src.play_api.play_audit import audit_state
            audit_state(state, self.policy)
            return len(state["history"])
