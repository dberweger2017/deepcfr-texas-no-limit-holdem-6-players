"""Stream deterministic hands from the pinned pokers engine to the native rules checker.

Successful traces are hashed, not accumulated in memory or stored as huge fixtures.
The first failing complete hand is retained for replay; any mismatch fails the gate.
This is a rules test, with privileged deal access, never an inference interface.
"""

import argparse
from collections import Counter
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path
from random import Random
import subprocess
import time

import pokers as p

ENGINE_COMMIT = "5db20e3d5d6862b32a7402035c1340b622d3b005"
BINARY = Path("native/hu20-trainer/target/release/examples/nplayer_parity")
ACTIONS = {"fold": p.ActionEnum.Fold, "check": p.ActionEnum.Check,
           "call": p.ActionEnum.Call, "raise": p.ActionEnum.Raise}
STREETS = [(p.Stage.Preflop, "preflop"), (p.Stage.Flop, "flop"), (p.Stage.Turn, "turn"), (p.Stage.River, "river")]
CARDS = {int(c.rank) * 4 + int(c.suit): c for c in p.Card.collect()}


def engine_identity():
    identity = json.loads(distribution("pokers").read_text("direct_url.json") or "{}")
    if identity.get("vcs_info", {}).get("commit_id") != ENGINE_COMMIT:
        raise RuntimeError("pokers installation is not verifiably pinned to " + ENGINE_COMMIT)
    return identity


def snapshot(state):
    players = state.players_state
    finished = state.final_state
    legal = None
    if not finished:
        actor = players[state.current_player]
        maximum = int(actor.bet_chips + actor.stake)
        bounds = [min(int(state.min_bet + state.min_raise), maximum), maximum]
        legal = {"check": p.ActionEnum.Check in state.legal_actions,
                 "call": int(min(max(0, state.min_bet - actor.bet_chips), actor.stake)),
                 "raise_to": bounds if p.ActionEnum.Raise in state.legal_actions else None}
        expected = [p.ActionEnum.Fold, p.ActionEnum.Check if legal["check"] else p.ActionEnum.Call]
        if legal["raise_to"] is not None:
            expected.append(p.ActionEnum.Raise)
        if state.legal_actions != expected:
            raise RuntimeError("unexpected engine legal action set")
    return {"actor": None if finished else state.current_player, "finished": finished,
            "board": [int(c.rank) * 4 + int(c.suit) for c in state.public_cards],
            "street": None if finished else next(name for stage, name in STREETS if stage == state.stage),
            "pot": int(state.pot), "wager": None if finished else int(state.min_bet),
            "min_raise": None if finished else int(state.min_raise),
            "players": [[int(x.stake), int(x.bet_chips), int(x.bet_chips + x.pot_chips), not x.active] for x in players],
            "holes": [[int(c.rank) * 4 + int(c.suit) for c in x.hand] for x in players],
            "legal": legal, "rewards": [int(x.reward) for x in players] if finished else None}


def pot_structure(contributions, folded):
    """Public contribution accounting, also used by the maintained Python hand interface."""
    from src.game.types import Player, pots_for
    players = tuple(Player(i, str(i), amount, 0, contributed=amount, folded=folded[i])
                    for i, amount in enumerate(contributions))
    refunds = [0] * len(players)
    pots = []
    for pot in pots_for(players):
        if pot.refund_to is not None:
            refunds[pot.refund_to] += pot.amount
        else:
            pots.append([pot.amount, sum(1 << i for i in pot.eligible_seats)])
    return {"contributions": contributions, "refunds": refunds, "pots": pots}


def record(seed, players=3, *, preset=None):
    rng = Random(seed)
    mode = seed % 8
    deck = list(range(52))
    rng.shuffle(deck)
    button = rng.randrange(players)
    sb, bb = rng.choice([(1, 2), (3, 7), (50, 100)])
    if mode in (0, 1):
        stacks = [rng.randint(1, 200 * bb) for _ in range(players)]
    elif mode in (2, 3):
        stacks = [rng.choice([1, sb, bb, bb + 1, 2 * bb - 1, 2 * bb + 1]) for _ in range(players)]
    elif mode == 4:
        stacks = [20 * bb] * players
    elif mode == 5:
        stacks = [100 * bb] * players
    elif mode == 6:
        stacks = [rng.randint(2 * bb, 20 * bb) for _ in range(players)]
    else:
        stacks = [rng.randint(1, 10**12) for _ in range(players)]
    forced = mode == 0
    scripted = []
    if preset is not None:
        deck, stacks, button, sb, bb = (preset[k] for k in ("deck", "stacks", "button", "sb", "bb"))
        players = len(stacks)
        scripted = preset.get("actions", [])
        forced = preset.get("force_all_in", True)
    state = p.State.from_deck(players, button, sb, bb, 1, [CARDS[c] for c in deck],
                              chip_unit=1, stakes=stacks)
    states, actions = [snapshot(state)], []
    history = [[name, "board"] for size, name in [(3, "flop"), (4, "turn"), (5, "river")]
               if len(states[0]["board"]) >= size]
    # Both blinds can be all-in at construction, which already clears engine pots.
    sb_seat = button if players == 2 else (button + 1) % players
    bb_seat = (sb_seat + 1) % players
    contributed = [0] * players
    contributed[sb_seat] = min(sb, stacks[sb_seat])
    contributed[bb_seat] = min(bb, stacks[bb_seat])
    while not state.final_state:
        if len(actions) >= 10000:
            raise RuntimeError(f"hand {seed} exceeded the termination bound")
        view = states[-1]
        legal = view["legal"]
        quiet = "check" if legal["check"] else "call"
        kind, target = quiet, 0
        bounds = legal["raise_to"]
        if len(actions) < len(scripted):
            kind, target = scripted[len(actions)]
        elif forced:
            if bounds:
                kind, target = "raise", bounds[1]
        else:
            # Mode 1 reaches late streets; other modes deliberately sample free folds,
            # arbitrary off-menu sizes, minimum raises, short raises and exact all-ins.
            passive = 0.88 if mode == 1 else 0.45
            if rng.random() >= passive:
                kind = rng.choice(["fold", quiet] + (["raise"] if bounds else []))
                if kind == "raise":
                    lo, hi = bounds
                    pot_target = min(hi, max(lo, view["wager"] + view["pot"] + legal["call"]))
                    target = rng.choice([lo, hi, pot_target, rng.randint(lo, hi)])
        actor = state.current_player
        paid = target - int(state.players_state[actor].bet_chips) if kind == "raise" else legal["call"] if kind == "call" else 0
        contributed[actor] += paid
        amount = target - int(state.min_bet) if kind == "raise" else 0
        state = state.apply_action(p.Action(ACTIONS[kind], amount))
        if state.status != p.StateStatus.Ok:
            raise RuntimeError(f"oracle generator chose invalid action at {seed}: {actions}, {[kind, target]}")
        actions.append([kind, target])
        states.append(snapshot(state))
        history.append([view["street"], actor, kind, target, paid])
        for size, name in [(3, "flop"), (4, "turn"), (5, "river")]:
            if len(view["board"]) < size <= len(states[-1]["board"]):
                history.append([name, "board"])
    if len(actions) < len(scripted):
        raise RuntimeError("script extends beyond settlement")
    return {"seed": seed, "mode": mode, "forced_all_in": forced, "stacks": stacks, "button": button,
            "sb": sb, "bb": bb, "deck": deck, "actions": actions, "states": states, "history": history,
            "settlement": pot_structure(contributed, [not x.active for x in state.players_state])}


def run(binary, hands, seed, players, out):
    out.mkdir(parents=True, exist_ok=False)
    identity = engine_identity()
    started = time.monotonic()
    digest = hashlib.sha256()
    counters = Counter()
    receipt = {"engine": identity, "players": players, "seed": seed, "requested_hands": hands,
               "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    (out / "admission.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with (out / "native.stderr").open("w") as errors, (out / "progress.jsonl").open("w", buffering=1) as progress:
        proc = subprocess.Popen([str(binary.resolve())], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=errors, text=True, bufsize=1)
        current = None
        try:
            for index in range(hands):
                current = record(seed + index, players)
                line = json.dumps(current, separators=(",", ":")) + "\n"
                proc.stdin.write(line)
                proc.stdin.flush()
                reply = proc.stdout.readline()
                result = json.loads(reply) if reply else {"ok": False, "error": "checker exited without response"}
                if not result.get("ok"):
                    raise RuntimeError(result.get("error", "invalid checker response"))
                if result["decisions"] != len(current["actions"]):
                    raise RuntimeError("checker did not check the entire action stream")
                digest.update(line.encode())
                counters["hands"] += 1
                counters["decisions"] += len(current["actions"])
                counters["forced_all_in_hands"] += current["forced_all_in"]
                counters["unequal_stack_hands"] += len(set(current["stacks"])) > 1
                counters["side_pot_hands"] += len(current["settlement"]["pots"]) > 1
                counters["refund_hands"] += any(current["settlement"]["refunds"])
                final = current["states"][-1]
                counters["multiway_showdowns"] += len(final["board"]) == 5 and sum(not x[3] for x in final["players"]) >= 3
                for state in current["states"][:-1]:
                    counters[state["street"] + "_decisions"] += 1
                if (index + 1) % 1000 == 0:
                    progress.write(json.dumps({**counters, "elapsed_seconds": time.monotonic() - started}) + "\n")
            proc.stdin.close()
            if proc.wait(timeout=10) != 0:
                raise RuntimeError("checker returned nonzero")
            receipt.update(status="passed", mismatches=0, **counters)
        except BaseException as error:
            if current is not None:
                (out / "failure-hand.json").write_text(json.dumps(current) + "\n")
            receipt.update(status="failed", error=repr(error), **counters)
            raise
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            receipt.update(elapsed_seconds=time.monotonic() - started, checked_trace_sha256=digest.hexdigest())
            (out / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=BINARY)
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=202610150000)
    parser.add_argument("--players", type=int, choices=range(2, 7), default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.hands < 1:
        parser.error("hands must be positive")
    print(json.dumps(run(args.binary, args.hands, args.seed, args.players, args.out), indent=2))


if __name__ == "__main__":
    main()
