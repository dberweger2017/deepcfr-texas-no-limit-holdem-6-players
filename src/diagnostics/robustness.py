"""Fixed reactive stress policies and observation-only one-step HU local response."""
from dataclasses import dataclass
from itertools import combinations
from random import Random
from time import perf_counter

import numpy as np

from src.blueprint.abstraction import choices, RANKS
from src.blueprint.search import DECK, _sample_world
from src.game.observation import ActionTaken, BoardDealt, replay
from src.game.showdown import hand_value
from src.game.types import Action, ActionKind


class ReactiveAttack:
    """Reconstructed rule; no adaptation or privileged policy queries."""
    def __init__(self, rule="pressure", contract="menu"):
        if rule not in ("pressure", "minraise", "passive") or contract not in ("menu", "native"):
            raise ValueError("Unknown fixed attack")
        self.rule, self.contract = rule, contract

    def choose_action(self, view):
        menu = choices(view, free_fold=False)
        raises = [c.action for c in menu if c.action.kind == ActionKind.RAISE]
        if self.contract == "native" and ActionKind.RAISE in view.legal_actions.kinds:
            raises = [Action(ActionKind.RAISE, view.legal_actions.min_raise_to)]
        if raises and self.rule != "passive":
            return min(raises, key=lambda a: a.raise_to)
        if ActionKind.CHECK in view.legal_actions.kinds:
            return Action(ActionKind.CHECK)
        if self.rule != "pressure" or qualifies(view):
            return Action(ActionKind.CALL)
        return Action(ActionKind.FOLD)


def qualifies(view):
    # Explicit reconstruction: any pair, any ace, or two ranks >= ten.
    # Claude's temporary source is unavailable; this is not an exact replication.
    a, b = (RANKS.index(c[0]) for c in view.hole_cards)
    return (a == b or max(a,b) == 12 or min(a,b) >= 8) if not view.board else hand_value(view.hole_cards + view.board)[0] >= 1


@dataclass(frozen=True)
class LBRConfig:
    chance_samples: int = 2
    max_seconds: float = 5.0
    def __post_init__(self):
        if not 1 <= self.chance_samples <= 128 or not 0 < self.max_seconds <= 60:
            raise ValueError("Invalid bounded LBR work")


def posterior(weights, likelihoods):
    """Zero evidence preserves the previous compatible belief, explicitly flagged."""
    updated = np.asarray(weights, dtype=float) * np.asarray(likelihoods, dtype=float)
    if not np.isfinite(updated).all() or (updated < 0).any():
        raise ValueError("Invalid likelihood")
    total = updated.sum()
    return (updated / total, False) if total > 0 else (np.asarray(weights)/sum(weights), True)


def checkdown_payoffs(view, action, outcomes):
    """Exact HU chip ledger after assumed call/checkdown, including uncalled refund.

    Both seats started equally and have no third-party dead money. At showdown
    matched contributions make an even pot, so an odd-chip split cannot arise.
    Utilities are whole-hand net chips, not fold-relative heuristic utilities.
    """
    hero, rival = view.players[view.seat], view.players[1-view.seat]
    if action.kind == ActionKind.FOLD:
        return np.full(len(outcomes), -hero.contributed, dtype=float)
    paid = (action.raise_to-hero.street_bet if action.kind == ActionKind.RAISE
            else view.legal_actions.call_amount if action.kind == ActionKind.CALL else 0)
    rival_paid = min(rival.stack, max(0, hero.street_bet+paid-rival.street_bet))
    matched = min(hero.contributed+paid, rival.contributed+rival_paid)
    return np.asarray(outcomes) * matched


class LocalBestResponse:
    """Lisy/Bowling one-step/checkdown response; actual target play stays fixed.

    Full opponent range, sampled future chance per holding, exact river ranks.
    A complete common-runout comparison batch is committed atomically. Deadlines
    are checked between batches; a completed batch can exceed the soft time
    budget. The campaign's external deadline/RSS guard is always authoritative.
    """
    def __init__(self, source, seed, config=LBRConfig()):
        self.source, self.random, self.config = source, Random(seed), config
        self.holdings = None
        self.weights = None
        self.processed = 0
        self.telemetry = []
        self.zero_likelihood = []
        self.cache = {}

    def probabilities(self, history, seat, pair):
        key = (history, seat, pair)
        if key not in self.cache:
            view = replay(history, seat, pair)
            self.cache[key] = self.source.distribution(view)
        return self.cache[key]

    def update(self, view):
        if len(view.players) != 2 or view.players[0].starting_stack != view.players[1].starting_stack:
            raise ValueError("LBR supports equal-stack heads-up only")
        if self.holdings is None:
            self.holdings = tuple(combinations([c for c in DECK if c not in view.hole_cards],2))
            self.weights = np.ones(len(self.holdings))/len(self.holdings)
        rival = 1-view.seat
        for i in range(self.processed, len(view.history)):
            event = view.history[i]
            if isinstance(event, BoardDealt):
                valid = np.array([not set(pair).intersection(event.cards) for pair in self.holdings])
                self.weights, zero = posterior(self.weights, valid)
                if zero:
                    raise ValueError("Public cards exclude all posterior mass")
            elif isinstance(event, ActionTaken) and event.seat == rival:
                likelihood = np.zeros(len(self.holdings))
                for j,pair in enumerate(self.holdings):
                    if self.weights[j] > 0:
                        menu, p, _ = self.probabilities(view.history[:i], rival, pair)
                        likelihood[j] = sum(v for c,v in zip(menu,p) if c.action == event.action)
                self.weights, zero = posterior(self.weights, likelihood)
                if zero:
                    self.zero_likelihood.append({"history_index":i,"action":repr(event.action),"handling":"preserve previous compatible prior"})
        self.processed = len(view.history)
        # Cache is local to this hand; completed histories aren't revisited.
        self.cache.clear()

    def _fold_probabilities(self, view, menu):
        result = []
        pair = next(pair for pair,w in zip(self.holdings,self.weights) if w>0)
        world = _sample_world(view, {1-view.seat: ((pair,1.0),)}, Random(0))
        for choice in menu:
            if choice.action.kind != ActionKind.RAISE:
                result.append(np.zeros(len(self.holdings))); continue
            child = world.apply(choice.action)
            folds = np.zeros(len(self.holdings))
            # A sampled future street must never enter the immediate fold query.
            if not child.finished and child.actor == 1-view.seat and child.observe(child.actor).street == view.street:
                for j,pair in enumerate(self.holdings):
                    if self.weights[j] > 0:
                        opts,p,_ = self.probabilities(child.events, child.actor, pair)
                        folds[j] = sum(v for c,v in zip(opts,p) if c.action.kind == ActionKind.FOLD)
            result.append(folds)
        self.cache.clear()
        return result

    def choose_action(self, view):
        start = perf_counter()
        self.update(view)
        menu = choices(view, free_fold=False)
        folds = self._fold_probabilities(view,menu)
        preparation = perf_counter()-start
        total = np.zeros(len(menu)); completed = 0
        batches = 1 if len(view.board)==5 else self.config.chance_samples
        for sample in range(batches):
            # At least one entire comparison is attempted; no action-order timeout bias.
            if completed and perf_counter()-start >= self.config.max_seconds:
                break
            outcomes = np.zeros(len(self.holdings))
            for j,pair in enumerate(self.holdings):
                if self.weights[j] <= 0:
                    continue
                available = [c for c in DECK if c not in view.hole_cards+view.board+pair]
                board = view.board + tuple(self.random.sample(available,5-len(view.board)))
                a,b = hand_value(view.hole_cards+board), hand_value(pair+board)
                outcomes[j] = (a>b)-(a<b)
            for k,c in enumerate(menu):
                utilities = checkdown_payoffs(view,c.action,outcomes)
                if c.action.kind == ActionKind.RAISE:
                    utilities = folds[k]*view.players[1-view.seat].contributed + (1-folds[k])*utilities
                total[k] += self.weights @ utilities
            completed += 1
        values = total/completed
        selected = int(np.argmax(values))
        elapsed = perf_counter()-start
        self.telemetry.append({"street":view.street.value,"samples":completed,"requested_samples":batches,
            "completed":completed==batches,"over_soft_budget":elapsed>self.config.max_seconds,
            "seconds":elapsed,"preparation_seconds":preparation,"positive_range_holdings":int((self.weights>0).sum()),
            "posterior_mass":float(self.weights.sum()),"values_chips":values.tolist(),"chosen":menu[selected].name,
            "zero_likelihood_events":len(self.zero_likelihood)})
        return menu[selected].action
