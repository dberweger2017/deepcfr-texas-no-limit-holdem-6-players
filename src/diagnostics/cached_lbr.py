"""Selectable shared-query cache for otherwise unchanged bounded LBR.

The native LocalBestResponse remains the reference/default. Only deterministic
saved-policy distribution queries move above the per-instance cache boundary.
Each LBR still owns its range weights, chance RNG, timer, comparison batches
and first-index argmax. A cache must be scoped to one immutable source object.
"""

from src.diagnostics.robustness import LBRConfig, LocalBestResponse
from src.game.observation import replay


class SharedProbabilityCache:
    def __init__(self, source):
        self.source = source
        self.entries = {}
        self.hits = 0
        self.misses = 0

    def probabilities(self, history, seat, pair):
        # The public event tuple includes board, bets, legal decision and
        # hand identity. Pair is the hypothetical *acting* player's own cards.
        # The source object binds all model/checkpoint and abstraction state.
        key = (history, seat, pair)
        try:
            result = self.entries[key]
        except KeyError:
            view = replay(history, seat, pair)
            result = self.source.distribution(view)
            self.entries[key] = result
            self.misses += 1
        else:
            self.hits += 1
        return result

    def telemetry(self):
        return {"entries": len(self.entries), "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / (self.hits + self.misses)
                            if self.hits + self.misses else 0.0}


class CachedLocalBestResponse(LocalBestResponse):
    """Same LBR algorithm with one explicitly shared deterministic-query cache."""

    def __init__(self, source, seed, shared_cache, config=LBRConfig()):
        if shared_cache.source is not source:
            raise ValueError("Shared LBR cache belongs to a different source")
        super().__init__(source, seed, config)
        self.shared_cache = shared_cache

    def probabilities(self, history, seat, pair):
        return self.shared_cache.probabilities(history, seat, pair)
