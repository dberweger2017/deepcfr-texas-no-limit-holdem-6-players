"""Uniform reservoir admission; minibatch sampling has a separate random stream."""

from random import Random

import numpy as np


class Reservoir:
    def __init__(self, capacity: int, seed: int):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Reservoir capacity must be a positive integer")
        self.capacity = capacity
        self.seen = 0
        self.size = 0
        self.random = Random(seed)
        self.infos = np.empty(capacity, dtype=np.int64)
        self.iterations = np.empty(capacity, dtype=np.int64)
        self.targets = np.empty((capacity, 3), dtype=np.float32)

    def add(self, info: int, iteration: int, target: np.ndarray) -> None:
        if (
            info < 0
            or iteration < 1
            or target.shape != (3,)
            or not np.isfinite(target).all()
        ):
            raise ValueError("Invalid reservoir sample")
        self.seen += 1
        if self.size < self.capacity:
            slot = self.size
            self.size += 1
        else:
            slot = self.random.randrange(self.seen)
            if slot >= self.capacity:
                return
        self.infos[slot] = info
        self.iterations[slot] = iteration
        self.targets[slot] = target

    def means(self, info_count: int) -> tuple[np.ndarray, np.ndarray]:
        totals = np.zeros((info_count, 3))
        weights = np.zeros(info_count)
        ids = self.infos[: self.size]
        iterations = self.iterations[: self.size]
        np.add.at(totals, ids, iterations[:, None] * self.targets[: self.size])
        np.add.at(weights, ids, iterations)
        means = np.divide(
            totals, weights[:, None], out=totals, where=weights[:, None] > 0
        )
        return means, weights
