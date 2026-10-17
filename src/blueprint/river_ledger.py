"""Persist full public policies and availability decisions for exact hand replay."""

from dataclasses import asdict
import json
from pathlib import Path
import threading

import numpy as np
import psutil

from src.arena.schedule import digest
from src.blueprint.abstraction import Choice
from src.blueprint.hu20_turn_solver import PolicyMatrix
from src.game.types import Action, ActionKind


class DecisionLedger:
    def __init__(self, directory, *, replay=False):
        self.directory = Path(directory)
        self.replay = replay
        if not replay:
            self.directory.mkdir(parents=True, exist_ok=True)
        elif not self.directory.is_dir():
            raise ValueError("Decision replay ledger is missing")

    def read(self, key):
        row = json.loads((self.directory / (key + ".json")).read_text())
        checksum = row.pop("sha256")
        if digest(row) != checksum or row["key"] != key:
            raise ValueError("Decision ledger identity or checksum differs")
        return row

    def write(self, key, row):
        payload = json.loads(json.dumps(dict(key=key, **row), allow_nan=False))
        target = self.directory / (key + ".json")
        if target.exists():
            previous = self.read(key)
            if any(previous[k] != payload[k] for k in payload if k != "telemetry"):
                raise ValueError("Repeated public decision changed its pinned policy")
            return
        with target.open("x") as stream:
            json.dump(dict(payload, sha256=digest(payload)), stream, allow_nan=False, sort_keys=True)
            stream.write("\n")


def encode_matrix(matrix):
    return dict(menu=[asdict(c) for c in matrix.menu], holdings=matrix.holdings,
                probabilities=matrix.probabilities.tolist())


def decode_matrix(value):
    menu = tuple(Choice(c["name"], Action(ActionKind(c["action"]["kind"]),
                                         c["action"]["raise_to"])) for c in value["menu"])
    holdings = tuple(tuple(h) for h in value["holdings"])
    p = np.asarray(value["probabilities"], dtype=np.float64)
    if (p.shape != (len(holdings), len(menu)) or not np.isfinite(p).all()
            or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=2e-5, rtol=0)):
        raise ValueError("Invalid decision ledger matrix")
    p.setflags(write=False)
    return PolicyMatrix(menu, holdings, p)


class FamilyPeak:
    """Sample simultaneous worker plus descendants; never sum lifetime maxima."""

    sample_seconds = .02

    def __enter__(self):
        self.peak = 0
        self.minimum_available_bytes = psutil.virtual_memory().available
        self.failure = None
        self.stop = threading.Event()
        self.sample()

        def watch():
            while not self.stop.wait(self.sample_seconds):
                self.sample()

        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        try:
            root = psutil.Process()
            total = 0
            for process in [root, *root.children(recursive=True)]:
                try:
                    total += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    pass
            self.peak = max(self.peak, total)
            self.minimum_available_bytes = min(self.minimum_available_bytes,
                                               psutil.virtual_memory().available)
        except psutil.AccessDenied as exc:
            self.failure = repr(exc)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=2)
        self.sample()
        if self.failure is not None:
            raise RuntimeError("Cannot observe owned process-family memory: " + self.failure)
