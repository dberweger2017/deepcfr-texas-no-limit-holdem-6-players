"""Read-only, windowed blueprint extraction over published training profiles."""

import gzip
import heapq
import json
import sqlite3
from collections import Counter
from hashlib import sha256
from math import fsum, isfinite
from pathlib import Path
from random import Random

from src.blueprint.abstraction import BUTTON_ZERO_COMPAT_LOOKUP, choices, information_key
from src.blueprint.solver import regret_match
from src.game.hand import Hand
from src.game.types import Street

EXTRACTION = "windowed-blueprint-extraction-v1"


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_one(root, target: int, random: Random, adapter, counters: dict):
    """One UPDATE-STRATEGY root: own actions sample; opponent actions branch.

    An abstract key aggregates all reached underlying histories. Because the
    current abstraction is lossy, these counters do not imply perfect recall.
    """
    visited = 0

    def visit(state):
        nonlocal visited
        visited += 1
        if adapter.stop(state, target):
            return
        actor = adapter.actor(state)
        menu, key, policy = adapter.decision(state)
        names = tuple(item.name for item in menu)
        if actor == target:
            index = random.choices(range(len(menu)), weights=policy, k=1)[0]
            previous = counters.get(key)
            if previous is None:
                previous = (names, [0] * len(names))
                counters[key] = previous
            elif previous[0] != names:
                raise ValueError("Collector action menu mismatch")
            previous[1][index] += 1
            visit(adapter.advance(state, menu[index]))
        else:
            for action in menu:
                visit(adapter.advance(state, action))

    visit(root)
    return visited


class NativePreflopAdapter:
    def __init__(self, trainer):
        self.trainer = trainer

    def stop(self, hand, target):
        return (hand.finished or hand.observe(target).players[target].folded
                or hand.observe(hand.actor).street != Street.PREFLOP)

    def actor(self, hand):
        return hand.actor

    def decision(self, hand):
        view = hand.observe(hand.actor)
        menu = choices(view, raise_cap=self.trainer.config.raise_cap)
        key = information_key(view, menu, schema=self.trainer.config.abstraction)
        node = self.trainer.nodes.get(key)
        names = tuple(item.name for item in menu)
        if node is not None and node.names != names:
            raise ValueError("Collector action menu mismatch")
        policy = regret_match(tuple(node.regrets)) if node else (1 / len(menu),) * len(menu)
        return menu, key, policy

    def advance(self, hand, choice):
        return hand.apply(choice.action)


def collect_preflop(trainer, profile: int, roots_per_seat: int, seed: int,
                    counters: dict, *, max_visited: int | None = None) -> dict:
    if roots_per_seat < 1:
        raise ValueError("Collector needs a positive equal root count")
    adapter = NativePreflopAdapter(trainer)
    visited = 0
    for seat in range(trainer.table.capacity):
        for sample in range(roots_per_seat):
            prefix = f"{EXTRACTION}/{seed}/{profile}/{seat}/{sample}"
            deal_seed = int.from_bytes(sha256((prefix + "/deal").encode()).digest()[:8], "big")
            action_seed = int.from_bytes(sha256((prefix + "/actions").encode()).digest()[:8], "big")
            hand = Hand.start(trainer.table, hand_id=f"extract-{profile}-{seat}-{sample}",
                              seed=deal_seed)
            visited += collect_one(hand, seat, Random(action_seed), adapter, counters)
            if max_visited is not None and visited > max_visited:
                raise RuntimeError("Collector resource preflight bound exceeded")
    return {"profile": profile, "roots_per_seat": roots_per_seat,
            "visited_states": visited, "counter_keys": len(counters),
            "action_counts": sum(sum(row[1]) for row in counters.values())}


def write_snapshot(trainer, path: Path) -> str:
    """Write one complete published profile; never mutates trainer state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as out:
        for key in sorted(trainer.nodes):
            node = trainer.nodes[key]
            out.write(json.dumps([key, node.names, regret_match(tuple(node.regrets))],
                                 separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)
    return _hash(path)


def _rows(path):
    last = ""
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            key, names, probabilities = json.loads(line)
            if key <= last:
                raise ValueError("Snapshot keys are not strictly sorted")
            last = key
            yield key, tuple(names), tuple(probabilities)


def build_index(snapshots: list[Path], counters: dict, path: Path) -> dict:
    """Merge sorted streams; absent profiles contribute a uniform strategy."""
    if len(snapshots) != 8:
        raise ValueError("The window needs eight complete snapshots")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    db = sqlite3.connect(temporary)
    try:
        db.execute("CREATE TABLE policies (key TEXT PRIMARY KEY, names TEXT NOT NULL, "
                   "current TEXT, snapshot TEXT, preflop TEXT, trained_profiles INTEGER NOT NULL)")
        db.execute("CREATE INDEX covered ON policies(trained_profiles)")
        streams = [_rows(item) for item in snapshots]
        def tagged(stream, index):
            for key, names, probs in stream:
                yield key, index, names, probs

        merged = heapq.merge(*(tagged(stream, i) for i, stream in enumerate(streams)))
        batch = []
        group_key = None
        rows = []

        def flush():
            if group_key is None:
                return
            names = rows[0][1]
            if any(row[1] != names for row in rows):
                raise ValueError("Snapshot action-menu mismatch")
            for _, _, probs in rows:
                if (len(probs) != len(names) or any(not isfinite(p) or p < 0 for p in probs)
                        or abs(fsum(probs) - 1) > 1e-8):
                    raise ValueError("Invalid snapshot probabilities")
            uniform = 1 / len(names)
            totals = [uniform * (8 - len(rows)) for _ in names]
            for _, _, probs in rows:
                for j, p in enumerate(probs):
                    totals[j] += p
            average = [p / 8 for p in totals]
            final = next((row[2] for row in rows if row[0] == 7), None)
            collected = counters.get(group_key)
            if collected and collected[0] != names:
                raise ValueError("Collector action-menu mismatch")
            count = collected[1] if collected else None
            batch.append((group_key, json.dumps(names), json.dumps(final),
                          json.dumps(average), json.dumps(count), len(rows)))
            if len(batch) >= 10_000:
                db.executemany("INSERT INTO policies VALUES (?,?,?,?,?,?)", batch)
                batch.clear()

        for key, index, names, probs in merged:
            if key != group_key:
                flush()
                group_key, rows = key, []
            if rows and rows[-1][0] == index:
                raise ValueError("Duplicate key in snapshot")
            rows.append((index, names, probs))
        flush()
        if batch:
            db.executemany("INSERT INTO policies VALUES (?,?,?,?,?,?)", batch)
        db.commit()
        db.execute("PRAGMA journal_mode=DELETE")
        stats = dict(db.execute("SELECT trained_profiles, COUNT(*) FROM policies GROUP BY trained_profiles"))
    finally:
        db.close()
    temporary.replace(path)
    return {"artifact_sha256": _hash(path), "profile_coverage": stats,
            "entries": sum(stats.values())}


class WindowedDistribution:
    """Hash-verified, read-only SQLite probability index for C/P/F/A."""
    def __init__(self, path: Path, manifest: dict, arm: str):
        if arm not in ("C", "P", "F", "A"):
            raise ValueError("Unknown extraction arm")
        if manifest.get("schema") != EXTRACTION or _hash(path) != manifest.get("artifact_sha256"):
            raise ValueError("Extraction artifact identity mismatch")
        self.db = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        self.arm = arm
        self.raise_cap = manifest["raise_cap"]
        self.abstraction = manifest["abstraction"]
        self.coverage = Counter()

    def distribution(self, view):
        menu = choices(view, raise_cap=self.raise_cap)
        key = information_key(view, menu, schema=self.abstraction,
                              lookup_mode=BUTTON_ZERO_COMPAT_LOOKUP)
        row = self.db.execute("SELECT names,current,snapshot,preflop,trained_profiles "
                              "FROM policies WHERE key=?", (key,)).fetchone()
        names = tuple(item.name for item in menu)
        if row is not None and tuple(json.loads(row[0])) != names:
            raise ValueError("Extracted action menu differs from observation")
        current = json.loads(row[1]) if row and row[1] != "null" else None
        if view.street == Street.PREFLOP and self.arm in ("P", "A"):
            counts = json.loads(row[3]) if row and row[3] != "null" else None
            if counts and sum(counts):
                probs = tuple(value / sum(counts) for value in counts)
                self.coverage["preflop_collected"] += 1
            else:
                probs = tuple(current) if current else (1 / len(menu),) * len(menu)
                self.coverage["preflop_final_fallback"] += 1
        elif view.street != Street.PREFLOP and self.arm in ("F", "A"):
            probs = tuple(json.loads(row[2])) if row else (1 / len(menu),) * len(menu)
            self.coverage[f"snapshot_profiles_{row[4] if row else 0}"] += 1
        else:
            probs = tuple(current) if current else (1 / len(menu),) * len(menu)
        trained = current is not None
        self.coverage["trained" if trained else "untrained"] += 1
        return menu, probs, trained

    def close(self):
        self.db.close()
