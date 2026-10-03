"""Disk-backed immutable average policy for bounded-memory M1 diagnostics."""

from functools import lru_cache
import gzip
import json
from math import fsum, isfinite
from pathlib import Path
import sqlite3

from src.blueprint.artifact import FrozenBlueprint
from src.blueprint.abstraction import HU20_UNCAPPED_SCHEMA
from src.blueprint.solver import HU20_UNCAPPED_GAME
from src.diagnostics.cfr_average import FORMAT, EXTRACTION, checked_header
from src.diagnostics.saved_hu20 import file_hash


def build_index(spec, inputs, path):
    path = Path(path)
    source = Path(inputs) / spec["path"]
    if file_hash(source) != spec["sha256"]:
        raise ValueError("Average export hash differs")
    if path.exists():
        raise FileExistsError("Preserve the existing immutable index")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    db = sqlite3.connect(temporary)
    try:
        db.execute("PRAGMA cache_size=-16384")
        db.execute("CREATE TABLE nodes(key TEXT PRIMARY KEY, row TEXT) WITHOUT ROWID")
        db.execute("CREATE TABLE metadata(value TEXT)")
        with gzip.open(source, "rt") as handle:
            metadata = json.loads(handle.readline())
            header = metadata["checkpoint_header"]
            checked_header(header, {"seed": spec["seed"], "iteration": header["iteration"]})
            if metadata["format"] != FORMAT or metadata["extraction"] != EXTRACTION:
                raise ValueError("Unexpected average extraction")
            count = 0
            batch = []
            for text in handle:
                key, names, p, total, visits = json.loads(text)
                if (len(key) != 32 or not names or len(names) != len(p)
                        or len(set(names)) != len(names) or type(visits) is not int
                        or visits < 0 or not isfinite(total) or total < 0
                        or not all(isfinite(v) and v >= 0 for v in p)
                        or abs(fsum(p) - 1) > 1e-8):
                    raise ValueError("Invalid immutable average row")
                batch.append((key, text.strip()))
                count += 1
                if len(batch) == 4096:
                    db.executemany("INSERT INTO nodes VALUES (?,?)", batch)
                    batch.clear()
            db.executemany("INSERT INTO nodes VALUES (?,?)", batch)
        db.execute("INSERT INTO metadata VALUES (?)", (json.dumps({
            "source": spec, "metadata": metadata, "rows": count}),))
        db.commit()
    finally:
        db.close()
    temporary.replace(path)
    return {"path": str(path), "rows": count, "source_sha256": spec["sha256"],
            "index_sha256": file_hash(path), "bytes": path.stat().st_size}


class DiskAverage(FrozenBlueprint):
    def __init__(self, path, spec):
        self.db = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
        self.db.execute("PRAGMA cache_size=-16384")
        stored = json.loads(self.db.execute("SELECT value FROM metadata").fetchone()[0])
        if stored["source"] != spec:
            raise ValueError("Index policy identity differs")
        header = stored["metadata"]["checkpoint_header"]
        self.players = 2
        self.raise_cap = None
        self.abstraction = HU20_UNCAPPED_SCHEMA
        self.game = HU20_UNCAPPED_GAME
        self.identity = header["identity"]
        self.entries = self
        self.description = {"weights_sha256": spec["sha256"], "training_seed": spec["seed"],
                            "iteration": header["iteration"], "strategy": EXTRACTION,
                            "entries": stored["rows"], "abstraction": self.abstraction}

    @lru_cache(maxsize=8192)
    def get(self, key):
        row = self.db.execute("SELECT row FROM nodes WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        _, names, p, _, _ = json.loads(row[0])
        return tuple(names), tuple(p)
