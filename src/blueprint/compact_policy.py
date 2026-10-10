"""Compact read-only storage for large exported HU20 policies.

A dict of Python strings and tuples costs about 800 bytes per key, so a 10B-node average
(7.2M keys) needed over 6 GiB to load. Here keys are 16-byte digests in one sorted array,
each key's action names are an index into the few distinct menus, and probabilities are
the exact float64 values in one flat array: about 50 bytes per key. Lookups return the
same `(names, probabilities)` tuples, with bit-identical floats, as the dict did.
"""

from array import array
from collections.abc import Mapping, Set

import numpy as np


class _Index:
    """Sorted 16-byte keys; hex strings in, positions out."""

    def __init__(self, keys: np.ndarray):
        self.keys = keys                                   # dtype S16, sorted, unique
        self.raw = keys.view(np.uint8).reshape(-1, 16)    # exact bytes (S16 elements drop trailing NULs)

    def find(self, key):
        if not isinstance(key, str) or len(key) != 32:
            return None
        try:
            digest = bytes.fromhex(key)
        except ValueError:
            return None
        i = int(np.searchsorted(self.keys, digest))
        return i if i < len(self.keys) and self.raw[i].tobytes() == digest else None

    def hex(self, i):
        return self.raw[i].tobytes().hex()

    def __len__(self):
        return len(self.keys)


class CompactEntries(Mapping):
    """key -> (names, probabilities), as `FrozenBlueprint.entries` holds them."""

    def __init__(self, index, menus, menu, offsets, probabilities):
        self.index, self.menus, self.menu, self.offsets, self.probabilities = index, menus, menu, offsets, probabilities

    def _row(self, i):
        names = self.menus[self.menu[i]]
        start = int(self.offsets[i])
        return names, tuple(self.probabilities[start:start + len(names)].tolist())

    def __getitem__(self, key):
        i = self.index.find(key)
        if i is None:
            raise KeyError(key)
        return self._row(i)

    def get(self, key, default=None):
        i = self.index.find(key)
        return default if i is None else self._row(i)

    def __contains__(self, key):
        return self.index.find(key) is not None

    def __iter__(self):
        return (self.index.hex(i) for i in range(len(self.index)))

    def __len__(self):
        return len(self.index)


class CompactFlags(Set):
    """The keys whose flag is set, e.g. zero-mass averages."""

    def __init__(self, index, flags):
        self.index, self.flags = index, flags
        self.count = int(flags.sum())

    def __contains__(self, key):
        i = self.index.find(key)
        return i is not None and bool(self.flags[i])

    def __iter__(self):
        return (self.index.hex(int(i)) for i in np.flatnonzero(self.flags))

    def __len__(self):
        return self.count


class CompactCounts(Mapping):
    """key -> integer count, e.g. training visits."""

    def __init__(self, index, counts):
        self.index, self.counts = index, counts

    def __getitem__(self, key):
        i = self.index.find(key)
        if i is None:
            raise KeyError(key)
        return int(self.counts[i])

    def get(self, key, default=None):
        i = self.index.find(key)
        return default if i is None else int(self.counts[i])

    def __contains__(self, key):
        return self.index.find(key) is not None

    def __iter__(self):
        return (self.index.hex(i) for i in range(len(self.index)))

    def __len__(self):
        return len(self.index)


class CompactBuilder:
    """Collects validated rows in compact buffers, then sorts once."""

    def __init__(self):
        self.digests = bytearray()
        self.menu_ids = {}
        self.menus = []
        self.menu = array("H")
        self.offsets = array("Q")
        self.probabilities = array("d")
        self.counts = array("Q")
        self.flags = bytearray()

    def add(self, key, names, probabilities, count, flag):
        names = tuple(names)
        menu = self.menu_ids.get(names)
        if menu is None:
            menu = self.menu_ids[names] = len(self.menus)
            self.menus.append(names)
        self.digests += bytes.fromhex(key)
        self.menu.append(menu)
        self.offsets.append(len(self.probabilities))
        self.probabilities.extend(probabilities)
        self.counts.append(count)
        self.flags.append(1 if flag else 0)

    def build_sorted(self):
        """Use native sorted exports without a second full table or sort array.

        Verify strict ordering in bounded chunks before exposing views. NumPy
        retains each buffer, so deleting the builder leaves immutable inference
        ownership with the compact containers, exactly like build().
        """
        keys = np.frombuffer(self.digests, dtype="S16")
        for start in range(1, len(keys), 65536):
            end = min(len(keys), start + 65536)
            if bool(np.any(keys[start:end] <= keys[start-1:end-1])):
                raise ValueError("Native policy keys are not strictly sorted")
        index = _Index(keys)
        return (CompactEntries(index, tuple(self.menus),
                    np.frombuffer(self.menu, dtype=np.uint16),
                    np.frombuffer(self.offsets, dtype=np.uint64),
                    np.frombuffer(self.probabilities, dtype=np.float64)),
                CompactFlags(index, np.frombuffer(self.flags, dtype=np.uint8).view(np.bool_)),
                CompactCounts(index, np.frombuffer(self.counts, dtype=np.uint64)))

    def build(self):
        """(entries, flagged keys, counts); raises on a duplicate key."""
        keys = np.frombuffer(bytes(self.digests), dtype="S16")
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        raw = keys.view(np.uint8).reshape(-1, 16)
        if len(keys) > 1 and bool(np.any(np.all(raw[1:] == raw[:-1], axis=1))):
            raise ValueError("Duplicate policy key")
        index = _Index(keys)
        menu = np.frombuffer(self.menu, dtype=np.uint16)[order]
        offsets = np.frombuffer(self.offsets, dtype=np.uint64)[order]
        counts = np.frombuffer(self.counts, dtype=np.uint64)[order]
        flags = np.frombuffer(bytes(self.flags), dtype=np.uint8)[order].astype(bool)
        probabilities = np.frombuffer(self.probabilities, dtype=np.float64)
        return (CompactEntries(index, tuple(self.menus), menu, offsets, probabilities),
                CompactFlags(index, flags), CompactCounts(index, counts))
