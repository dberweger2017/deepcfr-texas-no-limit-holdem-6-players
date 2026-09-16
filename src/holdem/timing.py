"""Process measurements kept outside reproducible training state."""

import resource
import sys
from contextlib import contextmanager
from time import perf_counter


@contextmanager
def measure(record, name):
    started = perf_counter()
    try:
        yield
    finally:
        record[name] = record.get(name, 0.0) + perf_counter() - started


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024
