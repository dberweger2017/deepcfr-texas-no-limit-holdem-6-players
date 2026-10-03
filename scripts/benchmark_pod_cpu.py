"""Portable CPU comparison for hosts and rented pods, standard library only.

Prints ``BENCH`` lines: single-process zlib throughput, a pure-Python loop rate,
and aggregate zlib throughput for several process counts. zlib is C code, so it
compares CPUs across Python versions; the loop rate depends on the interpreter.
"""

import os
import platform
import random
import sys
import time
import zlib
from multiprocessing import Pool

BLOCK_MB = 8
REPEATS = 3
BLOB = None


def data(seed=1, size=BLOCK_MB * 1024 * 1024):
    rng = random.Random(seed)
    words = [bytes(rng.getrandbits(8) for _ in range(rng.randint(3, 9))) for _ in range(4096)]
    out = bytearray()
    while len(out) < size:
        out += rng.choice(words)
    return bytes(out[:size])


def init():
    # Build the input before timing so worker start-up never enters a measurement.
    global BLOB
    BLOB = data()


def zwork(_):
    if BLOB is None:
        init()
    started = time.perf_counter()
    for _ in range(REPEATS):
        zlib.compress(BLOB, 6)
    return time.perf_counter() - started


def pyloop(n=6_000_000):
    started = time.perf_counter()
    total = 0
    for i in range(n):
        total += (i * i) % 7
    return n / (time.perf_counter() - started) / 1e6


def parse_cpu_max(text):
    """CPUs granted by a cgroup v2 ``cpu.max`` line, or None when unlimited."""
    quota, period = text.split()
    return None if quota == "max" else int(quota) / int(period)


def usable_cpus():
    """Scheduler affinity and cgroup quota; containers often see every host thread."""
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    quota = None
    try:
        quota = parse_cpu_max(open("/sys/fs/cgroup/cpu.max").read())
    except (OSError, ValueError):
        try:
            q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            quota = None if q < 0 else q / p
        except (OSError, ValueError):
            pass
    return affinity, quota


def process_counts(affinity):
    return sorted({n for n in (2, 4, 8, 16, 32) if n <= affinity} | {affinity})


def main():
    affinity, quota = usable_cpus()
    print("BENCH host", platform.node(), platform.machine(), "python", sys.version.split()[0],
          "os.cpu_count", os.cpu_count(), "affinity", affinity, "cgroup_quota", quota, flush=True)
    zwork(0)
    single = min(zwork(0) for _ in range(3))
    megabytes = REPEATS * BLOCK_MB
    print(f"BENCH single_zlib_MBps {megabytes / single:.1f}", flush=True)
    print(f"BENCH single_pyloop_Mops {max(pyloop() for _ in range(2)):.2f}", flush=True)
    for n in process_counts(affinity):
        with Pool(n, initializer=init) as pool:
            pool.map(zwork, range(n))
            started = time.perf_counter()
            pool.map(zwork, range(n * 2), chunksize=2)
            elapsed = time.perf_counter() - started
        aggregate = n * 2 * megabytes / elapsed
        print(f"BENCH procs {n} aggregate_zlib_MBps {aggregate:.1f} per_proc {aggregate / n:.1f}", flush=True)
    print("BENCH done", flush=True)


if __name__ == "__main__":
    main()
