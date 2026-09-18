# M4 fitting thread check

Increasing PyTorch's thread count does not establish a fitting speedup for the
current width-32 learner. The owner asked why the ten-core M4 was mostly idle.
The active trainer deliberately uses one Torch thread and performs both
collection and the six role fits sequentially. The M4 reports four performance
cores and six efficiency cores.

A bounded diagnostic loaded role 0's full 4,096-record replay from the active
seed's completed iteration-64 checkpoint. It refit the same initialized network
for 256 steps, batch 32, learning rate 0.001 and clipping 1.0, once each at
1/2/4/8 threads and again in reverse order. All work ran in a separate process;
the active campaign and its saved checkpoint were unchanged. The supervisor
completed in 70.19 seconds, including checkpoint loading, below its 300-second
limit; peak sampled RSS was 4.31 GB. Raw measurements and the pinned input hash
are in [the JSON record](holdem-m4-threads.json).

| Torch threads | First pass, seconds | Reverse pass, seconds |
| --- | ---: | ---: |
| 1 | 2.040 | 1.525 |
| 2 | 1.535 | 1.545 |
| 4 | 1.568 | 1.588 |
| 8 | 1.887 | 1.779 |

The first one-thread fit has visible startup/warmup overhead. Do not interpret
the two-observation medians as a reliable two-thread improvement. Eight threads
consume approximately 13.6–14.2 CPU-seconds per fit versus 1.7–1.9 for one thread,
without a wall-time benefit. Reported loss and gradient metrics agree across all
eight fits; this check does not compare every model byte or establish invariance
for other replay samples. The live single-thread trainer was also running, so
these are indicative small-model measurements, not an isolated machine benchmark.

The active run recently spent approximately 8.6 seconds per iteration collecting,
9.2 fitting and 0.5 in replay handling. More Torch threads do not parallelize its
Python traversal loop or its sequential role loop. A useful implementation path
is process-based parallel collection and independent role fits, with explicit
seed streams, stable result ordering and atomic whole-iteration publication.
It needs measured IPC/memory costs and equality/recovery checks; changing a thread
count does not provide that implementation. Keep the active campaign intact.
