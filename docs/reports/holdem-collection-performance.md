# Hold’em collection performance

**The same 10,000-node prefix is 1.78× faster, with 56% lower median process
peak memory. The difficult traversal remains incomplete.** These improvements
reduce repeated computation without changing actions, sampling or learning
signals. They do not establish scalable training or stronger play.

The [profiling command](../holdem-collection-performance.md) and
[machine-readable results](holdem-collection-performance.json) retain every
measurement, source fingerprint, checkpoint identity and failure. All work was
local; rental spend was **$0**, leaving **$7.33** of CPU authorization.

## Workload and method

The [first baseline report](holdem-baseline.md) failed during iteration 2 of
unequal-stack seed 103. We load its verified iteration-1 checkpoint and replay
`collection-2-4-0`: physical/participant seat 4, sample 0, the original deal and
opponent-action streams, and the rotated button. Stacks remain
20/40/60/100/150/200 BB. The saved policies come from one completed iteration of the width-16 implementation check. No model is refitted and no training state is published.

- Checkpoint SHA-256: `acd341bb5cdfd9d3809e2f0948f51ee754ab56f3108f7c7a25f04836ce708cfc`.
- Current-profile SHA-256: `24f0123ee59473c4c8fd17d0d243d9dba74f1172dc038990420742b834bf2936`.
- Reference revision: `f85a3e3` (profiling command, original collector behavior).
- Optimized revision: `35e0e59` (replay and immutable-feature changes).
- Hardware: Apple M1, 16 GiB memory, CPU float32 Torch with one thread.
- Three fresh-process runs per version, alternating reference then optimized.
  Each stops after the same 10,000 visited nodes, under a 60-second ceiling.
  Full tests were not run concurrently with these measurements.
- The reference worktree was clean. Optimized reports mark the working directory
  dirty because of documentation edits and unrelated untracked workspace files;
  all three optimized source fingerprints match, and the runtime changes are
  committed at the revision above. The full environment and historical checkpoint
  manifest are retained in JSON.

Node-limited prefixes deliberately return no training batch. Failure cleanup is
included in collection timing. Setup and record hashing are separate. Peak RSS
is the whole process's high-water mark, including imports and checkpoint loading;
it is not the collector's isolated allocation count. Fresh processes start with
cold caches; cache reuse within a traversal is part of the measured improvement.

## Equal-work results

| Repeat | Original seconds | Optimized seconds | Original peak MB | Optimized peak MB |
| --- | ---: | ---: | ---: | ---: |
| 1 | 15.506 | 9.052 | 760.0 | 323.7 |
| 2 | 15.216 | 8.558 | 719.2 | 315.8 |
| 3 | 13.834 | 8.035 | 725.2 | 313.3 |
| **Median** | **15.216** | **8.558** | **725.2** | **315.8** |

The median runtime ratio is **1.778×**, and median process peak RSS falls
**56.46%**. These are local measurements of one fixed prefix, not a universal
speedup, cloud throughput estimate or an end-to-end training benchmark.

## Where the work went

Separate instrumented runs use the same first 3,000 nodes. Instrumentation adds
overhead; these numbers explain call costs and are not the speed benchmark above.
Cumulative times overlap and must not be summed as independent cost categories.

| Operation | Original calls | Optimized calls | Original cumulative seconds | Optimized cumulative seconds |
| --- | ---: | ---: | ---: | ---: |
| Public replay | 9,001 | 9,001 | 3.290 | 0.937 |
| Decision encoding | 1,522 | 1,522 | 2.155 | 0.896 |
| Neural policy distribution | 1,522 | 1,522 | 1.900 | 1.844 |
| Showdown hand-value calculation | 4,908 | 4 | 1.003 | 0.001 |

Public replay previously allocated a new immutable player after every historical
action and street reset. It now uses local scratch arrays and publishes immutable
players once. There is no shared mutable replay state.

Bounded pure caches reuse visible-card canonicalization, exact event feature rows
and showdown values. Sibling branches share long immutable event prefixes;
retaining those feature rows once also reduces memory. Complete histories, exact
source records and ownership remain intact. Cache keys include the information
that changes each calculation, including pot, blind, relative seat, betting street
and canonical revealed cards. Cache state is not model or checkpoint state.

The network itself is unchanged. Its similar call count and time show that neural
inference remains substantial after the Python overhead reduction. Batching is a
future opportunity; a GPU speedup is not measured here.

## Larger bounded probes and the remaining blocker

Before optimizing, a diagnostic with a 150,000-node / 180-second ceiling reached
**86,549 nodes** without finishing. Collection plus failure cleanup took
183.13 seconds; peak process RSS was 2,566.9 MB.

After optimizing, a separate 150,000-node / 120-second probe reached
**109,901 nodes** without finishing. Collection plus cleanup took 120.27 seconds;
peak process RSS was 872.6 MB. These limits were deliberate diagnostic extensions,
not retries of the original training campaign. The original result remains
incomplete. The two probes use different amounts of work, so their runtime and
memory ratios are not the equal-work performance claim.

The difficult root alone exceeds the original **50,000-node whole-phase budget**.
Faster hardware cannot make it fit that node limit. Raising the limit also does
not bound the tree's actual cost: the collector enumerates every traverser action
at every later traverser decision, including zero-probability branches. This
experiment gives a lower bound on that root's size, not its completed size or
expected cost over all deals.

**There is not yet a defensible budget for a substantial learning campaign.**
The work measured here excludes role fitting, replay admission, archive growth,
evaluation and other training roots. No hours-to-strength or GPU rental estimate
is inferred from a truncated traversal.

## Correctness evidence

Three complete saved-policy roots use seed 101, iteration 2, traverser 0 in the
four-, five- and six-player 100 BB scenarios. Each finishes in both versions:

| Players | Nodes | Traverser targets | Full traversal digest |
| --- | ---: | ---: | --- |
| 4 | 50 | 8 | identical |
| 5 | 47 | 6 | identical |
| 6 | 70 | 8 | identical |

Digests cover the owner's root observation, every candidate/feature/source record,
execution, policy probability and value/regret target, plus result and node counts.
These are small completed roots; they do not prove completion or full-result
identity of the difficult root. Every digest and checkpoint hash is in JSON.

Twenty seeded hand histories, including all observers across heads-up and
four-/five-/six-player unequal-stack hands, retain observation digests generated
by the pre-optimization implementation. Existing tests cover hidden-world and
suit invariance, public identifiers, long histories, side pots/disclosures,
zero-probability traversal, exact expected regrets, rollback and fresh-process
recovery. The profiling test verifies instrumented/plain equality, saved-checkpoint
integrity and failure retention. Focused validation: **78 tests pass**. The full repository suite passes **455 tests** (122.63 seconds). Ruff and `git diff --check` pass.

## Decision

Keep the optimizations, preserve all failures and stop extending these probes.
The next PR should compare a **bounded-work sampling design** against the current
exhaustive-traverser reference. Specify action inclusion probabilities and target
corrections, verify expected regret targets on tractable trees, and compare
variance, runtime and memory on retained difficult roots before adopting it.
Discarding expensive deals, pruning zero-probability traverser actions or assigning
invented values to unfinished branches would change the learning problem and is
not an acceptable shortcut.

A meaningful multi-seed Hold’em campaign follows that decision and a declared
budget. Small-game readiness remains passed; milestone 4's Hold’em learning exit
remains open. No model is promoted and no professional-strength claim is made.

## Retained evidence

Local raw reports and profiler files remain under
`results/collection-performance/`, outside Git. JSON records their paths and
SHA-256 hashes. The historical baseline checkpoint remains under
`results/holdem-baseline-v1/scenario-3-seed-103/`; it has not been uploaded to durable
storage. Reproduction uses the command above on the corresponding Git revisions,
with the recorded environment. Timing is hardware/load dependent; policy and
record comparisons use hashes rather than wall-clock equality.
