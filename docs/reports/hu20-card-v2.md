# HU20 card abstraction v2 — pre-training evidence

I refine only the postflop private-card representation over merged #139/#142.
The [prospective protocol](../hu20-card-v2-protocol.md) and
[resource plans](../../configs/diagnostics/hu20-card-v2-preflight-v2.json)
precede any strength outcomes. Draft #143 is separate from active #136.

## Model-free collision and growth gate

All retained #139 files pass their byte/hash checks. No model is loaded and
no equity is re-estimated. On the same 3,072 uniform-card holdings, all five
highlighted concrete same-board failures separate. V2 retains v1 as a prefix:
old buckets can split, but cannot merge. Suit and card-order invariance pass.

| Street | Sample holdings | v1 buckets/keys | v2 buckets/keys | Same-board different-made-value pairs separated / old pairs |
| --- | ---: | ---: | ---: | ---: |
| flop | 1024 | 43 | 508 | 2056/2246 |
| turn | 1024 | 69 | 609 | 1219/1351 |
| river | 1024 | 35 | 519 | 1654/1783 |

**4,929/5,380 (91.62%)** sampled pairs with different made-hand values separate;
**451 residual pairs remain**. These are pair counts clustered within boards,
not independent examples. The largest residual retained river collision has
uniform-equity spread .1818 (8 versus T as third kicker in the same band).
Splitting measured collisions is a representation check, not proof of strength
or a causal explanation of historical losses.

Key counts use one identical check-through public/menu template per street,
varying only private cards/current board; therefore they match descriptor
counts in this audit. This is not reached-state occupancy or a mature-table
projection. [All row descriptors/keys](hu20-card-v2-artifacts/collision-audit/descriptors.jsonl),
[full audit and residual collisions](hu20-card-v2-artifacts/collision-audit/summary.json)
retain the source hash and every distinction. Audit: **0.56 seconds / 56.7 MiB**.

## Sequential M1 resource prefixes

No playing outcomes. Both use the retained seed 2026093001 and original recipe:
K1, uncapped native menu, unchanged history, 3M safety entry limit, 250k
per-iteration node limit and 300-second iteration watchdog. Each path starts
from zero and stops at the first complete iteration crossing each node target.

| Representation | Complete nodes | Iterations | Entries | Mean visits/key | Training nodes/sec | Training seconds | Whole prefix seconds | Peak with save/export |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 2,000,143 | 5,549 | 193,295 | 2.155 | 14,945 | 133.83 | 138.60 | 291.20 MiB |
| v2 | 2,000,268 | 5,070 | 351,592 | 1.147 | 15,055 | 132.86 | 140.78 | 504.34 MiB |

| Representation | Preflop keys | Flop keys | Turn keys | River keys |
| --- | ---: | ---: | ---: | ---: |
| v1 | 9,317 | 15,152 | 59,466 | 109,360 |
| v2 | 9,312 | 57,062 | 112,420 | 172,798 |

V2 has **1.819×** as many entries at this prefix and mean visits/key falls
2.155→1.147. Actual nodes/iterations differ slightly because new postflop
strategies change later sampled paths; both use the same completed-node rule,
not the same iteration count. The coarse preflop partition is unchanged.
Equal nodes is equal traversal-work accounting, not equal convergence or
identical CPU seconds. Training coverage/fallback counters and all four
milestones are in [v1](hu20-card-v2-artifacts/preflight-v1/summary.json) and
[v2](hu20-card-v2-artifacts/preflight-v2/summary.json); full iteration logs are compressed without changing their contents. The retained
per-prefix manifests pin their original bytes; [transport hashes](hu20-card-v2-artifacts/iteration-log-transport.json)
pin both the original and compressed logs.
M1 macOS arm64, Python 3.11.15, pinned native engine; both workers exited.

## Resource admission — pending, no paid run yet

The early 1M→2M v2 entry-growth exponent is **0.944**. Extrapolating that
local curve gives **14.11M keys** at 100M; holding the last absolute growth
rate gives **16.89M**. Neither is a mature-table forecast: saturation
can reduce growth, different seeds/strategies can increase it. The retained v1
3M safety limit is therefore a material risk; I do not silently raise it or
coarsen v2. Scaling observed save/export peak by the linear entry estimate gives
about **23.66 GiB**,
before headroom.

The [exact resource proposal](../../configs/diagnostics/hu20-card-v2-resource-proposal.json)
requests a **20M safety entry limit**, **three CPU5-memory 8-vCPU/64-GB pods**,
one worker/pod, **5-hour absolute rental cutoff**, **$10 total cap**. The live
catalog compute rate is $0.52/h/pod; total admitted rate must be ≤$0.57/h/pod
including disk. Three full five-hour allocations at that ceiling cost $8.55,
leaving reserve. Extra vCPUs buy the memory shape; no single-worker speedup is
assumed. Prefix timing projects 1.85 training hours/seed, with 1.5× slowdown
and 75 minutes setup/recovery/save/retrieval reserve below the cutoff.

This changes only an inactive abort threshold, not regret math, sampling,
actions, history or the 100M stopping rule. Approval is needed because the
original recipe's safety capacity and the resource/cost allocation differ.
Stop on cap/RSS/swap/disk/time failure; preserve partials. No silent coarsening,
extra nodes, paid follow-on or model promotion. Linux parity/recovery and the
final paired panel budget remain pending until admission.

## Validation and remaining work

24 focused descriptor/existing HU20 tests pass: the five concrete collisions,
card order/suit permutation, hidden-world observation isolation, own-card/
draw features, unchanged menu/history/seeds, artifact schema rejection,
current-export guard and next-iteration checkpoint recovery. The new schema
is isolated; production defaults and #142's river player stay unchanged.
CI is tracked in draft #143. No policy-strength results have been opened.

I notified Doctor Research and appended to the existing M4 coordination note.
That was a brief network-only coordination action; no M4 compute, allocation,
campaign-file changes or extra transfers occurred. I use already cached M1
baseline artifacts, separate RunPod ownership and cost accounting. #136 stays
untouched.

## Approved execution freeze (October1, before rentals)

I approved three CPU5 8vCPU/64GB pods, one worker each, $10 total /five hours
maximum per pod and a20M entry safety ceiling. The descriptor and scientific
100M/seed budget remain frozen. The [run plan](../../configs/diagnostics/hu20-card-v2-run.json)
and [protocol](../hu20-card-v2-protocol.md#approved-rental-and-evaluation-freeze)
pin the recovery, resource, checkpoint and paired-evaluation rules. The
additional generated-fixture campaign tests exercise native replay, v1/v2
schema isolation, exact tail/paired arithmetic, outcome-blind timing and
archive verification. No paid rental or strength result has run at this
checkpoint; the final evidence will report actual admission, failures and cost.
