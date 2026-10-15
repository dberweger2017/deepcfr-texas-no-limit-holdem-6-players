# Native three-player size measurement and bounded smoke

[PR291](https://github.com/dberweger2017/CFR-NLHE-6max/pull/291), October 15, 2026.
[Predeclared protocol](../nplayer-size-smoke-protocol.md). The owner cleared the
independent traversal/information/regret review of PR289 at `5df55c52`; both CI
shards passed before this dependent work. No merge is performed here.

## Size gate

All eight fixed configurations reach their 20M-node target without a capacity
or resource stop. Source/binary/input pins are in [admission](nplayer-size-smoke-artifacts/size-admission.json);
seed **2026101521**, exact actual nodes, street counts, capacities, payloads and
full curves are in [results](nplayer-size-smoke-artifacts/size-results.json).
Two 100k-node pilots, distinct seed 2026101520, admitted conservative per-run
quotes of 64/62 seconds against a 600-second cap. Each main job stays within
that cap; the entire size study has a 60-minute outer deadline.

A wrapper around the reviewed `State` interface counts every distinct decision
key reached by real training traversals, including opponent-only keys excluded
from average storage. It has no policy, RNG or update access. At N=2–6,
instrumented and plain training give identical tables, counters and checkpoint
bytes. Scope-0 and scope-3 main runs also produce identical reached-key and
visited-regret fingerprints at matching complete-iteration endpoints.

| Trainer / depth | Average scope | Reached keys | Retained entries | Retained requested MiB | Bytes/entry |
|---|---|---:|---:|---:|---:|
| Generic HU20 | all streets | 1,906,936 | 1,906,936 | 345.3 | 189.9 |
| Generic HU100 | all streets | 6,577,449 | 6,577,449 | 837.6 | 133.5 |
| Three-player 20 BB | all streets | 4,907,804 | 4,907,804 | 740.0 | 158.1 |
| Three-player 100 BB | all streets | 9,668,463 | 9,668,463 | 1,507.8 | 163.5 |
| Three-player 20 BB | preflop only | 4,907,804 | 2,209,039 | 314.5 | 149.3 |
| Three-player 100 BB | preflop only | 9,668,463 | 3,646,141 | 357.0 | 102.7 |
| Production HU20 K50 | all streets | — | 1,698,921 | 154.3 | 95.2 |
| Production HU100 K50 | all streets | — | 6,285,547 | 512.8 | 85.6 |

![Measured key and storage growth](nplayer-size-smoke-artifacts/size-growth.svg)

Production HU uses its unchanged compact store, key format and RNG; generic HU
uses the new key/traversal convention. These are two comparisons, not a claim
that player count alone explains every difference. The generic HashMap has
power-of-two capacity jumps, so bytes/entry varies with occupancy; HU20's
higher per-entry figure does not mean its entries intrinsically cost more.
Production-HU allocation figures also include retained scratch and worker
storage. Its checkpoint/save costs are in the results.

The meter counts requested allocations, not allocator metadata or OS residency.
Changes surviving a single-threaded generic step are attributed separately to
trainer storage and the observer. A deep-clone test independently verifies the
trainer attribution. Hash capacity plus actual regret/average arrays also
reconciles exactly. Process RSS snapshots and the on-host guard complement
these measurements. K50's fixed requested allocation is about 1.29 GiB; the
observer and checkpoint sorting are not silently charged to entry payloads.

Preflop-only averaging saves more than average arrays: postflop keys reached
only at sampled opponents no longer need a retained trainer entry. Such keys
are still counted by the survey and use the same uniform regret policy until
visited by their own traverser. At 20 BB the retained requested storage falls
**57.5%**; the regret strategy and reached-key set remain byte-identical.

## Projection and Phase 2 recommendation

[Projection inputs and calculations](nplayer-size-smoke-artifacts/projections.json)
include a local power fit, last-interval linear growth and origin-linear
scenarios. These are one-seed extrapolations, not statistical bounds or proof
of saturation. The last interval is 10M to 20M nodes. Do not use its power fit
alone to admit a much longer run.

The more conservative origin-linear scenario for three-player 20 BB with
preflop averages gives **22.1M /110.4M /552.2M retained entries** at
**200M /1B /5B nodes**. Planning peaks are **5.9 /21.4 /138.6 GiB** after adding
hash-growth overlap, 32 bytes per allocated vector for allocator overhead,
sorted checkpoint references and fixed inputs. Corresponding 100-BB preflop
scenarios are **10.1 /39.1 /165.9 GiB**. Full-street 100-BB averages project
**157.6 GiB at 1B**, already beyond this host. Actual RSS and new growth evidence
must govern admission; these scenarios do not guarantee a bound.

Recommend **three players, 20 BB, 50/100 blinds, the unchanged HU v1
minimum/pot/conditional-all-in menu, preflop-only averages, and 1B nodes** for
a first Phase 2 campaign. Use a **150M-entry capacity stop**, an **80-GiB family
RSS guard**, the existing <=8-CPU/85°C limits and checkpoints at 200M/500M/1B.
This leaves substantial room below the host's 125 GiB for allocation jumps,
recovery/export and the OS. Prefer a reported capacity stop to silently
freezing an unexamined key set. A 5B default is not justified by this study.
This recommendation authorizes no long campaign and makes no strength claim.

## Fixed smoke admission

The completed size gate admits the already requested **200M-node, three-player
20-BB** smoke: seed **2026101531**, **preflop-only averages**, a fixed
**30M-entry stop** (no freezing), pinned K50 and the same v1 menu. Run fresh to
200M, a separate 100M prefix, and resume that prefix to the identical 200M
endpoint. Compare complete compressed bytes. Node targets end at complete
iterations; no outcome-based extension is allowed.

The measured 20M traversal takes 18.98 seconds with census overhead. Scaling
its fresh+split work to 400M nodes quotes about 380 seconds of traversal;
allow approximately **20 minutes total** for growing tables, three saves,
recovery, export, loader/replay validation and hashing. The entire smoke batch
has a fixed **30-minute on-host deadline**, **32-GiB family RSS cap**, **48-GiB
free-disk floor**, <=8 assigned CPUs and an 85°C stop. Recheck host campaigns
before starting. A capacity/time/resource stop is reported, not retried or
extended. The 200M origin-linear estimate is about 22.1M entries /5.9 GiB
planning peak; the entry and host limits provide separate protection.

The play check is fixed at **128 hands**, seed **2026101532**, with three
separate policy instances, an exact repeated trace and independent pinned-engine
replay of every transition. It measures legal/reproducible play, not strength.

## Export and loader boundary

`nplayer-policy-v1` records N, full game/table identity, source checkpoint SHA256,
strategy mode, averaging scope and missing-key behavior. Exports are sorted,
atomic and deterministic. Both checkpoint and policy loads require the complete
expected SHA256. The loader validates metadata, sorted unique keys, owner/street,
action dimensions, finite normalized probabilities and row count.

`average` refuses a checkpoint without all-street averages. `scoped-average`
uses stored average probabilities through the chosen street and **current
regret-matching probabilities afterward**, named `current-after-average-scope`
in the header. Zero-mass and missing rows play uniform over the exact legal
menu. This is a minimal research continuation, not a full-street average or
search. The same export command also supports explicitly selected `current`.

Each seat owns its action RNG; identical models can share immutable weight
storage. Different matching-identity model files may be supplied per seat.
`Policy::choose` accepts only the acting seat's immutable observation and public
K50 tables. Decks, other holes, chance RNG and privileged replay records belong
to the harness, never the policy. The shared observation-to-key helper is a
mechanical extraction from the reviewed cash adapter. At both 20/100 BB and
scope 0/3, complete 100k-node checkpoints match the reviewed PR289 binary exactly.

Rust fixtures cover N=2–6, strategy scope, deterministic export and malformed
metadata/rows/probabilities. Real-K50 tests compare every exported probability
with its source accumulator, reject an unavailable full average, and check
three-instance repeatability plus independent engine parity. A streaming Python
audit separately recalculates every exported probability; a deliberately wrong
probability row proves it rejects corrupted output.

