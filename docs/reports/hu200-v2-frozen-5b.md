# HU200 frozen 5B overbet-menu comparison

[PR270](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/270)
· [Predeclared protocol](../hu200-v2-frozen-5b-protocol.md)

**5B training and every-row audits complete; paired evaluation in progress.** This experiment asks
whether representing off-menu bet sizes and training the retained keys more
densely improves HU200 against those sizes. Only the menu changes from #250/#265.
The primary evidence is the paired seven-opponent panel. Different-menu direct
play is descriptive because v1 can encounter histories it never trained.

## Recipe copied from #250/#265

| Value | Fixed recipe |
|---|---|
| Game | Heads-up, 200 BB, 20,000 chips per seat |
| Blinds /rake /ante | 50/100 chips /0 /0 |
| Training seed | 2026101302 |
| Roots per seat | 1 |
| CFR weighting | Linear, no floor/DCFR/training options |
| Lifetime averaging | Opponent-sampled |
| Cards | #163 pinned K50 flop/turn/river |
| Memory budget | 11,000,000,000 bytes |
| Treatment menu | v2: min, half, pot, 1.5-pot, 2-pot, conditional jam; schema-owned v2 history |
| Freeze | Cap-reaching complete iteration; update retained keys, uniform missing decisions |
| Node target | 5B cumulative, including pre-freeze growth |
| Export fallback /evaluation translation | Uniform zero mass /off |

The [first report](hu200-k50-first.md) and its
[exact input artifact](hu200-k50-first-artifacts/inputs.json) pin these tables:

| Table | Bytes | SHA256 |
|---|---:|---|
| Flop | 12,867,944 | `084e243ded9fe93ad03cd59601f22f16c3210592f51e1e3646a60983b572c111` |
| Turn | 139,600,524 | `c0bc6a8aa0535553118109d18a32d3b4dc6880e937c263cdc87472b1ae9f168f` |
| River | 1,231,562,564 | `70c1cb5ecf4cd292e4c89dc2200f7c8381100be3b0288ceeb0a12a3e83977629` |

#250's historical v1 cap was
`floor((11,000,000,000−1,400,000,000−500,000,000)/(96.9×1.1)) = 85,373,862`.
Its terminal overshot by 45 entries: 752,593,175 nodes, 493,496 iterations,
85,373,907 keys, 1.7446 visits/key, 44.29% zero visits.
[#265](hu200-k50-frozen.md) reused that exact table and recipe, reaching 5B
at 5,000,000,031 nodes /3,748,464 iterations, 9.7324 visits/key and 26.41%
zero visits. Its average is 2,756,501,598 bytes, SHA256
`43aa3b3cd3c7365c5be1b177dcc8a7c04e01c8470403c38327aa845922726512`.
That frozen 5B average is the fixed comparison, rather than the 20B final.

## Build, exact recovery and measured cap

Native source is current-main base `e2bcdc63`; research orchestration is
`bbfa6a1df1fc70f323b1335ccf2146804d0dcc30`. No native/game/regret changes.
The path-independent M4 build is 1,492,512 bytes, SHA256
`4f6abcbb954c3c744448c2a3542ab3164e0e1a015cb05ecebeda077525991cd4`,
matching #265. [Build receipt](hu200-v2-frozen-5b-artifacts/build-receipt.json).

Fresh recoverable 20M and 10M→20M are byte-identical:
**20,001,042 nodes /2,299 iterations /10,427,061 entries**, 343,463,165 bytes,
SHA256 `d0ab3813e65001cc331058cb6bdeccaf878c619fe2ffae4073ff1a468dab5b52`.
The fresh native segment took 34.75 seconds including its 29.30-second save;
resume took 32.83 including 30.00 seconds saving. Table load precedes that timer.
[Recovery and all memory measurements](hu200-v2-frozen-5b-artifacts/gates-complete.json).

Native process RSS was sampled every 50 ms for the gate, independently of the
five-second campaign guard. Fresh train peak: **2,307,260,416 bytes**; save peak:
**2,311,684,096 bytes**. As in #260, reserve the fixed table bytes separately
and retain all other overhead in the per-entry measurement:

```
B/entry = (max(2,307,260,416, 2,311,684,096) − 1,384,000,000) / 10,427,061
        = 88.96889507024079
cap = floor((11,000,000,000 − 1,384,000,000 − 250,000,000) / (88.96889507024079 × 1.1))
    = floor(9,366,000,000 / 97.86578457726488)
    = 95,702,497 entries
```

This cap is fixed before main training. RSS is a sampled planning measurement,
not a promise that all future frozen-table regret arrays fit the same footprint.

## Predeclared reading rule

Recommend Slumbot confirmation only if the paired 1.5× pot contrast's nominal
Student-t 95% lower bound exceeds zero and no panel contrast upper bound is below
−20 BB/100. Half-pot is the other key contrast. The direct match cannot decide
this rule. Cost-only pilots precede a posted canonical count plan; every final
hand independently replays and exactly reproduces before outcome readout.

The Slumbot shared loader previously forced v1 and its admission list refused
v2. This PR selects an admitted HU200 schema from the average header, then
verifies the complete hash/header/table identity through AveragePolicy. Both
play and analysis use that loader; HU100 and mismatched-card models remain
refused. A real synthetic K50 v2 model passes load, warm read-only cache,
Slumbot admission and recorded-hand analysis; corrupt hashes are rejected.
These are offline checks, with no account, credential or service calls.

## Frozen training and stored-key density

The fresh cap parent is **223,536,509 nodes /27,759 iterations /95,707,874 entries**,
5,377 entries above the measured nominal cap at a complete iteration. The exact
parent checkpoint is 3,274,048,973 bytes, SHA256
`bc72d3e0cc34c8903d784702fcfe96c5e5cba796e6092f3caf871a3e4bcb3284`.
Resume freezes at those actual entries, with no intervening update. This gives
the same training updates as freezing immediately after that iteration in a
fresh run. [Exact frozen command](hu200-v2-frozen-5b-artifacts/frozen-command.json).

The frozen run reaches 5B, preserving the entry count at both saves.
[Complete training receipts](hu200-v2-frozen-5b-artifacts/training-complete.json),
[density and freeze telemetry](hu200-v2-frozen-5b-artifacts/milestones.json).

| Save | Actual nodes | Iterations | Entries | Visits/key | Zero visits | Zero average mass | After-save RSS GiB | Write s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Freeze parent | 223,536,509 | 27,759 | 95,707,874 | 0.4401 | 62.16% | 34.03% | 1.846 | 315.59 |
| Frozen 2B | 2,000,002,806 | 396,493 | 95,707,874 | 1.3774 | 50.01% | 28.02% | 2.336 | 316.30 |
| Frozen 5B | 5,000,000,090 | 1,120,316 | 95,707,874 | 2.8011 | 42.55% | 25.11% | 2.516 | 340.05 |

Frozen continuation took **2,506.53 seconds including both writes**. The first
frozen segment ran at 2.60M nodes/s before its 316.30-second write. Final
checkpoint: **4,435,470,459 bytes**, SHA256
`39355b6c8f2653b11a8db44c99b540e1ada60fde2b3cbd5d96c7cb706f2ace75`.

At 5B, cumulative discarded traverser visits are **662,807,927** and discarded
merged updates **1,810,112,718**. The 2B→5B interval contributes 418,098,708
discarded visits and 1,133,688,692 discarded updates. Missing decisions stay
uniform; these counters measure omitted updates, not newly stored keys.

| Interval | Role | Preflop misses | Flop | Turn | River |
|---|---|---:|---:|---:|---:|
| Fresh growth to cap | Traverser | 38.436% | 48.015% | 76.806% | 92.814% |
| Fresh growth to cap | Opponent | 23.171% | 40.076% | 70.751% | 90.304% |
| Cap→2B | Traverser | 24.485% | 34.194% | 64.660% | 87.606% |
| Cap→2B | Opponent | 15.853% | 27.436% | 57.478% | 84.424% |
| 2B→5B | Traverser | 25.574% | 36.010% | 65.483% | 87.693% |
| 2B→5B | Opponent | 13.698% | 23.921% | 52.794% | 83.134% |

Pre-freeze misses are recorded before new keys receive their complete-iteration
updates. The fresh-growth interval spans the entire growing run; it is not the
short pre-cap interval reported for #265. After 2B, traverser miss shares rise
slightly while opponent shares fall. Training still reaches unsupported late
histories frequently.

**5B does not mean uniformly dense v2 support.** The v2 table has 2.801 visits/key
and 42.55% zero visits, versus v1 5B's 9.732 and 26.41%. Both node and 11 GB
budgets are matched; visits, iterations and key sets are not. V2 freezes much
earlier (223.5M versus 752.6M nodes) and has more actions and finer history
labels. The panel answers this fixed-budget menu comparison; a negative result
cannot establish that a more densely trained v2 could never help.

## Every-row exports

`export --zero-mass uniform` produced both policies. Existing
`src.diagnostics.cfr_average.audit` verifies **all 95,707,874 rows**, including
current regrets and stored average totals. It does not reconstruct historical
increments. Positive-average-mass rows: **71,680,253**; uniform zero-mass rows:
**24,027,621**. Unweighted mean current/average total variation is 0.2376390;
this row-weighted policy difference is not a strength measure.
[Full audit receipt](hu200-v2-frozen-5b-artifacts/5b.audit.json).

| 5B export | Bytes | SHA256 |
|---|---:|---|
| Average | 2,876,030,727 | `ba4811a4e2bdb82fca382758a3713bca6701bfe2c13c661131abb3105e28b6b3` |
| Current | 2,517,248,308 | `7077fdf83b4c674aea8b11fae5611984a4f6d7cd7d4fe24878cde6afd7d8f847` |

The final average is the only v2 model admitted to evaluation.
[Exact treatment](hu200-v2-frozen-5b-artifacts/treatment.json),
[source pins](hu200-v2-frozen-5b-artifacts/source-pins.json).
Evaluation/planning source is `c004f8235dfd7dee993d859b4a8ce48e915fb19d`.
After native training, this adds full canonical-plan posting and the separately
tested Slumbot loader. Native, game, blueprint, registry and shared evaluation
functions are byte-unchanged from the training source.

## Counts frozen before final hands

The cost-only pilot used 1,024 direct blocks and 64 paired blocks per script.
Only timing, memory and paired SDs informed sizing. Cache builds took 423.94 s
for v2 and 368.41 s for v1, serially and without hands. The fixed model pair
uses both menu schemas with each policy's own keys; translation and optional
check/call fallback remain off.

**Full requested counts fit; no reduction.** The complete
[canonical plan](hu200-v2-frozen-5b-artifacts/canonical-plan.json), SHA256
`8b3d240a1ffb078cff152c0f613e74e10519b7cec94ef08747fa7064d87994d2`,
was posted in the PR description and exactly read back at **10:33:59 Madrid**,
before final hands. The 1.25× padded remaining-science projection is
**5,236.02 seconds /1.454 hours**, before the 21:30 science end and 22:00
closeout deadline. [Posting/readback receipt](hu200-v2-frozen-5b-artifacts/plan-posted.json).

Each script has 16,384 independent four-hand blocks: v2 and v1 in both seats on
shared physical deals and logical-role private action streams. Direct play has
131,072 two-hand duplicate blocks. This gives **458,752 panel +262,144 direct
=720,896 final primary hands**, plus exact reproduction; independent replay
checks every original hand. Different-menu direct play is descriptive only.

A required-spec-field omission stopped the initial pilot at model admission
**before any hand or outcome**. Entries, iteration and source checkpoint hash
were recovered from the already verified baseline audit/header. The original
input, failed admission and stop record remain preserved; no failed hands were
replaced and no scientific outcome was retried. A new real-model admission test
and existing HU200 tests pass **26 checks**. The continuation retains the
original swap baseline, locks and guard thresholds, and reuses verified caches.
[Correction receipt](hu200-v2-frozen-5b-artifacts/admission-correction.json),
[complete baseline spec](hu200-v2-frozen-5b-artifacts/baseline-spec-corrected.json).
Final evaluation source is `9477419a0d33711c8156b9ecebfc8a16a1c1e580`; only
admission metadata/control changed. The shared evaluation functions, native
binary, game and blueprint remain unchanged.
