# HU200 frozen 5B overbet-menu comparison

[PR270](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/270)
· [Predeclared protocol](../hu200-v2-frozen-5b-protocol.md)

**V2 nearly removes the 1.5× pot loss, but it is not a general fix at this budget.**
The paired pressure gain is **+238.59 [201.61,275.57] BB/100**; v2's absolute
pressure result is **−1.83 [−23.96,20.30]**, versus v1's −240.42. Half-pot gets
worse, and tight-aggressive, loose-aggressive and matched 1× pot fail the
predeclared regression limit. **The Slumbot-confirmation rule fails.** No server
copy, cache preparation, launcher, account, credentials or live Slumbot run was
made for this candidate. Training and evaluation are complete; local archive
readback passes and cloud upload acceptance is pending. This experiment asks
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

## Paired seven-opponent results

All rates are BB/100, with nominal block-paired Student-t 95% intervals and no
multiplicity adjustment. Each row has 16,384 independent blocks. These intervals
condition on the two fixed policies, one training seed and these scripted bots.
The half-pot and 1× pot controls hold #263's thresholds, bluffing, raise
frequencies and two draws per decision fixed; only their size differs from the
1.5× bot. They are not approximations of Slumbot's strategy.

| Opponent | V1 frozen 5B absolute, 95% | V2 frozen 5B absolute, 95% | V2 − V1 paired, 95% |
|---|---|---|---|
| Random | +189.33 [+114.77, +263.89] | +182.70 [+112.26, +253.13] | -6.64 [-51.04, +37.77] |
| Check/call | +146.50 [+135.08, +157.92] | +369.00 [+342.89, +395.11] | +222.50 [+197.46, +247.54] |
| Tight aggressive | +68.42 [+55.71, +81.14] | +21.66 [+2.78, +40.55] | -46.76 [-65.71, -27.81] |
| Loose aggressive | +127.38 [+102.54, +152.21] | -8.29 [-45.63, +29.05] | -135.67 [-175.40, -95.94] |
| 1.5× pot | -240.42 [-274.75, -206.09] | -1.83 [-23.96, +20.30] | +238.59 [+201.61, +275.57] |
| Half pot | +5.49 [-12.41, +23.39] | -26.83 [-51.73, -1.93] | -32.31 [-59.14, -5.48] |
| Matched 1× pot | +74.54 [+58.49, +90.59] | -9.12 [-33.33, +15.08] | -83.66 [-109.12, -58.21] |

**The primary pressure lower bound passes:** +201.61 >0. The large v1 loss is
reduced to a result consistent with break-even; an absolute v2 win is not
established. **The regression condition fails** for tight aggressive
(upper −27.81), loose aggressive (−95.94) and matched 1× pot (−58.21), all below
−20 BB/100. Half-pot is a nominally detected loss of −32.31 BB/100; its upper
bound −5.48 does not meet the declared severe-regression flag.

Therefore **do not recommend Slumbot confirmation under this rule**. The menu
addresses the tested 1.5× support weakness, with substantial tradeoffs. This
5B/11 GB v2 table is still sparse overall, so the experiment does not settle
an arbitrarily densely trained v2 or promise that more training removes the
regressions. No outcome-dependent extension or model selection followed.

## Descriptive direct match and coverage

Direct duplicate v2 versus v1 is **+244.53 [229.38,259.67] BB/100**, 131,072
blocks. It does not enter the decision rule. V1 encounters many missing keys
against v2's larger menu; that makes this different-menu match unsuitable as
the deciding benchmark.

[Complete coverage CSV](hu200-v2-frozen-5b-artifacts/coverage.csv) retains every
model/street/contrast, exact counts and all four visit bands. Missing and
zero-mass shares use all model decisions. Visit bands use known decisions,
including known zero-mass rows and excluding misses; missing keys are not
mislabeled as known zero visits. The Git CSV uses LF line endings; the original
archived CSV bytes remain unchanged.

| Direct model | Street | Decisions | Missing | Zero mass | Known visits 0 | 1–9 | 10–99 | 100+ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v1-frozen-5b | preflop | 262,591 | 9.428% | 0.006% | 0.000% | 0.008% | 0.222% | 99.770% |
| v1-frozen-5b | flop | 156,503 | 18.979% | 0.015% | 0.000% | 0.015% | 0.442% | 99.543% |
| v1-frozen-5b | turn | 80,941 | 27.245% | 0.014% | 0.000% | 0.136% | 1.805% | 98.059% |
| v1-frozen-5b | river | 47,222 | 36.062% | 0.025% | 0.040% | 0.649% | 5.508% | 93.803% |
| v2-frozen-5b | preflop | 285,950 | 0.178% | 0.000% | 0.008% | 0.382% | 2.994% | 96.616% |
| v2-frozen-5b | flop | 152,461 | 0.679% | 0.002% | 0.050% | 1.282% | 9.045% | 89.623% |
| v2-frozen-5b | turn | 80,191 | 5.027% | 0.009% | 0.454% | 5.085% | 24.271% | 70.189% |
| v2-frozen-5b | river | 45,592 | 23.079% | 0.042% | 1.839% | 12.697% | 39.900% | 45.563% |

V1's direct missing shares reach 18.98% flop /27.24% turn /36.06% river. V2
still misses 23.08% of direct river decisions, and only 45.56% of its known
river decisions have 100+ visits, versus v1's 93.80%. Different policies reach
different histories; these are decision-weighted observations, not a paired
causal effect on an identical set of states.

| Script /model | Street | Decisions | Missing | Zero mass | Known visits 100+ |
|---|---|---:|---:|---:|---:|
| 1.5× pot /v1-frozen-5b | flop | 8,516 | 32.316% | 0.000% | 99.948% |
| 1.5× pot /v1-frozen-5b | turn | 4,504 | 46.048% | 0.000% | 99.794% |
| 1.5× pot /v1-frozen-5b | river | 2,244 | 58.779% | 0.000% | 99.568% |
| 1.5× pot /v2-frozen-5b | flop | 5,896 | 0.017% | 0.000% | 95.047% |
| 1.5× pot /v2-frozen-5b | turn | 2,245 | 1.203% | 0.045% | 81.064% |
| 1.5× pot /v2-frozen-5b | river | 1,001 | 11.289% | 0.000% | 63.401% |
| Half pot /v1-frozen-5b | flop | 9,678 | 1.147% | 0.021% | 99.634% |
| Half pot /v1-frozen-5b | turn | 5,769 | 3.207% | 0.017% | 98.532% |
| Half pot /v1-frozen-5b | river | 3,248 | 6.650% | 0.062% | 94.723% |
| Half pot /v2-frozen-5b | flop | 6,574 | 0.791% | 0.000% | 94.818% |
| Half pot /v2-frozen-5b | turn | 2,819 | 4.541% | 0.035% | 82.386% |
| Half pot /v2-frozen-5b | river | 1,352 | 16.346% | 0.074% | 69.054% |
| Matched 1× pot /v1-frozen-5b | flop | 8,946 | 0.000% | 0.000% | 99.497% |
| Matched 1× pot /v1-frozen-5b | turn | 4,931 | 0.000% | 0.000% | 97.850% |
| Matched 1× pot /v1-frozen-5b | river | 2,759 | 0.072% | 0.000% | 93.870% |
| Matched 1× pot /v2-frozen-5b | flop | 6,268 | 0.223% | 0.000% | 93.844% |
| Matched 1× pot /v2-frozen-5b | turn | 2,484 | 3.301% | 0.000% | 81.391% |
| Matched 1× pot /v2-frozen-5b | river | 1,162 | 13.941% | 0.000% | 66.200% |

Against 1.5× pot, v2 misses fall to **0.017% /1.203% /11.289%** on
flop/turn/river, versus v1's **32.316% /46.048% /58.779%**. This is consistent
with improved menu/history support accompanying the pressure gain. Half-pot
and 1× pot coverage deteriorate on later streets; their v2 river misses are
16.35% and 13.94%, respectively. Coverage does not isolate the cause of their
losses from changed actions, ranges, visit density or restriction to the frozen
table.

## Verification, guards and evidence

**Every one of the 720,896 final primary hands independently replayed and
exactly reproduced before readout.** All menus, keys, private action streams,
public transitions and settlements pass; no failed hand was replaced.
[Global verification](hu200-v2-frozen-5b-artifacts/all-final-verified.json),
[complete result summary](hu200-v2-frozen-5b-artifacts/summary.json).
The full requested counts remained fixed after posting. No guard breach,
numerical failure or outcome-driven retry occurred. The pre-hand metadata
admission failure and administrative setup failures remain in the archives.

All work ran detached on M4, with temporary awake assertions. Both locks were
held during science and archiving; the correction preserved the original swap
baseline. Guard readings are retained every five seconds, with creation-time
checks before any process signal. Continuous observed 60-second breaches stop
for >13 GiB family RSS, critical pressure, <8% free memory or >4 GB swap growth;
disk <16 GiB stops immediately. Gaps reset the sustained timers.

Through scientific/archive closeout: **1,751 readings**, sampled family peak
**10.645 GiB**, minimum free 39%, maximum swap growth **1.336 GB**, minimum disk
**26.836 GiB**, pressure levels 1/2, zero critical samples and zero sampled or
sustained resource breaches. Maximum observed gap is **5.060 s**. After-save
RSS is distinct from this sampled family peak.
[Closed resource summary](hu200-v2-frozen-5b-artifacts/resource-summary.json).
Science and local archive closeout finished around **11:01 Madrid**, before
the fixed 22:00 deadline. [Closeout](hu200-v2-frozen-5b-artifacts/detached-closeout.json).

| Local ZIP | Bytes | Whole SHA256 |
|---|---:|---|
| Full evidence, 271 members | 18,420,960,366 | `2197b5691542376a3eb03915bb9e4d432648df959439424001a75a6bea253276` |
| V2 5B average only, one model | 2,876,032,489 | `bbd1abc9abdf550504ca2c3b5a6ffc72b162c8b8fea8aadc04364a9718ed623a` |

Both ZIPs have embedded manifests and every member passes local size/SHA256
readback. [Full receipt](hu200-v2-frozen-5b-artifacts/archive-receipt.json),
[average receipt](hu200-v2-frozen-5b-artifacts/average-receipt.json).
They are in `~/Local/Research-Cloud/PR-270-hu200-v2-frozen-5b/`.
Cloud upload is in progress; local integrity is not upload acceptance. The
closed archive/upload observer and receipts will be included in a small
closeout supplement. Remote archive bytes have not been downloaded or verified.
All originals and caches remain, and no synced file was deleted or evicted.
[Archive/retrieval index](../../RESULTS_INDEX.md).

The Slumbot loader fix remains offline-tested through the client and analysis
entry points, including a real synthetic K50 v2 model, read-only warm cache,
recorded-hand analysis, depth/card-schema admission and bad-hash rejection.
The failed decision rule means no db-ocean preparation was undertaken. Final
reviewed-head CI is required before handoff; the owner decides merge. No
release, default change, account, live Slumbot play or further training.
