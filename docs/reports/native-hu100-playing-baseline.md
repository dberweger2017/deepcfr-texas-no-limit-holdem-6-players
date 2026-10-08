# HU100 playing baseline: complete, with limited playing strength

**The fixed 40,960-hand baseline completed and fully replayed/reproduced.** The
11,042,440-node average improves on the native-menu uniform reference against
check_call and loose_aggressive in this paired sample. Its differences against
random, tight_aggressive and pot_pressure are inconclusive. The trained policy
loses against tight_aggressive, loose_aggressive and pot_pressure; the last two
losses are substantial. This is a development baseline, not a release gate,
exploitability measure or general-strength claim. No additional training,
checkpoint selection, outcome-driven extension or merge occurred.

[Protocol](../native-hu100-playing-baseline.md) · [PR197](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/197)
· [Complete compact evidence](native-hu100-playing-baseline-artifacts/scientific-summary.json)
· [Decision rates, actions and latency](native-hu100-playing-baseline-artifacts/decision-diagnostics.csv).

## Fixed comparison and results

Heads-up, 100-BB stacks (10,000 chips) reset each hand, blinds 50/100, no rake.
Five unchanged registry opponents; each has 2,048 independent duplicate-deal
blocks, two target seat rotations and both policies: 8,192 hands/opponent.
The physical deals are paired across arms/seats; action streams are independent
per opponent/block/rotation/arm/logical player and private from observations.
Each table reports **BB/100 [block-based 95% interval]**. The paired difference
is average minus uniform, using the difference of seat-averaged block outcomes.
Student-t intervals are unadjusted for five opponents. The four hands in a block
are never counted as four independent observations.

| Opponent | Trained average | Native-menu uniform | Paired difference |
|---|---:|---:|---:|
| random | +69.26 [-35.15, +173.67] | +49.35 [-53.59, +152.30] | +19.91 [-126.04, +165.86] |
| check_call | +76.32 [+41.89, +110.75] | -29.41 [-71.84, +13.03] | +105.73 [+51.75, +159.70] |
| tight_aggressive | -42.25 [-84.16, -0.33] | -87.33 [-130.06, -44.59] | +45.08 [-6.55, +96.71] |
| loose_aggressive | -310.89 [-396.21, -225.57] | -491.49 [-576.66, -406.32] | +180.60 [+71.89, +289.31] |
| pot_pressure | -133.08 [-189.75, -76.41] | -131.96 [-189.09, -74.82] | -1.12 [-72.00, +69.75] |

The uniform reference uses exactly the trained native menu and existing weighted
Random.choices sampler. Random the opponent remains distinct: it chooses legal
action kinds uniformly and splits a raise between distinct minimum and all-in
maximum targets. Its deep-stack all-ins are retained even when absent from the
native reference menu. Check_call and the three styles keep their original
heuristics, thresholds and private sampling; HU100 fixture tests and the pilot
confirmed legality and complete settlements.

## Pilot, frozen count and correctness

Only timing/completeness was inspected from the separately seeded 320-hand pilot.
Pilot root `2026100819711`; final root `2026100819712`. The pilot completed all
1,299 decisions, passed independent replay, then reproduced every hand and
deterministic decision trace. It is excluded from the reported final outcomes.
The cost-only receipt froze all 2,048 blocks/opponent before the first final hand:
2× measured complete play/replay/reproduction and fixed loading cost forecast
**216.61 seconds**, plus **180 seconds** reserved for closeout, versus
**1,770.75 seconds** remaining. No count reduction or subsequent extension.
Frozen receipt SHA256
`33d6541bca16e224cc0f338aaefd675b7b5a2ea9cf64fb967c1a105cb1e58916`.

Independent replay verified **40,960 final hands /160,425 actions**, all starts,
legal actions, complete public events, chip conservation and final settlements.
It also independently recomputed the block estimates/intervals and recounted
coverage/action rates. A fresh sequential reproduction reloaded the pinned model
and reproduced every hand plus every deterministic decision field, including
lookup classifications and exact probabilities; measured latency is excluded
from equality. No failed, omitted or unaudited final hands.

## Lookup exposure and inference

Rates below use decisions within each policy/opponent/street as the denominator.
Known means a positive-mass stored key; zero and missing use the unchanged uniform
fallback. Uniform-arm exposure is measured against the pinned trained table,
without using learned probabilities. Opponent lookup coverage is inapplicable.
Telemetry observes after the original draw and consumes no RNG; fixture tests
confirm exact actions and RNG state against the uninstrumented sampler.
Chooser latency excludes the subsequent telemetry lookup; wall cost includes it.
Full counts, action frequencies and mean/p50/p95/max chooser latency for both
policies and opponents are in the linked CSV and JSON. Conditional street rates
describe the states reached by each policy, rather than matched decision states.

| Opponent | Policy | Street | Decisions | Known positive % | Zero % | Missing % | Mean /p95 ms |
|---|---|---|---:|---:|---:|---:|---:|
| random | average | preflop | 4,214 | 82.56 | 0.07 | 17.37 | 0.0193 /0.0235 |
| random | average | flop | 1,274 | 79.20 | 0.00 | 20.80 | 0.0310 /0.0367 |
| random | average | turn | 469 | 81.45 | 0.21 | 18.34 | 0.0434 /0.0510 |
| random | average | river | 176 | 82.39 | 0.00 | 17.61 | 0.0659 /0.0864 |
| random | uniform | preflop | 4,223 | 81.74 | 0.76 | 17.50 | 0.0112 /0.0090 |
| random | uniform | flop | 1,290 | 80.54 | 0.23 | 19.22 | 0.0072 /0.0085 |
| random | uniform | turn | 451 | 79.60 | 3.10 | 17.29 | 0.0075 /0.0088 |
| random | uniform | river | 165 | 74.55 | 3.03 | 22.42 | 0.0076 /0.0092 |
| check_call | average | preflop | 4,096 | 100.00 | 0.00 | 0.00 | 0.0193 /0.0207 |
| check_call | average | flop | 3,538 | 99.83 | 0.03 | 0.14 | 0.0314 /0.0343 |
| check_call | average | turn | 3,538 | 98.67 | 0.31 | 1.02 | 0.0461 /0.0499 |
| check_call | average | river | 3,538 | 98.67 | 0.23 | 1.10 | 0.0560 /0.0843 |
| check_call | uniform | preflop | 4,096 | 100.00 | 0.00 | 0.00 | 0.0084 /0.0093 |
| check_call | uniform | flop | 3,621 | 99.75 | 0.08 | 0.17 | 0.0075 /0.0081 |
| check_call | uniform | turn | 3,621 | 98.18 | 0.55 | 1.27 | 0.0078 /0.0084 |
| check_call | uniform | river | 3,621 | 97.74 | 0.30 | 1.96 | 0.0080 /0.0087 |
| tight_aggressive | average | preflop | 2,420 | 99.92 | 0.04 | 0.04 | 0.0322 /0.0232 |
| tight_aggressive | average | flop | 667 | 99.10 | 0.00 | 0.90 | 0.0319 /0.0354 |
| tight_aggressive | average | turn | 312 | 95.51 | 0.96 | 3.53 | 0.0430 /0.0498 |
| tight_aggressive | average | river | 136 | 98.53 | 1.47 | 0.00 | 0.0732 /0.0841 |
| tight_aggressive | uniform | preflop | 2,429 | 99.05 | 0.82 | 0.12 | 0.0088 /0.0093 |
| tight_aggressive | uniform | flop | 758 | 98.15 | 1.45 | 0.40 | 0.0078 /0.0087 |
| tight_aggressive | uniform | turn | 286 | 90.56 | 5.94 | 3.50 | 0.0081 /0.0094 |
| tight_aggressive | uniform | river | 106 | 86.79 | 5.66 | 7.55 | 0.0084 /0.0096 |
| loose_aggressive | average | preflop | 3,670 | 99.56 | 0.35 | 0.08 | 0.0201 /0.0231 |
| loose_aggressive | average | flop | 1,687 | 98.81 | 0.53 | 0.65 | 0.0313 /0.0367 |
| loose_aggressive | average | turn | 1,001 | 95.40 | 1.30 | 3.30 | 0.0417 /0.0510 |
| loose_aggressive | average | river | 764 | 93.59 | 2.62 | 3.80 | 0.0623 /0.0855 |
| loose_aggressive | uniform | preflop | 3,663 | 97.35 | 2.24 | 0.41 | 0.0086 /0.0093 |
| loose_aggressive | uniform | flop | 1,702 | 97.59 | 1.59 | 0.82 | 0.0078 /0.0089 |
| loose_aggressive | uniform | turn | 1,052 | 88.88 | 5.32 | 5.80 | 0.0082 /0.0096 |
| loose_aggressive | uniform | river | 711 | 86.64 | 6.05 | 7.31 | 0.0083 /0.0098 |
| pot_pressure | average | preflop | 2,805 | 90.55 | 0.11 | 9.34 | 0.0202 /0.0220 |
| pot_pressure | average | flop | 851 | 70.74 | 0.12 | 29.14 | 0.0314 /0.0352 |
| pot_pressure | average | turn | 368 | 53.53 | 0.27 | 46.20 | 0.0430 /0.0489 |
| pot_pressure | average | river | 202 | 31.19 | 0.00 | 68.81 | 0.0679 /0.0830 |
| pot_pressure | uniform | preflop | 2,849 | 89.51 | 0.25 | 10.25 | 0.0087 /0.0093 |
| pot_pressure | uniform | flop | 863 | 69.87 | 0.12 | 30.01 | 0.0079 /0.0090 |
| pot_pressure | uniform | turn | 352 | 50.85 | 0.57 | 48.58 | 0.0082 /0.0094 |
| pot_pressure | uniform | river | 130 | 43.85 | 0.00 | 56.15 | 0.0084 /0.0099 |

Pot_pressure exposes the largest coverage gap: average missing-key rates are
29.14% on the flop, 46.20% on the turn and 68.81% on the river (202 river
decisions). These are exposure diagnostics, not evidence of reliable unseen-key
play or convergence. Random’s retained min-raise/all-in behavior also produces
missing keys. One small training seed, one evaluation root, limited scripted
opponents and wide intervals cannot establish general playing strength.
No external benchmark, LBR, exploitability audit or release comparison was run.

## Source, resources and storage

Evaluation source **`009b5d92529792b0db32798407cf56c25416beb5`**. Configuration
SHA256 `ec2a2ba9f3cf0851eac9bc361110fbc9e5664e951a097bc0338fbd61ee7e9670`.
Average: **79,195,090 bytes**, SHA256
`ffd53decdd4af5bffc0ae34e98144d43e27eef92a49033a7d4008615576a93be`,
iteration 7,722, 3,255,387 entries, checkpoint SHA256
`234628a0390502f6b17f4bad3486c47c2a5aa7a297fc611e00287533ea1f4567`.
Manifests pin source/environment/model/opponent implementations; schedules and
explicit private-stream maps pin every block. Model byte identity and training
identity were checked on each load/snapshot. Existing HU20 equivalence and HU100
training/capacity results remain unchanged.

All heavy work used the isolated free M4 checkout, one sequential evaluation
worker. A new clock started **11:23:30 Madrid**, hard stop **11:53:30 Madrid**.
Pilot/model loads/final play/full audits/reproduction completed **11:25:53**
(**142.59 seconds**); local archive verification/atomic native copy completed
**11:27:34**, archive guard closed **11:27:35**, within the same 30-minute cap.
Continuous external guards covered the whole owned process family at 10-GiB RSS,
normal system pressure, original 448.81-MiB swap baseline/+0.5 GiB, ≥15.5-GiB
disk and AC. Across 29 five-second science samples: sampled peak **0.840 GiB**,
swap usage **8 MiB below the original baseline**, minimum free disk **89.831 GiB**,
all AC/normal pressure, minimum system free percentage **81%**. Sampling does
not measure every transient RSS peak. No guard or correctness failure; both
execution and archive guards closed successfully. Permanent claims prohibit
duplicate launches/retries; all four old timers remain disabled.

Exact-source M4 qualification passed **52 focused Python tests** and artifact
checks. Independent source review 14 closed review 13’s configuration-substitution,
alternate-root duplicate/retry and M4-identity findings; no P1/P2 source blockers.
[Source review](native-hu100-playing-baseline-artifacts/source-review.json) is
static only; runtime and byte proofs come from separate retained receipts.
Final evidence review and final-head CI are pending; no merge is authorized.

[Accepted Research-Cloud archive](https://drive.google.com/file/d/1QEHJJ_yPJbfHYkX1Taa3wM4RnblVPdK1/view)
`native-hu100-playing-baseline-20261008.zip`: **574,958,858 bytes /227 locally
verified members**, whole SHA256
`a2c6f3106bdff486ece8ed1cec53521c235fe3eaf0c9debbae8b901e5ea5421d`.
Embedded `ARCHIVE-MANIFEST.json.gz` decompressed SHA256
`3ae2ec4cd2d8b5b41fc3e1f81c5bc1df9d75840a916a243d06f2259e895aee61`;
compressed member SHA256
`82d3870d094cbdec5341250d8c3ede6cefdb5115e20ef17a5868164dcc1f63ff`.
All member sizes/hashes were read back locally. Native uploaded1/uploading0/
conflicts0 and independent cloud ID/name/size/parent metadata were accepted.
Remote bytes were not downloaded; upload metadata and local byte verification
are separate proofs. The archive preserves pilot/final/reproductions, raw traces,
model snapshots, manifests, resources, qualifications, review findings/responses
and selected evaluation source. Original research archives/files are retained.
[Archive receipts](native-hu100-playing-baseline-artifacts/archive-upload-acceptance.json)
and [restoration instructions](../../RESULTS_INDEX.md) name exact members/commands.
No paid compute, cleanup, release, publication or merge.
