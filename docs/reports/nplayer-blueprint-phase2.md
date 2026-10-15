# Three-player Phase 2: first full-average blueprint

**The predeclared learning rule passes across all three seeds. Cloud archive closeout is pending.**
[PR292](https://github.com/dberweger2017/CFR-NLHE-6max/pull/292) follows merged
PR289/PR291. The [protocol](../nplayer-blueprint-phase2-protocol.md) fixes the
owner's all-street averages, three sequential fresh seeds and learning rule.
No search, release or merge is authorized.

## Harness proven before training

The native host fixes one deal and plays all six permutations of three labeled
policy instances. Every instance occupies each physical seat twice; the two
copies of a model share immutable weights but have separate action RNGs.
Only acting-seat observations enter model and script methods. Deck/action
streams are domain-separated, and policies cannot see duplicate identities,
roots or other players' hidden cards.

Reported BB/100 and uncertainty use independent six-hand **block means**, not
individual correlated hands. Overall and per-seat results distinguish the
singleton from each player's share of the pair. Three primary lower-bound
checks must all exceed zero. The complete 42-contrast matrix must pass before
readout; all other intervals are descriptive and nominal.

Controls are menu-uniform random, check/call, tight-aggressive and matched-pot
pressure. The latter two retain the heads-up public card-score heuristics and
style thresholds/frequencies; their wagers use one pot after calling, clamped
to exact legal bounds. They are controls, not optimal multiplayer opponents.

Admission on bb128: **47 native and 380 Python checks pass**, including
800 scripted random hands and hidden-card invariance on every street. Fourteen
focused Python admission tests cover two distinct full-average models in both
lineups, all-hand pinned-engine replay, exact reproduction, corrupted settlement
rejection, correct block/seat denominators, refusal to continue partial training
and synthetic VRM stop behavior. All 18 small HU20/HU100/HU200
checkpoint/current/average identities match the
qualified baseline. Fresh/resumed HU100 retain SHA256
`792a675ce6d45d4de8d1b7f3fc6d976548610810f11da8f336c3926f37f8d416`.
The fresh20M and resumed10M→20M #250 checkpoints both retain SHA256
`4009bbbce8d95a722e4e63e96dfc6e2b67957095f06c53487a29e4bf3d3acc8b`.
The rebuilt HU executable itself differs, so these were freshly executed
identity checks, not a claim of executable-byte reuse. The existing HU20
1B release-identity regression passed, retaining the required 142,677,367-byte
average SHA256 `571e198266eabc6d8bb9de2d1aa76d9a68be0b2512222ea96874168989c6b74d`
in 780.66 guarded seconds; it changes no release. After training,
23 focused harness/controller/statistics/archive/watchdog checks passed in
34.24 seconds under a 35.36-second successful supervisor. This includes the
two-model real-K50 fixture and the full controller/replay/readout fixture.

The exact PR291 scoped-average smoke input, SHA256
`fcbe8bd5733572fe9f9046840d6e277a0b11ce0ac96936546df0815be218485e`, completes nine
fixed 64-block cases (self-play and both lineups against each script), twice.
All **6,912 executed hands** reproduce byte-for-byte and **3,456 hands** replay
independently in the pinned engine, with no mismatch. This proves the harness;
it is not a Phase 2 strength comparison. Total **263.62 seconds**, peak **2.339
GiB RSS**, CPU **65°C**, VRM **66.5°C**, fan1 **337–611 RPM**. Loading the 8.25M-row
model takes about 12.5 seconds per process, versus roughly 0.03 seconds to play
384 hands and 0.08 seconds for independent replay.
[Per-case measurements](nplayer-blueprint-phase2-artifacts/harness-costs.json).

## Safety, budget and retained failures

The on-host exclusive supervisor allows one workload, at most four CPUs,
records nct6687 Thermistor 15 and fan1 in every resource row, and requests a clean
stop after VRM >=95°C for 30 continuous seconds. CPU >=85°C and resource limits
remain immediate stops. Clean thermal stopping still marks the attempt failed;
there is no automatic restart. CPU/memory/time supervision stays active through
checkpoint saving, with a bounded 180-second clean-stop grace.

The 39.1-GiB figure discussed in planning was the 100-BB preflop-average scenario.
The uncapped 20-BB full-average origin-linear scenario was 79.1 GiB. At the
owner's 150M-entry terminal cap, the same allocation assumptions give **43.41
GiB**, still a scenario rather than a guarantee. Keep the 80-GiB RSS guard.

No research attempt has failed so far. Tests were written before implementation
and initially failed on absent modules as intended. A light M1 statistics test
could not run because that interpreter lacks SciPy (and later pytest); the
guarded bb128 run passed using its installed research environment. The existing
M1 project environment subsequently ran 12 small controller/archive/statistics
checks in 0.82 seconds; three native integration checks correctly skipped there.
No dependencies were installed or native workload run on the laptop. An
optional all-history seed scan was canceled when it became too costly for the
laptop; the prior repository/host inventory and a direct check of PR286's
2026101501/1502 lineages found no collision with the reserved cash seeds. Synthetic corruption and
thermal tests are deliberately failing inputs, not undisclosed campaign reruns.

## Training milestones

All three fresh seeds reached 1B nodes with all nine scheduled checkpoints,
full-average exports and every-row audits passing. All used training source
`8ad2729c1fc9edad4afe8e406b28dd680b50a989` and the same admitted binaries.
The whole-seed supervisors passed without thermal or resource stops.

| Seed | Actual terminal nodes | 200M entries | 500M entries | 1B entries | Whole-seed minutes | Peak family GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026101541 | 1,000,000,057 | 15,621,814 | 22,245,256 | 29,025,056 | 62.10 | 6.250 |
| 2026101542 | 1,000,000,341 | 15,811,662 | 22,599,817 | 29,370,052 | 61.91 | 9.200 |
| 2026101543 | 1,000,000,244 | 15,826,732 | 22,578,942 | 29,668,448 | 62.08 | 9.585 |

Mean traverser regret-update visits per retained key (opponent-only average
entries may have zero regret visits):

| Seed / save | Preflop | Flop | Turn | River |
| --- | ---: | ---: | ---: | ---: |
| 2026101541 / 200M | 8.095 | 5.299 | 3.081 | 1.719 |
| 2026101541 / 500M | 14.130 | 9.426 | 5.589 | 3.094 |
| 2026101541 / 1000M | 21.830 | 14.509 | 8.700 | 4.810 |
| 2026101542 / 200M | 8.025 | 5.192 | 3.032 | 1.703 |
| 2026101542 / 500M | 14.179 | 9.203 | 5.483 | 3.047 |
| 2026101542 / 1000M | 22.159 | 14.421 | 8.640 | 4.734 |
| 2026101543 / 200M | 8.098 | 5.292 | 3.054 | 1.688 |
| 2026101543 / 500M | 14.122 | 9.371 | 5.532 | 3.038 |
| 2026101543 / 1000M | 21.870 | 14.376 | 8.545 | 4.678 |

Missing training-decision shares **since the previous save**, with the initial
200M row measured from the start:

| Seed / save | Preflop | Flop | Turn | River |
| --- | ---: | ---: | ---: | ---: |
| 2026101541 / 200M | 3.028% | 5.911% | 11.281% | 20.829% |
| 2026101541 / 500M | 0.506% | 1.253% | 2.664% | 5.909% |
| 2026101541 / 1000M | 0.287% | 0.771% | 1.613% | 3.513% |
| 2026101542 / 200M | 3.048% | 6.026% | 11.459% | 21.030% |
| 2026101542 / 500M | 0.479% | 1.277% | 2.754% | 6.051% |
| 2026101542 / 1000M | 0.262% | 0.727% | 1.570% | 3.567% |
| 2026101543 / 200M | 3.031% | 5.935% | 11.405% | 21.222% |
| 2026101543 / 500M | 0.486% | 1.265% | 2.718% | 6.036% |
| 2026101543 / 1000M | 0.294% | 0.788% | 1.688% | 3.696% |

The per-seed [admission, milestone and result receipts](nplayer-blueprint-phase2-artifacts/)
retain exact model/checkpoint hashes, every-row audit counts, cumulative and
segmented decisions/misses, and zero-regret-visit entries by street.
These are training coverage measurements; final played lookup misses are
measured separately during evaluation. No final comparison readout yet.

## Final evaluation admission

The [committed manifest](nplayer-blueprint-phase2-artifacts/evaluation-manifest.json)
pins all nine audited full averages and the unchanged arena executable, with
42 fixed contrasts ×8,192 six-hand blocks. Manifest SHA256
`b46b197a9b68a4b0b189121e477eaf34095443c97c5e1c86c961a6224aa8613c`.
Projected cost is 83.70 minutes of fresh model loads, 8.25 minutes of two-pass
play and 10.22 minutes of engine replay: **102.17 minutes total**. The fixed
whole-evaluation cap is **215 minutes**, one job/four CPUs/80 GiB family RSS
with the same thermal guards. This declaration precedes final hands. No
comparison readout is available yet.

## Final evaluation results

The predeclared learning rule **PASSES**.
All 42 fixed contrasts completed. Every one of the **2,064,384 final hands**
independently replayed in the pinned engine and reproduced exactly in a fresh
process (another 2,064,384 hands) before the complete readout was admitted.
The outer evaluation guard passed with no stop. These are conditional
three-player playing comparisons, not an equilibrium or exploitability claim.

| Primary: 1B singleton versus two own-200M copies | BB/100 [95% interval] | Lower bound >0 |
| --- | ---: | --- |
| 2026101541 | +18.39 [+12.49, +24.29] | yes |
| 2026101542 | +21.39 [+15.61, +27.17] | yes |
| 2026101543 | +21.61 [+15.62, +27.60] | yes |

All remaining intervals below are descriptive, nominal two-sided 95%
Student-t intervals over 8,192 independent six-hand deal blocks. Paired-model
results are per player, not the two-player team total.

| Seed | Earlier model | One 1B vs two earlier | Two 1B vs one earlier, per 1B player |
| --- | --- | ---: | ---: |
| 2026101541 | 200M | +18.39 [+12.49, +24.29] | +11.04 [+7.99, +14.09] |
| 2026101541 | 500M | +10.62 [+5.48, +15.76] | +5.01 [+2.45, +7.57] |
| 2026101542 | 200M | +21.39 [+15.61, +27.17] | +7.85 [+4.90, +10.80] |
| 2026101542 | 500M | +8.60 [+3.54, +13.67] | +4.63 [+2.10, +7.16] |
| 2026101543 | 200M | +21.61 [+15.62, +27.60] | +12.24 [+9.28, +15.20] |
| 2026101543 | 500M | +9.66 [+4.48, +14.85] | +4.85 [+2.24, +7.47] |

| Seed | Script | One model vs two bots | Two models vs one bot, per model |
| --- | --- | ---: | ---: |
| 2026101541 | random | +124.88 [+115.55, +134.20] | +75.53 [+70.47, +80.58] |
| 2026101541 | check-call | +156.94 [+144.71, +169.17] | +88.47 [+83.06, +93.88] |
| 2026101541 | tight-aggressive | +38.81 [+35.33, +42.29] | +14.11 [+12.48, +15.73] |
| 2026101541 | pot-pressure | +23.79 [+19.28, +28.30] | +11.30 [+9.19, +13.41] |
| 2026101542 | random | +123.93 [+114.62, +133.25] | +75.20 [+70.10, +80.31] |
| 2026101542 | check-call | +169.14 [+156.78, +181.51] | +85.03 [+79.69, +90.37] |
| 2026101542 | tight-aggressive | +32.50 [+28.93, +36.08] | +14.71 [+13.15, +16.26] |
| 2026101542 | pot-pressure | +23.73 [+19.31, +28.15] | +11.96 [+9.83, +14.09] |
| 2026101543 | random | +124.98 [+115.47, +134.48] | +72.84 [+67.92, +77.77] |
| 2026101543 | check-call | +162.42 [+150.06, +174.79] | +90.53 [+85.07, +95.99] |
| 2026101543 | tight-aggressive | +35.83 [+32.25, +39.42] | +14.41 [+12.80, +16.03] |
| 2026101543 | pot-pressure | +24.42 [+19.77, +29.07] | +13.43 [+11.31, +15.55] |

| Singleton 1B seed | Paired 1B seed | Singleton BB/100 [95% interval] |
| --- | --- | ---: |
| 2026101541 | 2026101542 | -1.14 [-6.19, +3.90] |
| 2026101541 | 2026101543 | -0.01 [-5.10, +5.09] |
| 2026101542 | 2026101541 | -0.09 [-5.09, +4.92] |
| 2026101542 | 2026101543 | -0.95 [-5.90, +3.99] |
| 2026101543 | 2026101541 | -3.71 [-8.83, +1.40] |
| 2026101543 | 2026101542 | +2.25 [-2.86, +7.35] |

[Full verified readout](nplayer-blueprint-phase2-artifacts/evaluation-results.json)
includes overall, every physical seat and BTN/SB/BB position intervals.
[Flat evaluation table](nplayer-blueprint-phase2-artifacts/evaluation.csv)
also retains both roles and model lookup missing shares by street.

## Resources and archive closeout

Across 18,320 retained readings through verified readout, maximum CPU package
was **70°C**, VRM **78°C**, peak family RSS **9.585 GiB** and swap growth zero.
Every resource row includes VRM and fan1; no scientific guard stopped.
Evaluation fan1 was337–744RPM, training337–797RPM. The complete per-job
[resource receipt](nplayer-blueprint-phase2-artifacts/resources-complete.json)
retains peaks, minima, counts and source/input identities; raw readings remain
in the archives. Main sealing/readback took234.33seconds, then the small
archive-guard closeout took1.02seconds. bb128 compute was released at20:17Madrid.

Main ZIP:15,887,630,020bytes, SHA256
`ea57f0e4910c458c5597c194da25342aaf912f2f098452e8e0982b1121dddb3f`,
embedded manifest SHA256
`ae9903fa0a8fea8b59bddb55d0441528de7b87f66136071d4adc26fc4c44700c`.
All member sizes/SHA256/CRC and concatenated original transport-part SHA256
passed. Cloud acceptance is still partial. The upload bridge rejected a128MiB
input before action invocation because its limit is100MiB; verified buffers
are split into at most64MiB cloud pieces. The sealed ZIP and host originals
remain unchanged. The bounded M1 transport spool is at most2GiB.
[Operator events](nplayer-blueprint-phase2-artifacts/closeout-operator-events.json)
also retain the premature buffer assertion and delayed native metadata observation.
These transport issues changed no scientific run or result. No remote archive
bytes have been downloaded or verified, and no scientific original was removed.

## Outstanding work

The full evaluation and verified readout passed. Main archive and closeout
archive sealing/readback passed; upload and indexed cloud acceptance remain
outstanding. bb128 compute was released at approximately20:17Madrid onOctober15.
All working evidence is retained under
`/home/dberweger/nplayer-blueprint-phase2-20261015/`; its shared source/input
parents are protected. No cleanup has occurred.
