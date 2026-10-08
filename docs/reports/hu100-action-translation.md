# HU100 action translation: fixed paired evaluation

Inference-time translation is an explicit, off-by-default research option. The predeclared pot_pressure primary is inconclusive; it does not pass.
The random safeguard passes. All three on-menu controls have identical complete hand histories, actions and settlements. No released default, web runtime or HU20 model changed.

## Fixed comparison and results

One #204 terminal average: **39,438,279 nodes /7,643,261 entries**, iteration 30,080, original training seed. Pinned SHA256 `ba62d13536120a9d549f2f3ff84bcb2a96fbd8143ac2fc8477addab368dee0c4`. Same weights and private action streams with translation disabled/enabled. Final root **2026100820512**, **2,048 independent paired deal blocks/opponent**, both swapped target positions; reset 100 BB, blinds 50/100, no rake. Each block averages the two rotations before a Student-t 95% interval. One formal primary, no pooling or extensions.

| Opponent | Disabled BB/100 [95%] | Translated BB/100 [95%] | Translated minus disabled [95%] | Role |
|---|---:|---:|---:|---|
| check_call | +97.09 [+69.20, +124.99] | +97.09 [+69.20, +124.99] | +0.00 [+0.00, +0.00] | descriptive control; identical |
| loose_aggressive | -199.11 [-276.73, -121.49] | -199.11 [-276.73, -121.49] | +0.00 [+0.00, +0.00] | descriptive control; identical |
| pot_pressure | -131.74 [-185.55, -77.92] | -104.15 [-156.78, -51.52] | +27.59 [-6.49, +61.67] | primary |
| random | +72.07 [-26.07, +170.21] | +72.07 [-26.07, +170.21] | +0.00 [+0.00, +0.00] | secondary safeguard |
| tight_aggressive | -7.63 [-48.00, +32.75] | -7.63 [-48.00, +32.75] | +0.00 [+0.00, +0.00] | descriptive control; identical |

Random predeclared safeguard: paired lower bound >−20 BB/100; a paired upper bound below −20 is a severe regression. The controls require complete candidate hands identical, rather than merely similar means.

## Translation rule and limits

Only missing exact keys may translate; exact stored positive-mass and zero-mass keys keep their existing behavior. Current cards, player flags and menu always come from the real observation. A pure public betting reducer replays fixed actors/action kinds/street transitions. On-menu past raises retain their menu names; off-menu raises branch over min/pot/conditional jam. A witness must reach the same actor, folded/all-in flags and current menu names. It supplies only raise-label overrides to the existing key encoder.

Ordering prefers fewer all-in-status changes, then summed `abs(x/(1+x) − y/(1+y))` for observed/witness paid/pre-action-pot fractions, then lexicographic raise-to targets and labels. Exact rational costs make ties deterministic. Limits are 512 popped states and 128 public events; no lookup outcome depends on a timer or policy RNG. Sampling retains exactly one existing weighted draw.

Preserving real all-in player flags is a deliberate limit: a first-action open jam cannot be witnessed by the conditional-jam training menu and remains uniform. Translation also cannot fix sparse visits at supported keys, poor card abstraction or opponent modeling. No claim of general strength, multi-seed robustness, HU20 adoption or release eligibility follows.

## Decision telemetry

Counts/rates refer to acting target decisions, not hands. Zero-mass stored keys are uniform; exact rates here mean positive-mass stored exact keys. Distance is the sum over past raises; it is not a BB payoff distance.

| Option | Opponent | Street | Decisions | Exact % | Translated % | Uniform % |
|---|---|---|---:|---:|---:|---:|
| disabled | random | preflop | 4051 | 83.39 | 0.00 | 16.61 |
| disabled | random | flop | 1254 | 83.01 | 0.00 | 16.99 |
| disabled | random | turn | 464 | 81.03 | 0.00 | 18.97 |
| disabled | random | river | 166 | 86.14 | 0.00 | 13.86 |
| enabled | random | preflop | 4051 | 83.39 | 0.00 | 16.61 |
| enabled | random | flop | 1254 | 83.01 | 0.00 | 16.99 |
| enabled | random | turn | 464 | 81.03 | 0.00 | 18.97 |
| enabled | random | river | 166 | 86.14 | 0.00 | 13.86 |
| disabled | check_call | preflop | 4096 | 100.00 | 0.00 | 0.00 |
| disabled | check_call | flop | 3440 | 99.91 | 0.00 | 0.09 |
| disabled | check_call | turn | 3440 | 99.65 | 0.00 | 0.35 |
| disabled | check_call | river | 3440 | 99.88 | 0.00 | 0.12 |
| enabled | check_call | preflop | 4096 | 100.00 | 0.00 | 0.00 |
| enabled | check_call | flop | 3440 | 99.91 | 0.00 | 0.09 |
| enabled | check_call | turn | 3440 | 99.65 | 0.00 | 0.35 |
| enabled | check_call | river | 3440 | 99.88 | 0.00 | 0.12 |
| disabled | tight_aggressive | preflop | 2388 | 100.00 | 0.00 | 0.00 |
| disabled | tight_aggressive | flop | 707 | 99.86 | 0.00 | 0.14 |
| disabled | tight_aggressive | turn | 270 | 99.26 | 0.00 | 0.74 |
| disabled | tight_aggressive | river | 123 | 98.37 | 0.00 | 1.63 |
| enabled | tight_aggressive | preflop | 2388 | 100.00 | 0.00 | 0.00 |
| enabled | tight_aggressive | flop | 707 | 99.86 | 0.00 | 0.14 |
| enabled | tight_aggressive | turn | 270 | 99.26 | 0.00 | 0.74 |
| enabled | tight_aggressive | river | 123 | 98.37 | 0.00 | 1.63 |
| disabled | loose_aggressive | preflop | 3542 | 99.92 | 0.00 | 0.08 |
| disabled | loose_aggressive | flop | 1596 | 99.87 | 0.00 | 0.13 |
| disabled | loose_aggressive | turn | 989 | 98.79 | 0.00 | 1.21 |
| disabled | loose_aggressive | river | 713 | 98.32 | 0.00 | 1.68 |
| enabled | loose_aggressive | preflop | 3542 | 99.92 | 0.00 | 0.08 |
| enabled | loose_aggressive | flop | 1596 | 99.87 | 0.00 | 0.13 |
| enabled | loose_aggressive | turn | 989 | 98.79 | 0.00 | 1.21 |
| enabled | loose_aggressive | river | 713 | 98.32 | 0.00 | 1.68 |
| disabled | pot_pressure | preflop | 2797 | 90.60 | 0.00 | 9.40 |
| disabled | pot_pressure | flop | 837 | 69.89 | 0.00 | 30.11 |
| disabled | pot_pressure | turn | 353 | 53.54 | 0.00 | 46.46 |
| disabled | pot_pressure | river | 179 | 40.22 | 0.00 | 59.78 |
| enabled | pot_pressure | preflop | 2795 | 90.66 | 6.33 | 3.01 |
| enabled | pot_pressure | flop | 846 | 68.91 | 26.00 | 5.08 |
| enabled | pot_pressure | turn | 376 | 51.06 | 37.77 | 11.17 |
| enabled | pot_pressure | river | 189 | 39.15 | 41.80 | 19.05 |

| Summed translation distance | Decisions |
|---|---:|
| 0-.05 | 0 |
| .05-.1 | 427 |
| .1-.25 | 80 |
| .25-.5 | 105 |
| .5-1 | 6 |
| >1 | 0 |

Largest observed search: **46 states**. Bound reached: **0 decisions**. Successful witness all-in changes: **0**.

## Lookup latency and resources

Measured latency includes menu/key generation and translation lookup, before the unchanged single sampler draw; measurement consumes no policy RNG. The work bound is deterministic; these elapsed times are observations on this M4, not hard real-time guarantees.

| Option | Opponent | Street | Mean ms | p95 ms | p99 ms | Max ms |
|---|---|---|---:|---:|---:|---:|
| disabled | random | preflop | 0.020 | 0.024 | 0.027 | 0.094 |
| disabled | random | flop | 0.032 | 0.038 | 0.042 | 0.059 |
| disabled | random | turn | 0.045 | 0.054 | 0.057 | 0.067 |
| disabled | random | river | 0.068 | 0.087 | 0.090 | 0.091 |
| enabled | random | preflop | 0.039 | 0.120 | 0.161 | 13.134 |
| enabled | random | flop | 0.061 | 0.210 | 0.267 | 0.375 |
| enabled | random | turn | 0.089 | 0.283 | 0.349 | 0.414 |
| enabled | random | river | 0.110 | 0.335 | 0.417 | 0.584 |
| disabled | check_call | preflop | 0.020 | 0.021 | 0.023 | 0.075 |
| disabled | check_call | flop | 0.033 | 0.036 | 0.038 | 0.073 |
| disabled | check_call | turn | 0.047 | 0.051 | 0.053 | 0.066 |
| disabled | check_call | river | 0.058 | 0.085 | 0.089 | 0.131 |
| enabled | check_call | preflop | 0.020 | 0.021 | 0.023 | 0.084 |
| enabled | check_call | flop | 0.032 | 0.035 | 0.038 | 0.103 |
| enabled | check_call | turn | 0.048 | 0.051 | 0.054 | 0.177 |
| enabled | check_call | river | 0.061 | 0.085 | 0.088 | 0.235 |
| disabled | tight_aggressive | preflop | 0.020 | 0.024 | 0.027 | 0.083 |
| disabled | tight_aggressive | flop | 0.033 | 0.038 | 0.042 | 0.061 |
| disabled | tight_aggressive | turn | 0.045 | 0.053 | 0.055 | 0.058 |
| disabled | tight_aggressive | river | 0.073 | 0.087 | 0.090 | 0.093 |
| enabled | tight_aggressive | preflop | 0.020 | 0.024 | 0.027 | 0.164 |
| enabled | tight_aggressive | flop | 0.033 | 0.037 | 0.040 | 0.056 |
| enabled | tight_aggressive | turn | 0.046 | 0.051 | 0.053 | 0.185 |
| enabled | tight_aggressive | river | 0.076 | 0.086 | 0.160 | 0.225 |
| disabled | loose_aggressive | preflop | 0.021 | 0.024 | 0.027 | 0.200 |
| disabled | loose_aggressive | flop | 0.032 | 0.038 | 0.042 | 0.065 |
| disabled | loose_aggressive | turn | 0.043 | 0.052 | 0.055 | 0.079 |
| disabled | loose_aggressive | river | 0.065 | 0.088 | 0.090 | 0.097 |
| enabled | loose_aggressive | preflop | 0.021 | 0.024 | 0.027 | 0.111 |
| enabled | loose_aggressive | flop | 0.033 | 0.037 | 0.041 | 0.167 |
| enabled | loose_aggressive | turn | 0.045 | 0.052 | 0.057 | 0.237 |
| enabled | loose_aggressive | river | 0.069 | 0.086 | 0.212 | 0.260 |
| disabled | pot_pressure | preflop | 0.020 | 0.022 | 0.026 | 0.076 |
| disabled | pot_pressure | flop | 0.032 | 0.036 | 0.040 | 0.048 |
| disabled | pot_pressure | turn | 0.044 | 0.050 | 0.053 | 0.056 |
| disabled | pot_pressure | river | 0.067 | 0.084 | 0.087 | 0.098 |
| enabled | pot_pressure | preflop | 0.040 | 0.166 | 0.308 | 0.495 |
| enabled | pot_pressure | flop | 0.134 | 0.339 | 0.431 | 26.155 |
| enabled | pot_pressure | turn | 0.196 | 0.485 | 0.587 | 0.659 |
| enabled | pot_pressure | river | 0.314 | 0.654 | 0.753 | 0.801 |

Apple M4, 10 cores, 16 GB, AC. Final whole-family peak **1.537 GiB**, ceiling 3 GiB; largest swap growth **0.00 MiB**, ceiling 256; disk minimum **41.86 GiB**, floor 20; largest system used memory **8.135 GiB**, ceiling 10. All sampled pressure normal and AC present. Every polling iteration refreshed all guards, with 200 ms target cadence and actual timestamps retained.

Corrected timing-only pilot root **2026100820521**, 16 blocks/opponent, full play/replay/reproduction. No pilot outcomes inspected. Measured fixed loads 128.07s, scalable pilot cost 3.456s, projected final 570.47s. Budget posted on PR205 before final play: **1832s**, three times measured projection plus 120s reporting. Final closed in **287.32s** inside the original absolute deadline; no extension.

## Verification and provenance

Frozen scientific source **`809cf2d7d854037d2e7e4e21f037ec2b242024e9`**. Full ZIP/manifest/member SHA256 verification restored #204 average before use. All final deal roots are disjoint from every earlier HU100 baseline/curve/growth schedule and both timing pilots. **61,440 distinct hand evaluations** comprise 40,960 target hands plus 20,480 uniform reference hands reused exactly across options. Complete deterministic reproduction covers all hand events/actions/settlements and target telemetry except elapsed time. Independent engine replay checks every final hand; both options replay the reused reference, for 81,920 audited hand rows.

Forty focused tabular/arena/storage tests pass (two existing tests skipped); independent source review passes 42 tests plus default sampler and freshness assertions. The review corrected a guard cadence mismatch before final play. The first timing-only pilot remains archived separately at source 2cc06e8/root 2026100820511, with no outcomes inspected; corrected pilot alone froze the final. Preparation path/clean-source fixture errors remain in setup notes and logs. No information leak, invalid action, accounting failure, guard breach or final retry occurred.

CI initially could not collect because psutil was installed on M4 but absent from declared development dependencies. A separate M4 integration checkout adds its exact 7.2.2 pin to requirements-monitoring.txt while science remains at its reviewed source. No scientific code or outcomes changed. Full final-head CI and independent evidence review are required before merge.

Research ZIPs contain source snapshots, model restoration and hashes, both pilots, raw hands/decisions, schedules, full repeats/audits, guards, setup notes and reproduction tools. Archive receipts and native/cloud acceptance are linked in [RESULTS_INDEX](../../RESULTS_INDEX.md). Originals remain; no synced file was deleted or evicted. Final metadata/review acceptance is added after sealing; those later receipts do not rerun science or reset its deadline.

For portable reproduction, restore the accepted primary ZIP into a fresh ignored
folder and verify its whole ZIP hash and embedded member manifest. The later
metadata ZIP supplies `tools/reproduce_translation.py`; from a clean checkout
of scientific commit `809cf2d7d854037d2e7e4e21f037ec2b242024e9`, run:

```sh
python /ABS/RESTORED/tools/reproduce_translation.py \
  --restored /ABS/PRIMARY-RESTORED --out /ABS/NEW-REPRODUCTION
```

The helper verifies all required restored model/final members, rebinds only a
fresh config's model path, reproduces both options on the original schedule,
reuses the reproduced uniform reference and audits the new outputs. It preserves
original archived metadata and does not relaunch the one-use campaign. Use
`--check-only` for restoration verification without playing a hand. The full
installed engine, package pins and original deterministic reproduction receipts
are already sealed in the primary ZIP; this portable helper is an additional
restoration convenience, not a new final experiment.

Independent [evidence review](hu100-action-translation-artifacts/evidence-review.json) independently recounts all 81,920 raw hand rows and 313,565 decision rows, paired intervals, telemetry and guards. A bounded-memory scan of all 7,643,261 stored rows verifies all 400 unique translated keys have positive mass, matching menus and exact recorded probabilities. No open findings remain. The same reviewer’s [metadata acceptance addendum](hu100-action-translation-artifacts/evidence-review-metadata.json) clears the main metadata scope; the [final review receipt](hu100-action-translation-artifacts/evidence-review-complete.json) also clears the late receipt appendix. Exact final-head checks remain mandatory. [Integration CI](hu100-action-translation-artifacts/integration-ci.json) is green before the final report commit.

The later [metadata archive acceptance](hu100-action-translation-artifacts/metadata-acceptance.json) confirms the 43-member closeout ZIP is fully locally verified and uploaded with matching independent cloud metadata. Its snapshots precede the self-indexing receipt and final-head CI; later receipts and PR checks are authoritative.

[Complete setup and closeout notes](hu100-action-translation-artifacts/setup-notes.txt) retain the initial acceptance-helper time-module call error, fixed before acceptance, and the portable reproduction command correction. Those later notes are authoritative over the original primary snapshot.

The single independent evidence review is complete with **no open findings**; its final receipt is authoritative over earlier pending fields. Only exact final-head CI and the normal merge workflow remain. No further final play occurred.
