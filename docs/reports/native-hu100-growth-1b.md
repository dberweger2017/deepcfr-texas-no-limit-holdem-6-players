# HU100 growth to 1B: continued loose-aggressive gain

Fresh unchanged-recipe training on the M4 reached **1,000,002,065 nodes /41,010,014 entries**. Terminal minus the exact 39.4M parent improves loose_aggressive **+214.39 [122.28, 306.51] BB/100**; tight_aggressive **+37.05 [−15.00, 89.10]** is inconclusive. These are the two predeclared **97.5%** paired primary intervals. Separately, terminal translation on minus off improves pot_pressure **+62.45 [17.15, 107.76]**, paired **95%**. This is one training seed and a scripted pool, with no general poker-strength or release claim.

[Protocol](../native-hu100-growth-1b.md) · [PR207](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/207) · [training receipt](native-hu100-growth-1b-artifacts/training-result.json) · [paired readout](native-hu100-growth-1b-artifacts/playing-result.json) · [audited model index](native-hu100-growth-1b-artifacts/model-index.json).

**Closeout stopped on a later archive swap-growth guard breach. Science is verified; archive acceptance, end evidence review and merge remain pending.**

## Exact execution and resource admission

All science ran on **Apple M4, 10 cores, 16 GiB**, at ~/Local/hu100-1b-growth-20261008. The M1 only coordinated SSH/GitHub operations. Isolated current-main base 9da625f9b376dcf02420680a2d4cb68bf4bd1b2c; frozen execution source **bd0e7a417064f736091dc2b667954b50becb4b69**. Native runtime SHA256 **7650ad60bbf2437622ea3c39d37c7d56686d00bac11680e44a6e47dab509a262**. No native trainer source, game rules, observation boundary, regret recipe, menu, abstraction or old campaign script changes. Seed 2026100601, one root/seat, linear CFR, opponent-sampled average and uniform missing/zero fallback remain unchanged.

The fresh capacity-stop gate exactly reproduces #204's checkpoint SHA256 **792a675ce6d45d4de8d1b7f3fc6d976548610810f11da8f336c3926f37f8d416**. Its uniform-zero average exactly reproduces #205's pinned 39,438,279-node parent SHA256 **ba62d13536120a9d549f2f3ff84bcb2a96fbd8143ac2fc8477addab368dee0c4**. No Drive parent retrieval was needed. Full parent and every main save's current/average audit passed, checking every stored regret/total and serialized export. Historical increments are not reconstructed by those audits; exact recovery preserves the recipe lineage.

Main training started fresh, without any resume, and saved the first complete iteration at each requested 100M/250M/500M/1B target. Entry cap **57,658,644** came from **110 B/entry +100 MB**, a **6 GiB soft family ceiling**, **8 GiB hard ceiling**, and 2 GiB headroom between them. No doubling allowance. Native exit 0 reached the target; no capacity or soft-stop request. Guards retained normal system pressure, >=15% system-free, <=512 MiB swap growth from the original **563.56 MiB** baseline, **15.5 GiB disk floor** and AC. Sampling target was 200 ms; actual timestamps and kernel command peaks remain archived.

| Requested nodes | Actual nodes | Entries | Save s | Family peak GiB | Forecast GiB | Recent nodes/s excluding save |
| --- | --- | --- | --- | --- | --- | --- |
| 100000000 | 100001959 | 13291612 | 41.63 | 1.092 | 1.455 | 2,161,000 |
| 250000000 | 250000540 | 21552330 | 71.76 | 1.785 | 2.301 | 1,839,386 |
| 500000000 | 500000323 | 30027422 | 103.03 | 2.735 | 3.169 | 1,777,145 |
| 1000000000 | 1000002065 | 41010014 | 144.08 | 3.562 | 4.294 | 1,742,094 |

Main train/save supervision took **15.38 minutes**: native nonsave training 9.26min and four atomic checkpoint writes 6.01min, plus supervision. Peak sampled whole-family RSS **3.563 GiB**, kernel command peak **3.530 GiB**. Every save stayed below its memory forecast. Peak bytes/entry were 88.2 /88.9 /97.8 /93.3, rather than assuming the earlier 87–93 range held exactly. Terminal entries are 9.2% below the 45.16M extrapolation; terminal checkpoint compression is **43.63 B/entry**, with measured bytes indexed.

Streaming export/full audit costs at 100M/250M/500M/1B were respectively **136.2/297.4s**, **222.0/483.6s**, **309.6/672.2s**, **433.6/923.0s**. Total **57.96min**, below the quoted 120.08min. Closed scientific operations all passed; peak across gates/pilots/training/evaluation was **4.467 GiB**, maximum swap growth **0.5 MiB**, minimum system-free **71%**, normal pressure/AC throughout. The minimum disk over preparation and science was **32.99 GiB**; final-stage minimum was 49.57 GiB before archive creation. [Resource aggregates](native-hu100-growth-1b-artifacts/resources.json) bind full archived sample streams.

## Fixed paired evaluation

A 16-block/opponent timing-only pilot (root **2026100810011**) inspected no winnings and froze **2,048 independent blocks/opponent/policy**. Final root **2026100810012** was checked against every earlier #197/#200/#203/#204/#205 schedule and the current pilot. Frozen schedule SHA256 **c20877b2ef46ff49b52055931bbb5d225472985c63a175a2464d497faed369cf**; final freeze SHA256 **4aeba06df0fa2987cd0a2a796fda7f721f31bf591b1348fb8e739e1c9f27b0c5**.

The prospective protocol and measured quote were posted before final play: **5h19 upper campaign budget**, including training 42.50min, saves 10.86min, exports/audits 120.08min, play/replay/reproduction 135.05min and 10min closeout reserve. Final-stage budget froze **8,703s** (measured play allowance plus reporting reserve); actual final play through strict reporting took **31.48min**, 21:11:36–21:43:05 UTC. The original storage forecast incorrectly block-scaled fixed snapshots and admitted no training; the corrected forecast required **58.02 GiB free**, including retained originals and ZIP copies above the disk floor. Renewed pre-launch admission had **72.3 GiB**, and the owner explicitly said **“Start now”**. No inherited 30-minute cap, arbitrary training cutoff, extension, pooling or checkpoint selection.

Heads-up 100-BB stacks reset each hand, blinds 50/100, no rake; five unchanged scripted opponents; both swapped seats per block; paired private action streams across policies; the same uniform reference reused exactly. Translation stayed off for the full five-average learning curve. Every hand/action/settlement independently replayed, every hand/probability/classification deterministically reproduced, excluding only measured latency. Clean frozen-source manifests, model hashes, streams, schedules, menus, keys and accounting all pass the normal strict tools.

There are **143,360 distinct final hand evaluations**, including the six candidate policy panels and one uniform reference. Stored replay rows, including reference copies, are **245,760**, with **957,490 actions** (off curve: 795,608; translated panel: 161,882); reproduction covers them all. There were no failed final hands or reruns.

| Opponent | Parent BB/100 | Terminal BB/100 | Terminal minus parent | Scope |
| --- | --- | --- | --- | --- |
| random | +90.59 [-11.47, +192.65] | +140.09 [+38.58, +241.60] | +49.50 [-5.26, +104.26] | 95% descriptive |
| check_call | +96.48 [+69.06, +123.91] | +110.27 [+81.14, +139.39] | +13.78 [-18.93, +46.49] | 95% descriptive |
| tight_aggressive | -12.89 [-56.28, +30.50] | +24.16 [-2.45, +50.76] | +37.05 [-15.00, +89.10] | 97.5% primary: inconclusive |
| loose_aggressive | -147.35 [-223.65, -71.05] | +67.04 [+13.21, +120.87] | +214.39 [+122.28, +306.51] | 97.5% primary: improvement |
| pot_pressure | -101.23 [-157.15, -45.32] | -83.96 [-134.91, -33.01] | +17.27 [-35.52, +70.07] | 95% descriptive |

The only formal primary family is terminal-minus-parent tight/loose: two-sided paired Student-t **97.5%** intervals, Bonferroni familywise alpha .05 for two tests; improvement needs lower >0. Absolute policy and other contrasts have descriptive ordinary 95% intervals. Terminal loose is profitable on this fixed scripted sample, **+67.04 [13.21, 120.87] BB/100**; tight profitability remains inconclusive. Translation-off pot_pressure still loses, **−83.96 [−134.91, −33.01]**.

### Full learning curve, descriptive

| Opponent | 39,438,279 nodes | 100,001,959 nodes | 250,000,540 nodes | 500,000,323 nodes | 1,000,002,065 nodes |
| --- | --- | --- | --- | --- | --- |
| random | +90.59 | +123.86 | +150.20 | +150.59 | +140.09 |
| check_call | +96.48 | +86.63 | +100.45 | +117.61 | +110.27 |
| tight_aggressive | -12.89 | -5.18 | +18.14 | +33.26 | +24.16 |
| loose_aggressive | -147.35 | -66.43 | +17.99 | +83.45 | +67.04 |
| pot_pressure | -101.23 | -90.48 | -91.16 | -86.22 | -83.96 |



The 500M point estimate is higher than 1B against both primaries; that does not select a checkpoint or establish a decline. The terminal was fixed prospectively, and all milestone results are retained.

### Separately predeclared translation secondary

Terminal translation on minus off against pot_pressure is **+62.45 [17.15, 107.76] BB/100**, ordinary paired **95%**, outside the two-test primary family. #205 default TranslationOptions remain unchanged (512 states, 128 events, one private draw). Translation-on absolute pot result is **−21.51 [−67.60, 24.58]**, so profitability is inconclusive despite the paired improvement. Translation stays off by default; no production policy adoption or release follows this campaign. Other translated panels are controls/descriptive.

## Coverage and visit bands

| Actual nodes | Entries | Zero traverser visits % | Positive average keys % | Mean traverser visits/key |
| --- | --- | --- | --- | --- |
| 39438279 | 7643261 | 52.22 | 66.80 | 1.035 |
| 100001959 | 13291612 | 48.49 | 67.14 | 1.515 |
| 250000540 | 21552330 | 44.63 | 67.53 | 2.334 |
| 500000323 | 30027422 | 41.63 | 67.67 | 3.335 |
| 1000002065 | 41010014 | 38.70 | 67.78 | 4.847 |


Stored zero-visit share falls 52.22%→38.70%, while table size grows 5.37x and mean traverser visits/key grows 1.035→4.847. The table still contains many unvisited keys.

Reached target-decision coverage is policy-dependent, descriptive, and includes all streets and zero/missing decisions in its denominator:

| Opponent | Positive-mass % parent→terminal | Missing parent→terminal | Zero-mass parent→terminal | Decisions parent→terminal |
| --- | --- | --- | --- | --- |
| random | 82.90→83.16 | 1031→1055 | 2→0 | 6040→6264 |
| check_call | 99.88→100.00 | 12→0 | 6→0 | 14416→14797 |
| tight_aggressive | 99.66→100.00 | 8→0 | 4→0 | 3532→3914 |
| loose_aggressive | 99.42→99.99 | 31→1 | 9→0 | 6925→7432 |
| pot_pressure | 80.91→78.58 | 819→1012 | 3→1 | 4306→4729 |

Known decisions with fewer than ten traverser visits, including known zero-mass keys:

| Opponent | Preflop % parent→terminal | Flop % | Turn % | River % |
| --- | --- | --- | --- | --- |
| tight_aggressive | 1.67→0.00 | 4.60→0.12 | 17.23→1.05 | 33.33→3.10 |
| loose_aggressive | 4.22→0.00 | 5.07→0.40 | 25.75→2.05 | 37.69→3.42 |


Reached known late-street decisions become much denser. Pot missing/zero coverage remains substantial and is descriptively worse along the changed trajectory; more visits do not demonstrate repair of unsupported public histories. #201 diagnosed off-menu support limits on its earlier sample, but this campaign does not causally relabel every new missing key. Full bands by street/opponent/checkpoint and policy-dependent payoff exposures are in the compact readout; they cannot assign causal branch gains.

## Review, retained failures and limits

Qualification passed **70 Python tests**, **11 Rust tests**, the release build and staged repository artifact guard. One independent source review cleared the exact execution source before final play. End evidence review and green final-head CI are still required before merge. The archive blocker leaves that review pending and this PR unmerged.

Preserved preparation failures: duplicate prior roots refused setup before pilot hands; an initial excessive storage forecast scaled fixed model snapshots by blocks; locked HTTPS keychain required the existing SSH push path. Corrected and successful receipts remain separately identifiable. No numerical/exactness/information/action/accounting error, scientific-phase guard breach, failed final hand or scientific retry occurred. The later archive-only guard breach below stops closeout.

This is one fixed seed and five scripted opponents, without an owner-defined external benchmark, independent training seeds, exploitability certificate or general-strength confirmation. The remaining unvisited keys, existing abstraction and unsupported histories remain limits. There is no recipe/menu/abstraction change, automatic follow-on training, paid compute, release or tag.

## Archive guard breach: closeout stopped

At **21:47:00 UTC**, direct Research-Cloud ZIP creation recorded **829.38 MiB swap growth**, over the unchanged **512 MiB** limit. Family RSS was only 43.61 MiB, pressure normal, free-memory 86%, AC connected and free disk 45.42 GiB; these do not waive the guard or establish its cause. The runner latched campaign-failure.json and stopped the packer. Later SIGKILL cleanup raised PermissionError, preventing the normal archive receipt from being finalized; the exception, admission and raw samples remain. No active campaign/packer process remains. [Stop receipt](native-hu100-growth-1b-artifacts/archive-stop-readback.json) · [PR issue and SOMA](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/207#issuecomment-6069695396).

The **3,354,661,317-byte partial ZIP** remains untouched at ~/Local/Research-Cloud/PR-207-hu100-1b/hu100-1b-campaign-M4-20261008.zip in the [existing folder](https://drive.google.com/drive/folders/1qhlOHmphBGSFfiM82S7T4B_KhdabyRUS). It has **no accepted final manifest, integrity readback or cloud acceptance** and must not be used for restoration. Every nonsynced original, scientific receipt, source/runtime snapshot and raw hand/reproduction remains in the isolated M4 work root. No archive retry, guard/baseline relaxation, end evidence review or merge follows this failed admission.

SOMA: prepare a separately reviewed archive-only readmission from the retained failure evidence, preserving the original guard/baseline, partial ZIP and unchanged scientific outputs. This does not rerun or extend science. [RESULTS_INDEX](../../RESULTS_INDEX.md) records original model paths/sizes/hashes and fresh-path retrieval commands; archive member names are planned only. Later compact receipts remain in Git, with no extra metadata seals. No other PR files, synced deletions or forced offloading.
