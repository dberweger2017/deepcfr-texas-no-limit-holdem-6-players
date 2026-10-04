# HU20 turn search — completed thirty-second calibration

**No configuration qualifies under either the strict or relaxed thirty-second gate.** All 1,728 final coordinates completed: 48 original roots × three B500M average lineages × both bot seats × six configurations. No final coordinate or reference was excluded. Average remains the Part A base. The heartbeat is PAUSED, and no replacement worker, sampled-river phase or paid arena has started.

The [owner-approved thirty-second-only scope](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/148#issuecomment-5970668869) requires stopping and asking if no qualifier results. A 120-second campaign requires separate approval; the owner’s preference for future parallel RunPod work does not authorize paid compute.

## Complete native curve

All final settings use the timing-selected native menu, six threads and uncompressed storage. Quality measures the exported turn strategy with internal river continuation against full-native deviations under #145’s original range law. Weights are the original reference reach weights. Cold latency includes preparation/parsing and reconstructed native duration for exact shared requests; separate quality solves consume the journal but do not enter play latency.

| Iterations | Opponent floor | Cold median / p95 / p99 / max (s) | Weighted residual mean / p95 (% pot) | Timeout fallback | Strict / relaxed |
| --- | --- | --- | --- | --- | --- |
| 25 | 0 | 4.706 / 9.334 / 12.589 / 12.699 | 7.2876 / 15.8243 | 0/288 (0.00%) | Fail / fail |
| 25 | 0.01 | 4.897 / 9.677 / 12.897 / 13.156 | 7.2502 / 15.7097 | 0/288 (0.00%) | Fail / fail |
| 50 | 0 | 8.953 / 18.023 / 24.509 / 24.792 | 1.3905 / 2.9293 | 0/288 (0.00%) | Fail / fail |
| 50 | 0.01 | 9.480 / 18.942 / 25.294 / 26.020 | 1.4360 / 2.8444 | 0/288 (0.00%) | Fail / fail |
| 100 | 0 | 17.347 / 30.036 / 30.070 / 30.160 | 8.2252 / 62.2773 | 32/288 (11.11%) | Fail / fail |
| 100 | 0.01 | 17.648 / 30.040 / 30.081 / 30.220 | 8.3657 / 62.2773 | 35/288 (12.15%) | Fail / fail |

Strict requires cold p95 ≤30 seconds and residual mean/p95 ≤0.5% pot. Relaxed requires mean ≤1%, p95 ≤2%, and residual below 10% of blueprint loss on every eligible coordinate. At 50 iterations every coordinate meets the relative blueprint condition, but aggregate mean and p95 exceed the relaxed quality limits. At 25 iterations both aggregate quality and many relative conditions fail. At 100 iterations, 32/288 epsilon-zero and 35/288 epsilon-0.01 decisions time out; cold p95 exceeds 30 seconds. The quality of the legal average fallback is measured from the original full-native reference, rather than omitted or treated as zero loss.

## Range-law sensitivity and support

These sensitivity aggregates cover completed solves only, with timeout coordinates explicitly missing. They cannot qualify a configuration or replace the complete reference-law curve above.

| Iterations | Floor | Completed / missing search-law coordinates | Search-law mean / p95 (% pot) | Completed-only reference mean / p95 (% pot) |
| --- | --- | --- | --- | --- |
| 25 | 0 | 288 / 0 | 7.2876 / 15.8243 | 7.2876 / 15.8243 |
| 25 | 0.01 | 288 / 0 | 7.2694 / 15.6785 | 7.2502 / 15.7097 |
| 50 | 0 | 288 / 0 | 1.3905 / 2.9293 | 1.3905 / 2.9293 |
| 50 | 0.01 | 288 / 0 | 1.4166 / 2.8470 | 1.4360 / 2.8444 |
| 100 | 0 | 256 / 32 | 0.3437 / 0.6995 | 0.3437 / 0.6995 |
| 100 | 0.01 | 253 / 35 | 0.3741 / 0.7214 | 0.4178 / 0.8889 |

Every final reference retains its full mass within the frozen 1e-6 tolerance. No missing reference, whole-range zero-support fallback, unsupported holding, memory refusal, invalid output or turn-conditioning gap occurred in final rows. All 67 final play failures are timeout fallbacks; final quality requests have no recorded failure. These roots have no prior turn actions, so zero conditioning gaps here do not substitute for the separate live river validation.

Coverage remains cause-specific. Missing blueprint keys and zero-average-mass uniform handling are distinct. Zero own/opponent likelihood counts below count observed holding/action factors, not whole-range failures; opponent flooring can restore opponent factors while leaving own likelihoods and structural card incompatibilities intact. Counts repeat across the frozen root/lineage/seat coordinates and are not counts of unique hands. Timeout preparation may not return coverage, so its unreported factors remain unmeasured.

| Iterations | Floor | Missing keys | Zero average mass | Zero likelihood own / opponent | Floored opponent factors |
| --- | --- | --- | --- | --- | --- |
| 25 | 0 | 48 | 1562 | 916 / 916 | 0 |
| 25 | 0.01 | 48 | 1562 | 916 / 916 | 102477 |
| 50 | 0 | 48 | 1562 | 916 / 916 | 0 |
| 50 | 0.01 | 48 | 1562 | 916 / 916 | 102477 |
| 100 | 0 | 48 | 1562 | 897 / 900 | 0 |
| 100 | 0.01 | 48 | 1562 | 885 / 894 | 95056 |

Whole-range support failures are 0/1,728 (0%). Some compatible individual holdings retain zero posterior mass under the frozen action factors; this is separate from whole-range failure. Counts below cover coordinates with returned range telemetry. Live holding-query fallback rates remain unmeasured until the arena. Structural board-card incompatibilities are removed before the 1,128 compatible holdings per seat and are never revived by flooring.

| Iterations | Floor | Compatible holdings with zero posterior mass: seat 0 / seat 1 |
| --- | --- | --- |
| 25 | 0 | 896 / 936 |
| 25 | 0.01 | 448 / 468 |
| 50 | 0 | 896 / 936 |
| 50 | 0.01 | 448 / 468 |
| 100 | 0 | 881 / 916 |
| 100 | 0.01 | 433 / 452 |

## Full screening evidence, reuse and verification

The full [1,966-row curve](hu20-turn-search-artifacts/calibration-03-full-curve.jsonl) publishes every final root, lineage, seat, residual, range law, coverage, receipt, failure and original reference provenance, plus the 238 hash-bound retained screen rows. The [compact result and independent arithmetic proof](hu20-turn-search-artifacts/calibration-03-result.json) include all six aggregate rows and exact gate decisions. The retained screen covers 128 native and 110 reduced-menu coordinates across 1/2/4/6 threads, compression and both floors; eighteen reduced coordinates remain unattempted. All 64 retained screen play failures are timeouts, and 110 reduced rows lack full-native quality/extra speculative LBR costing, so no reduced setting is qualified. The original [screen and timing forecast](hu20-turn-search-calibration-02-budget.md) remain separate, with no calibration-01 timing stitching.

There are 2,557 native receipts and 832 shared receipts (play and quality) from successful exactly identical unlocked epsilon-zero requests. Epsilon=0.01 and failed requests are separate. Completed play versus quality requests have maximum observed full-matrix absolute difference zero. Conservative reconstructed cold latency, actual wall time and retained native receipt duration are all published; cached wall time does not establish cold performance.

The native attempt consumed 29549.600 charged seconds (about 8.208 hours), with peak owned family RSS 5.877 GiB below its 8-GiB admission and no resource guard stop. Worker, native children and sidecar exited. Swap fell from 859.75 MiB to 843.75 MiB. Verification and retrieval are separately charged. Final cumulative M4 journal use is **42294.351 seconds (11.748 hours)**; **12.252 hours remain** of the original 24. The three-hour river reserve remains unconsumed.

All 15,292 raw manifest members / 29,486,803,777 bytes passed independent size/SHA checking on M4. Original raw records remain at `/Users/dberweger/Local/hu20-turn-search-20261003/calibration-03`. The complete local copy at `/Users/dberweger/Local/hu20-turn-search-20261002/m4-calibration-03-complete` independently passed exact size/streaming SHA256 checks for **all 15,292 members**. [Retrieval/verification and final clock proof](hu20-turn-search-artifacts/calibration-03-retrieval-verification.json) retain both checks, conservative full-wall-time charges and TensorBoard hashes. No local or M4 original was deleted. The manifest SHA256 is `dd39458bbd629ab0ca3416506a48b8210081e20cf5e9c9f0b07c25850ff3015a`. Source 93d55eb, immutable settings, source CI, original references and external binary are unchanged. AGPL solver/harness remain outside MIT; the original #145 binary and earlier failed attempts remain preserved.

## Required owner decision

Recommend preparing a separate native 120-second RunPod calibration proposal and measured quote, with all roots/lineages/seats and strict quality unchanged. The successful 100-iteration subset is descriptive only; timeout roots must be measured before any research qualifier can be selected. A new deadline or paid campaign needs explicit approval and a prospective freeze. No research configuration is currently selected. River validation and the 82,944-hand arena quote await a qualifying search configuration; arena counts remain unchanged. Paid arena production still requires its own approved quote and actual-pod macOS/x86_64 Linux parity.
