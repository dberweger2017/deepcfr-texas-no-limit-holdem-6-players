# Trained global K50 bench: stopped before final scoring

**Unclassified, partial evidence.** All four v1/K50 fold trajectories reached 10M iterations, all 1M/3M timing-prefix files reproduced byte for byte, and all 40 label adapters completed. The scoring pilot stopped after 3.24 seconds when its process-family supervisor raised `psutil.AccessDenied(pid=11563)`. No complete scoring timing quote, final 40-root evaluation, E/Q curve or primary contrast exists. There was no scientific retry.

The [prospective protocol](../hu20-equity-bench.md) fixed the primary before any scoring: matched K50 opponent-sampled loss minus v1 at 3M must have a 95% upper bound below −0.10 BB and a negative 10M point contrast. No pass/fail/inconclusive classification is assigned without final scores. Abstraction step 3 remains unresolved; the #190 witness E = 0.3917 has no trained-policy comparison here.

The pilot passed the lock tool’s V1 exactness gate with no mismatches, then was terminated. Cleanup reported no surviving owned children. PID 11563 had exited before diagnosis, so its identity is unknown. The installed psutil parent discovery scans every host PID and can raise on unrelated protected processes. The subsequent infrastructure correction derives ancestry from `ps` before querying only creation-verified owned identities, retains children of reparented workers, and still fails closed on unreadable owned processes. It is used only for archival closeout and independent Stage B; A scoring remains stopped.

## Training and matched visits

The same 40 #149 roots and two frozen halves use B500M seed 2026093001 ranges, native seed 202610050001 and the unchanged recipe, menu and history. Pinned global K50 tables match every native SHA256. The 3M pilots fixed K50 checkpoints 4,000,977 and 3,855,889 before any scoring. No later adjustment was made.

| Fold | Export | Keys | Total visits | Mean | p0 | p10 | p25 | p50 | p75 | p90 | p99 | max |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | iteration-1000000/equity.current | 90,386 | 12,791,926 | 141.53 | 0.00 | 1.00 | 2.00 | 11.00 | 49.00 | 186.00 | 1951.15 | 50051.00 |
| 0 | iteration-1000000/v1.current | 65,469 | 12,843,007 | 196.17 | 0.00 | 0.00 | 1.00 | 7.00 | 47.00 | 222.00 | 2706.32 | 154376.00 |
| 0 | iteration-3000000/equity.current | 102,245 | 37,236,679 | 364.19 | 0.00 | 1.00 | 3.00 | 18.00 | 97.00 | 428.00 | 4891.52 | 149432.00 |
| 0 | iteration-3000000/v1.current | 77,055 | 37,426,084 | 485.71 | 0.00 | 0.00 | 2.00 | 12.00 | 88.00 | 458.00 | 6632.28 | 480179.00 |
| 0 | iteration-4000977/equity.current | 104,534 | 49,594,539 | 474.43 | 0.00 | 1.00 | 3.00 | 21.00 | 115.00 | 549.00 | 6485.01 | 199555.00 |
| 0 | iteration-10000000/equity.current | 115,212 | 123,601,833 | 1072.82 | 0.00 | 1.00 | 5.00 | 32.00 | 196.00 | 1147.90 | 15371.36 | 498853.00 |
| 0 | iteration-10000000/v1.current | 87,851 | 123,724,502 | 1408.34 | 0.00 | 0.00 | 2.00 | 24.00 | 212.00 | 1198.00 | 19566.00 | 1641451.00 |

Fold 0 actual matched K50 mean is 474.435 versus v1 at 3M, 485.706: a −2.321% residual.

| Fold | Export | Keys | Total visits | Mean | p0 | p10 | p25 | p50 | p75 | p90 | p99 | max |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | iteration-1000000/equity.current | 85,786 | 12,839,782 | 149.67 | 0.00 | 1.00 | 2.00 | 10.00 | 47.00 | 190.00 | 2144.15 | 70161.00 |
| 1 | iteration-1000000/v1.current | 64,164 | 12,937,413 | 201.63 | 0.00 | 0.00 | 1.00 | 6.00 | 44.00 | 219.00 | 2773.00 | 127781.00 |
| 1 | iteration-3000000/equity.current | 97,460 | 38,133,158 | 391.27 | 0.00 | 1.00 | 3.00 | 17.00 | 90.00 | 441.00 | 5719.53 | 215046.00 |
| 1 | iteration-3000000/v1.current | 74,317 | 37,373,845 | 502.90 | 0.00 | 0.00 | 1.00 | 10.00 | 82.00 | 460.00 | 7143.36 | 383901.00 |
| 1 | iteration-3855889/equity.current | 99,574 | 49,120,546 | 493.31 | 0.00 | 1.00 | 3.00 | 19.00 | 110.00 | 547.00 | 7385.05 | 273432.00 |
| 1 | iteration-10000000/equity.current | 111,004 | 126,747,136 | 1141.82 | 0.00 | 1.00 | 5.00 | 30.00 | 189.00 | 1141.00 | 18219.07 | 714028.00 |
| 1 | iteration-10000000/v1.current | 85,682 | 123,836,862 | 1445.31 | 0.00 | 0.00 | 2.00 | 20.00 | 189.00 | 1173.00 | 20443.31 | 1277163.00 |

Fold 1 actual matched K50 mean is 493.307 versus v1 at 3M, 502.898: a −1.907% residual.

All zero/1–9/10–99/100–999/1000+ bands are in [visits.json](hu20-equity-bench-artifacts/visits.json). Native 10M operations took 614.41/607.02 seconds for v1 and 681.01/688.82 seconds for K50. The 3M pilots took 184.21/181.81 and 207.01/206.78 seconds (about 16.3k/14.5k iterations/s). Scoring cannot be quoted from an interrupted 3.24-second pilot.

## Guards, earlier preparation failures and restoration

All 23,714 retained samples satisfy resource thresholds: peak family RSS 3.794 GiB, maximum total swap 1.085 GB, minimum system free 74% and minimum disk free 52.365 GiB. The scoring failure is loss of process visibility; no measured threshold breach occurred. Original baselines are byte-identical. The 7/9 GiB family, total 3 GB swap, normal pressure/15% free, 16 GiB disk and AC guards were unchanged.

Two prior preparation failures are retained separately: a standalone helper import failed before archive access, then a manifest path omitted the ZIP’s top directory. The pinned #169 archive SHA passed; corrected restoration verified all 120 selected members’ full sizes/SHA256 before keeping 80 atomic B/L/P results and 40 exact equilibrium completion records. No native trajectory/scoring was retried. Those failures, original latches, readmissions and partial solver outputs are part of the ZIP.

Native trainer source is unchanged from main 3ca6f81c064595b27e5c1fcaa665769f7c5b4ba5. Executed scoring source is d06105a with later metadata-only commits until the stop; archival guard correction is recorded separately. Source-code snapshots omit grandfathered research payloads while retaining their exact Git-revision/blob restoration. All models/raw outputs are ignored; originals and full preparation source archives remain local. The member-hashed partial ZIP, independent cloud acceptance and restoration commands are indexed in [RESULTS_INDEX](../../RESULTS_INDEX.md). No default/recipe/release/tag/publication, paid compute or other-PR cleanup occurred. Independent [Stage B](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/223) proceeded under its own protocol.

## Validation and independent evidence review

The single combined [end evidence review](hu20-equity-bench-artifacts/evidence-review.json) is clear, with no correctness findings open. It independently reproduces the statistics, resource extrema, coverage, restoration locators and archive acceptance. Scientific operations were not rerun. Merge requires every final-head host check to pass.
