# Sampled Hold’em pilot

**All 12 training jobs complete; all 24 checkpoint hashes and all 12 final evaluations verify.** This supports proceeding to a longer exploratory run for v0.5. Playing strength remains unproven: all twelve final paired comparisons are inconclusive.

## Protocol and scope

The [protocol](../holdem-sampled-training.md) and [plan](../../configs/holdem/sampled-pilot.json) were frozen with implementation commit `f0ee7c4b1fe2760977ae7cc1d2268376e8486337` before measurements. Three seeds (211, 223, 227), four table configurations, six iterations, four roots per role per iteration, width 32, sixteen fitting steps, and fixed evaluations at iterations 3 and 6. No settings changed after results. No failures, retries, unattempted training jobs, checkpoint selection or model promotion.

This is a short pipeline/resource pilot. It is not the longer learning experiment required for v0.5 and does not close milestone 4’s learning exit. The previous incomplete external-sampling report remains incomplete; this uses fresh seeds and a different loss normalization, so it is not a paired sampler-strength comparison.

## Training and resource results

| Measure | Result |
| --- | --- |
| Completed jobs / iterations | 12 / 72 |
| Scheduled traverser roots / terminal continuations | 1,512 / 7,729 |
| Visited nodes / admitted decision records | 57,223 / 7,106 |
| Retained replay | All 7,106 records; this pilot did not reach its per-role capacity. Separate recovery tests force reservoir replacement. |
| Invalid evaluation actions / failed hands | 0 / 0 |
| Initial evaluation hands, both arms and checkpoints | 7,560 (3,780 at the final checkpoint) |
| Largest inverse own sampling reach | 4,295.29 |
| Largest absolute importance-corrected regret update | 867,255.26 BB (four players, seed 211) |
| Largest pre-clipping gradient norm | 163,991,232 |
| Clipped optimizer steps | 6,048 / 6,048 |
| Experiment runtime | 165.15 seconds; 167.35 seconds including process startup/shutdown |
| Evaluation time within the experiment | 21.48 seconds |
| Peak process RSS | 457,474,048 bytes (436.28 MiB) |
| Uncompressed experiment artifacts | 310,756,128 bytes |
| Rental cost | $0; $7.33 conservative CPU authorization remains |

Hardware: Apple M1, 16 GiB RAM, deterministic CPU Torch with one thread. The remaining 143.67 seconds include collection, fitting, replay validation, checkpoint serialization and other overhead; they are not an isolated fitting benchmark. Archive and replay costs will grow in longer runs, so extrapolation from six iterations is only a starting estimate.

Every fit and model remained finite. The gradient norm limit of 1 was active at every optimizer step. Correct expected gradients before clipping do not guarantee equivalent optimizer behavior after clipping. Large updates are estimator values, not physical hand payoffs. A longer run should record these tails and clipping alongside held-out poker results; neither the raw regression loss nor this finite run establishes convergence.

## Poker results

Each checkpoint uses 30 independent deal blocks, rotates every starting seat and compares snapshot-average play with uniform-candidate play against the same style pool. The table reports all checkpoints. Confidence intervals below are nominal paired 95% intervals over deal blocks; checkpoints share the schedule, and these are not independent confirmations or multiplicity-adjusted campaign claims. Raw per-hand outcomes, all intervals and per-role fitting reports are retained.

| Scenario | Seed | Iteration | Candidate BB/100 | Difference vs uniform | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| four-100bb | 211 | 3 | -247.08 | -242.50 | [-1027.90, 542.90] |
| four-100bb | 211 | 6 | -207.08 | -202.50 | [-1006.07, 601.07] |
| four-100bb | 223 | 3 | 93.75 | 98.33 | [-683.57, 880.24] |
| four-100bb | 223 | 6 | -115.00 | -110.42 | [-774.65, 553.82] |
| four-100bb | 227 | 3 | -295.83 | -291.25 | [-931.62, 349.12] |
| four-100bb | 227 | 6 | -800.00 | -795.42 | [-1615.27, 24.44] |
| five-100bb | 211 | 3 | -356.33 | 335.00 | [-599.27, 1269.27] |
| five-100bb | 211 | 6 | -659.33 | 32.00 | [-739.54, 803.54] |
| five-100bb | 223 | 3 | -1108.33 | -417.00 | [-1093.22, 259.22] |
| five-100bb | 223 | 6 | -830.67 | -139.33 | [-911.85, 633.18] |
| five-100bb | 227 | 3 | -735.33 | -44.00 | [-729.46, 641.46] |
| five-100bb | 227 | 6 | -644.33 | 47.00 | [-595.72, 689.72] |
| six-100bb | 211 | 3 | -230.28 | 314.17 | [-343.64, 971.97] |
| six-100bb | 211 | 6 | -629.72 | -85.28 | [-863.46, 692.91] |
| six-100bb | 223 | 3 | -176.94 | 367.50 | [-407.28, 1142.28] |
| six-100bb | 223 | 6 | -563.06 | -18.61 | [-872.17, 834.94] |
| six-100bb | 227 | 3 | -607.22 | -62.78 | [-855.25, 729.69] |
| six-100bb | 227 | 6 | -756.39 | -211.94 | [-866.23, 442.34] |
| six-unequal | 211 | 3 | -681.67 | 348.06 | [9.90, 686.21] |
| six-unequal | 211 | 6 | -834.17 | 195.56 | [-353.12, 744.23] |
| six-unequal | 223 | 3 | -742.22 | 287.50 | [-232.36, 807.36] |
| six-unequal | 223 | 6 | -1319.44 | -289.72 | [-820.40, 240.96] |
| six-unequal | 227 | 3 | -750.83 | 278.89 | [-201.94, 759.72] |
| six-unequal | 227 | 6 | -793.06 | 236.67 | [-271.52, 744.85] |

All final candidate point estimates are negative against this style pool, and four of twelve final paired differences are positive. The intervals are wide. These results give no basis for calling the model strong or choosing a winning seed. They also do not establish that this recipe cannot improve with more data and fitting.

## Verification and artifacts

553 tests pass, including exact mixed-iteration/reservoir gradients, sparse and empty roots, sampled checkpoint validation, whole-iteration rollback, fresh-process byte-identical recovery, and CLI reproduction of training plus arena outcomes. Changed files pass Ruff and formatting. Repository-wide Ruff also reports two existing issues in unchanged `scripts/__init__.py` and `tests/test_pokers_regressions.py`; they are outside this change.

All 24 saved training checkpoints and corresponding inference exports were loaded and hash-checked. Every final export and hand outcome was reproduced exactly after recovery, with no new fitting, in 50.38 seconds. The first verification helper stopped after comparing an in-memory tuple with its JSON list representation; its serialized evaluation and outcomes already matched byte for byte. The helper was corrected, that initial attempt and its outputs were retained, and verification completed within the original total 600-second ceiling. No training job was rerun.

The manifest marks the workspace dirty because the unrelated untracked `.claude/` directory remained present. All training implementation and configuration changes were committed before measurements; the manifest also pins source bytes and the installed environment.

The local retained archive is `results/sampled-pilot.tar.gz` (138,909,769 bytes), SHA-256:

```text
4fa85d4a1a648239498ff8b56027e5e3c9aabf2c04cea112f532ea035f07e716
```

It contains all checkpoints, policy exports, per-hand outcomes, manifests, reports, test logs, timing logs, verification outputs (including the initial helper failure), and reproduction helpers. Large model files remain outside Git. The [compact JSON report](holdem-sampled-pilot.json) retains every iteration’s diagnostics, every evaluation summary and artifact hashes. To inspect the retained run, extract the archive into a separate directory. To repeat training, check out the source revision and use a fresh output directory with the protocol command.

## Decision and next experiment

**Move to longer Hold’em training.** The next experiment should use one fixed recipe and three independent seeds in the primary six-handed 100 BB game, collect more roots per iteration, retain periodic checkpoints, and evaluate on a fixed validation schedule. Measure collection/fitting time, replay/archive memory and throughput during that run. Keep four/five-player and unequal-stack evaluations explicit as coverage expands.

Set a fixed runtime and spending cap from measured throughput before rental; the remaining CPU authorization is available, and GPU funding remains separate. Do not require a positive win rate to start or continue within the declared budget. Stop on invalid play, non-finite state, recovery failure or resource limits. Publish weak results and failures. Further sampler or architecture comparisons should answer a concrete problem observed in this longer run rather than becoming another prerequisite.

The release sequence is v0.5 (reproducible research workflow and longer training evidence), then v1.0 (the documented lower-end professional qualification standard). This pilot supplies the implementation/resource evidence for proceeding, not either release by itself.
