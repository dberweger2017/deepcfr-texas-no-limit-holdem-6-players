# HU20 cap A/B: all six training runs completed

**Training is finished; evaluation and the final audit are still running.** Snapshot captured at **2026-09-28T21:35:28.093365+00:00**. The three fixed paired seeds each completed fresh capped A and native-reopening B training at the frozen 20M-node budget. The last run finished at **2026-09-28T21:12:25.490842+00:00**, approximately **23:12 Madrid**. No training attempt failed or was retried, and all 24 fixed milestone checkpoint/export pairs were recorded. Completed work totals **120,001,431 nodes**, including **1,431 nodes** of whole-iteration overshoot.

These are resource and representation measurements. No playing-strength improvement or LBR non-inferiority is claimed. The fixed final-current C policy remains the evaluated candidate for every seed.

## Resource and work measurements

| Arm / seed | Complete nodes | Overshoot | Entries | Iterations | Minutes | Peak RSS GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A / 2026093001 | 20,000,188 | 188 | 270,548 | 66,168 | 20.79 | 0.367 |
| B / 2026093001 | 20,000,557 | 557 | 730,010 | 55,291 | 21.70 | 0.991 |
| A / 2026093002 | 20,000,094 | 94 | 270,949 | 65,011 | 20.93 | 0.367 |
| B / 2026093002 | 20,000,238 | 238 | 729,165 | 54,709 | 21.57 | 0.968 |
| A / 2026093003 | 20,000,086 | 86 | 270,774 | 69,118 | 20.67 | 0.367 |
| B / 2026093003 | 20,000,268 | 268 | 723,259 | 57,701 | 21.90 | 0.962 |

A takes 20.7–20.9 minutes per run; B takes 21.6–21.9. B builds roughly 2.7 times as many entries while completing fewer outer iterations at equal node work. Every run reports unchanged swap, and peak training RSS stays below 1 GiB. The expanded tree is affordable under the frozen limits on these runs; that is not a strategic-quality result.

## Same-observation coverage across all three seed pairs

All six final policies are measured on the same **4,248 independently frozen observations**, generated before main training from cap2-uniform, native-uniform, passive and later-repeated-minraise paths. Coverage is a decision-weighted trained-entry hit rate. The fixtures deliberately include unsupported histories and do not estimate a representative whole-game hit rate.

| Street | A seed 1 | B seed 1 | A seed 2 | B seed 2 | A seed 3 | B seed 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| preflop | 97.35% | 100.00% | 97.35% | 100.00% | 97.35% | 100.00% |
| flop | 75.80% | 99.74% | 75.80% | 99.66% | 75.54% | 99.57% |
| turn | 35.28% | 81.52% | 35.18% | 88.64% | 35.18% | 84.98% |
| river | 31.36% | 37.82% | 31.36% | 54.98% | 31.25% | 51.69% |

The third seed repeats the broad flop/turn coverage increase. B's river coverage is higher in every seed but varies from 37.8% to 55.0%; repeated-raise river histories remain a limitation. B also has lower mean visit counts on the common observations at equal node work. Broader representation and less repeated refinement remain simultaneous effects; neither alone establishes strategy quality.

The [machine-readable snapshot](hu20-native-reopening-training-artifacts/completed-training.json) retains all four milestones for every seed, including independent street coverage, visit quantiles and trajectory-specific exposure. The [earlier two-pair report](hu20-native-reopening-preliminary.md) is retained as a historical snapshot.

## Evaluation progress at capture

All **135 cheap panels**, totaling **1,105,920 hands**, completed. The first original-cap2 LBR target finished its 512 paired blocks; the second target was running, with 65 completed blocks in the last saved progress snapshot. The remaining LBR targets, original six-opponent secondary panel, every-hand native replay, independent primary arithmetic check and fixed-first-seed candidate human demo/replay still follow sequentially.

Profit estimates await completion and audit. The prespecified native-pressure B−A contrast and original-cap2 LBR safeguard retain their two-sided 97.5% rotation-block intervals, with seed contrasts averaged inside each shared block. Non-inferiority still requires the lower bound above −10 BB/100. The rough preflight precision estimate may leave that safeguard inconclusive.

## Provenance and retained limits

All six original result hashes match their per-run sealed M4 checksums. Every result reports zero initial entries/iterations, the correct arm-specific game/cap, the frozen plan digest `114fba80cda43604774ce689c3c064d85709c5a76f8998444854ee2adb407a39`, and source `a87e9f8805d211e2b21dbb710339ada9083eedf6`. The final full artifact/lineage/hand audit is pending; this publication does not claim it has already passed.

The running source and plan remain unchanged. The original hard deadline remains **2026-09-29 04:37:43 UTC / 06:37 Madrid**, with 10.5-GiB RSS, 8-GiB free disk and 0.5-GiB swap-growth guards. Raw logs, checkpoints and hand rows stay in the established M4 directory; see [retrieval instructions](hu20-native-reopening-preflight.md#retention). Existing HU20/TP20 demos and prior results are preserved.

Draft PR #115 remains open for review. No merge, model promotion, rental, recipe change or follow-on campaign.
