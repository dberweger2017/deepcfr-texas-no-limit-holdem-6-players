# Windowed K1 blueprint extraction on M4

**Draft PR #111, September 27, 2026.** The frozen campaign completed. Eight-snapshot extraction did not establish an improvement over the final current policy on the scripted primary suite. Both are still far below the tight-aggressive reference there. No player is promoted.

## Frozen protocol and provenance

The [plan](../../configs/blueprint/windowed-extraction-m4.json) has canonical JSON SHA-256 `38b81b096d25eeaedb3dba13a03d309fd309e6012943c5c8661901fd7b35ba83`. The executing source was `85227b9`; later commits add only the read-only direct parity audit and this report. The original shared parent remains SHA-256 `94d41c737d0e8f41a548d58c64dd219dfb9593459c80d7a7382569bf6963e33a`, rehashed after the run. Three saved K1 continuations from #110 supplied the source checkpoints; these are **not independent from-scratch training seeds**.

Each source checkpoint was replayed for 20 million additional traversal nodes with the unchanged K1 trainer. Eight current-policy snapshots were captured at fixed milestones from 10M through 20M nodes. A separate fixed collector sampled 16 preflop roots per seat per capture. The extracted policy `A` uses collected preflop action mass and an arithmetic mean of the eight postflop snapshot distributions, with uniform probability for a missing key in a given snapshot. This is **not** the exact own-reach-weighted CFR average. `C` is final current play, `P` the preflop-collected alternative, and `F` the snapshot alternative. All evaluation arms used the same button-zero-compatible lookup and no-free-fold wrapper.

The 15 arms shared fresh paired schedules: 4,096 six-rotation scripted blocks and 1,024 six-rotation random blocks per arm. The two scripted primary contrasts are `A−C` and `A−U_safe`; their two-sided 97.5% intervals account for two claims. Each contrast averages the three saved-policy differences **within each independent block** before estimating uncertainty. Random and other arm contrasts are descriptive.

## Playing results

| Scripted primary comparison | Paired BB/100 | 97.5% interval |
| --- | ---: | ---: |
| Eight-snapshot `A` minus final current `C` | −10.57 | [−26.53, +5.40] |
| Eight-snapshot `A` minus `U_safe` | −15.51 | [−53.46, +22.43] |

Neither interval excludes zero. The seed-pair `A−C` effects were −16.30, −11.24 and −4.15 BB/100; `A−U_safe` was −18.92, −11.24 and −16.38. The scripted absolute rates were −478.17 BB/100 for mean `A`, −467.60 for mean `C`, −462.65 for `U_safe`, −480.73 for the shared parent and +42.72 for the tight-aggressive hero. Thus, the extraction change has no demonstrated playing gain, and the trained policies remain weak against this pool.

On the secondary random suite, aggregate `A−C` was −19.27 BB/100 [95% CI −54.50, +15.96]. `A−U_safe` was +101.60 [+13.15, +190.05]. The random result does not establish the scripted primary improvement or a general competence claim. Full per-seed and per-arm rates are in [report.json](blueprint-windowed-extraction-m4-artifacts/report.json).

## What the snapshots cover

All three continuations reached the 20M-node target and produced eight captures. Their index sizes were 8.217M–8.225M keys. About 7.036M–7.038M keys appear in only one of the eight snapshots, while roughly 168k–169k appear in all eight. This records considerable missing snapshot support: a late-created key receives uniform distributions for absent earlier captures. The direct audit recomputed 10 deterministic sampled keys per index, including late-created keys, from the eight source files and matched the saved index exactly, with zero observed probability error. The index builder's menu, hash and missing-profile checks remain covered by focused tests; this sampled audit does not assert a full independent recalculation of all eight million rows.

On `A`'s actual evaluation trajectories, trained-key hit rates were about 62% preflop, 45% flop, 15–16% turn and 8% river. These are **decision-weighted, policy-dependent** observations, not whole-range coverage. The three `A` arms made 38.6k–38.8k preflop decisions each, about 13k flop, 6.7k–7.2k turn and 3.8k–4.1k river. Their collected preflop lookup occurred about 14.5k–14.7k times each; the explicit final-current preflop fallback occurred about 24.0k–24.2k times. Training-table misses and postflop absent-profile behavior remain substantial. The [machine report](blueprint-windowed-extraction-m4-artifacts/report.json) retains the per-arm telemetry.

## Audit, resources and artifacts

All **18 sequential attempts** completed: three replays and 15 evaluations. Every replay checkpoint is byte-identical to its #110 source checkpoint, each source lineage and policy index hash matches its manifest, all 24 captures reached their declared node milestones, and all checked file hashes match. Every arm completed 30,720 hands: **460,800 legal, chip-accounted hands** in total. The report found no missing, duplicated or unpaired schedule row; the same deal seed, button and opponents were used at every corresponding arm/rotation. No attempt or resource failure was discarded.

The three extractions took 2,208–2,219 seconds each and peaked at 6.19–6.22 GiB process RSS. The complete campaign took 7,453 seconds (2.07 hours), below its ten-hour ceiling; peak reported system swap stayed unchanged at 761.38 MiB used. The 10.5-GiB process guard was not reached. The repository's focused windowed tests pass (`6 passed`), including a late-created-key parity fixture. The campaign itself ran before the added read-only parity script was committed.

The [compact artifact directory](blueprint-windowed-extraction-m4-artifacts) contains the verified report, campaign record, three direct-parity reports, manifests, result and capture summaries, and the [152-file inventory](blueprint-windowed-extraction-m4-artifacts/inventory.json). All 62 compact files copied here matched the inventory hashes. The full 11.41 GB of hand rows, snapshots, checkpoints, indexes and logs remain on M4 at:

```text
ssh m4
/Users/dberweger/Local/blueprint-extraction-pr111/results/windowed-campaign-m4-20260927
```

The inventory gives each relative path, byte count and SHA-256. Retrieve a large file through the `m4` SSH alias by appending that relative path to the directory above.

**Conclusion:** the extraction and artifact mechanics passed their declared checks, but this fixed comparison did not show that arithmetic windowed extraction improves the old six-player blueprint. The observed sparse late-street support and weak scripted play belong to these old trajectories. The next owner-authorized task is a separate from-zero, versioned heads-up 20BB learning baseline; its results cannot be inferred from this comparison.
