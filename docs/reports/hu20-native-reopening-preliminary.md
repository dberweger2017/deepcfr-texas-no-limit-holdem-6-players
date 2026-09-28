# Preliminary HU20 cap A/B training diagnostics

Captured **2026-09-28T20:51:28.905172+00:00**, while the campaign remained in training. This snapshot includes the first two fixed paired seeds, each with both arms completed at 20M nodes. These are training coverage and resource diagnostics. **No pressure-profit or LBR result is reported here.** The third seed and the frozen evaluation/audit campaign remain required; this snapshot does not change the plan or select a checkpoint.

## Comparable public observations

All four policies are measured on the same **4,248 frozen observations**: 1,131 preflop, 1,161 flop, 1,012 turn and 944 river. They come from the prespecified cap2-uniform, native-uniform, passive and later-repeated-minraise paths, independently of trained outcomes. These deliberately stressful paths are not a representative whole-game distribution. Coverage is the decision-weighted fraction with at least one trained visit, not strategic quality or exact-card coverage.

| Street | A seed 1 | B seed 1 | A seed 2 | B seed 2 |
| --- | ---: | ---: | ---: | ---: |
| preflop | 97.35% | 100.00% | 97.35% | 100.00% |
| flop | 75.80% | 99.74% | 75.80% | 99.66% |
| turn | 35.28% | 81.52% | 35.18% | 88.64% |
| river | 31.36% | 37.82% | 31.36% | 54.98% |

The flop and turn coverage increase repeats in both seeds. River coverage improves, but varies substantially between B seeds. On the later-repeated-minraise path specifically, each later street has 640 observations: A has zero trained turn/river entries in both seeds; B seed 1 has **457 turn / 57 river** hits, and B seed 2 **531 turn / 221 river**. Repeated-raise river histories therefore remain sparsely represented, especially in seed 1.

## Cost and work distribution

| Run | Complete nodes | Entries | Complete iterations | Minutes | Peak RSS GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| A / 2026093001 | 20,000,188 | 270,548 | 66,168 | 20.79 | 0.367 |
| B / 2026093001 | 20,000,557 | 730,010 | 55,291 | 21.70 | 0.991 |
| A / 2026093002 | 20,000,094 | 270,949 | 65,011 | 20.93 | 0.367 |
| B / 2026093002 | 20,000,238 | 729,165 | 54,709 | 21.57 | 0.968 |

Both B runs build about 2.7 times as many entries with roughly 4% more elapsed time. Peak memory remains under 1 GiB and both completed pairs report unchanged swap. No traversal failure is recorded. The expanded game completes fewer outer iterations at the same completed-node budget; equal nodes do not imply equal strategic coverage or elapsed work.

Mean visits across the common observations, **including zero-visit misses**, are lower for B. River means are **94.9 / 85.4** for B seeds 1/2 versus **151.7 / 141.4** for A; turn means are **126.9 / 134.8** versus **212.1 / 197.0**. These distributions use each arm's own keys on the same observations, with no intersection of schema hashes. They show broader but less repeatedly updated representation; they do not establish poorer strategy quality or convergence.

The first B seed's traversal logs additionally record **2,347,515 decision visits** and **924,366 regret-update visits** after more than two raises on the current street. A has none in that category. These are repeated traversal visits, not distinct information sets or independent poker hands.

## Interpretation and provenance

Removing the cap demonstrably permits training on previously excluded histories at modest measured runtime cost. The repeated coverage improvement across the first two seeds supports that structural finding. Whether B resists native pressure better, and whether its original-cap2 LBR safeguard passes, is still unknown. The primary 97.5% paired intervals and fixed 10-BB/100 margin remain unchanged.

The [machine-readable snapshot](hu20-native-reopening-preliminary-artifacts/two-seed-training.json) preserves each completed result, its original M4 path and SHA-256, checkpoint/export hashes, all milestone density measurements, source revision and frozen-plan digest. The frozen M4 source remains `a87e9f8805d211e2b21dbb710339ada9083eedf6`; this publication changes documentation only. The four original result hashes match their sealed M4 per-run checksums. Final independent artifact and hand audits are pending. Every raw log/checkpoint remains in the established M4 campaign directory; see [retrieval instructions](hu20-native-reopening-preflight.md#retention).

Draft PR #115 remains an experiment. No model promotion, merge, resource extension or follow-on campaign.
