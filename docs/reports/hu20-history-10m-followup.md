# HU20 history density: owner-approved 10M follow-up

**The unchanged 10M density gate passes in all three seed pairs and in the descriptive pooled assessment.** The original frozen 2M gate remains failed. This is coverage evidence, not poker strength, a convergence proof or permission to rent C/D training hosts.

The [prospective amendment](../hu 20-history-10m-amendment.md) continued all six retained 2M states to 10M total nodes each; no fresh starts, changed schema, seed selection or thresholds. Source `c48edc861e98a60e6f5943ec6453a8be7969e639`, Python 3.11.14, engine 5db 20e 3. Fixed 5M/7M measurements are descriptive only.

## Common corpus and gate

The same model-independent public corpus contains 16,681 decision encounters, including 545 river encounters, from 2,048 deals with both button rotations. Each seed uses the same corpus: the 1,635 pooled river encounters repeat those 545 across three lineages and do not create 1,635 independent observations. There are no strength confidence intervals.

| River metric at 10M | Full history | Compressed history |
| --- | ---: | ---: |
| Median visits/key | 8 | 21 |
| Zero visits | 11.50% | 3.18% |
| Below 10 visits | 53.09% | 29.36% |
| Below 100 visits | 94.43% | 88.99% |

| Seed | Median full→compressed | <10 reduction (pp) | <100 reduction (pp) | Gate |
| --- | ---: | ---: | ---: | --- |
| 2026093001 | 7→21 | 24.22 | 5.32 | pass |
| 2026093002 | 8→21 | 23.85 | 5.50 | pass |
| 2026093003 | 9→21 | 23.12 | 5.50 | pass |

The gate required compressed median≥max(2,2×full median), ≥5 percentage-point reductions in both low-visit fractions, the pooled assessment and at least two complete seed pairs. All three pass, but the <100 margins are narrow (5.32–5.50 points). The observed pooled reduction is 5.44 points, below the earlier conditional linear-growth projection of 6.85; the projection was planning context, not an outcome or threshold. Most river encounters still have fewer than 100 visits (88.99% compressed). No claim that starvation is eliminated.

## Fixed 2M-policy reach: mixed secondary evidence

Before continuation, two frozen 2M current-policy families per seed generated 512 paired blocks/both target positions versus uniform native menu play, root 202610110002. Terminal payoffs were neither recorded nor inspected. Probe both 10M tables on each identical reference family; these are fixed 2M paths, not 10M policy occupancy or an independent strength schedule. River samples are small.

| Seed | Reference 2M policy | River encounters | Median full→compressed | <100 full→compressed |
| --- | --- | ---: | ---: | ---: |
| 2026093001 | full | 68 | 11→21 | 91.18%→80.88% |
| 2026093001 | compressed | 52 | 6→17 | 84.62%→84.62% |
| 2026093002 | full | 64 | 13→30 | 87.50%→87.50% |
| 2026093002 | compressed | 74 | 15→29 | 93.24%→86.49% |
| 2026093003 | full | 71 | 12→22 | 90.14%→83.10% |
| 2026093003 | compressed | 53 | 12→18 | 81.13%→84.91% |

Every family improves median and <10 coverage, but<100 is unchanged for seed 1/compressed and seed 2/full reference paths, and **worsens** for seed 3/compressed:81.13%→84.91%. This mixed reach result is retained; it does not overturn or replace the prospectively defined common-corpus gate. Zero coverage is unchanged 9.62% for seed 1/compressed-reference paths. Coverage differences do not identify a unique causal mechanism or imply playing improvements.

## Resources and retained state

| Cell | Seed | Total nodes | Entries | Step nodes/sec | Wall sec | Peak MiB | Save sec / final MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 2026093001 | 10,000,220 | 507,613 | 16,134 | 536.60 | 789.5 | 4.66 / 21.80 |
| compressed | 2026093001 | 10,000,220 | 277,990 | 15,104 | 555.03 | 466.5 | 2.58 / 13.29 |
| full | 2026093002 | 10,000,441 | 506,401 | 16,126 | 537.03 | 788.4 | 4.65 / 21.74 |
| compressed | 2026093002 | 10,000,093 | 276,218 | 15,176 | 552.34 | 468.6 | 2.60 / 13.17 |
| full | 2026093003 | 10,000,367 | 505,270 | 15,857 | 546.30 | 780.3 | 4.72 / 21.68 |
| compressed | 2026093003 | 10,000,096 | 276,861 | 14,965 | 561.08 | 459.7 | 2.68 / 13.23 |

Mean table entries fall 45.30%; measured compressed step throughput is 5.97% lower. Total sequential run plus the initial policy-reach probe took 55.34minutes. Wall includes loading, probing, saving and export; step timing is separately retained. This is a 10M M1 prefix, not a 100M resource forecast. No paid spend, guard failure, partial worker or dropped lineage occurred.

Stored-key street tables explicitly leave 58,369/57,053/57,763 full and 18,974/18,775/19,102 compressed retained keys unclassified at 10M. Original 2M checkpoints contain hashed keys without a street map. Classified-only stored histograms do not silently include those unknown keys; common-corpus and policy-reach encounters have exact streets.

## Independent verification

All six workers and the policy-reach phase closed with exit 0. The verification pass rechecked the frozen source/plan, original parents and corpus hashes; all 48 recovery checkpoints and six exports; contiguous iteration suffixes, completed node/street/terminal work, entry growth and visit increments. Each final checkpoint reloads and reserializes byte-identically and reproduces its current export exactly. Two independent fresh processes produce identical next-iteration work, checkpoint and export bytes for every lineage. Those single-iteration checks are recovery evidence, not additional scientific training or new output candidates. The 29 prelaunch correctness/information-isolation/public-separation checks already passed; no gameplay evaluation was added.

[Verification](dr 2x 2-history-artifacts/followup-10m-20261001/verification.json), [density gate](dr 2x 2-history-artifacts/followup-10m-20261001/density-gate.json), [resources](dr 2x 2-history-artifacts/followup-10m-20261001/resources.csv), [all streets/milestones](dr 2x 2-history-artifacts/followup-10m-20261001/density-by-street.csv), [policy reach](dr 2x 2-history-artifacts/followup-10m-20261001/fixed-policy-reach.csv) and [final raw inventory](dr 2x 2-history-artifacts/followup-10m-20261001/final-manifest.json) preserve every seed and limitation. The original 2M compact artifacts remain unchanged. The machine field`paid_training_authorized_by_density` denotes this density prerequisite only; it does not supply a rental budget or waive remaining admission checks.

## Next authorized preparation

Integrate D using the exact frozen#143 v 2 descriptor, test public/history isolation and deterministic recovery, and measure compressed-v 2 growth on M1. Validate C/D Linux parity before main training. Then publish current CPU/RAM quotes, measured runtime/memory projections, transfer/storage reserve and a conservative all-in hard cap for owner approval **before rentals**. New Guy owns B and its separate ledger; do not repeat or alter it. The common fresh four-cell evaluation and fixed restricted-river range law remain to be frozen. No automatic 100M rental,300M extension, merge or promotion.

M4 remains#136-only; its frozen scientific queue and ACTIVE90-minute supervision continue. All future owned C/D pods/volumes/roots use`dr2x2-`.

## Retrieval

Large data remain at the following M1 root; the committed final manifest records exact relative paths, sizes and SHA256. Download a required checkpoint with normal file copy/scp and verify its recorded SHA256 before loading. Preserve originals until a separate archive destination is independently verified.

```text
/Users/dberweger/Local/dr2x2-history-10m-20261001/results/dr2x2-history-10m-m1-20261001
```

```sh
shasum -a 256 "/Users/dberweger/Local/dr2x2-history-10m-20261001/results/dr2x2-history-10m-m1-20261001/compressed-2026093001/checkpoint-10000000.json.gz"
```
