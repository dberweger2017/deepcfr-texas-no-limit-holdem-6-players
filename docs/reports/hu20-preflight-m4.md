# Fresh heads-up 20BB M4 resource preflight

**Draft PR #112, September 27, 2026.** This is an outcome-free resource measurement for the new game. No playing comparison has been run or inspected.

The [proposed plan](../../configs/blueprint/hu20-m4.json) passed its 1M-node preflight and is now frozen unchanged for the main campaign. Its canonical JSON SHA-256 is `0da3cb6075d328acd748cf79a7e5f2d2422815c0574fdc3dbf7bb8bbb3908141`. The M4 source was `2f94cc7` for the original preflight and `893cd6a` for a second identical-seed pass that added only read-only independent-collector visit telemetry. Both started from zero regrets. The deterministic current inference export has the same SHA-256 in both passes: `84b4b6256930717f31cb0d6bde0ada6954d5f80f98617a7e1fec6115fd713cc9`.

| Measurement | Result |
| --- | ---: |
| Completed traversal nodes / outer iterations | 1,000,268 / 3,606 |
| Wall time including snapshot, collector and export | 55.8–56.6 s |
| Entries at stop | 80,438 |
| New / revisited key contributions | 80,438 / 134,130 |
| Snapshot, collector and export time | 0.69–0.70 s |
| Peak process RSS | 0.127 GiB |
| System swap used before/after | 761.38 MB / 761.38 MB |
| Independent preflop collector actions | 106 across 102 keys |
| Collector decision-weighted trained / revisited actions | 103 / 98 |
| Collector decision-weighted visit p25 / p50 / p75 / p90 | 4 / 7 / 12 / 22 |

The collector uses separately seeded deals and action randomness, so these preflop counts are outside the training trajectories. The sample is small and only covers preflop; it is a work-allocation diagnostic rather than evidence of playing strength. Street-specific new/revisited counts, system pressure, manifests and checksums are retained in the [compact preflight artifacts](hu20-preflight-m4-artifacts). Both preflight directories passed their five-file checksum checks with zero mismatches. The source snapshots, exports and iteration rows remain on M4 at `/Users/dberweger/Local/hu20-pr112/results/hu20-m4-20260927`.

The measured 1M-node throughput projects roughly 19 minutes per 20M-node seed before larger-table and repeated capture costs. Three sequential seeds, extraction and the declared evaluation panel have substantial room under the existing ten-hour limit; the projection is not a runtime guarantee. Entry growth at 1M nodes is far below the 4M cap, but the later growth curve and RSS will be checked at every iteration. The fixed main budget remains **20M nodes per seed**, checkpoints at **2/5/10/20M**, eight captures from **10M to 20M**, **4,096 confirmation blocks per opponent**, **10.5 GiB RSS**, and a **ten-hour total deadline**. No paid host or additional algorithm change is part of this freeze.
