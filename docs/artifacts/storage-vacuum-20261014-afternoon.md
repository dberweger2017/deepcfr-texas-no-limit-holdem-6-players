# Uploaded cache cleanup during active HU100 work — October 14 afternoon

The owner requested another storage pass using the standing preservation rules.
This pass released **25.425 GB**: M1 **3.574 GB**,
M4 **21.851 GB**. Verified free space at the receipt
snapshot: **M1 60.510 GB /M4 83.941 GB**.
Filesystem availability changes with the running work; per-path allocation deltas
are the accountable cleanup total.

## Scope and preservation

- M4: four confirmed uploaded PR270/271 archive caches, 21.851 GB.
- M1: 56 historical research payload caches, 2.993 GB,
  mapped to merged owners through the exact top-level RESULTS_INDEX inventory
  rows; every listed owner has a fresh merged-status check. Thirty-five unmapped
  loose files remain retained.
- M1: two inactive Rust `target/debug` caches, 0.582 GB;
  retained source revisions and rebuilding commands are below.

Active PR272 and both worktrees remain untouched. Its recorded inputs reference
PR260 archives/tables, PR246 averages and released v0.5.2; PR272/260/246/163 were
excluded from native cache release. PR265 averages, current Slumbot inputs and
the final shared HU200 cache remain; PR270 final average and unarchived final
closeout guard are retained. Source, environments, release executables, shared
Git and uncertain/unarchived evidence are preserved. The PR272 agent separately
removed four archived PR270 originals and owner-approved derived caches; this
pass neither touches those paths nor credits that cleanup. No research original
is removed here.

Each native release checks uploaded1/uploading0/conflicts0, unpinned,
not downloading/paused/excluded/trashed, matching size and cloud ID/name/parent,
unchanged identity/time and fresh open-file admission. Supported per-file
`FileManager.evictUbiquitousItem` leaves each cloud file and same-size dataless
placeholder. Fresh cloud readback confirms every identity; all 60 placeholders
remain and the observed residual allocation is excluded from the credited delta.
No synced deletion, forced offload, unpinning, archive download or repeated
whole/member hash audit. No personal cleanup, application shutdown or schedule.

For build output, no active cargo/rustc or open selected path was found. The full
file/inode/time inventory was unchanged immediately before removal. All hardlinks
were internal to the same selected cache; unique file inodes were counted once.
There were no symlinks or external hardlinks in either selected cache. Only the
two exact `target/debug` directories were removed; checkouts/source remain.

## Same-file cache restoration

Open/download the linked unchanged Drive file to restore its native cache at the
same path. Legacy aliases and cloud organization are preserved. For subsequent
research retrieval into fresh ignored nonsynced working storage, use the existing
inventory/archive/member hash pins and verification rules before computing.
Cache release does not claim these loose files form new scientific archives.

| Mac | Native cached path | Unchanged Drive file | Allocated bytes released |
| --- | --- | --- | ---: |
| M1 | `~/Local/Research-Cloud/Historical-experiments/snapshot-readiness.tar.gz` | [snapshot-readiness.tar.gz](https://drive.google.com/file/d/1UkhLIJ2hLz3duSAP2S59tuL2wYCRDfMa/view?usp=drivesdk) | 1,869,869,056 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/m4-overnight-20260919/branching-continuous-2026091901/scenario-0-seed-2026091901/outcomes-1088.json.gz` | [outcomes-1088.json.gz](https://drive.google.com/file/d/1l_PS-PWAGhd3GfTn3kISd9tzI1XcTGXQ/view?usp=drivesdk) | 6,414,336 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/m4-overnight-20260919/branching-continuous-2026091901/scenario-0-seed-2026091901/outcomes-832.json.gz` | [outcomes-832.json.gz](https://drive.google.com/file/d/1r1WGAk4tpezKHv8c7brbAxyW3YxhKQrK/view?usp=drivesdk) | 6,414,336 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/m4-overnight-20260919/branching-continuous-2026091901/scenario-0-seed-2026091901/outcomes-1536.json.gz` | [outcomes-1536.json.gz](https://drive.google.com/file/d/1X3qds9sKbEg18JykDnkQiBT2C0uMKR98/view?usp=drivesdk) | 6,418,432 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/m4-overnight-20260919/branching-continuous-2026091901/scenario-0-seed-2026091901/outcomes-1600.json.gz` | [outcomes-1600.json.gz](https://drive.google.com/file/d/1b1YV_pLD5uzdhNrVkzU80GBdEWvFf8Qs/view?usp=drivesdk) | 6,422,528 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/m4-overnight-20260919/branching-continuous-2026091901/scenario-0-seed-2026091901/outcomes-128.json.gz` | [outcomes-128.json.gz](https://drive.google.com/file/d/1Y5_jfDKWbSrpUV3kjM60oBV670OS-e9w/view?usp=drivesdk) | 6,488,064 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/m4-overnight-20260919/replay-continuous-2026091901/scenario-0-seed-2026091901/outcomes-448-random.json.gz` | [outcomes-448-random.json.gz](https://drive.google.com/file/d/1wq2l6CiRCJOSleMgwwNXqAC0t0s-sd2_/view?usp=drivesdk) | 6,377,472 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/m4-overnight-20260919/replay-continuous-2026091901/scenario-0-seed-2026091901/outcomes-576-random.json.gz` | [outcomes-576-random.json.gz](https://drive.google.com/file/d/1My7QyS8L-hCVxKUoTMPdCVHaLA7_oIkw/view?usp=drivesdk) | 6,389,760 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/holdem-baseline-resumed-v1/scenario-0-seed-101/training-2.pt` | [training-2.pt](https://drive.google.com/file/d/18P0Zc00c4E9_IqE-sIfWGXXf0wFUQlaJ/view?usp=drivesdk) | 4,124,672 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/holdem-baseline-resumed-v1/scenario-2-seed-103/training-2.pt` | [training-2.pt](https://drive.google.com/file/d/1pP_R2f43Pb1mPrXn3HwZvuv8AsKbw5a_/view?usp=drivesdk) | 3,006,464 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/holdem-baseline-resumed-v1/scenario-0-seed-103/training-2.pt` | [training-2.pt](https://drive.google.com/file/d/132pEbe3wlHsn1RQo--Woom9KfYrEOHIV/view?usp=drivesdk) | 9,261,056 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/holdem-baseline-resumed-v1/scenario-3-seed-101/outcomes-2.json` | [outcomes-2.json](https://drive.google.com/file/d/1RDOj2kxtJxqGrX1LKysluzEjiqYKbEd3/view?usp=drivesdk) | 2,748,416 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/holdem-baseline-v1/scenario-2-seed-103/training-1.pt` | [training-1.pt](https://drive.google.com/file/d/1WornFzx3ayCWd46Fz_FAFXmEHC2ld6w0/view?usp=drivesdk) | 2,572,288 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/holdem-baseline-v1/scenario-2-seed-101/outcomes-2.json` | [outcomes-2.json](https://drive.google.com/file/d/1l6Nrx4RkH1nHhL9sINMa8l1EZ_Z8lHK-/view?usp=drivesdk) | 2,969,600 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/neural-readiness-retrieved/neural-readiness/jobs/job-000/campaign/training/iteration-000360.pt` | [iteration-000360.pt](https://drive.google.com/file/d/1EAPg7SKL3rsCZET5HLxllnCsnCc6o4bn/view?usp=drivesdk) | 17,072,128 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/neural-readiness-retrieved/neural-readiness/jobs/job-008/campaign/training/iteration-000080.pt` | [iteration-000080.pt](https://drive.google.com/file/d/1I-sr2x6y3RRplprD1Gjaq1WsYhO3qbDn/view?usp=drivesdk) | 8,036,352 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/neural-readiness-retrieved/neural-readiness/jobs/job-012/campaign/training/iteration-000020.pt` | [iteration-000020.pt](https://drive.google.com/file/d/1dfIsAO2599II5lZhT2bQOhCStQuVDneG/view?usp=drivesdk) | 3,633,152 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/neural-readiness-retrieved/neural-readiness/jobs/job-012/campaign/training/iteration-000040.pt` | [iteration-000040.pt](https://drive.google.com/file/d/1yeQrwwpZg9hTBk6XHA3KPXA-muwTLQsW/view?usp=drivesdk) | 5,754,880 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/neural-readiness-retrieved/neural-readiness/jobs/job-014/campaign/training/iteration-000040.pt` | [iteration-000040.pt](https://drive.google.com/file/d/1y11xXpynL-czCWFrpagUawsG3VfXlMPy/view?usp=drivesdk) | 5,734,400 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/neural-readiness-retrieved/neural-readiness/jobs/job-014/campaign/training/iteration-000060.pt` | [iteration-000060.pt](https://drive.google.com/file/d/1htJauoZYN1fFkbdgHJN8i-GUgMbcjDLB/view?usp=drivesdk) | 7,163,904 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/neural-readiness-retrieved/neural-readiness/jobs/job-003/campaign/training/iteration-000360.pt` | [iteration-000360.pt](https://drive.google.com/file/d/1Xp-5pWOSgb7VP6D-nv4bZGwqWaIeerxG/view?usp=drivesdk) | 17,072,128 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/neural-readiness-retrieved/neural-readiness/jobs/job-004/campaign/training/iteration-000360.pt` | [iteration-000360.pt](https://drive.google.com/file/d/1e3cZ86KznHyOFlB_o78GW91JJID0aKAt/view?usp=drivesdk) | 17,072,128 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/strategy-fitting-inputs/257/replay.npz` | [replay.npz](https://drive.google.com/file/d/1EtwNoQmlVQsWJvswSaapMI8S6_CXZtxA/view?usp=drivesdk) | 1,093,632 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/cpu-pilot-remote-v1/training-4/job-002/training/iteration-000008.pt` | [iteration-000008.pt](https://drive.google.com/file/d/15f2hXTfHHIkDRQ7nxqwxrW6m-H_c3Uza/view?usp=drivesdk) | 1,282,048 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/cpu-pilot-remote-v1/training-1/job-003/training/iteration-000008.pt` | [iteration-000008.pt](https://drive.google.com/file/d/177SzDxhTBnlH_7icpffLL7UI-cEmNdMn/view?usp=drivesdk) | 1,400,832 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/longer-calibration/scenario-0-seed-997/outcomes-4-previous.json` | [outcomes-4-previous.json](https://drive.google.com/file/d/1KQtlDsAhvAiQZzV0oLLyMsCQD6Z1PtoU/view?usp=drivesdk) | 2,703,360 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/longer-calibration/scenario-0-seed-997/outcomes-4.json` | [outcomes-4.json](https://drive.google.com/file/d/1bO_ua4d0d_WnhNVHuGqpiNUyBHDXdsN_/view?usp=drivesdk) | 2,768,896 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/snapshot-readiness-retrieved/snapshot-readiness/jobs/job-007/campaign/training/iteration-000060.pt` | [iteration-000060.pt](https://drive.google.com/file/d/1m-zbmNguEpFaKNbBS7zXqtLRMHBv6dFC/view?usp=drivesdk) | 20,291,584 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/snapshot-readiness-retrieved/snapshot-readiness/jobs/job-000/campaign/training/iteration-000240.pt` | [iteration-000240.pt](https://drive.google.com/file/d/1rx9Koy9_imkrGK-lU3X-6x5sHVG0yk7c/view?usp=drivesdk) | 32,309,248 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/snapshot-readiness-retrieved/snapshot-readiness/jobs/job-000/campaign/training/iteration-000180.pt` | [iteration-000180.pt](https://drive.google.com/file/d/13F4h4u1-rrAkdMydj482tiK5RCsZmtAn/view?usp=drivesdk) | 28,454,912 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/snapshot-readiness-retrieved/snapshot-readiness/jobs/job-001/campaign/training/iteration-000120.pt` | [iteration-000120.pt](https://drive.google.com/file/d/15W31-laKSAxlZ55hNf70R-_X98WigpZE/view?usp=drivesdk) | 24,600,576 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/snapshot-readiness-retrieved/snapshot-readiness/jobs/job-006/campaign/training/iteration-000120.pt` | [iteration-000120.pt](https://drive.google.com/file/d/1B2_qugA24hBKZWE-Rgkyr7JnNjLgyLQn/view?usp=drivesdk) | 24,600,576 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/snapshot-readiness-retrieved/snapshot-readiness/jobs/job-012/campaign/training/iteration-000020.pt` | [iteration-000020.pt](https://drive.google.com/file/d/1MWGyU7Dvi03mygn1n6kI_tGaQVJE2RyF/view?usp=drivesdk) | 4,890,624 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/snapshot-readiness-retrieved/snapshot-readiness/jobs/job-013/campaign/training/iteration-000100.pt` | [iteration-000100.pt](https://drive.google.com/file/d/1-M5zraYYjQWy1uWDKVZVzr4X5uvFcjIC/view?usp=drivesdk) | 14,909,440 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/snapshot-readiness-retrieved/snapshot-readiness/jobs/job-013/campaign/training/iteration-000080.pt` | [iteration-000080.pt](https://drive.google.com/file/d/1Qc6b6o2i77wO6lZHNVqXE7sdgWUhmibd/view?usp=drivesdk) | 13,119,488 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/snapshot-readiness-retrieved/snapshot-readiness/jobs/job-002/campaign/training/iteration-000060.pt` | [iteration-000060.pt](https://drive.google.com/file/d/1YZqvZkIdMJz-Xz37UmLAC4luDv3nQfHH/view?usp=drivesdk) | 20,217,856 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/snapshot-readiness-retrieved/snapshot-readiness/jobs/job-002/campaign/training/iteration-000420.pt` | [iteration-000420.pt](https://drive.google.com/file/d/11p8O6uDcvVLIgI-GmIAh6JN9zHCRb23n/view?usp=drivesdk) | 43,868,160 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/snapshot-readiness-retrieved/snapshot-readiness/jobs/job-002/campaign/training/iteration-000300.pt` | [iteration-000300.pt](https://drive.google.com/file/d/1Ax5xIDU2V4PXkmRsyicU9694PgA0CmM-/view?usp=drivesdk) | 36,159,488 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/snapshot-readiness-retrieved/snapshot-readiness/jobs/job-011/campaign/training/iteration-000100.pt` | [iteration-000100.pt](https://drive.google.com/file/d/1A2B_3y-4Y4b9KL1Q_EqkEO3eGeIGfNl7/view?usp=drivesdk) | 14,909,440 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/policy-comparison/seed-307/unclipped64-vs-clip64/outcomes.jsonl.gz` | [outcomes.jsonl.gz](https://drive.google.com/file/d/1y2-MHlqTdbS8phNoibKs_dn8u4mz9Of3/view?usp=drivesdk) | 1,703,936 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/policy-comparison/seed-307/current-vs-average/outcomes.jsonl.gz` | [outcomes.jsonl.gz](https://drive.google.com/file/d/1zE6LZqkmUcQL0CGxXmZ9JDZe_8KAXAaQ/view?usp=drivesdk) | 1,716,224 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/policy-comparison/seed-313/current-vs-average/outcomes.jsonl.gz` | [outcomes.jsonl.gz](https://drive.google.com/file/d/1OT8mgqTcVBAkWZJEtY0ZT--F70yjog2L/view?usp=drivesdk) | 1,806,336 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/representation/report.json` | [report.json](https://drive.google.com/file/d/1yZAPUKX2cROHK2ECGlVRurEGvYCy-hp_/view?usp=drivesdk) | 42,086,400 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/sampled-pilot/scenario-3-seed-211/outcomes-3.json` | [outcomes-3.json](https://drive.google.com/file/d/1Bq6qv5gJXY9Bs7m7nFNy6i6xQnMGh5JA/view?usp=drivesdk) | 2,822,144 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/sampled-pilot/scenario-2-seed-227/average-6.pt` | [average-6.pt](https://drive.google.com/file/d/19hjdaex9ZNnXhDjCmKBMAL7idN3cBsl1/view?usp=drivesdk) | 3,231,744 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/sampled-pilot/scenario-3-seed-223/training-6.pt` | [training-6.pt](https://drive.google.com/file/d/17l-sieXMzXdswsOeRVx6pZ9Yt_Igof87/view?usp=drivesdk) | 15,044,608 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/sampled-pilot/scenario-2-seed-223/outcomes-6.json` | [outcomes-6.json](https://drive.google.com/file/d/11iyz9vQDUdIv_xdqDUt6iJhnRJYqy-XU/view?usp=drivesdk) | 2,895,872 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/sampled-pilot/scenario-1-seed-227/training-6.pt` | [training-6.pt](https://drive.google.com/file/d/1BqxET9Ev5PnrM2dLqRaAAsP3D2IbDmkF/view?usp=drivesdk) | 9,560,064 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/sampled-pilot/scenario-1-seed-227/average-3.pt` | [average-3.pt](https://drive.google.com/file/d/1SWQcq6jqLuEm8TzqmjTrbVE8r8ICXhfe/view?usp=drivesdk) | 1,081,344 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/sampled-pilot/scenario-0-seed-227/average-6.pt` | [average-6.pt](https://drive.google.com/file/d/1a7CAUlLQgd2r9p8d1Y55FvcqcIj3AKjZ/view?usp=drivesdk) | 2,154,496 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/persistent-critic/river-reference-503-phase-1.pt` | [river-reference-503-phase-1.pt](https://drive.google.com/file/d/1qf0Unuu3VrUMOIBbty88EdTNSYCYICn2/view?usp=drivesdk) | 37,580,800 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/persistent-critic/full-river-509-phase-2.pt` | [full-river-509-phase-2.pt](https://drive.google.com/file/d/1EUMlPH29p_WZ3hCYGTwIokA0O4b_WCWn/view?usp=drivesdk) | 25,202,688 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/persistent-critic/full-turn-521-phase-2.pt` | [full-turn-521-phase-2.pt](https://drive.google.com/file/d/1b8K04OGyMbm8pIOmZB-qibPWQZVKOern/view?usp=drivesdk) | 20,451,328 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/blueprint-m4-slice-v1-continued-m4/checkpoint.json.gz` | [checkpoint.json.gz](https://drive.google.com/file/d/1-GAPpGKW6xzctJ7BqgXVvR5R01kKA-AK/view?usp=drivesdk) | 252,235,776 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/collector-branching/coverage.jsonl` | [coverage.jsonl](https://drive.google.com/file/d/10Bmcsq2RaZUSkBI4Oh0dQCxf03fYAvMq/view?usp=drivesdk) | 10,473,472 |
| M1 | `~/Local/Research-Cloud/Historical-experiments/blueprint-runpod-check-v1/blueprint-runpod-worker-1/checkpoint.json.gz` | [checkpoint.json.gz](https://drive.google.com/file/d/1nQTO-dJbmcyy8ukMWXRSywffUjZFaqxu/view?usp=drivesdk) | 254,046,208 |
| M4 | `~/Local/Research-Cloud/PR-270-hu200-v2-frozen-5b/hu200-v2-frozen-5b-M4-20261014.zip` | [hu200-v2-frozen-5b-M4-20261014.zip](https://drive.google.com/file/d/1zvA11Jr_6pwDehqMnVIGPcSGrPZdQMpb/view?usp=drivesdk) | 18,420,961,280 |
| M4 | `~/Local/Research-Cloud/PR-270-hu200-v2-frozen-5b/hu200-v2-frozen-5b-average-M4-20261014.zip` | [hu200-v2-frozen-5b-average-M4-20261014.zip](https://drive.google.com/file/d/1XwE9T6K84_tZPv00soy_d8VtQAHSUEGD/view?usp=drivesdk) | 2,876,035,072 |
| M4 | `~/Local/Research-Cloud/PR-271-hu200-k50-translation/hu200-k50-translation-M4-20261014.zip` | [hu200-k50-translation-M4-20261014.zip](https://drive.google.com/file/d/1k3lH3i0uwyOIXx4ySdZdMkYvzJLrXkeU/view?usp=drivesdk) | 552,636,416 |
| M4 | `~/Local/Research-Cloud/PR-270-hu200-v2-frozen-5b/hu200-v2-frozen-5b-closeout-M4-20261014.zip` | [hu200-v2-frozen-5b-closeout-M4-20261014.zip](https://drive.google.com/file/d/1N1q4Svr1CqRnPsMv81ufwRbyUGfs3p5t/view?usp=drivesdk) | 1,085,440 |

## Build-cache restoration

- `~/Local/deepcfr-hu200-slumbot/native/hu20-trainer/target/debug`: 247,693,312 allocated file bytes; owner PR216 merged. Source checkout `~/Local/deepcfr-hu200-slumbot`, retained HEAD `3d74724d2f39d1d11e3fc1fc9b69c97a196a7d08`. From that checkout run `cargo build --manifest-path native/hu20-trainer/Cargo.toml` with the retained Rust toolchain to rebuild debug output.
- `~/Local/native-frozen-table/native/hu20-trainer/target/debug`: 333,860,864 allocated file bytes; owner PR262 merged. Source checkout `~/Local/native-frozen-table`, retained HEAD `1d9fade1c7ee2d6aa16606e51de6a934bf028cfe`. From that checkout run `cargo build --manifest-path native/hu20-trainer/Cargo.toml` with the retained Rust toolchain to rebuild debug output.

[Full private receipt](https://drive.google.com/file/d/1KN0Puu2kDFcSgiec5cF5xazC6W46UTdb/view?usp=drivesdk), ID `1KN0Puu2kDFcSgiec5cF5xazC6W46UTdb`, **361,557 bytes**, SHA256 `45adea400fce865b7cfa46a252e93aaa4d046b4069f635a7e32fe56908b8553a`. Independent metadata confirms exact name/size and Research-Cloud parent `188bEt6i0RHqegCCdvpf3wPzUiRw78N2s`. The receipt preserves every historical scope/owner mapping, native/cloud acceptance and before/after stats, exact cache IDs/paths, source/rebuild provenance, dependency review and verification.
