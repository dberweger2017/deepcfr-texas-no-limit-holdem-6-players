# M4 blueprint scaling slice

The [one-hour protocol](../blueprint-m4-slice.md) began seed `2026092402` for the six-player, 100 BB tabular blueprint on the always-on M4 (10 CPU cores, 16 GiB RAM). The purpose was to measure useful training throughput, table growth, checkpoint cost, and lookup coverage. It is not a poker strength test. No rental was used.

## Execution

The first phase used one worker, four independent roots per seat per iteration, a 10 GiB process RSS guard, a 30 GiB free-disk guard, and 15-minute timed checkpoints. Its original two-million-entry cap would have ended the run well before the one-hour wall cap. At iteration 2,333 the table held 1,611,738 entries, the checkpoint occupied 69,766,280 bytes and took 12.27 seconds to save. A clean signal stop at iteration 2,522 saved a resumable checkpoint with 1,740,153 entries. The run had no numerical or invalid-action failure.

The [entry-cap amendment](../../configs/blueprint/m4-slice-v1-entry-cap-amendment.json) raised only the operational table ceiling to eight million entries. The seed, abstraction, roots, sampling, node/time limits, and evaluation recipe remained fixed. A resume-equivalence test compares the amended resumed run with an uninterrupted amended run. The second phase resumed the exact first-phase checkpoint, with 2,580 seconds of wall time to retain the approximately one-hour total slice. Both phases and their artifacts are retained separately.

## Measurements

The first phase completed 2,522 iterations and 12,670,232 traversal nodes in 974.9 seconds of step time, or approximately 13,000 nodes/second. Its peak parent RSS was 1,313,177,600 bytes (1.22 GiB). The final first-phase checkpoint is 75,307,992 bytes and took 13.21 seconds to save.

In the resumed phase, the 15-minute checkpoint reached iteration 4,796 with 3,244,277 entries. It was 140,264,536 bytes and took 23.32 seconds to save. At the 30-minute checkpoint, iteration 7,014 held 4,708,503 entries; the checkpoint was 203,578,422 bytes and took 34.5 seconds. Peak parent RSS there was 3,592,060,928 bytes (3.35 GiB). The second phase processed 23,319,807 traversal nodes in 1,800.5 seconds of step time through that checkpoint, or approximately 12,952 nodes/second. Neither checkpoint showed a collapse in step throughput.

The wall cap stopped the resumed phase at iteration 8,733 with 5,834,622 entries. Across both phases, the trainer processed **45,157,550 traversal nodes** in 3,485.1 seconds of step time (approximately 12,957 nodes/second). The final phase checkpoint was 252,235,464 bytes and took 42.87 seconds to save. Peak parent RSS was 4,551,933,952 bytes (4.24 GiB). The final status is `stopped / wall_time`; neither the 8-million-entry cap nor the RAM/disk guards tripped. Free disk remained 99,068,162,048 bytes (92.3 GiB).

Lookup hits during sampled training increased from the first 500 iterations to the last 500:

| Street | Iterations 1–500 trained lookups | Iterations 8,234–8,733 trained lookups |
| --- | ---: | ---: |
| Preflop | 16.79% | 66.07% |
| Flop | 4.25% | 36.60% |
| Turn | 0.25% | 7.39% |
| River | 0.08% | 3.19% |

These are hits in the trainer's sampled traversals, not held-out game coverage or playing strength. Turn and river still depend overwhelmingly on fallback decisions. Entry growth remained approximately 0.33–0.35 million entries per 500 iterations in these windows.

## Recovery and artifacts

The final checkpoint's SHA-256 is `94d41c737d0e8f41a548d58c64dd219dfb9593459c80d7a7382569bf6963e33a`. Its file hash matches both the checkpoint index and `result.json`, and a fresh Python process loaded all 5,834,622 entries at iteration 8,733. The first-phase checkpoint hash is `2001750431a06a777ceff4365c052794d42574fe2c5369eaced24334a9c50129`. Both phases remain in `results/blueprint-m4-slice-v1/` and `results/blueprint-m4-slice-v1-continued/` on the M4 under the repository root; hash-verified copies are in local `results/blueprint-m4-slice-v1-m4/` and `results/blueprint-m4-slice-v1-continued-m4/`. The manifest in each directory records its revision, plan, and environment. These result directories are ignored by Git. The capped run intentionally did not export an inference policy or execute the arena, so there is no poker strength estimate and no model promotion.

## Decision

The M4 can continue this seed to the frozen 10,000-iteration target: the last 500 iterations added 323,010 entries in 205.9 seconds of step time, implying roughly nine more minutes of steps and an approximately 6.7-million-entry table if that local rate persists. The measured memory slope from 1.74 to 5.83 million entries projects about 6 GB of process RSS at the current eight-million-entry ceiling, below the 10 GiB guard; Python allocation jumps and export memory can raise the actual peak. The one-hour slice does **not** bound memory for a much longer campaign or the still unmeasured large-table inference export.

The next useful paid measurement is a bounded RunPod 64 GB worker and export test using this same checkpoint, with a live-price spending cap. It should measure nodes per dollar and aggregate memory before choosing an AWS instance or setting a two-seed work target. Continuing M4 training is reasonable while that test is prepared, but it is not a substitute for the worker measurement. No AWS campaign budget or finish date follows from this one-worker result.
