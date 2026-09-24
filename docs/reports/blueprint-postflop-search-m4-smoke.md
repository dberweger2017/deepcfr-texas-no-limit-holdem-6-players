# M4 postflop search smoke run

## Purpose and host

Exercise PR #103's complete paired comparison path on a trained six-player blueprint before the checkpoint-0.4 campaign finishes. This was inference and evaluation only; no new training was started and the active RunPod training recipe was untouched. The fixed [smoke plan](../../configs/blueprint/postflop-search-m4-smoke.json) uses eight fresh validation blocks against random opponents and eight against the scripted pool. Each block rotates six seats and runs both the search and unchanged blueprint arms. Search settings match the proposed larger comparison: 12 sampled worlds, 96 candidate hands per opponent, four continuation styles and one second per postflop decision. The smoke guard was 15 minutes and 8 GiB process RSS.

The M4 has an Apple M4 CPU and 16 GiB physical RAM. System memory was 78% free before the run and 76% free afterward; the existing RunPod backup and TensorBoard processes remained active. The source checkpoint contained 5,834,622 entries and matched SHA-256 `94d41c737d0e8f41a548d58c64dd219dfb9593459c80d7a7382569bf6963e33a`. The final clean-checkout run used commit `6aa3e22208879a701d8b474907e31e803e3c4aa4`; its manifest reports `dirty: false`.

## Result

Both schedules finished in about 26 seconds of observed end-to-end wall time. All 192 scheduled hands completed with zero invalid actions or failures. Peak process RSS was 5,322,817,536 bytes (4.96 GiB), within the 8 GiB guard. The previous runs on the same schedule produced the same per-benchmark outcome hashes; timing and peak RSS vary between processes.

| Opponents | Candidate / baseline hands | Searches completed | Fallbacks | Search-only p95 / max | Paired BB/100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 48 / 48 | 8 / 8 | 0 | 34 / 64 ms | +12.50 |
| Scripted pool | 48 / 48 | 34 / 34 | 0 | 100 / 127 ms | +195.31 |

The scripted schedule exercised search on flop (21 decisions), turn (7) and river (6); random play exercised eight flop searches. The arena returned **no confidence interval** for either paired profit estimate because eight blocks are too few. These profit numbers are descriptive smoke output, not evidence that search improves playing strength. The purpose of the run was to verify legal play, exact checkpoint loading, paired schedule completion, search execution and memory/latency headroom on the smaller table. The final 0.4 checkpoint is expected to be much larger and still needs its planned rental-backed comparison.

## Artifacts and follow-up

The [machine-readable result](blueprint-postflop-search-m4-smoke.json) retains both arena reports, schedule and outcome hashes, street coverage, latency and RSS. Its SHA-256 is `5e645a47464c91636fae9da4ba0b40af2c0519449038416dba119d68d0a109d7`. The final output directory is retained on the M4 at `~/Local/blueprint-search-pr103/results/m4-smoke-final-20260924/` and copied locally under the ignored `results/blueprint-search-m4-smoke-final-20260924/`. All six copied files matched the M4 `checksums.json`; compact hand rows and the manifest remain there. The draft PR stays open for the final 0.4 checkpoint and the declared larger comparison. No model is promoted by this smoke run.
