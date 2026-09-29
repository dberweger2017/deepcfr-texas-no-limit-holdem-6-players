# High-world corrected-rollout control calibration

The [frozen plan](../../configs/blueprint/river-rollout-highworld-calibration-m4.json)
extends the [initial resource calibration](river-rollout-calibration-m4.md)
on three previously examined development roots. It uses an evaluation-only
world-count override; production corrected search is unchanged. Source
`e1d9214` was clean, and [all raw rows and hashes](river-rollout-highworld-calibration-m4/)
were copied from the M4 and verified.

| Development root | 8,192 worlds | 9,216 worlds |
| --- | ---: | ---: |
| Dry check | 25.747 s | 28.944 s |
| Paired flop | 22.528 s | 25.233 s |
| Deep mixed high-card range | 24.008 s | 26.964 s |

All six decisions completed without fallback. The run took 209.30 seconds
including one checkpoint load, with 7.68 GiB peak RSS, below the 10.5-GiB
guard. System swap did not increase. We select **8,192 worlds with a
30-second per-decision cap** for the fresh compute control. The 9,216-world
setting leaves just 1.06 seconds of margin on the slowest development root;
8,192 allows some case-to-case variation. The fresh run will retain actual
seconds, completed worlds and fallbacks, so any failure to match compute on
held-out roots remains visible.
