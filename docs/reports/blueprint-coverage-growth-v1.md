# Blueprint lookup coverage across the M4 slice

This exploratory check asks whether additional training filled information sets the bot would use in play. It is a lookup measurement, not a poker-strength result. The [M4 slice report](blueprint-m4-slice-v1.md) records the two source checkpoints and the trainer's sampled lookup rates.

The iteration-8,733 **current** policy played 1,024 fresh six-player, 100 BB self-play hands with deal seeds `2026092700..2026093723` and action seeds `2026092800..2026093823`. At each decision, the same abstraction generated the information key. The 13,705 resulting keys were then checked for membership in both saved tables. This keeps the decision sample fixed across the comparison; it favors states reached by the later policy and does not say how the earlier policy would have played. The source SHA-256 values are `2001750431a06a777ceff4365c052794d42574fe2c5369eaced24334a9c50129` (iteration 2,522) and `94d41c737d0e8f41a548d58c64dd219dfb9593459c80d7a7382569bf6963e33a` (iteration 8,733). Both were verified on the M4 before this check.

| Street | Decisions | Iteration 2,522 trained | Iteration 8,733 trained |
| --- | ---: | ---: | ---: |
| Preflop | 7,905 | 4,798 (60.7%) | 6,200 (78.4%) |
| Flop | 3,917 | 1,238 (31.6%) | 1,984 (50.7%) |
| Turn | 1,479 | 92 (6.2%) | 239 (16.2%) |
| River | 404 | 10 (2.5%) | 28 (6.9%) |

The table grew from 1,740,153 to 5,834,622 entries. Coverage improved on the same played decisions, including turn and river, so more work is reaching useful states. Most postflop decisions still use the uniform fallback, especially late in a hand. This is a consequential sampling and abstraction risk for a larger blueprint: the key retains the full public action sequence, which can create many rare histories even with coarse card and bet buckets. The two checkpoints do not support a reliable extrapolation to a target iteration count or RAM size. The [RunPod check](../blueprint-runpod-check.md) will measure coverage under varied opponents and the memory and worker speed needed to choose the large-run host.
