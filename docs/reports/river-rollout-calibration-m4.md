# Conditional river rollout resource calibration

The [frozen calibration](../../configs/blueprint/river-rollout-calibration-m4.json)
used the unchanged 12M checkpoint and three previously examined development
roots. It timed corrected rollout decisions under the same declared joint
private-card law as the river CFR solver. This is a resource measurement,
not a playing comparison. The [raw rows, manifest, result and hashes](river-rollout-calibration-m4/)
were copied from the M4 and verified against the retained checksums. Source
`8dd0aec` was clean.

| Development root | 8 worlds | 128 | 512 | 1,024 | 2,048 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dry check | 0.026 s | 0.413 s | 1.605 s | 3.208 s | 6.424 s |
| Paired flop | 0.022 s | 0.354 s | 1.394 s | 2.795 s | 5.584 s |
| Deep mixed high-card range | 0.024 s | 0.376 s | 1.509 s | 2.991 s | 5.984 s |

All 15 decisions completed their requested world count without fallback in
89.04 seconds including one checkpoint load. Peak process RSS was 7.52 GiB,
below the 10.5-GiB guard; system swap did not increase. The normal control
uses eight worlds. Even 4,096 worlds would take only about 11–13 seconds by
extrapolation, below the selected 30-second CFR budget. A separate bounded
high-world calibration will empirically set an evaluation-only control count;
it does not change production SearchPlayer settings.
