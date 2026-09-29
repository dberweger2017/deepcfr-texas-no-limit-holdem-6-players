# Profile-retaining conditional river preflight

This rerun of the frozen [two-root preflight](../river-conditional-comparison.md)
used clean source `bbab576` after adding saved average profiles and the
predeclared root-cluster report analyzer. The roots are previously examined
development cases and their returns are excluded from confirmation. The
[complete artifacts](river-conditional-profile-preflight-m4/) include every
root and arm row, two solved profiles, manifest, result, and checksums. The
copied files and the two profile hashes were verified.

| Root | Complete sweeps / solver time | Normal control | Matched control | Status |
| --- | ---: | ---: | ---: | --- |
| Dry check | 682 / 30.000 s | 0.041 s / 8 worlds | 25.856 s / 8,192 worlds | 3/3 legal |
| Paired flop | 1,245 / 30.000 s | 0.078 s / 8 worlds | 19.453 s / 8,192 worlds | 3/3 legal |

The whole run took 162.57 seconds including checkpoint load. Both matched
controls completed all requested worlds without fallback. The candidate had
no off-tree delegation. Peak process RSS was 7.40 GiB, below the 10.5-GiB
limit, and swap was unchanged. The report analyzer returned `complete` after
checking row completeness, deal/opponent pairing and all artifact hashes.

The clean-source repository suite passed **831 tests**. The earlier full-suite
attempt had one source-hash reproducibility failure because files changed
between its two fresh-process invocations while the suite was running; that
test passed in isolation after edits stopped, and all 831 tests passed in the
final stationary-source run.
