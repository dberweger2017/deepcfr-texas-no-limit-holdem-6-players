# D integration and resource prefix

D uses `hu20-native-reopening-compressed-history-card-v2`, combining the unchanged
`hu20-earlier-streets-public-summary-v1` with #143's exact descriptor file:
SHA-256 `190d530ce66d65a031334d95600ce0f005d81324a7353170dcebf1d64a6ccd92`.
Preflop keeps B's exact full-v2 key; postflop compresses earlier public streets
and keeps the current street ordered. A/B/C compatibility is checked against
retained executables using exact checkpoint and current-export bytes.

Before rental, three D resource prefixes use seeds 2026093001/02/03 from zero,
with fixed 250k/500k/1M/2M completed-node milestones. This measures entries,
visits by street, throughput, peak memory and checkpoint/export costs, without
strength outcomes. B's existing prefix is retained; no B training is repeated.
The numerical history acceptance gate remains the separately approved C 10M
measurement, with its original 2M failure preserved.

Run sequentially on M1 with pinned Python 3.11.14 / engine 5db20e3, one worker,
6 GiB RSS / 8 GiB free-disk guards and a two-hour total engineering limit.
The 4M entry ceiling is a prefix safety guard, not the eventual 100M allowance.
Hash-check every prefix artifact, exact reload/export and two independent fresh
process next iterations. Preserve any failed phase without automatic retry.

Resource admission must project C and D separately, include serialization peaks,
and retain enough disk for full checkpoints, recovery and retrieval. Linux parity
and live provider shape/rate admission precede paid 100M traversal. No rentals
until the owner approves the quoted all-in cap; no schema, seed or node-work
changes to fit hardware. #136 retains M4 and its 90-minute schedule.
