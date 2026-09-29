# M4 local CFR scaling preflight

The [scaling protocol](../blueprint-local-cfr-scaling.md) was frozen at
revision `68c4f604c65dc002ec87b978dcd79cdd2cbd3f12`, with the saved 12M
blueprint and public range artifact SHA-256
`6770fc6871fa286864a31c685172b5cacd1f27ba1bd3e48f2277f4978636272c`.
The source tree was clean. The first two preselected observations were each
solved once with targeting on and off, using a 20-second solve limit.

All four attempts retained their rows and completed **710–1,029 full cycles**,
well beyond 128. The 5- and 15-second completed-cycle snapshots were present
in every attempt; the last completed-cycle state was retained near 20 seconds.
Targeting on produced the actual decision policy in both cases. Targeting off
produced one policy and left the actual information set unvisited in the other.
This is an expected diagnostic outcome, not a lost attempt. The process took
135.0 seconds including checkpoint load and reached **8.08 GiB peak RSS**,
below the 10.5-GiB guard. The four attempts each used about 20 seconds.
No unexpected exception or invalid observation occurred.

This preflight checks that the longer budget, frozen ranges, snapshots and
resource guard execute on the M4. It is not combined with the overnight run
or used to infer playing strength. Its [manifest](local-cfr-scaling-preflight-m4-manifest.json),
[four attempt rows](local-cfr-scaling-preflight-m4-attempts.jsonl) and
[result](local-cfr-scaling-preflight-m4-result.json) are versioned. The attempt
file SHA-256 is `dee34f31b03e89c04b48662551d8695cda3b2698a03cd68b15a9c4b6bb897807`.
The full output, including TensorBoard events and checksums, remains on the M4
at `/Users/dberweger/Local/local-cfr-scaling-preflight-107`.
