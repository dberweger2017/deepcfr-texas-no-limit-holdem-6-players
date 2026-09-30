# Luna browser experiment — time-budget amendment

On September 30, 2026, at about 15:35 UTC, I requested: “actually lets stop it
at 400, then lets do the random 100 hands, this is taking longer than I thought”.
The public server state reported 381 completed hands when the stop instruction
was dispatched. Progress and provisional scores had already been seen.

The [original protocol](luna-browser-benchmark-protocol.md), prompt, harness and
frozen configuration remain preserved. This amendment changes the primary
stopping boundary to **400 completed B100M hands** for elapsed-time reasons.
It does not retroactively make 400 a prespecified target. The server's original
500-hand target is retained, and its normal early-end operation will record
`ABORTED` after hand 400 settles. No hand 401 is authorized.

The same Luna HIGH child/context continues to that boundary. It receives only
the technical stop instruction, without scores, poker advice or hidden records.
All completed hands, browser interruptions and observable failures are retained.
The report must identify this as a shortened, interrupted exploratory run,
not successful completion of the original autonomous 500-hand protocol.

After the primary stops, the separately identified uniform restricted-menu
random adapter and its tests may be added. The planned **100-hand calibration**
still uses a fresh Luna HIGH context and the unchanged frozen poker prompt.
No additional primary run or calibration extension is authorized.

For the ended-early primary only, the reporting command requires
`--allow-aborted --expected-completed-hands 400`. It rejects an active session,
an unfinished-hand abort, a missing explicit count or a mismatched completed
count. Actual HTTP export, native replay and the original status/target remain
authoritative.
