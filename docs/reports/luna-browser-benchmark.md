# GPT-6 Luna browser poker experiment

## Preflight — completed September 30, 2026

I ran a separate ten-hand preflight before the planned primary experiment.
The runtime confirmed `gpt-6-luna` with `high` reasoning. It used only rendered
table observations and browser controls, with no inherited parent conversation.
The frozen [protocol](../luna-browser-benchmark-protocol.md) records the prompt,
game, model hash, run order and sampling rules.

| Check | Preflight evidence |
| --- | --- |
| Completed hands | 10 / 10; five button/SB, five BB |
| Player net | 0 chips / 0 BB / 0 BB per 100 |
| Wins / losses / ties | 3 / 7 / 0 |
| Native replay | All ten settlement amounts, chip conservation and public digests matched |
| Human decisions | All 32 attempted actions reconciled with accepted native actions |
| Card reading | All 32 reported human-card/board observations matched the native human view |
| Bot lookup | 31 trained lookups, zero fallback; inspected only after completion |
| Browser tools | 56 CUA calls; no prohibited tool or capability detected |
| Parent poker intervention | None |

Three setup tool failures occurred: an existing-tab lookup, unsupported live
visibility in a child thread, and an unbound table handle after tab creation.
Luna recovered by binding the same local table before its first poker action.
No subsequent UI error was reported. Tool restrictions are
instructional, not an enforced tool allowlist; the transcript audit is required.
I do not publish private reasoning or the unfiltered model transcript.

### Raw evidence

- [Sanitized HTTP benchmark export](luna-browser/preflight/export.json).
- [Every human decision](luna-browser/preflight/decisions.csv): player-visible cards/board, menu and
  attempted button, accepted native action/target, timestamps and UI confirmation.
- [Per-hand results and public digests](luna-browser/preflight/hands.csv).
- [Tool, configuration, replay and usage summary](luna-browser/preflight/audit.json).

Last rendered-result receipt to action attempt averaged 5.19 seconds (median
5.01 seconds, p95 6.51 seconds). This observable decision interval excludes
earlier reads and the first-ready wait; it does not measure private reasoning.
UI acknowledgment separately averaged 298 ms (median 300 ms, p95 385 ms).

The service's initial sampled RSS was 1,065,728 KiB (1.016 GiB), followed by a
much lower idle resident sample. These are snapshots, not a measured peak or
total logical memory footprint; macOS compression/pageout can reduce RSS.
The longer runs will collect periodic service RSS/CPU and system memory/swap
samples. M1 has 16 GiB RAM. M4 is not used.

Usage counters reported 3,772,472 input tokens (3,644,416 cached), 11,017 output
tokens and 2,623 reasoning-output tokens. These are environment counters, not
published reasoning or a calculated monetary charge.

## Primary and calibration — pending

The 500-hand restricted B100M session will use a fresh, persistent Luna context.
Only after it finishes will I add and test the separately identified uniform
restricted-menu random opponent, then run a fresh 100-hand context. Results will
remain separate. This ten-hand preflight is integration evidence, not a playing
strength estimate or a substitute for either planned session.

### Primary technical supervision log

The primary child ended its first turn at 9/500 with no UI failure. I resumed
the **same agent and context**, using only: “Continue playing until the benchmark
session ends. Keep using the same poker table and the same context. The frozen
player prompt and browser-only technical harness remain unchanged. This
continuation provides no poker advice or result information.” This is recorded
as an orchestration intervention, not silently described as uninterrupted
autonomous execution. No poker action or strategy was supplied.

The child ended another turn after 192 completed hands. A same-context
continuation then failed because its browser tab had disappeared; both the
bound handle and exact-URL lookup failed. The server's public state still
reported 192/500, ACTIVE. I marked this attempt interrupted/incomplete rather
than calling it a completed autonomous run. I approved technical recovery of
the existing session and player context, with the interruption retained in
the report. No new benchmark, changed poker prompt or parent poker action is
authorized by that recovery.

The reopened table retained the original session without any parent poker
action. The same child/context resumed. After 351 completed hands another
browser recovery restored that same table; the child stopped after confirming
recovery, then resumed on a technical continuation. This included an idle gap
of about twenty minutes (15:01–15:21 UTC), which remains part of wall time.
No replacement 500-hand benchmark or context was created. These interruptions
remain explicit qualifications of the eventual result.

At my request for live viewing, I opened a temporary loopback spectator on port
8766. It polls only the existing public state/history and reuses the table
presentation with action controls disabled. Its server rejects every POST (405),
foreign Origin (403) and unknown filesystem path (404). It cannot deal, wager,
advance the bot or end the benchmark. It does not reveal running totals. Luna
remains on the original released UI at port 8765; no game/runtime source changed.
The additional polling load is included in subsequent resource samples.

[Machine and native dependency identity](luna-browser/environment.json) records
the exact environment. The focused service/audit/report/tally suite passed 27 tests.
The earlier `f7ac53e` GitHub CI candidate passed all 952 tests and its existing
CLI/reproduction gates. Subsequent reporting changes receive their own CI run.

When I requested a running score, I tallied only individual hand outcomes already
rendered in Luna's browser. Through hand 106, every hand was represented once
after deduplication, with no conflicting values: Luna was +50 chips / +0.5 BB,
with 31 wins, 74 losses and one tie. This provisional public-UI tally was not
sent to Luna, did not use the active private journal and did not change the
fixed 500-hand target. The final native/server report remains authoritative.

I also questioned whether resetting stacks could corrupt this tally. The service
adds `terminal human stack - 2000` once per completed hand; a fresh deal does not
add profit. Eight full rendered terminal-stack observations independently agreed
with their displayed hand profits. A generated two-hand regression loses 50
chips in the small blind, resets both stacks, then loses 100 in the big blind:
the accumulated result is correctly -150 chips / -1.5 BB. It also replays both
hands. These checks support the accounting; full primary replay remains pending.

## Reproduction and audit

Use the public `v0.4.0` source/model and the [local launch instructions](../play-web.md).
Freeze and hash the linked prompt/harness before outcomes, create the exact-N
restricted benchmark, and dispatch the named model with no inherited context.
After completion, run `python -m scripts.audit_luna_browser ROLLOUT --out PRIVATE_AUDIT`
and `python -m scripts.report_luna_browser PRIVATE_DB ROLLOUT TOKEN_FILE OUTPUT_DIR`.
The report command requires one isolated completed session and the running local
service; it fetches the actual sanitized HTTP export. `--export-file SAVED_EXPORT`
supports later offline analysis of that recorded HTTP response. Private database, token and
rollout remain outside Git. Decision CSVs include only cards available in the
human observation at that decision. Unrevealed bot cards, future cards, seeds,
RNG state, private policy keys and private reasoning are omitted. Failed/incomplete
runs must be retained.

For a provisional running score, `python -m scripts.tally_luna_public ROLLOUT`
reads only rendered individual-hand results and the rendered progress label.
It deduplicates repeated observations and rejects missing or conflicting hand
results. It reads neither the active private journal nor private reasoning.
Do not send its running totals to the player. Reconcile them against the final
native/server export after the fixed session ends.

At hand 175 the initial tally rejected conflicting player metadata: an automatic
bot fold could finish the next hand before the player emitted metadata labelled
with the previous hand. The tally now uses the released UI's progress semantics
(`completedHands + 1` while active), with a regression for this case. The rendered
results cover all 192 completed hands once, without gaps or conflicting amounts:
+2,350 chips / +23.5 BB, 68 wins / 122 losses / 2 ties. The earlier through-106
tally remains +50 chips. These are provisional public observations, not private
journal validation or information supplied to Luna.

Some player observations named their rendered button list
`visibleLegalButtonLabels` rather than `legalButtonLabels`. The reporting helper
accepts either spelling, retaining the actual labels and requiring a valid list.
This is a reporting correction; neither the player harness nor game runtime
changed. Native action counts, hand order and replay checks remain required.
