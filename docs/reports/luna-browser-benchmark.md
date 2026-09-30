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

## B100M primary — stopped at 400, verified

The authoritative result is **+10,700 chips / +107 BB / +26.75 BB per 100**
for Luna against the released first-seed B100M. This is an unfavorable observed
result for B100M on this run, not a confidence-qualified relative-strength claim.
The original 500-hand protocol was not completed. The time-budget amendment
was made after progress/scores were known, and browser interruptions required
technical supervision in the same context.

| Measure | Verified primary result |
| --- | --- |
| Completed / original target | 400 / 500; ABORTED after completed hand 400 |
| Wins / losses / ties | 164 / 231 / 5 |
| Button/SB | 200 hands, +5,450 chips / +54.5 BB |
| BB | 200 hands, +5,250 chips / +52.5 BB |
| Average terminal pot | 709.25 chips / 7.0925 BB |
| Native replay / conservation / payoff / public digests | All 400 passed |
| Accepted human decisions / raw attempts | 978 / 979; one explicit retry retained |
| Attempt/native mismatch | One: hand 343 declared Call 1 BB, accepted Fold; outcome retained |
| B100M lookups | 894 trained, zero fallback (0%) |
| Tools | 1,488 CUA calls; no prohibited tool/capability detected |
| Parent poker decisions | Zero |
| Benchmark elapsed time | 3h 27m 26s, including setup, pauses and technical recovery |

### Primary raw evidence

- [Actual HTTP benchmark export](luna-browser/primary/export.json), retaining target 500/status ABORTED.
- [All accepted native human decisions and logged attempts](luna-browser/primary/decisions.csv).
- [All per-hand results and digests](luna-browser/primary/hands.csv).
- [Sanitized public event history](luna-browser/primary/public-history.json), reconstructed after replay with the service serializer, including bot actions and legitimate reveals.
- [Sanitized raw browser metadata](luna-browser/primary/browser-metadata.json), including deal/retry observations.
- [Tool/configuration/replay/lookup/usage audit](luna-browser/primary/audit.json).
- [Resource samples](luna-browser/primary/resources.csv) and [measurement summary](luna-browser/primary/resources.json).

All 978 accepted human decisions were correlated with native replay. The hand-160
retry is explicit in the raw table; it is not a second wager. Two decisions lack
an emitted observation record, and one further record lacks its observation
timestamp. Those three acknowledgment intervals remain missing, not invented.
Hand 400 used `hand`/`decision` field spellings; the audit normalizes those aliases
while retaining their original values. No full rendered tree or private reasoning
is published. The first stopping acknowledgment confused the UI's post-settlement
progress with the completed count; the public server confirmed 399, and the same
child played the one remaining hand before the technical end operation.

Last rendered receipt to attempted action: mean **6.64 s**, median **6.12 s**,
p95 **9.98 s** across 978 decisions. Emitted acknowledgment intervals: mean
4.92 s, median 0.416 s, p95 10.28 s across 975 timestamps. These include logging
and tool delays and do not isolate engine/network latency or private reasoning.

The service's sampled `top MEM` ranged **1356M–1365M** (about 1.3 GiB). Sampled
RSS ranged 5,504–271,280 KiB after startup; compression/pageout makes that
inappropriate as the whole model footprint. Host swap ranged 693–5,378 MiB,
including other applications. Thirty-second CPU samples peaked at 0.2% for the
service; this can miss brief action bursts. I stopped the service, spectator,
sampler and its owned caffeinate process after export. M4 was not used.

Surfaced thread counters: **213,616,234 input tokens** (210,588,416 cached),
349,129 output tokens and 75,280 reasoning-output tokens. These are usage
counters, not reasoning content or a monetary charge. The runtime did not
provide a separately attributable experiment dollar cost.

## Uniform-random calibration — pending

The separate uniform control was added only after the primary stopped and was
uploaded. Its [definition and reproduction guide](../luna-uniform-random-calibration.md)
use the same concrete restricted menu and persisted per-session bot RNG, with
no B100M modification or model load. The server accepts only restricted
benchmark sessions for this opponent. Trained/fallback lookups are inapplicable.

A fresh `gpt-6-luna` / `high` context is playing a new exact-100 session, using
the same frozen prompt/harness and no primary history. The
[calibration configuration](luna-browser/calibration-config.json) pins runtime
source `7ce9ba600ec9a1822bea19a9396b3c540e6797af` and the algorithm-definition
hash. All **969 full fixture tests** passed on M1 in 474.65 seconds at that
source, and its GitHub CI passed. The later reporting-only tally change adds
one focused test; **37 focused checks** pass. Results remain separate and
calibration replay/exports await completion.

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
the exact environment. Before the random adapter, the focused service/audit/report/tally suite passed 31 tests.
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

The tool audit also counts expired browser handles during play, rather than
only failed URL lookups. It publishes sanitized failure categories and call
identifiers, without copying browser-session identifiers or tool error traces.

[Active HTTP boundary evidence](luna-browser/primary-active-boundary.json)
records public state/history requests during the primary session. Forbidden
private fields were absent; final result/export requests returned 409 and
diagnostics returned 403. This is a bounded check of the actual service, alongside
the fixture tests; it is not a replacement for final native replay and disclosure
validation. No active private journal was read.
