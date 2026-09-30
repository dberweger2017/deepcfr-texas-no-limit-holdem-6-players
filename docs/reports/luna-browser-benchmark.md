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

## Uniform-random calibration — completed and verified

A fresh `gpt-6-luna` / `high` context completed exactly **100/100 hands** against
uniform random, earning **+1,300 chips / +13 BB / +13 BB per 100**. This weak
control has its own deal stream and no inherited primary memory. Its result is
not combined with B100M's result, and I report no confidence interval.

| Measure | Verified random result |
| --- | --- |
| Status / hands | COMPLETE; 100 / 100 |
| Wins / losses / ties | 41 / 59 / 0 |
| Button/SB | 50 hands, +4,400 chips / +44 BB |
| BB | 50 hands, -3,100 chips / -31 BB |
| Average terminal pot | 880 chips / 8.8 BB |
| Native replay / conservation / payoff / public digests | All 100 passed |
| Human decisions / raw attempts | 217 / 218; one explicit failed-click retry |
| Attempt/native or reported-menu mismatches | Zero |
| Missing observation records / timestamps | Zero |
| Bot decisions | 189; trained/fallback classification is inapplicable |
| Tools | 331 CUA calls; no prohibited tool/capability detected |
| Parent poker decisions / session restarts | Zero / zero |
| Elapsed time | 39m 05s, including setup and a hand-12 idle gap |

The unchanged frozen prompt/harness was submitted to a fresh child. The
[configuration](luna-browser/calibration-config.json) pins the runtime source
`7ce9ba600ec9a1822bea19a9396b3c540e6797af`, distinct from the released primary.
The [adapter definition and reproduction guide](../luna-uniform-random-calibration.md)
specify uniform weights over exactly the same concrete restricted legal menu,
using the normal persisted per-session bot RNG, native engine and journal path.
The definition hash identifies this algorithm, not trained model bytes. The
server permits only restricted benchmark sessions for this adapter. No B100M
artifact was loaded for the control.

Three setup failures occurred (two existing-tab lookups and unsupported child
live visibility). On hand 12 a click failed; a fresh visible observation confirmed
that the turn remained active and Luna retried the same Fold. The native journal
contains one Fold. The raw retry marker was a descriptive string rather than a
boolean; the reporting helper now retains and recognizes that explicit marker,
with a regression test. Its initial attempt-to-final-ack interval includes the
retry. The child's final statement of no errors does not override this evidence.
No parent continuation or browser recovery was required during calibration.

### Random raw evidence

- [Actual HTTP benchmark export](luna-browser/calibration/export.json).
- [All accepted decisions and raw attempts](luna-browser/calibration/decisions.csv).
- [Per-hand results/digests](luna-browser/calibration/hands.csv).
- [Actual completed HTTP public history](luna-browser/calibration/public-history.json).
- [Sanitized browser metadata](luna-browser/calibration/browser-metadata.json).
- [Replay/configuration/tool/usage audit](luna-browser/calibration/audit.json).
- [Actual completed HTTP allowlist/disclosure validation](luna-browser/calibration/http-validation.json)
  and [active HTTP denial checks](luna-browser/calibration-active-boundary.json).
- [Resource CSV](luna-browser/calibration/resources.csv) and [summary](luna-browser/calibration/resources.json).

All 100 HTTP history entries exactly matched the native human-view allowlist and
legitimate disclosures after completion. Active result/export returned 409 and
benchmark diagnostics returned 403. Reporting metadata now uses a field allowlist,
so arbitrary embedded rendered trees are excluded as well as private reasoning.

Observed decision interval mean/median/p95: **7.62 / 5.66 / 9.23 seconds**.
Attempt-to-emitted-ack interval: **0.345 / 0.300 / 0.405 seconds**. Neither is an
isolated engine/network latency or private reasoning measurement. The mean
includes a long hand-12 delay; it is not silently removed.

The random service's 91 samples showed `top MEM` **19M–24M**, RSS
**9,296–27,968 KiB**, and sampled service CPU at most 0.2%. Host swap was
2,942–3,926 MiB, including other applications and concurrent fixture tests; this
is not random-service allocation. Its final RSS snapshot was 27,856 KiB.
I stopped the server, sampler, spectator and owned caffeinate process after
export. There was no M4, training or paid RunPod use.

Surfaced usage: **45,702,080 input tokens** (45,019,904 cached), 62,348 output
tokens and 11,228 reasoning-output tokens. No separately attributable dollar
charge was supplied; these counters do not disclose private reasoning.

## Fixed qualitative review — both results frozen

The frozen selection uses first ten hands, every 50th ordinal, and the five
largest positive and negative **terminal human payoffs**, with ordinal tie-breaking
and deduplication. This is not selection by pot size. The manifests retain every
selected public event, human-visible decision and selection reason:
[28 primary hands](luna-browser/primary/qualitative-sample.json) and
[21 random hands](luna-browser/calibration/qualitative-sample.json).
No solver, private reasoning or unrevealed bot cards were used for this review.

| Observation | Evidence and practical limit |
| --- | --- |
| Legal controls and readable game state | All accepted wagers were restricted-menu legal; no recurring rule violation or typed wager exists in these restricted runs. The hand-343 attempted-call/accepted-fold mismatch is a computer-use error, not a strategy judgment. |
| Cautious folds versus large bets | Primary hands 6, 151, 166 and 210 fold after earlier investment. Hand 6 folds river top pair to a pot-sized bet; this is observable caution, not proof of an EV error without the opponent strategy. |
| Weak holdings sometimes enter raised pots | Primary 4/8/10 call with K2/K4/K5, and random 4 calls J7 then folds the flop. These are candidates for reviewing preflop selectivity; the sample alone cannot label every call a mistake. |
| Small bets and calls with strong made hands | Primary 5 bets one BB on the flop, checks the turn and calls one BB on the river with trip aces; 46 bets one BB on the river then calls the shove with a full house. Random 20/61 call large raises with trips. This does not prove missed value or optimal sizing. |
| Coherent showdown calls | Primary 200 calls three one-BB bets and wins with A-high on a trips board. Random 41 calls the river shove on a five-spade board holding two spades and wins. These are compatible with coherent bluff catching/hand reading, not evidence of private reasoning. |
| Losses from observable calls | Primary 188 calls small flop/river bets with unpaired AK and loses to a paired six; random 40 calls a pot-sized river bet with Q5 and loses to Q9's kicker. Random 74 calls a turn shove with QQ and loses when the river gives the opponent trips. The last example is a bad outcome, not proof that the call was wrong. |
| Large winners materially affect the aggregate | Primary has twelve +20-BB hands and no -20-BB hand; random has five +20-BB hands and one -20-BB hand. The top-five positive sample alone contributes +100 BB in each run, underscoring variance rather than establishing strength. |

I cannot quantify how much profit would have changed without the computer-use
error: no counterfactual hand was played. Primary hand 343's actual Fold and
loss remain in the ledger; retries did not create duplicate bets. Poker actions
account for the recorded outcomes, but an action's strategic quality is not
identified by its realized payoff. The most useful recurring review candidates
are weak-hand preflop calls and investing before folding to large bets. There is
no supported ranking of “biggest mistakes” by EV in this experiment.

A post-hoc action-count check across primary quarters finds raises **46/251,
69/250, 69/227, 77/250** decisions (about 18%, 28%, 30%, 31%). This is an observed
change in action mix, not proof of adaptation: cards, positions and opportunities
also changed. The corresponding net quarters were +5.5, +13, +73 and +15.5 BB.
Random quarters were +37, -14.5, -3 and -6.5 BB. Counts are retained in the sample
files; no strategic memory or coaching was added between opponents.

## Handoff answers

1. **500 autonomous hands?** No. I explicitly shortened the primary to 400 after seeing progress/scores. It retains original target 500/status ABORTED and technical interruptions in one context. The separate random target 100 completed.
2. **Information boundary?** No prohibited tool or hidden-information access was detected. Restrictions are instructional; actual HTTP boundary checks, native disclosure comparison and fixture hidden-world tests provide bounded evidence, not a mechanically enforced child tool sandbox.
3. **Versus B100M?** Luna +107 BB over 400, raw +26.75 BB/100; all hands verified.
4. **Versus random?** Luna +13 BB over 100, raw +13 BB/100; all hands verified. Separate opponent and fresh context.
5. **Poker versus computer use?** One primary intent/accepted mismatch, one retry in each run, missing primary metadata and tool recoveries are retained. No counterfactual attribution of net profit is justified.
6. **Recurring mistakes?** The observable review candidates above merit investigation; there is no solver-backed EV mistake ranking or demonstrated recurring rule misunderstanding.
7. **Adaptation?** Later primary aggression increased in a post-hoc action mix; adaptation cannot be established from that alone.
8. **B100M fallback?** 0/894 (0%); inspected only after ending the primary.
9. **Latency/runtime?** Primary decision mean/median/p95 6.64/6.12/9.98 s; wall time 3h27m26s. Random 7.62/5.66/9.23 s and 39m05s. Logging intervals have the limits stated above.
10. **M1 sufficient?** Yes for both services, browser play and tests on this 16-GiB M1. B100M's sampled logical memory was about 1.3 GiB; compression/host swap and child tab lifecycle still affected observability/stability. All owned processes are stopped.
11. **Extend random to 500?** More hands could clarify this noisy +13-BB control, but the current position split and large-pot sensitivity do not establish a stable advantage. No extension was run or authorized.
12. **One next recommendation:** A preregistered, time-budgeted replication of the restricted HU20 Luna-versus-B100M experiment after fixing browser tab lifecycle/recovery, with the same frozen prompt and exact completion target.

No three-player run is included: the featured B100M is heads-up only. A future
three-player experiment would need a separately verified TP20 policy and protocol.

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
CLI/reproduction gates. The latest reporting candidate `fd90a1c` passed GitHub CI with 998 tests in 509.86 seconds, CLI checks and both solver-reference/replay gates; GitGuardian also passed. The platform-specific full M1 count at the calibration runtime was 969.

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
hands. These checks support the accounting; the completed primary native replay also passed for all 400 hands.

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
