# HU200 river search: amended cost gate passes; strength pending

[PR #317](https://github.com/dberweger2017/CFR-NLHE-6max/pull/317), October 17, 2026.
**Draft; final strength study not declared or run.** The owner-approved disjoint
128 direct plus seven by 32 calibration completed under its hard two-hour cap.
This is a cost/engineering readout, not a final strength result or a release
recommendation. The original latency failure remains below; the owner-amended
10-second gate passes. The owner chooses final counts/venue before final hands
and decides merge.

## Finding

River-only search on #306's audited HU200 K50 blend completes all **24 fixed
real decisions** on the 16 GiB Apple M4. Memory and telemetry pass; the
**3-second p95 target fails** by a wide margin. No iteration, range, menu or
sample reduction was made after timing. No Slumbot play or strength hands ran.

| Measured on the fixed sample | Result |
|---|---:|
| Complete decision p50 | **15.03 s** |
| Complete decision p95 | **36.79 s** |
| Complete decision maximum | **39.94 s** |
| Sampled simultaneous process-family peak | **4,411,490,304 bytes /4.41 GB** |
| Minimum measured host available memory | **5,579,259,904 bytes /5.58 GB** |
| Search completions | **24/24** |
| Fallbacks /timeouts | **0/0** |
| Successful decisions missing model EV | **0** |
| Native solves | **48** |
| Actual-request public-tree sizes | **2,121–9,651 nodes** |

The percentile is the empirical linear percentile of these 24 selected spots,
not a population latency guarantee. Host available memory includes reclaimable
memory. Family RSS is sampled every 20 ms, alongside the outer M4 guard;
it is not a sum of separate process lifetime maxima. The guard did not stop.

The [prospective protocol](../hu200-river-search.md) and
[committed sample declaration](hu200-river-search-artifacts/cost-declaration.json)
fix six free and six facing-bet decisions from each of #304 and the blend's
verified exploratory prefix, selected by public-decision hash without outcomes.
Each spot gets a fresh policy. Earlier hero river actions in those blueprint
journals remain blueprint locks. Both ranges enumerate all card-compatible
holdings; all observed opponent river prefixes use the declared blueprint
likelihood law. Native work stays **50 iterations, six threads, uncompressed,
full menu, 2 GiB allocation ceiling, 300,000 public-node limit**, with a
120-second search-work deadline. Fallback, ledger, comparison-action and output
costs remain in total service time. Model/cache/table verification precedes play.

Scientific source is `11fa4fa424ec577cf8492338fd61c60ed7f3c97c`.
The [input receipt](hu200-river-search-artifacts/inputs.json) binds the exact
6,282,968,150-byte blend, model SHA256, read-only cache and K50 tables.
[Summary](hu200-river-search-artifacts/cost-summary.json) and
[all 24 compact decision rows](hu200-river-search-artifacts/cost-decisions.csv).

## Where the time goes

The 24 complete decisions total **427.81 seconds**. Native process work,
including solving, EV finalization and JSON profile export, totals **142.73 s**.
Adapter startup, profile parsing and evidence I/O outside the native completion
timer add **223.62 s**. Public-tree preparation totals **24.52 s**, range
conditioning **14.26 s**, and opponent-lock preparation **6.21 s**. These are
measured components, not isolated causal speedup estimates.

The inherited resolver tries an earlier solution before rebuilding a request
with observed river likelihood locks. This produces 48 native solves for 24
decisions. Full JSON profiles retain complete holding strategies and EVs at
all public nodes; their output/parsing cost is substantial. The data identify
engineering candidates, but do not establish that either will meet 3 seconds.

Search changes the same-uniform-draw comparison action at **11/24** sampled
spots. Conditional model EV delta averages **+59.02 chips**, median **+2.44**;
all individual deltas are in the CSV. These compare both mixtures against the
same solved continuation at the same spot. They are **not realized gains,
causal action values or a strength estimate**; this selected cost sample is not
a frequency estimate for normal play.

## Correctness and design limits

Native EV and best response match the independent Python reference within
**0.0004 chips** under the stipulated small ranges, including exact wagers
7,333 and **12,000 chips (1.5× the 8,000-chip pot)**. Prior hero matrices stay
locked, native repetitions are identical, and exported root action EVs agree
with independent profile EV. Constrained responses preserve request locks;
unrestricted best responses are reported separately. A complete searched river
continuation reproduces identical actions and settlement from fresh player
state using persisted matrices and availability records, without replay solves.

Expanded M4 regressions pass 118 checks, a separate allocation-fallback check,
and 61 focused checks after the external-journal boundary repair. Independent
Python review findings are addressed. [Correctness](hu200-river-search-artifacts/correctness.json),
[searched-hand replay](hu200-river-search-artifacts/native-hand-replay.json),
[review scope](hu200-river-search-artifacts/independent-review.json).

The external solver has no safe re-solving gadget. This implementation uses
ordinary joint-range CFR with complete earlier hero locks and observed
opponent-prefix likelihood locks. Future opposing responses are equilibrium
responses. It can still mis-model value callers and future folds, as in the
HU20 stackoff diagnosis. No seven-opponent or pot_pressure effect, direct gain,
interval or losing-range diagnosis exists yet: the cost gate blocks those hands.

Wall-clock availability is a recorded input to exact replay. Fixed seeds and
work reproduce completed native matrices; seeds alone do not guarantee the
same timeout behavior under a different machine load. The persisted full
policy/support/failure ledger refuses missing or changed replay evidence.
Tiny off-path conditional EVs can be unavailable; their mask is preserved.
All 24 actual searched decisions here have finite EV telemetry.

## Failures retained

The first full cost attempt reached **zero native solves**: all 24 journal
roots failed strict synthetic replay because their hand-start events omitted
default physical-seat metadata. Its p95 2.01 seconds is fallback plumbing time
and never qualifies the search. The repair expands only validated Table
defaults in a synthetic copy; strict betting/decision/board replay remains,
with a corrupted-payment regression. The same frozen spots and solver work
then complete in the reported second attempt.

Earlier preparation/reference plumbing failures are also retained: unsupported
rsync display flag, action-label/order comparison, mismatched constrained versus
unrestricted BR objectives, optional EV normalization and missing test dependencies.
Completed native profiles were reused after request/member verification where
possible. No research outcome was rerun, tuned or extended. [Failure record](hu200-river-search-artifacts/failures.json).

## Owner options

### Owner-selected engineering continuation

The owner selected lossless transport/adapter optimization, followed by exact
parity on the same 24 spots and an amended **p95 at most 10 seconds** gate,
retaining the 3 GB headroom requirement. The original 3-second result above is
unchanged. No strength hands are permitted before an owner-approved count
proposal, even if the amended cost gate passes.

The first transport implementation passes complete fixed 24 output parity at
source `51c46f7a`: final requests/ranges/menus/locks, every raw and normalized
finite strategy/EV bit, unavailable masks and diagnostic weights agree with
the retained original. Full played matrices/support and same-seed actions agree;
fresh player replay invokes no solver. [Parity receipt](hu200-river-search-artifacts/optimization-parity.json).
The external executable `9e1efe275c2eab216e3af9da20d55dea9436e88aacb057a41b86b32c171cf914`
changes export only; no solve, iteration or finalization logic changes.
Strategies and diagnostics travel as the same promoted float64 values, retaining
all nodes and the existing normalization. Source/dependency pins remain in the
[external receipt](hu200-river-search-artifacts/optimization-external-source.json).

Certified off-menu ancestor elimination reduces native solves 48 to 35 on this
sample. An absent next action proves the old candidate cannot be reused;
existing candidates and uncertified on-menu ancestors remain authoritative.
Unconditionally removing the latter could change rare reusable policies, so
their omission is not claimed as an equivalent optimization.

The first amended timed run, source `817de444` with the same entire search
source tree as the parity run, still **fails 10 seconds**: p50 **5.64 s**, p95
**10.766536 s**, maximum **13.04 s**. All24 searches finish without fallback,
timeout or missing actual EV. Peak family RSS is **3,356,459,008 bytes**;
minimum host available memory is **5,000,101,888 bytes**. Memory passes.
[Failed attempt receipt](hu200-river-search-artifacts/optimization-cost-attempt-01.json).

The authorized parser/I/O continuation at `214695a3` shares the two validated
holding tables and reads numeric views through a read-only mapping, retaining
the same owned copies and normalization. Native executable and search settings
are unchanged. 72 focused M4 checks pass and independent review finds no blocker.
Its full 24 parity gate now passes: all scientific requests, raw and normalized
finite bits, unavailable masks, diagnostic weights, played policies/actions/EVs
and fresh replay match the original. Native solves remain 35. A restart
interrupted the first pending validation without a surviving result; the
owner-approved continuation restores exactly the same source. A guarded
read-only continuity audit verifies 919 files /27,098,539,771 bytes with zero
hash mismatches and loads the cache against the pinned model SHA. The original
and first amended completed failures remain unchanged.
[Continuity](hu200-river-search-artifacts/restart-continuity.json),
[full parity](hu200-river-search-artifacts/optimization-parity-02.json).

The fixed amended cost gate then **passes** at the same source:

| Same fixed 24 spots | Result |
|---|---:|
| Complete p50 |4.727273 s|
| Complete p95 |**9.648689 s ≤10 s**|
| Maximum |11.106579 s|
| Search-family peak RSS |2,912,206,848 bytes|
| Minimum host available memory |5,323,390,976 bytes|
| Complete /fallback /missing actual EV |24 /0 /0|

The combined parity-and-cost guard completes without breaches, with a sampled
combined-stage family peak of 3,640,115,200 bytes. All 72 focused post-restart
checks pass. The external executable, search work, ranges, menus and locks are
unchanged. The fixed-sample percentile is an engineering gate result, not a
population latency guarantee. The original 3-second target still fails.
[Cost](hu200-river-search-artifacts/optimization-cost-attempt-02.json),
[inputs](hu200-river-search-artifacts/optimization-cost-inputs-02.json),
[combined guard](hu200-river-search-artifacts/optimization-parity-cost-guard-02.json),
[post-restart checks](hu200-river-search-artifacts/optimization-post-restart-test-guard.json).

Complete service time totals **126.97 seconds** across the 24 spots: native
process work totals **78.62 seconds**, and adapter transport outside that timer
adds **14.70 seconds**, versus the original 223.62 seconds. Ancestor elimination
and transport both contribute; this is not an isolated causal parser benchmark.
Same-draw action changes remain **11/24** and model EV deltas remain identical
(mean +59.02 chips, median +2.44), as required by parity. These conditional
solved-continuation estimates are not realized strength gains.
[Detailed M4 readout](hu200-river-search-artifacts/optimization-cost-readout-02.json),
[all 24 decision rows](hu200-river-search-artifacts/optimization-cost-decisions-02.csv).

No concurrent pilot or strength hands run. The unchanged disk floor blocks
another full-profile concurrency pilot at current capacity. The
[strength-count proposal](../hu200-river-search-strength-proposal.md) quotes
both runtime and evidence volume, labels historical variance/frequency proxies,
and identifies storage plumbing that must be resolved before play. Transport
engineering has observed no poker strength outcomes and changes no strategy.

### Original response choices and remaining scope

1. **Recommended: a separate M4 engineering slot**, preserving 50 iterations,
   all ranges and menus. Investigate compact lossless matrix transport and
   avoiding discarded ancestor work. Require decoded-policy/EV/request parity
   where claimed and another declared cost measurement before strength. An
   unseen ancestor occasionally could have been reused, so eliminating that
   path universally is a policy change to identify and validate explicitly.
2. **Authorize a slower offline research budget.** Keep the measured search,
   waive the interactive latency target explicitly, and first quote variance
   pilot, powered counts, replay and total M4 time. Do not start a study over
   about 24 hours without owner-approved counts. No such counts are declared.
3. **Amend this PR to stop at the feasibility finding.** Preserve the failed
   cost result and omit strength claims. This requires an explicit scope
   decision; the current PR remains draft while planned strength is unrun.

The owner explicitly selected option 1 with the 10-second amendment above.
The original M4 preservation stage completed and its reservation released at 22:58
Madrid, October 16, after successful archival guard closure, verified cable
copy and a fresh common-lock availability check; the engineering continuation
uses a separate reservation. The verified local ZIP and
pending Drive acceptance are distinguished in [RESULTS_INDEX](../../RESULTS_INDEX.md#pr317--hu200-river-search-completed-cost-gate-strength-pending).
The large upload waits for existing Slumbot play under the shared-uplink policy;
no upload acceptance is claimed. All originals remain protected. Final-review
readiness remains unresolved until owner-approved strength research and
storage closeout actually finish. The amended cost gate is complete.

## Incremental preservation and next decision

The completed engineering evidence is preserved uncompressed on bb128 and in
a 2,359,632,223-byte ZIP; all 1,507 payload files /19,478,922,634 bytes pass
independent disk and ZIP-member SHA256 readback. A supplemental 43-member ZIP
retains closed guard/readout/continuity administration. Both archives and all
originals remain. Drive upload still waits for existing Slumbot play; no cloud
acceptance or final research completion is claimed.
[Receipts and both restoration routes](../../RESULTS_INDEX.md#pr317--hu200-river-search-completed-cost-gate-strength-pending).

The next owner decision is evidence I/O resource plumbing, followed by a fixed,
separately declared variance/cost calibration before choosing the final strength
counts. The proposal suggests 128 direct duplicate blocks and 32 paired blocks
per each of seven scripts for that calibration; no such hands are authorized or
played here. The much larger historical-proxy count/time/storage estimates are
planning evidence, not a commitment to a five-day campaign.

## Evidence retention and measured concurrency, October 17

The owner approved evidence I/O plumbing before a separate variance calibration.
The scientific-request SHA256 rule selects full native bodies independently of
outcomes at a fixed 1% rate; every native failure retains its full generated
body. Every decision retains full policy/support/fallback ledgers, telemetry
and original-byte checksums. Retained requests, headers and selected/failed
bodies are losslessly compressed. No range, menu, iteration, lock, game rule or
solver calculation changes.

All 24 fixed spots retain exact scientific-request, numeric-body, normalized
probability/EV/mask, played-matrix/support/action and fresh-replay parity. Source
`5b6b2f6c` measures sequential p50/p95/max **4.640/9.834/11.511 s**, peak family
RSS **2,898,165,760 bytes**, minimum host available **6,221,905,920 bytes**;
all 24 complete with zero fallbacks or missing EVs. The original 3 s failure
and all earlier cost results remain unchanged. These empirical percentiles do
not guarantee population latency.

| Concurrent searches | Decisions/s | Complete-work wall | p95 per decision | Peak family RSS | Minimum host available | Full parity |
|---|---:|---:|---:|---:|---:|---|
|2|0.318685|75.309 s|10.711 s|4.719 GB|5.999 GB|24/24|
|3|0.419357|57.231 s|11.510 s|6.279 GB|6.074 GB|24/24|

The offline throughput choice is three workers, frozen before calibration
outcomes. Each native solve still uses six threads; concurrency contention is
reported separately from the sequential interactive cost gate. Fixed24 evidence
is **54,319,419 bytes** for 35 native calls, including one selected full body;
other tree/sample/failure distributions can require more storage.
[Engineering readout](hu200-river-search-artifacts/evidence-io-engineering-readout.json).

Intermediate plumbing stops remain preserved: runtime RSS fields made raw
header hashes differ although scientific/numeric outputs matched; later the
usual disk floor stopped a three-worker attempt before any completed spot.
The corrected comparison excludes only runtime RSS semantics, while retaining
original-byte checksums, and selected/failed bodies are also losslessly
compressed with fresh disk admission. The owned-array diagnostic found no
retained memory-map leak; the parser was unchanged. Completed native outputs
were reused only after scientific input, raw numeric SHA and header semantics
matched. Full24 parity precedes the final cost/concurrency measurements.

The [prospective calibration protocol](../hu200-river-calibration-protocol.md)
and pinned execution/continuation declarations precede any played calibration
action. The initial evaluator callback wiring failure occurs before any action,
hand or native solve; its repair and fixture setup correction pass all **84**
focused M4 checks. The original absolute two-hour deadline is retained rather
than restarted. Final hands remain zero.

## Completed calibration; proposed finals, not a strength result

All **352 blocks / 1,152 primary hands** replay and reproduce with fresh players
and **zero native reproduction solves**, crossing the global checksum/identity
barrier before readout. The original two-hour clock includes the launch repair:
worker closure takes **604.32 s**. No completed research is rerun or extended.

Three-worker peak family RSS is **10.326 GB**, minimum host available **6.033 GB**.
All memory checks pass. There are **202 searched decisions, 67 changed actions,
zero fallbacks/timeouts and zero missing successful EVs**. Full decision/EV and
policy/failure evidence remains archived; calibration mean effects are withheld
from variance planning, and its roots/deals stay excluded from finals.

Direct block SD **1,086.84 BB/100**, normal-model 95% interval
**[968.04, 1,239.15]**, upper one-sided 95% SD **1,212.92** gives the declared
rounded recommendation: **131,072 direct blocks + 2,048 per each seven scripts**,
approximately **84.7%** power for +10 BB/100 under that model. M4 compute forecast
is **73.4 h** including local verification/setup and 25% headroom, or **87.4 h**
using an upper block-cost projection. Cygnus remains unbenchmarked while #301
computes: its pre-XMP trainer proxy gives **448.6 h**, with an explicitly
conditional one-worker scenario range **184.6–448.6 h**, within eight busy threads.
No final hand is authorized by those numbers.

The panel pilot is sparse: pot_pressure has only **five searched decisions and
two nonzero blocks**, and its projected seven-way halfwidth is about **117.3
BB/100** at 2,048 blocks. This panel count is exploratory, not a per-opponent
+10 power claim. Zero observed SD on two panels does not exclude unobserved tails.
The full SD uncertainty and every opponent are in the linked proposal/summary.

[Final-count/venue/storage proposal](../hu200-river-search-final-proposal.md),
[verified calibration summary](hu200-river-search-artifacts/calibration-summary.json),
[barrier](hu200-river-search-artifacts/calibration-all-hands-verified.json),
[calculated options](hu200-river-search-artifacts/calibration-final-count-proposal.json).
The recommended evidence projection is **90.6 GB**, beyond M4 local admission;
closed-block transfer/verified local retirement needs the owner's specific
choice before implementation and qualification. No old research is removed.

The M4 was released **October 17 at 08:49:03 Madrid**. The new **13,496-member,
1.642 GB** uncompressed bb128 evidence and **633,416,285-byte** ZIP both pass all
SHA256 readback checks, including every failure/partial/control/source pin.
ZIP SHA256 `efa83610c98cbb4d8f93245d7472673fe5bb56f051eb310768ad08102733ad64`;
manifest SHA256 `5a48332e0d228b50a6674868668863b76949fe6aaa387410f99a90148e426d85`.
Drive acceptance and supplemental administration receipts are recorded separately
in the index; no pending upload is treated as accepted. The PR stays draft for
owner choice before final research and final-review readiness.
