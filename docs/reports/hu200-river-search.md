# HU200 river search: correctness passes, latency gate fails

[PR #317](https://github.com/dberweger2017/CFR-NLHE-6max/pull/317), October 16, 2026.
**Draft; strength study not declared or run.** This is the completed cost-gate
readout, not a final strength result or a release recommendation. The owner
chooses the response to the failed latency gate and decides merge.

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
Its new full 24 parity gate and subsequent cost gate are **pending**. Neither
concurrency nor strength is admitted by the 10.766536-second finding. Transport
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
readiness remains unresolved until the amended cost gate, owner-approved
strength research and storage closeout actually finish.
