# HU200 translation: calibration stopped on system pressure

**The intended playing comparison was not reached.** Normal-pressure admission
passed, but macOS pressure changed from normal (`1`) to warning (`2`) during
calibration after the model had loaded. The guard stopped its owned worker.
There was no sample freeze, final hand or playing-gain estimate. Translation
remains **experimental and off by default**; this attempt supplies no evidence
for HU200 adoption, even if the implementation can reduce fallback exposure.

[PR218](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/218)
starts from current main `a3358fc` (merged #217). Scientific source is
`734b4f1093c963d15a81d12b3e5de209f95ba613` in the isolated M1 worktree
`/Users/dberweger/Local/hu200-action-translation-20261009`.
[Prospective protocol](../hu200-action-translation.md) and
[provenance](hu200-action-translation-artifacts/provenance.json) preserve the
fixed policy, schedule, comparison and admission rules.

## Implementation and independent correctness

The existing HU100 translator now explicitly accepts HU200's versioned schema,
game and 200BB table, rejecting mismatches even before bounded early returns.
HU100's version, search ordering, all-in preference, uniform fallback and
behavior are preserved. The current real legal menu and amounts remain
authoritative; only public past-raise labels can change. Search consumes no policy
RNG and retains one unchanged weighted sampling draw. Bounds remain 512 popped
states and 128 history events. Exact and stored zero-mass keys retain behavior;
entirely on-menu missing histories do not translate. Open jams remain a limitation
of the conditional-jam witness menu.

Independent [translator review](hu200-action-translation-artifacts/source-review-1.json)
passes 14 focused tests, 912 HU100 comparisons against main (22 positive witnesses),
and independent legal four-street engine witnesses at both stack depths and
button positions. Costs match exact rational minima, insertion-order reversal
changes nothing, hidden deals do not affect lookup, and RNG states agree.
Independent [evaluator/controller review](hu200-action-translation-artifacts/source-review-2.json)
passes 21 focused tests and eight synthetic nonuniform-policy hands, including
three translated witnesses checked by the real engine. The evaluator records
actual decisions once, validates current legal actions, reproduces all policies
and deterministic telemetry, and checks complete events/settlements. Its selected
witnesses are independently replayed through the real engine.

Before science, review found detached children could survive supervisor SIGTERM,
and the source snapshot omitted monitoring pins and a required schedule config.
Both were fixed and independently cleared. A real subprocess regression confirms
owned-child termination, failed receipt and latch. Two dirty-source compatibility
tests initially refused their frozen-source admission; both pass after committing,
as part of a separate 33-test compatibility run. The first broader focused set
passes 34 tests with two native-build-dependent skips; no new native campaign was
run. [Preparation notes](hu200-action-translation-artifacts/setup-notes.txt).

Post-closeout CI at `734b4f1` passes the first full shard but fails the second
and aggregate on the SIGTERM fixture: macOS `time -l` prevented its child from
launching on Linux. The test now isolates that wrapper for the portable case and
waits for the actual child before signalling; native and portable cases pass on
M1 in a **22-test focused run**. The M1-only campaign code and sealed scientific
source/evidence are unchanged. [CI failure/fix receipt](hu200-action-translation-artifacts/integration-ci-first-failure.json)
and [independent integration review](hu200-action-translation-artifacts/integration-review.json)
preserve the correction; final-head CI is separate.

## Intended comparison and what is unavailable

Freeze #216's **100,000,034-node /20,575,288-entry** opponent-sampled average,
seed 2026100905, iteration 52,136, SHA256
`1e9613547ad6721f2559e419caf296a781a914a4e92ecd4ef35f4b483c66060b`.
Its 510,277,592-byte file and checkpoint/header/audit identity pass the indexed
hash/header gate. Existing model/runtime paths are read directly; neither is
copied into this evidence archive. HU200 stacks, blinds, v1 cards,
min/pot/conditional-jam menu and uniform missing/zero fallback are unchanged.

The protocol predeclares pot-pressure on-minus-off as the sole paired 95%
primary; random's lower bound must exceed −20 BB/100, and check-call/tight/loose
must have identical complete behavior. The target is 2,048 fresh paired blocks
per opponent /40,960 hands with rotated seats and identical deal/private-action
streams. The timing-only 32-block sample uses root **2026100909**; intended final
root **2026100908** is unused. Freshness excludes **72,561** prior HU100/HU200 deal
seeds and current timing/final collisions. Calibration payoffs were not inspected.

**Zero final blocks/hands:** paired gains and uncertainty, absolute playing
results, final exact/translated/fallback rates by street, bound hits, distances
and latency are all **unavailable**. Do not substitute the excluded, incomplete
calibration for the primary comparison. No timing-only sample adjustment was
frozen, no outcomes motivated a retry, and no sample was extended. #217's
unsupported-history diagnosis motivates this intervention; it does not guarantee
a gain from it. This stopped attempt demonstrates neither fewer final fallbacks
nor improved play.

## Stop, partials and verification limits

The M1's fresh admission at **15:44:44 UTC** passed: AC power, normal pressure,
54% system-free, 4,525,031,424 available bytes, 26,657,628,160 free disk bytes and
1,128,267,776 swap bytes. The owner expressly retained the **15.5GiB disk floor**
and restored headroom. The exclusive research lock and process inventory found
no competing research worker. A separate cleanup thread was active; its roots
and work were untouched, with model/runtime dependencies retained. M4 was unused.
[#216/#217 live statuses](hu200-action-translation-artifacts/parent-217-live-status.json)
were merged before input use; PR218 remained open before archival.

The source snapshot and hash/header gates complete. Calibration loads the policy
and writes the off panel, then part of on. At **15:46:47 UTC /123.48s from
admission**, pressure changes `1→2`, despite family RSS below the unchanged
3/4GiB limits, system-free 42%, AC and zero swap growth. The worker is terminated
with SIGTERM and science latched off. The archived timing log has no kernel
command peak because the timing wrapper was also terminated; sampled family
peak is **1,629,061,120 bytes /1.517GiB**, not an exact process-family supremum.
This evidence does not establish why the system pressure changed.

Count-only prefix readback retains **439 complete calibration JSON records**:
320 off /119 on, 1,195 /592 actions and 557 /286 candidate decisions. Each record
is written only after that hand's legal observation/action checks, in-memory
complete-event/accounting replay, policy/telemetry reproduction and selected
witness checks return. On's gzip footer is interrupted; streaming returns the
119 complete records before EOFError. Its native fixture stream has **118 rows**,
because termination can fall between the hand flush and fixture flush. Off has
320 matching fixtures. The exact interrupted originals remain unchanged; no
missing record, fixture or hand is regenerated.

There is **no complete timing costs receipt and no independent native parity
run**. The **117.40s interrupted worker duration** includes completed loading and
partial play; it cannot be called isolated loading time or scaled as a per-hand
rate. Fixed and scalable costs remain explicitly unavailable; no assumption
uses #217's six-minute runtime. Native/action/accounting verification for an
intended full final sample cannot be claimed. Independent final evidence review
checks the preserved prefixes and receipts, without producing calibration payoff
results or replaying new hands.

The primary agent initially inferred “no hands generated” from the absence of a
worker completion receipt. That status was wrong and was promptly corrected after
count-only inspection. Archived `partial-summary.json` preserves the initial
mistake; `partial-summary-corrected.json` preserves the first correction. The
latter still inaccurately labels pressure as “during loading,” and its top-level
zero counts mean final counts. **This report and
[final partial receipt](hu200-action-translation-artifacts/final-partial-summary.json)
are authoritative:** loading completed; 439 recoverable calibration records,
zero final hands, payoffs generated but not inspected, no isolated load timer.
The sealed archive is not rewritten to erase these reporting mistakes.

## Local closeout, archive and recommendation

Original immediate archive admission also refuses warning pressure; no ZIP starts.
Once pressure returns to normal, archive-only closeout uses the **same original
swap/disk limits and original admission time anchor**, with the scientific latch
still terminal. No science readmission, model reload, hand or baseline reset occurs.
Local seal/readback takes **1.08s**; closeout finishes **339.77s /5.66min** from
original admission, within the single 60-minute cap. Later report/review/PR work
is administrative; it does not restart science.

Across **221** retained samples, 220 pressure readings are normal and one warning;
maximum within-operation gap **0.583s**, minimum free memory **42% /3,524,345,856
available bytes**, minimum disk **26,589,147,136 bytes**, zero swap growth and AC
throughout. Archive-only family peak is 189,005,824 bytes, kernel command peak
102,252,544 bytes; these scopes differ. [Resources](hu200-action-translation-artifacts/resources.json),
[costs](hu200-action-translation-artifacts/costs.json),
[original failure](hu200-action-translation-artifacts/failure.json),
[completed closeout](hu200-action-translation-artifacts/local-closeout-completed.json).

The **1,728,155-byte /27-member** ZIP, SHA256
`93469e766ff8227fc601aa2e618ed4323134df24718999f2b5391ab994628987`, is placed in
[PR218's Research-Cloud folder](https://drive.google.com/drive/folders/1LrrpXK2BWB52WNOUYICBg6ctOuc3K3sb).
Every member's size/SHA256 and embedded `ARCHIVE-MANIFEST.json` readback pass;
manifest SHA256 `d8991862c8d72466b6168dc4a3f590044c48bc6730356946db58b550f7e5d1d6`.
Manifest records original paths and mtimes for restoration/dependency review.
The ZIP contains reviewed source, input identities/restoration pointers,
interrupted calibration streams, fixtures, raw resources, failures, corrections
and original closeout. The archive-operation monitor, completed closeout,
independent final review and final correction are separate compact Git lifecycle
receipts. No duplicate model or runtime archive is created.

**Upload-pending owner/later-agent handoff:** one-time Drive metadata identifies
the actual folder; archive ID/metadata availability is recorded separately in
[handoff](hu200-action-translation-artifacts/drive-handoff.json). No native uploaded/
no-pending/no-conflict status or cloud upload acceptance is claimed; no remote
archive bytes are downloaded or verified. There is no upload wait, deletion or
forced offload. [Restoration index](../../RESULTS_INDEX.md#pr218-hu200-action-translation-calibration-stop--october-9-2026).

Recommend **retaining translation as experimental**. No HU200 playing evidence
here supports adoption or rejection of the hypothesis. A separately admitted,
owner-directed repeat should first provide more idle M1 memory headroom and
normal pressure; keep the same scientific controls, fresh excluded timing/final
roots and a newly declared budget. The current attempt is terminal and is not
restarted. Do not claim sparse covered learning or unsupported histories have
been fixed. No training, live Slumbot, release or automatic adoption follows.

Independent final evidence review and final checks are recorded in
[review](hu200-action-translation-artifacts/evidence-review.json). The initial
handoff presented PR218 unmerged as a preserved, resource-stopped attempt.
The owner subsequently requested merge. All five checks passed on reviewed head
`14de46b215e2f34b2beb320902d679868999f415`; [CI run](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/actions/runs/37957947013)
and independent final reconciliation had no open findings. PR218 merged at
`31cf2490144581992d0d334d5132daeebfa7918d` on October 9, 2026, 18:29:15 UTC.
This owner-authorized merge supersedes the earlier unmerged handoff. The original
playing-comparison goal remains incomplete, translation stays experimental and
off by default, and archive upload acceptance/cleanup remains a pending handoff.
No new experiment or archive modification accompanies merge.
