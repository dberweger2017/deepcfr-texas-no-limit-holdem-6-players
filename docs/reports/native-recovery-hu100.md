# Native recovery verified; HU100 stopped at measured capacity

**Final follow-up result: HU20 recovery is equivalent; HU100 reached 11,042,440
actual nodes and 3,255,387 entries, then stopped at the measured capacity ceiling.**
All four early pilot sets and the final atomic capacity-stop set passed complete
current/average export audits. **No 1B milestone or 10B endpoint was reached.**
This measures recovery correctness, table growth and resource capacity; it makes
no poker-strength, exploitability or convergence claim.

[Protocol](../native-recovery-hu100.md) · [PR197](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/197)
· [scientific receipts and checkpoint/resource diagnostics](native-recovery-hu100-artifacts/followup-scientific-closeout.json)
· [all 15 model/export member paths, sizes and hashes](native-recovery-hu100-artifacts/followup-model-index.json).

## Separately authorized follow-up

The [owner's October 8 comment](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/197#issuecomment-6054725121)
authorized **one retained-file verification attempt**, without repeating reference
or recovery training, under **10 GiB aggregate RSS for the entire owned family**
and continuous system-pressure guards on free 16-GiB M4. The original **448.81-MiB
swap baseline**, ≤0.5-GiB growth, ≥15.5-GiB disk, AC and **12:00 Madrid /10:00 UTC
deadline** remained unchanged. Original 5.5-GiB failure/state/archives were preserved.
All heavy work used the isolated M4 checkout/environment; M1 handled light metadata.

Verifier/HU100 scientific source: `c4058b7f14a6df85e01f8354de676091572ab1bc`.
Retained HU20 training source: `64318398fea423a9c43c0db8635a3724e05bd55f`.
The native binary remained exactly
`d6ecd69ce54b1afaf50e6df64edf104f1d14a77681de4d2c227b759015b2cf29`.
Exact-source qualification passed **110 Python tests, 7 Rust tests, release build
and artifact check**. Independent source rounds 6–9 resolved permanent-claim,
original-baseline, fresh per-audit admission, terminal-stop, successful-closeout,
whole-family coordinator and fresh stage/phase time-budget findings. Round 9
found no blocking P1/P2 source issues. [Receipt](native-recovery-hu100-artifacts/followup-source-review.json).
[Full CI](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/actions/runs/37746167499)
passed at the executed source. Final independent evidence review is pending.

## HU20 equivalence verified

The single guarded comparison completed in **167.66 s**. At **1,000,000,110 actual
nodes**, iteration **2,126,271**, every one of **4,319,080 training rows** matched
by IEEE-754 bits, including signed zero: keys, menus, regrets, average accumulators,
mass and visits. Both complete current/average exports were independently audited
in sequential fresh processes. Current file bytes matched; every average
probability/mass/visit row matched exactly. The original average stayed pinned to
**142,677,367 bytes**, SHA256
`571e198266eabc6d8bb9de2d1aa76d9a68be0b2512222ea96874168989c6b74d`.

Only declared recovery metadata differed: checkpoint `native_state`, average
`source_checkpoint_sha256` and `checkpoint_header.native_state`. Recovery coverage
started at `[1095942, 500000417, 98373121]`; all final counters/street deltas were
validated against the reference and retained parent. The changed verifier source
is separate from the executed original training source/binary. A successful
verification guard and durable CLOSEOUT gated HU100, not the existence of a
partial equivalence file. The original interrupted comparison remains a failure.

## HU100 growth and capacity stop

Seed **2026100601**, linear CFR, opponent-sampled averaging, existing v1 abstraction
and action menu, and uniform zero-mass fallback were unchanged. Fresh HU100 pilot
saved all four early endpoints and audited **all eight export/audit jobs** before
continuation. Growth resumed that fresh campaign's audited 10M parent, with
coverage still starting at `[0,0,0]`, aiming at each 1B through 10B total nodes.

| Requested save | Actual nodes | Iteration | Entries | New entries | Checkpoint bytes | Write seconds | Full current/average audit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 100000 | 100,691 | 64 | 53,743 | 53,743 | 1,640,195 | 0.141 | Verified |
| 1000000 | 1,001,382 | 690 | 460,885 | 407,142 | 15,096,813 | 1.334 | Verified |
| 5000000 | 5,001,210 | 3,406 | 1,777,063 | 1,316,178 | 60,894,602 | 5.159 | Verified |
| 10000000 | 10,001,922 | 6,983 | 3,023,624 | 1,246,561 | 106,120,927 | 8.966 | Verified |
| Capacity stop (1B requested) | 11,042,440 | 7,722 | 3,255,387 | 231,763 | 114,633,184 | 9.774 | Verified |

The measured capacity forecast limited entries to **3,255,354**, reserving a
**9.800-GiB export/audit RSS forecast** with the required 2× allowance plus 10%
additional headroom. All eight pilot timings/peaks and the complete prerequisite
manifest were bound and rehashed before the growth claim. Save reserve was
**21.237 s**, serialization reserve **188,408,211 bytes**, export/audit reserve
**247.669 s**, archive/closeout reserve **293.607 s**, and disk reserve
**6,799,495,975 bytes**. The training RSS soft stop was **9.325 GiB**, with the
continuous external 10-GiB/pressure/swap/disk/AC guard retained.

The native trainer reached the entry ceiling at a complete iteration and saved
atomically: **3,255,387 entries**, a **33-entry iteration overshoot**, at
**11,042,440 nodes**. It exited with the intentional resource-limit code **3**;
the supervisor therefore recorded an incomplete target and operator failure,
with **no external guard failure**. Those receipts/logs remain preserved. This is
an admission-capacity result, driven by reserved export/audit memory; native
training did not exhaust RAM. No limit was increased and no training was retried.

Its filename requests the next 1B save, but telemetry is **`incomplete-target`**.
The final export/audit used **11,042,440 actual nodes** and verified the atomic
checkpoint and both full policies once. It is a verified capacity-stop checkpoint,
**not a verified 1B checkpoint**. There were no interrupted saves. The table already
held 3,023,624 entries at 10,001,922 nodes; continuation added 231,763 entries over
1,040,518 further nodes. No 1B–10B growth curve is inferred from this early stop.

The last checkpoint has **2,158,927 positive-average-mass keys**, **1,096,460 zero-
mass keys**, and traverser visits by preflop/flop/turn/river of
`[75437, 247030, 587061, 1281740]`. Per-save street decisions, visitation, visit
histograms, average mass, actual nodes/iterations, new entries, throughput,
current native RAM, nearest resource sample/offset, sampled peak through save,
swap/disk, checkpoint size and write time are in the linked scientific receipt.
Samples are labeled sampled; exact intersample peaks are not claimed.

## Resource evidence, storage and closeout

Across follow-up qualification/verification/pilot/growth/final audit, **102
continuous five-second samples** recorded a maximum aggregate RSS of
**5,999,837,184 bytes /5.588 GiB**,
maximum campaign swap growth **zero**, minimum free disk
**98,008,903,680 bytes /91.278 GiB**, all AC,
normal pressure level **1**, and minimum system-free percentage **79%**.
No follow-up hard resource guard fired. All scientific workers exited; the
terminal latch prevents further training, and **all four campaign timers are disabled**.
Original evidence/inputs and open PR roots remain retained; nothing was cleaned.

A wrapper import-path failure occurred before any preparation/claim/science and
was preserved separately. Its environment was corrected before the one actual
verification claim. Qualification attempts, independent-review findings, native
capacity exit, operator traceback, partial requested target and all outputs are
retained. The archive's two provenance metadata fields have an explicit
[correction receipt](native-recovery-hu100-artifacts/followup-provenance-corrections.json):
the original historical retrieval receipt is `research/inputs/retrieval.json`,
and its handwritten PR-check time is not treated as an exact observed timestamp.

[Follow-up science ZIP](https://drive.google.com/file/d/1kMXJIUUB6YkYphhHSKsRp3Xacz_Oxno2/view):
**722,130,867 bytes /164 members**, SHA256
`cd2e3c197ec2adbf8e161b1aaca39eccff017fc9aaf4ff0cfe667122e186b16f`; embedded `ARCHIVE-MANIFEST.json` SHA256
`84fadd58a44f8dcdbfcf157beb262797ead77767c544324d14589f151c4abfe4`. Every member's size/hash was read back locally.
Native upload is complete/no pending/no conflicts; independent cloud ID/name/
size/parent confirms acceptance. Remote archive bytes were **not** downloaded.
[Archive receipt](native-recovery-hu100-artifacts/followup-archive-receipt.json).
Retained HU20 inputs refer to the already accepted original archive and exact
member hashes; no original training was repeated. [Restoration index](../../RESULTS_INDEX.md).

Post-archive guard/upload/review/report receipts will be sealed in a closeout
supplement. No merge, publication/release, arenas, paid compute, further limit
increase, deadline extension or cleanup is authorized by this result.

---

## Original campaign record (before the separately approved follow-up)

The recovery comparator was terminated by the external aggregate-RSS guard at
**08:10:02 Madrid /06:10:02 UTC on October 8, 2026**. This resource-ceiling result
is terminal for this campaign: no retry, raised limit, or dependent HU100 launch.
There is no detected correctness verdict to reinterpret as success. No poker
strength, convergence, merge or publication claim follows.

[Protocol](../native-recovery-hu100.md) · [PR197](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/197)
· [compact scientific receipts](native-recovery-hu100-artifacts/scientific-closeout.json)
· [all checkpoint/resource diagnostics](native-recovery-hu100-artifacts/milestone-summary.json).

## Qualification and admission

The owner initially reserved M4 to #188 through merge/closeout. The campaign's
04:00 Madrid wake disabled itself and admission checks waited every 15 minutes.
On October 8 the owner explicitly released M4 before #188 merge and extended
the hard deadline from 10:00 to **12:00 Madrid /10:00 UTC**. Those amendments did
not change the scientific recipe, resources, dependent gates or no-retry rule.
Actual M4 inspection found no competing heavy research worker, AC power and
about 100 GiB free disk. #188 subsequently merged at 05:34:01 UTC.

An isolated checkout/environment preserved shared source and other agents'
dependencies. Heavy work ran only on free M4; M1 coordinated light metadata.
Reviewed/qualified scientific source was
`64318398fea423a9c43c0db8635a3724e05bd55f`, based on merged #196
`97d0cb9deb25e7937cf232f458dcea6fc71ff800`. Release build, **7 Rust tests**, **81
Python fixtures** and repository-artifact check passed under the external guard;
qualification sampled peak was **356,220,928 bytes**, with zero swap growth.
Executed binary SHA256:
`d6ecd69ce54b1afaf50e6df64edf104f1d14a77681de4d2c227b759015b2cf29`.
Later report commits did not replace the worker's qualified source/binary.

## HU20 reference and retained-parent recovery

Seed **2026100601**, linear CFR, opponent-sampled averaging, existing v1
abstraction/menu and uniform fallback were unchanged. Fresh reference training
started **07:57:12 Madrid** and deliberately retained the legacy header, without
recovery metadata. It reproduced the historical retained 500M checkpoint exactly
(171,794,336 bytes; SHA256
`a5318cd586c0b170a68334e4236111faddabaf7f686c071958757db888afab47`).

The reference finished at **1,000,000,110 actual nodes**, iteration **2,126,271**,
with **4,319,080 entries**. Both current and average exports were independently
audited across every stored row. The average exactly matched the pinned
**142,677,367 bytes** and SHA256
`571e198266eabc6d8bb9de2d1aa76d9a68be0b2512222ea96874168989c6b74d`.
There were **2,929,370 positive-average-mass keys** and **1,389,710 zero-mass
keys**. Stored visits and current regrets were checked; historical incremental
updates were not reconstructed. Reference training including saves took
**392.55 s**; export **20.29 s** and full audit **55.86 s**.

The historical #182 parent was retrieved into a fresh ignored M4 input after
checking #182 MERGED and verifying whole ZIP, embedded manifest and member hashes.
A separate fresh recovery attempt completed at the same actual node/iteration
endpoint and table size. Recovery training took **202.87 s**, then exports
**20.29 s**. Its current export has identical file bytes by SHA256 to the verified
reference (`3b75a6f7ffb49631d89fae9c17e6863be1e2016f1f327c5ee8159a5153c36740`),
as confirmed by archive member hashes. This byte identity does **not** establish
complete recovery-state or average-probability equivalence.

The comparator was intended to compare every key/menu/regret/average accumulator/
visit by IEEE-754 bits (including signed zero), both policy probabilities, and
only narrowly validated recovery metadata. `native_state` coverage and the
average's source hash/header were the declared recovery-only differences. The
comparator ran **76.19 s** before the RSS guard terminated it (exit **-15**).
**No equivalence receipt was written.** Complete training-state, resumed full-row
audit and average-probability/metadata equivalence therefore remain **unverified**;
no partial comparator progress is promoted to a scientific result.

| Attempt /requested checkpoint | Actual nodes | Iteration | Entries | Checkpoint bytes | Write seconds | Audit |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Reference /100M | 100,000,464 | 247,261 | 2,173,888 | 98,549,345 | 6.801 | Unaudited |
| Reference /500M | 500,000,417 | 1,095,942 | 3,543,310 | 171,794,336 | 11.827 | Unaudited; historical bytes reproduced |
| Reference /1B | 1,000,000,110 | 2,126,271 | 4,319,080 | 214,220,678 | 14.645 | Verified current + average |
| Recovery /1B | 1,000,000,110 | 2,126,271 | 4,319,080 | 214,221,240 | 14.789 | Unaudited; equivalence interrupted |

All four saves have native atomic-completion receipts, hashes and byte counts.
The linked milestone summary records actual throughput, new entries, current
process RAM, nearest aggregate sample/offset, peak through save, swap, disk,
street decisions/visitation and average-mass diagnostics. Sampled RAM is labeled
as sampled; no exact intersample peak is inferred. There were no interrupted
checkpoint-save partials. Completed exports remain retained even when unaudited.

## HU100 and resource outcome

**HU100 pilot and growth were not run.** Its prerequisite did not pass, so there
are no HU100 entries, throughput, checkpoints, growth curve or extrapolated
training result to report. The prepared 100k/1M/5M/10M pilot and every-1B-through-10B
protocol remains available; future execution is a separate campaign decision.

The continuous supervisor used the unchanged **5.5-GiB aggregate RSS threshold**,
**≤0.5-GiB campaign-wide swap growth**, **≥15.5-GiB free disk** and **AC-power**
guards. Across qualification, retrieval, reference and recovery it recorded
**248 samples** at five-second intervals. The comparator's detected aggregate
peak was **5,964,972,032 bytes /5.555 GiB**, exceeding the threshold by
**59,392,000 bytes** before termination. The guard stopped the owned process;
this is an observed RSS breach, not a claim that every sample stayed below 5.5 GiB.
Maximum swap growth was **zero**, minimum free disk **103,642,181,632 bytes
/96.524 GiB**, and every sample showed AC power. The worker/coordinator exited,
terminal guard receipts and durable closeout cleared the owned launch claim.
No other job was stopped. Growth's 4-GiB soft-stop/serialization reserve was never
used because no growth job launched. The source-only review and small fixtures
did not forecast the full comparator's live memory successfully; a future
memory-bounded audit would need separate qualification, not a retry of this stop.

## Failures, review and archive

All setup failures are retained: M1 system Python 3.9 lacked `zip(strict=True)`
(the supported environment passed), an incorrect operational runpy invocation,
and detached PATH missing `gh`. The first two operator launches failed before
scientific claims. Installed M4 `gh` then returned HTTP401; a narrowly scoped
ignored adapter accepts only fixed public PR188 status invocations, records live
GitHub REST response identity/provenance, and transfers no credentials. The
qualified scientific source/binary stayed unchanged. Failed comparator logs,
resource samples, all successful saves/exports and incomplete closeout remain.

Independent source review rounds 1–4 closed prior objections: stop-during-save,
recovery-gated pilot, all four early audits/both guards, baseline continuity,
source/binary identity, measured reserves, complete pilot prerequisite manifests
and all-eight-export/audit memory forecasting. Round 4 found no blocking source
findings at `6431839`. Runtime capacity failure is preserved despite those reviews.
Independent final evidence review round 5 found **no blocking findings** at
`05a4994`. It confirmed the verified/unaudited distinction, terminal RSS stop,
absent active workers/pilot and archive provenance.
[Review receipt](native-recovery-hu100-artifacts/independent-review-round5.json).
All four campaign timers are now disabled; no scientific work remains active.
[Timer receipt](native-recovery-hu100-artifacts/timers-disabled.json).

The [science archive](https://drive.google.com/file/d/1hNbCAU71BcYFxVBx2OSmqYtp0sXfhmcS/view)
in [PR197 Research-Cloud](https://drive.google.com/drive/folders/1D2f8JkP1oZYPmeph9AexXD5ZnSLgBqfI)
is **1,380,462,715 bytes /108 verified members**, SHA256
`4b4cc0e890756ee076a31737a3f5abc8af63d0f2a9065f28c3dc83baad38f88f`.
Embedded `ARCHIVE-MANIFEST.json` SHA256
`b5e506be66fb68700be6332155bc726835e2dc42e19fa4b746165477327aa3d5`.
It contains exact native source tar/binary, environment/qualification, retrieval
provenance/input, all attempts/logs/guards/resources and successful model files.
All member sizes/SHA256 were read back locally before the final archive was
accepted. Native upload is complete (uploaded=1, uploading=0, no conflicts), and
independent Drive ID/name/size/parent matches. Remote archive bytes were **not**
downloaded or rehashed. [Archive receipt](native-recovery-hu100-artifacts/archive-receipt.json).
The [closeout supplement](https://drive.google.com/file/d/14MOuquGcYRPyeI5PUDW2Tdbmg8bPPxR9/view)
is **99,275 bytes /24 verified members**, SHA256
`41534b63cb6f1aecaab0e1c9d0082f5b3c5e353c9465c49e99bd33714138b7f6`;
embedded manifest SHA256
`d6c5f21ba8d1854962e6c4dbad44d76f825bd3aa8d3f520c5109e99f6522f156`.
It seals derived diagnostics, independent review, timer disablement, science
archive member index/upload receipts, report/index snapshots and archival guard
records. All member sizes/hashes passed local readback; native upload complete
and independent cloud ID/name/size/parent accepted. No remote-byte download.
[Supplement receipt](native-recovery-hu100-artifacts/closeout-archive-receipt.json).

Closeout completed **08:35:20 Madrid**, before the owner deadline. Both archival
guards finished successfully, each below 35 MiB sampled aggregate RSS, zero swap
growth and AC power; the lowest archival free-disk sample was 93.926 GiB.
[Terminal receipt](native-recovery-hu100-artifacts/terminal-closeout.json).
All owned scientific/archive processes exited and all campaign timers are disabled.
Originals remain; no cleanup, merging or publication.
