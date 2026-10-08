# Native recovery and HU100 table growth — resource stop

**HU20 reference verified; recovery equivalence unverified; HU100 not run.**
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
Independent final evidence review is pending.

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
Final derived diagnostics/review/archival guard receipts will be sealed in a small
closeout supplement. Originals remain; no cleanup, merging or publication.
