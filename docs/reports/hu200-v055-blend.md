# Frozen HU200 v0.5.5 three-seed blend

**Owner-authorized continuation:** use raw `Node.average` values from the three
20B checkpoints. The original stop below remains part of the record. The fixed
blend and evaluation are unchanged; current work is input admission and export
reproduction. [Corrected protocol](../hu200-v055-blend-protocol.md).

## Original normalized-export stop

**Stopped before building a blend or playing any hand.** The requested prerequisite
is false: these average exports store normalized action probabilities, rather
than raw action accumulators. The owner explicitly instructed: “If the exports
are already normalized, stop and report before choosing anything else.” No
alternative formula has been selected.

[PR297](https://github.com/dberweger2017/CFR-NLHE-6max/pull/297).
Source inspected: current main `98658f7b` (full revision in the evidence receipts).
The required frozen, confirmation, translation and evaluation-harness reports,
repository workflow and artifact-storage contract were read before execution.
The exporter and reader establish this before model retrieval. Full current
input SHA256 verification and header/row inspection were **not completed**.
No training, union scan, merge, cache construction, pilot or final play occurred.
No Slumbot request, credentials, db-ocean or cygnus-x1 was used. The M1 performed
administration only; the sole admitted M4 worker began an archive read.

### Export and weighting semantics

`native/hu20-trainer/src/export.rs`, function `export_in`, reads each checkpoint
row as `[key, names, regrets, accumulated, visits]`. It computes
`total = fsum(accumulated)` and writes the average-export row as
`[key, names, p, total, visits]`, where `p[a] = accumulated[a] / total` for a
positive-mass row. Zero-mass rows use `1 / len(names)` with the selected uniform
fallback. The extraction label is
`normalize-lifetime-iteration-opponent-sampled-accumulator-v1`.

`src/blueprint/average.py`, `AveragePolicy.__init__`, validates that `p` is a
probability vector with sum one and loads it directly into inference storage.
`src/diagnostics/cfr_average.py` independently uses the same representation.
Thus the third field is normalized; the fourth is a key-local accumulated mass.

`native/hu20-trainer/src/cfr.rs`, `Options::weights`, returns `(t, t)` for linear
CFR. In `Traversal::visit`, opponent-sampled averaging adds `t * policy[a]` on
each sampled opponent-node visit, before sampling its action. There is no
own-reach multiplier in that branch. The per-key `total` is the sum over actions
of those accumulated updates; it is not a seed-wide total or the traverser
visit count. The nominal iteration-weight schedule through iteration `T` sums
to `T * (T + 1) / 2`; that schedule alone does not describe which sampled visits
a particular retained key received. No seed-wide normalization denominator has
been chosen for a blend.

The normalized probabilities and stored row mass permit a numerical estimate
of positive-mass raw accumulators by multiplication. That would require a
revised interpretation and is not performed under this instruction. A uniform
zero-mass probability vector still represents zero accumulator mass.

### Requested campaign, not run

The requested single rule was `sum_s accumulator_s(key, action) / W_s`, with
missing keys contributing zero, uniform zero-mass rows, and a distinct output
identity. The requested fixed evaluations were 131,072 direct duplicate blocks
against each of the three seeds and 16,384 blocks against each of seven scripts,
with all four models paired on common deals, translation 512/128, independent
replay and full reproduction before outcome readout. Those counts are retained
as scope, not represented as completed evidence.

Because the semantic stop precedes construction, union/overlap counts, export
and cache capacity projections, merge/cache resource peaks, played missing-key
shares and poker results are **not measured**. There is no capacity admission
or playing-strength conclusion. No release, tag, default change or merge.

### Original stop evidence and resources

The M4 `Mac16,1` has 17,179,869,184 bytes RAM. A bounded administrative
verification worker acquired both exclusive nonblocking research locks under
the existing five-second host guard. Model originals were absent at their
indexed paths; the first accepted #265 ZIP was an online-only Drive file.
The worker began an on-demand archive read. This restoration was unnecessary
after the semantic stop, and was explicitly terminated before any complete
archive/model SHA256 verification or model extraction. The intentional SIGTERM
is recorded by the guard as `RuntimeError('Supervisor interrupted: 15')`; it is
not a resource breach or a failed poker hand. Ordinary Drive hydration may
finish independently; no provider offloading or synced deletion was requested.

The initial 15-minute ETA was revised to 30–45 minutes upon discovering the
online-only archives, then superseded by this completed semantic-stop record.
The archive includes every available guard reading, the terminated worker's
receipts, input archive metadata/provenance, setup failures and source snapshots.
No sampled model row, union/overlap count or completed input hash is claimed.
The hash strings in the provenance are expected pins from the existing index.

Existing caches, original archives, tables and research roots are retained.
The cleanup receipt has an empty removed-path list. Merge/cache peaks are not
applicable; the resource summary describes only the aborted read-only preflight.
The PR remains open for the owner. Reinterpreting the normalized export as raw
accumulators, or selecting a reconstruction and normalization rule, needs a
revised instruction before this campaign can proceed.

The preflight has **46 readings**, sampled family peak **49,512,448 bytes
(47.22 MiB)**, minimum free disk **59,262,562,304 bytes (55.19 GiB)**,
minimum free memory **84%**, zero swap growth, normal pressure and no resource
breach. Maximum sample gap is **5.068 s**. Successful small-archive sealing
completes in **5.08 s**; its guard has two readings and no breach.
[Resource summary](hu200-v055-blend-artifacts/resource-summary.json).
[Failures](hu200-v055-blend-artifacts/admin-failures.json).
[Cleanup receipt](hu200-v055-blend-artifacts/cleanup-receipt.json).

The primary evidence ZIP is **157,264 bytes /35 members**, whole SHA256
`59667b090aa52987ab8658ee9c6b42577a3f78588f0665704fe1aa6591b407ec`.
`ARCHIVE-MANIFEST.json` SHA256 is
`c46abb7d35977c6dfdc55e980c7e30cda3e5fe72543e302a87eb96adb2668e2a`;
all member sizes/hashes and the manifest passed complete local readback.
The [archive](https://drive.google.com/file/d/1sNQv6L7WMcSMB73AEvy-robzwQwMieIg/view)
has native uploaded1/uploading0/no-conflicts and independent matching
ID/name/size/direct-parent confirmation. Output archive bytes were not downloaded
from Drive. This does not establish the uncompleted input archive hashes.
[Local archive receipt](hu200-v055-blend-artifacts/archive-receipt.json),
[native acceptance](hu200-v055-blend-artifacts/native-upload-observation.json),
[independent metadata](hu200-v055-blend-artifacts/cloud-upload-observation.json).

All owned M4 workers exited and both locks were available at **16:29:26 UTC /
18:29:26 Madrid, October 15**. Remaining work is administrative closeout.
[Release receipt](hu200-v055-blend-artifacts/final-compute-release.json).

## Corrected raw-input continuation (owner decision, October 15)

The owner authorized replacing the source with raw 20B checkpoint `Node.average`;
the blend rule and final evaluation counts remain fixed. The original stop's
non-completion statements above are dated observations, not a cancellation of
this continuation. [Predeclared weights and protocol](../hu200-v055-blend-protocol.md).
No merge or final hand has begun. Seed2026101421's existing 20B checkpoint
matches its full pinned SHA256; exact existing-exporter reproduction and the
169-root-infoset mass scan **pass** under both M4 locks/guards. Its average
reproduces all 85,373,916 rows in **724.79 seconds**, exact full SHA256
`c08d86a9615022df7015d34e5dc3de2705bbb8346ed6976bd72284e71d91f444`.
Root mass is exactly **140,199,312,593,278.0**, equal to scheduled W with zero
observed floating-point difference and zero missing root keys. The full
check takes **779.12 seconds**, 156 readings, peak family **72,056,832 bytes
(68.72 MiB)** and no resource breach.
[Reproduction receipt](hu200-v055-blend-artifacts/local-average-reproduced.json),
[root diagnostic](hu200-v055-blend-artifacts/local-root-mass-summary.json),
[resources](hu200-v055-blend-artifacts/local-seed-resources.json).

### Initial retrieval capacity observation (superseded)

Full remaining checkpoint archives were online-only. The Drive connected fetch
refuses the 26.698-GB PR265 ZIP with HTTP413 (256-MiB tool limit); native
Drive hydration is available, but its disk requirement must be admitted first.
The initial capacity observation is 59,400,392,704 free bytes. Even optimistically
reclaiming all 10,064,002,807 bytes of identified inactive PR296 policy caches,
full hydration of PR265/seed1422 archives, both extracted checkpoints, the
16-GiB floor and a 2-GiB margin require **76,896,637,928 bytes**, leaving a
**7,432,242,417-byte deficit** before outputs, union or cache construction.
The local reproduction adds its separate average output to future requirements.
Additional inspected build targets total under 2 GB and cannot close that gap.
No cache has been removed merely to approach a failing admission. At that observation, no full ZIP
was being downloaded for this continuation until storage was resolved; the
original preflight's on-demand download may continue under ordinary Drive.

This is a retrieval-capacity issue before the requested union gate, not a
measured union failure. Checkpoint-only archives or additional M4 capacity
would permit the next check without forced offloading or research deletion.
At that observation, final ETA awaited input admission, then union sizing and cost pilots.

The owner is arranging additional M4 storage and instructed continued work.
The native three-lookahead union/merge tool is implemented; guarded fixture
tests and its build follow the completed local-seed check, before real merging.

### Sequential intake admitted October 15, 17:52 UTC /19:52 Madrid

The owner instructed continued work while freeing M4 storage. The initial
simultaneous-download calculation was unnecessarily restrictive: the repository
records normal per-file `FileManager.evictUbiquitousItem` release of inactive,
accepted Drive download caches. After whole-archive and selected-checkpoint
hash verification and exact released-average reproduction, that ZIP cache can
become inactive. Normal release preserves its cloud file and same-size
placeholder; no force, unpinning, synced deletion or research-original removal.
It requires current merged source ownership, clean uploaded/no-conflict native
status, matching independent cloud ID/name/size/parent, unchanged file identity
and no open handles. A refused request is retained, with no forced retry.

Sequential intake's maximum requirement is **56,525,196,918 bytes**, including
retained new checkpoint/average outputs, one full archive, the 16-GiB floor
and 2-GiB margin. Initial free disk **72,950,648,832 bytes** admitted the first
restore. This is intake admission only; the actual union/export/cache/audit/
archive gate remains mandatory. The source PR265/278 live states were MERGED
before copying. Independent Drive metadata confirms both archive IDs/sizes/
parents. The full PR265 hash read is now running under both locks and guards.

The qualified M4 build passes **5 native tests and 31 focused Python tests**.
The five native checks exercise weighting, missing/zero mass, conflicting menus,
positive underflow refusal and a complete three-way stream/export fixture.
[Build receipt](hu200-v055-blend-artifacts/continuation-build-receipt.json).
Verified read-only APFS clones protect seed1421's checkpoint, the frozen exporter
and three pinned tables in this PR's own ignored root; originals are retained.
[Copy receipt](hu200-v055-blend-artifacts/continuation-copy-receipt.json).

The fixed 16-hour wall envelope starts at **17:52:37 UTC**; its final two hours
are reserved for preservation, and intake is capped at six hours.
[Clock](hu200-v055-blend-artifacts/continuation-campaign-clock.json).
The initial post-admission estimates were M4 release October 16 00:50–03:50
Madrid and review readiness 02:50–05:50. At 18:00 UTC, the roughly 4-MB/s
first-archive arrival rate widened the provisional estimates in the PR to M4
release **03:00–07:00 Madrid**, review readiness **05:00–09:00 Madrid**. Union sizing and cost-only
pilots must replace these estimates before any final hand. No final hand, full
blend or policy cache exists yet.

[Continuation setup and planning failures](hu200-v055-blend-artifacts/continuation-admin-failures.json)
are retained separately from the original stop; no scientific hand was retried.

At **19:36 UTC /21:36 Madrid**, the native Drive rate remains variable and the
first ZIP has about 20.3/26.7 GB resident. Before the second intake worker or
any merge/final hand, intake's initial six-hour allocation is revised to eight
hours **within the same 16-hour overall envelope**. The science deadline,
two-hour preservation reserve, memory/disk guards, formula and counts are
unchanged. The already running first worker keeps its original deadline.
[Original/revised clocks and owner-status receipt](hu200-v055-blend-artifacts/continuation-intake-budget-reallocation.json).
At 21:16 Madrid the provisional full-run ETA widened to M4 release **06:00–10:00
Madrid October 16**, review readiness **08:00–12:00**, conditional on union
admission and cost pilots; the PR description carries the latest estimate.


At **20:25 UTC /22:25 Madrid**, the seed 2026101302 full archive, embedded
manifest and extracted checkpoint all match their declared SHA256 pins. The
unchanged frozen exporter is checking exact reproduction of its released
average; that proof is still pending. Seed 2026101422 restoration and the actual
union/capacity gate follow. No blend or final hand has run. The PR description
was updated at 22:27 Madrid; provisional release/review windows remain those
above. The administrative controller was reattached while the same original M4
worker and guard continued; no scientific job, deadline or baseline was reset.
[Controller receipt](hu200-v055-blend-artifacts/continuation-controller-reattach-status.json).


At **20:37 UTC /22:37 Madrid**, seed 2026101302 reproduces its released average
SHA256 `9128f71c602f18a9cb7219b18cd43ff4e3f2b7ed6a30f9babe7ed033840ac7b6`
exactly with the existing frozen exporter (`--zero-mass uniform`). Export takes
718.29 seconds; full archive restoration, raw verification and reproduction take
9,876.81 seconds. All **1,956** resource samples are clear: sampled family peak
**57,212,928 bytes**, exporter kernel peak **4,587,520 bytes**, minimum sampled
free disk **45,590,859,776 bytes**. Two of three checkpoint/export proofs now pass;
all-three proof still precedes the merge.

After fresh source-PR and independent cloud metadata checks, one ordinary
FileManager cache-release request retires only the inactive **26,698,072,064-byte**
local ZIP cache. Its cloud ID, same-size placeholder, inode and modification time
remain; the restored raw checkpoint and exact average remain protected. No
research original or synced path is deleted. Free disk rises to **72.28 GB**.
The third seed's restore starts at **20:38 UTC /22:38 Madrid**, using the revised
intake deadline and the original resource baseline.
[Exact input, resource and cache-release receipt](hu200-v055-blend-artifacts/continuation-seed-2026101302-verified.json).


At **23:05 UTC October 15 /01:05 Madrid October 16**, seed 2026101422 also
reproduces its released average SHA256
`b191c5ed8b943d138e6076fb4a1b0117c98964d92dd2d6253e157f696f691f6f`
exactly. Its existing-exporter run takes **722.63 seconds**; full restoration,
raw verification and reproduction take **8,801.79 seconds**. All **1,743** guard
samples are clear: family peak **53,493,760 bytes**, exporter kernel peak
**4,636,672 bytes**, minimum sampled free disk **42,371,039,232 bytes**.

Fresh source-owner/native/cloud checks then permit one ordinary FileManager
request to release its **20,953,055,232-byte** inactive ZIP download cache.
The same cloud file, placeholder, inode and modification time remain; all three
raw inputs and exact reproduced averages remain protected. Free disk is
**63.32 GB**. The streaming union/overlap/compression/capacity stage starts at
**23:07:32 UTC /01:07:32 Madrid**. All three required input/export proofs now pass;
full blend, caches and final play remain unstarted pending that gate.
[Third-seed proof, resource summary and cache-release receipt](hu200-v055-blend-artifacts/continuation-seed-2026101422-verified.json).
