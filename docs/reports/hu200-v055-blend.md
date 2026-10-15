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

## Export and weighting semantics

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

## Requested campaign, not run

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

## Evidence and resources

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
