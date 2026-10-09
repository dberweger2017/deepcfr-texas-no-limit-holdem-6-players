# HU100 independent seeds: experiment complete; recipe unqualified

[PR215](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/215)
is open and unmerged. [Protocol](../hu100-independent-stages.md).
Both predeclared fresh seeds reached audited early and 1B endpoints, and all
three fixed lineages completed fresh paired evaluation at 4,096 blocks/opponent.
**Overall recipe qualification is not established:** seed 2026100902's tight
improvement is inconclusive under the predeclared adjusted interval. Loose
improvement and pot-pressure translation repeat with practical support in all
three lineages. Eight of nine formal contrasts pass; none shows formal decline.
No release, changed recipe, paid compute or automatic larger run.

Merged #211 retained a 1,000,373-node pilot, not fresh 1B qualification results.
This campaign used that actual partial only after provenance/full-audit/SHA256
checks and byte-exact resumed versus fresh-direct training to 2M. Main training
continued the original 1M partial, not the compatibility fixture. Seed 2026100902
started fresh. The unchanged #207 native binary remains
`7650ad60bbf2437622ea3c39d37c7d56686d00bac11680e44a6e47dab509a262`;
its actual source is `bd0e7a417064f736091dc2b667954b50becb4b69`.
Scientific controller source is `1d862d6f9ea2e5e56b23c84cac94561103b5da11`,
from current-main base `6e18043317817080fd38f400c5366fbf18fc6b53`.

## Fixed endpoints and paired comparisons

| Seed | Early actual nodes / entries | Terminal actual nodes / entries |
|---|---:|---:|
|2026100601|39,438,279 / 7,643,261|1,000,002,065 / 41,010,014|
|2026100901|39,439,801 / 7,722,785|1,000,000,506 / 40,752,103|
|2026100902|39,440,603 / 7,767,917|1,000,001,141 / 41,047,344|

All four new endpoints retain recovery/current/average files with full stored
entry, serialization and SHA256 audits. Targets permit only the first complete
iteration at/past 39,438,279 and 1B total nodes; no entry-cap or soft stop occurred.
[Audited model pins](hu100-independent-stages-artifacts/model-index.json).

Final root 2026100911013 is physically disjoint from the other 17 listed roots. All models share identical paired physical deals and private streams;
each block swaps seats. Freeze hashes bind sample, models, comparisons and
schedule before any final hand. No winnings or variance were inspected for
sample selection. [Freeze](hu100-independent-stages-artifacts/frozen-final.json),
[independent admission/schedule review](hu100-independent-stages-artifacts/admission-review.json).

All values below are **BB/100**. Growth is terminal/off minus early/off against
tight/loose: six contrasts, Bonferroni FWER 0.05, paired two-sided 99.1667% Student-t
intervals. Translation is terminal/on minus off against pot: separate three
contrasts, FWER 0.05, 98.3333% intervals. There is no union-FWER 0.05 claim. Lower >0
means improvement; lower >10 means practical support. All intervals condition on
these fixed policies and independent deal blocks; they do not include training
seed population uncertainty.

| Seed | Tight growth, adjusted interval | Loose growth, adjusted interval | Pot translation, adjusted interval |
|---|---:|---:|---:|
|2026100601|+48.16 [10.56, 85.76]|+254.27 [178.16, 330.39]|+90.15 [53.53, 126.77]|
|2026100901|+52.96 [17.73, 88.19]|+193.30 [117.47, 269.14]|+62.22 [22.86, 101.57]|
|2026100902|+32.54 [-5.11, 70.20]|+213.78 [135.95, 291.61]|+77.94 [40.17, 115.72]|

Tight passes, including practical support, for 2026100601 and 2026100901.
Seed 2026100902 is inconclusive at the formal 99.1667% interval. Its descriptive
95% interval is positive, **+32.54 [4.57,60.51]**, but cannot replace the fixed
formal decision. Loose and pot translation pass practical support for all three.
The sample was not extended after this finding.

| Contrast | Three-seed point-estimate range | Descriptive sample SD |
|---|---:|---:|
|growth-loose_aggressive|193.30–254.27|31.03|
|growth-tight_aggressive|32.54–52.96|10.67|
|translation-pot_pressure|62.22–90.15|14.00|

These are three descriptive estimates, not a pooled-hand or population interval;
common deals correlate the estimates. Full per-seed 95% and formal intervals,
absolute early/terminal/on results and control flags are in the
[compact result](hu100-independent-stages-artifacts/result-summary.json).

## Absolute play, controls and remaining coverage

Terminal/off absolute results below use descriptive 95% intervals, not additional
formal families. Translation/on is identical for all opponents except pot.

| Opponent |2026100601|2026100901|2026100902|
|---|---:|---:|---:|
|random|+76.89 [4.82, 148.96]|+78.01 [5.11, 150.91]|+95.36 [23.95, 166.77]|
|check_call|+100.51 [81.49, 119.54]|+117.55 [96.85, 138.25]|+100.23 [80.67, 119.78]|
|tight_aggressive|+36.30 [18.61, 53.99]|+43.04 [24.96, 61.12]|+32.90 [17.25, 48.55]|
|loose_aggressive|+57.10 [21.67, 92.54]|+35.27 [-0.51, 71.04]|+56.85 [20.17, 93.53]|
|pot_pressure|-99.48 [-136.78, -62.17]|-88.35 [-126.02, -50.68]|-100.20 [-138.35, -62.05]|

| Pot pressure | Terminal/off 95% | Terminal/on 95% |
|---|---:|---:|
|2026100601|-99.48 [-136.78, -62.17]|-9.33 [-39.10, 20.44]|
|2026100901|-88.35 [-126.02, -50.68]|-26.13 [-57.37, 5.11]|
|2026100902|-100.20 [-138.35, -62.05]|-22.25 [-54.22, 9.71]|

Off pot pressure still loses for every seed. On estimates remain negative and
all 95% intervals cross zero; translation's repeated relative gain does not
establish winning absolute play. Seed 2026100901's terminal loose interval also
crosses zero, despite its clear early→terminal gain.

Check_call/tight/loose on/off candidate actions, events and settlements match
exactly for every seed. Random also matches exactly: paired difference0 [0,0],
no predeclared severe-regression flag. This is evidence on these controls, not a
general noninferiority proof. The uniform reference is identical across all arms.
**409,600 unique primary hands** and their deterministic reproduction complete;
independent replay verifies737,280 rows and 2,878,098 actions including repeated
reference copies. Strict recount checks means, all paired coordinates, source,
model identities, probabilities, lookup classes, settlements and visit bands.

Pot uniform fallback falls21.00%→4.34%,21.88%→4.15%,21.28%→4.23% for the three
seeds (conditional observed decision rates; on/off paths can differ). No
translation bound was reached; maximum reported states65, below the fixed512
limit. Terminal/off native positive-mass river coverage against pot remains40.32–42.48%,
versus essentially100% on tight/loose; random river coverage is84.09–86.10%.
Global terminal zero-average-mass fractions remain about 32%; mean traverser
visits/key about 4.8. These diagnostics are not convergence or exploitability
bounds. [Coverage/visits](hu100-independent-stages-artifacts/coverage-summary.json).

## Costs and separate admission

Calibration uses existing #207 policies at 32 and 512 blocks, a 16x sample ratio,
with full replay/reproduction. Load once per checkpoint: early30.851 s,
terminal171.769 s. The validated registry serves both sizes and reproduction;
terminal also serves off/on. Fresh seeded policy instances are created per hand,
translation is explicitly reset, and full spec/game/model bytes are validated
at execution boundaries. Hash-verified hardlinks replace physical snapshot
copies; the archive stores one canonical model and reconstructible aliases.
Tiny-model regression proves reused versus fresh loading gives identical hands
and probabilities, including translation reset and corruption rejection.

| Arm | Fixed validation/snapshot/model hashes,32 /512 s | Play/replay/reproduction/raw hash/setup,32 /512 s |
|---|---:|---:|
| Early/off |0.604 /0.601|1.896 /20.383|
| Terminal/off |3.622 /3.623|1.738 /15.878|
| Terminal/on |3.625 /3.625|1.636 /16.035|

Initial loads are separate. Early plays the uniform reference; other arms reuse
its exact retained rows, while replay includes repeated reference rows. At512
blocks, primary play/report is0.732 ms/actually-played-hand early,0.914 ms terminal,
0.933 ms on; independent nonmodel-hash replay is0.385/0.485/0.482 ms/replayed-row.
These include bookkeeping, not isolated solver latency. Fixed model costs are
not multiplied by sample size. Block-scaled setup joins the conservative
variable bound; constant manifest work in that phase is conservatively included.
Full output inventory hashing, previously outside wall_seconds, is now timed.
[Phase receipts](hu100-independent-stages-artifacts/calibration-costs.json),
[denominators/rates](hu100-independent-stages-artifacts/calibration-rates.json).

Each training seed was separately admitted against 99.11min upper training,
save/export/audit cost plus the shared30min closeout reserve, unchanged entry
ceiling and complete originals/archive/transient disk quote. The combined
future evaluation estimate could not refuse this useful training. Only after
both audited lineages finished did evaluation use actual remaining time/models:
8,192 quote4.878 h +0.5 h closeout >4.522 h remaining;4,096 quote2.706 h +0.5 h passes.
Both disk quotes pass; selected required47.43 GiB against 85.00 GiB free.
The sample freeze uses cost receipts alone. [Admission](hu100-independent-stages-artifacts/evaluation-admission.json).

| Stage | Observed M4 wall minutes | Upper admission/reserve minutes |
|---|---:|---:|
|Calibration|4.62|30.00|
|Seed 2026100901 train/save/export/audit|38.66|99.10|
|Seed 2026100902 train/save/export/audit|38.93|99.10|
|Final play/replay/reproduction, all nine arms|32.58|141.24|
|Strict reporting|3.62|21.12|
|Local packing/hashing/readback|1.78|30.00|

The one original clock10:34:33–12:41:15UTC totals **2h06m42 s** (7,602.223 s),
including compatibility/freshness, preparation repair downtime and local
closeout; hard deadline16:34:33UTC, six hours. Initial stable admission301 samples
lasts60 s before model work. M1 tests/review/CI and metadata archival are separate;
post-campaign M4 upload metadata uses the same original deadline/baseline,
with explicit before/after guard snapshots and no additional scientific work.
Whole archive operation106.889 s includes pre-manifest hashes; inner seal receipt
90.637 s alone is not its full cost. Detailed train versus export/audit costs and
operation cleanup receipts are in the [resource summary](hu100-independent-stages-artifacts/resources-summary.json).

## Guards, failures and preservation

Reviewed limits remain6 GiB soft /8 GiB hard family RSS,57,658,644 entries,
normal pressure,>=15% system-free,3,000,000,000-byte swap growth from the fresh
fixed **1,429,408,317.44-byte baseline**,15.5 GiB disk floor and AC. Across36,158
recorded samples: peakRSS **5.11 GiB**, maximum swap growth **592,246,210.56 bytes**,
minimum disk **64.75 GiB**, minimum reported free **75%**, all pressure normal/all AC.
No guard breach, capacity stop, scientific retry or surviving owned child.

Initial source dd19995 stopped during preparation before any fixture/calibration
or final hand: its provenance gate compared #207's binary source to main's
inactive newer native tree (merged optional equity-bucket #210). The reviewed
repair archived/bound actual frozen bd0e7a source without changing/rebuilding the
executed trainer or HU100v1 branch. Original failure/receipt remain immutable.
Readmission requires exclusive ownership, dead old supervisor, zero prior
science, old guard passes and 301 fresh observations against the **original**
baseline. No clock reset, baseline rebase or guard waiver.

Readmission's reported idle gap370.914 s and actual adjacent-stream maximum gap
370.968 s differ by first-sample scheduling. This gap is disclosed and charged;
no continuous whole-clock claim covers it. All science/localpacking stages have
observed guard coverage. A later read-only progress probe attempted to read the
last row of still-empty save telemetry and raised IndexError; it changed no
campaign data or stage, was recorded, and the subsequent probe handles pending
telemetry. Original #207/#211 failures, inputs and active dependencies remain.
[Repair review/readmission](hu100-independent-stages-artifacts/preparation-repair-review.json).

## Storage, final review and unanswered questions

Primary native Research-Cloud bundle locally verifies1,422 unique members and 30
hardlink aliases:15,801,035,296 bytes, SHA256
`b0be9e7b2b7ad42b40fea6e95977f21f42c9aac4848c75f5a691448f41436edc`.
Manifest SHA256 `53cc1f8e058ea81e8141f50f3a111f8767908a4ee44d8728353dec4f8f44c4ad`.
[Local receipt](hu100-independent-stages-artifacts/archive-receipt.json) and
[asset/member pins](hu100-independent-stages-artifacts/model-index.json).
The primary archive is [accepted in Research-Cloud](https://drive.google.com/file/d/1gNvm6C1lkaxu1M82pVfcsoaYrOlYsYT9/view).
[Native uploaded/no-pending/no-conflicts](hu100-independent-stages-artifacts/native-upload-status.json)
and [independent cloud ID/name/size/parent confirmation](hu100-independent-stages-artifacts/cloud-acceptance.json) agree.
[Independent scientific/local archive review](hu100-independent-stages-artifacts/science-evidence-review.json) passes.
The [accepted M1 metadata bundle](https://drive.google.com/file/d/1B3LqWZyeg-nRdXtIyqqK-vHTQ8j1fyKU/view)
retains the complete closed guard stream, archive lifecycle, native M4 upload
evidence and dated report/scientific review: 147 fully readback-verified members,
939,179 bytes, whole SHA256 `7cc076f20ed95ad2d844a820ecec3698f5b1096b14b2f1a5b5b7c8c0b109f2f9`,
manifest SHA256 `4abec5dc49f7ec20c39728fc139ef637a9d6149a8094b82127d2c57b5e542140`.
[Local metadata seal](hu100-independent-stages-artifacts/metadata-archive-receipt.json),
[native upload](hu100-independent-stages-artifacts/metadata-native-upload-status.json),
[cloud confirmation](hu100-independent-stages-artifacts/metadata-cloud-acceptance.json).
Later acceptance/review/CI receipts remain in Git; the snapshot’s dated pending
statements are superseded by these current receipts. Metadata packing used M1
only (0.861s), with no model/main ZIP copy or additional M4 scientific work.
The final bounded M4 upload query retained the original baseline/deadline and
passed guards before/after; its final timestamp is 13:01:18.586 UTC, 2h26m45s
after original admission and inside the original six-hour deadline. Science/packing
closed at 12:41:15; no continuous observation is claimed after that close.
Remote bytes have not been downloaded/verified. All originals remain; no
cleanup, force-offload or changes to another owner's active root.

The fixed experiment is complete. The recipe remains unqualified because one
formal contrast is inconclusive; storage/review cannot change that statistical
finding. Remaining questions: whether tight growth repeats beyond these three
fixed lineages; whether translated pot-pressure play becomes winning in absolute
terms; and how sparse pot/late-history coverage relates to those weaknesses.
Any additional sample or training is a separately authorized experiment, not
continuation of this frozen campaign. Focused source qualification38 tests pass;
Final metadata storage review and exact-head CI remain pending.
