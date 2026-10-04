# Board pooling M4 execution — revision 3

Owner authorization October 4 replaces the RunPod plan. No rental exists; $5
unused. Read the [protocol](hu20-board-pooling-protocol.md) and tracked M4 budget.

Fresh campaign: `/Users/dberweger/Local/hu20-board-pooling-20261004` on M4, with
its own checkout. Python: `/Users/dberweger/Local/deepcfr-training/.venv/bin/python`.
Inputs: `/Users/dberweger/Local/hu20-board-pooling-20261004/inputs-restored-02`,
verify all three stored-average hashes. The shared original alias was archived;
use this isolated dependency folder. External tool stays outside checkout; copy/hash-verify
the retained Mac v2 and reference binaries/source receipt. No new M1 solves.

Record an external approval.json with source/binary/plan fingerprints, observed
swap baseline, shared append-preserving clock path, original start and unreset
24-hour deadline. One worker, six threads; owner-amended 7-GiB worker / cache-inclusive 8-GiB
family ceiling; 20-GiB disk floor. Use `scripts.guard_board_pooling` for every
stage; it reuses #148's admission and RunBudget. Detached jobs survive SSH.
Use `ssh -n m4` in loops. No deletion or CloudStorage changes.

1. Guarded prepare: `scripts.prepare_board_pooling --memory-gib 4`, all forty
   boards, three immutable disk-backed average policies, K100k, frozen codebooks.
2. Guarded qualify: `scripts.qualify_board_pooling`, retained singleton/native
   payoff/MES fixtures plus three fixed real-export 20k-deal V4 checks and
   convergence/memory/time pilots. Six native threads fixed. Linux parity not run.
3. Post pilot resource/results and conservative forecast on #149 before main.
   Update qualification hash in external approval only after all gates pass.
   `scripts.run_board_pooling` refuses forecast beyond the remaining reserve.
4. Guarded main: 120 collect solves, opposite-half fits, fresh lock-only passes
   (plus covered-context sensitivity) and 18 unchanged same-thread replay jobs.
   Stop owned work on any gate/guard failure; no automatic restart.
5. Retrieve/hash-verify all attempts, prepare final report and compact JSON,
   update roadmap and draft #149. No hypothesis decision if incomplete/common
   mask insufficient or missing-key coverage exceeds 5%. Report every outcome.
   Preserve raw evidence under the dated folder for RESULTS_INDEX Drive archive.

October 4 second owner readmission uses `approval-resume-03.json`,
`guard-prepare-03` and fresh `prepared-03`. Recovery checks the pinned second
retrieval manifest (SHA-256
`38710bfd86d9d18c73b93e059ef95bc67b0d501597a276406957507c7595d8f5`),
old index/source hashes, regenerated codebooks, menus, ranges and card features.
Only identical completed artifacts are shared by immutable hard links into the
fresh directory; old attempt evidence is never rewritten. Regenerate K and
features before accepting reuse. The missing third-lineage exports are newly
computed. This is deterministic exporter recovery, not changed science or
solver-result reuse. `continue_board_m4_03.py` queues `qualification-03` only
after successful guarded preparation; it never starts main or restarts failure.
The original clock retains both failed attempts (1871.3424450419989 seconds).

Qualification-03 mandatory stop: preparation completed 120/120 exports; K,
retained river/singleton fixtures, first real-export V4 (20,000 deals) and
first V5 (0.1832% pot) passed. Fresh first lock-only evaluation exceeded the
5-GiB worker RSS limit at 5.0034 GiB. No full qualification/forecast/main was
admitted. All owned processes exited; heartbeat paused; no automatic restart.
See the [stop report](reports/hu20-board-pooling.md) and third retrieval receipt.
The journal retains 3079.873457832997 guarded seconds with the original
absolute deadline. Memory engineering plus explicit owner readmission and
remaining gates are prerequisites for another attempt; current limits and
science are not waived.

Third explicit owner readmission increases only worker RSS to 7 GiB, with the
8-GiB family and 4-GiB arena unchanged. Use fresh `approval-resume-04.json`,
`qualification-04` and `guard-qualify-04`; reuse pinned `prepared-03` unchanged.
The qualifier reads the approved worker ceiling and writes per-pilot stage
resource records. All three fixed pilots and the retained fixtures run fresh.
Post forecast before main; failure stops without retry or further budget change.

Readmission-04 source `2dfde352bd2d229146fc894a0694ab4db552b724` verified
253 preparation/source members against retrieval-manifest-03 and all 120
request/compact pairs in place. Arena remains exactly 4 GiB. Fresh admission
allows the unchanged 8-GiB family with 93.88 GiB free disk; the original
allowance leaves about 18.1 hours before its one-hour reserve at admission.
Qualification guard PID 46437 runs alone; heartbeat is active for forecast/main
admission and eventual verified closeout. [Receipt](reports/hu20-board-pooling-artifacts/m4-resume-04.json).
Twenty-five Python tests pass, including fixed-budget amendment rejection and
separate solve/lock RSS retention on a simulated lock failure. All native
qualification is M4-only. A launch-helper import initially lacked PYTHONPATH;
it failed before admission or any native work and was rerun with the checkout
on PYTHONPATH. The absolute deadline includes that administration time.

Qualification-04 stopped under the owner's forecast condition after the first
fresh solve (256.901 seconds, 4.902 GiB) and lock-only pass (285.396 seconds,
5.568 GiB). Both fit 7 GiB; first V4/V5 and reference-lock check passed.
The frozen formula already lower-bounds main at 31.182 hours versus 17.899
remaining before reserve. Adding pilots cannot decrease either maximum.
First replay was deliberately interrupted; middle/last pilots unattempted.
No full qualification or main admission. Guarded total 3773.578917582996
seconds, original clock/deadline unchanged; heartbeat paused. All owned work
exited. Further execution needs owner-approved allowance/compute action;
no automatic restart or further RSS increase. Fourth fresh retrieval and
all three prior copies are hash-verified; see the latest report and receipt.

Owner-authorized engineering amendment (October 4, 16:45 UTC): no longer
allowance or rental. Up to about two hours of M1 development/short tests,
with no M4 solves during engineering. The external build caches each distinct
pooled file through its last use, clears native locks directly and traverses
with exact copies of interpreter state instead of replaying every prefix.
Lock normalization, solver arithmetic/storage, tree, original requests, 4-GiB
arena, six threads and all frozen science are unchanged. Sources remain
outside the MIT checkout; the old binary and raw responses stay immutable.
An independent timing sidecar records parsing, tree/arena construction, CFR,
policy construction/locking, BR, blueprint EV and serialization.

M1 zero-CFR tests compare 33,984 decision contexts across compressed/plain,
rainbow/monotone turn trees, including nonuniform locks and zero reaches;
weights/strategies and twelve BR comparisons match exactly. Input-only parsing
of the retained 112-MB pilot policy takes 4.472 seconds for five loads versus
0.677 seconds for one cached load, so repeated parsing alone cannot explain
285 seconds. The new traversal/locking speedup still needs measurement.
[External fingerprint](reports/hu20-board-pooling-artifacts/engineering-amendment-05.json).
Thirty-two Python tests pass, including strict response comparison and pinned
first-pilot recovery. No M1 production solve or M4 engineering solve occurred.

Next: the expressly authorized guarded M4 pilot-0 solve and lock-only reruns
use the **original request files**. Require exact scientific response hashes;
ignore only top-level elapsed time and peak RSS. Differences stop without
relaxing tolerance. Reforecast as `1.5 * 138 * (slowest solve + slowest lock)`.
Only if that fits the remaining original deadline minus the one-hour reserve,
run remaining fixed pilots/fixtures and the unfinished first replay. First
V4's fixed 20k-deal native MC may be reused only when every evidence hash is
pinned and the fresh blueprint EV matches exactly. A failed forecast stops
for the owner's allowance/compute decision. Production requires all original
gates and the final measured forecast; no automatic retry of a failure.


Engineering-05 closeout (October 4): exact pilot-0 solve/lock scientific hashes
match qualification-04. The external native binary is
`fb32974d9d211fa66001d1efb330ec4af2d825005d24b287dc6cf3c37fa8812f`;
source archive SHA is
`a5ee6998617b8b94a62f79835851c5f62e8ac4eb6c458ff5fbe297634abed5c8`.
Qualification-05 completed 61/61 gates, all three fixed V4 checks (20k deals
each), converged solves and fresh unfinished replays. Qualification SHA
`a1914726d352a179e55e75706b245eb867060bb3a5ae9242889664e224b128c0`.
Final forecast 55,952.809 s fit 60,058.807 s before reserve at admission and
was posted before main values. Main-05 used f2ad9e7 and prepared-03 unchanged.

Main-05 was deliberately stopped at 17:43:28 UTC upon discovering eager fit
retains all 120 huge parsed statistics. No guard threshold breach; one atomic
collect result and one partial retained, zero relock/common boards. All owned
processes exited; heartbeat PAUSED. Clock charges 5,497.8664519159975 guarded
seconds, original start/deadline/reserve unchanged. No automatic restart.

M1-only bounded fit repair uses original pooling arithmetic, streams each
lineage/root in original per-group order, and writes byte-identical policies.
A sequential owned Python child exits before native workers resume; same RSS,
family, swap, disk and clock guards apply. Three actual pilot fixtures with
synthetic split/identity metadata: 2.565→1.662-GiB peak, 26.043→32.394 seconds,
all three policy hashes equal. This is not a forty-board memory measurement
or a main scientific readout. Thirty-five Python tests pass, zero M1 CFR.
The driver repair is not deployed to M4. Owner readmission and a fresh
remaining-clock forecast precede any continuation; no extension/rental is
implied. [Current report](reports/hu20-board-pooling.md) records every outcome,
raw inventory, exclusions and unavailable intervals. Fifth fresh retrieval
verifies all 3,476 files and all four older immutable copies. Nothing deleted.
