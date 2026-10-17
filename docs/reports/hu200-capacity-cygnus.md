# HU200 capacity experiment on cygnus-x1

**Two seeds have saved20B; their final audits are pending.** Both completed10B recovery checkpoints (1421/1422) reproduce pre-shutdown compressed hashes exactly and pass every-row audits. Seed1302's10B checkpoint matches its known node/iteration identity; its full audit is still running before20B continuation. No failure or guard stop. No final evaluation hands yet.

October17,07:42Madrid snapshot: CPU59°C/VRM63.5°C,132.52GB available,158.37GB free NVMe,75MiB swap (74MiB growth over1MiB baseline, within256MiB guard). Two audits are active; full prospective output reservations currently limit further admission. Training/audits remain on the weekend forecast; report/archive target Monday19 evening, Tuesday20 upload contingency.

[Protocol](../hu200-capacity-cygnus-protocol.md) fixes three paired seeds,
301,153,954-entry capacity, guards and final sample sizes. Owner decides merge.

## Morning progress — October17,05:42UTC /07:42Madrid

[Verified status snapshot and source receipt pins](hu200-capacity-cygnus-artifacts/status-20261017-morning.json):

| Seed | Current saved endpoint | Entries | Verification/progress |
| --- | ---: | ---: | --- |
| 2026101302 |10,000,000,240 |257,683,983 | Full10B audit, final sorted validation/hashes phase |
| 2026101421 |20,000,001,428 |301,153,954 |10B exact reference/full audit passed;20B native export/audit active |
| 2026101422 |20,000,000,322 |301,153,964 |10B exact reference/full audit passed;20B save awaiting audit admission |

The ten-key cap overshoot for1422 is the unchanged complete-iteration freeze
semantics. Both20B tables are frozen. Final20B every-row verification is still
pending, so neither is yet claimed as a verified terminal model. No final
evaluation or playing-strength result exists.

Completed10B audits take4h11m and4h22m, matching the current cost projection.
Seed1302's audit started later at01:58UTC due resource admission and is now
finishing its sorted-stream checks. Native10B→20B stages for1421/1422 take
3h05m/3h11m including loads/writes. The first20B audit began05:23UTC.
The current original259GB plan has approximately257.32GB committed to owned
files, remaining output reservations,20GB growth and50GB floor, leaving1.68GB
estimated admission margin. A further23GB audit reservation cannot yet fit;
verified completion/retirement frees it. No disk budget or guard is relaxed.

Recovery trace through the snapshot peaks CPU69°C/VRM83°C, RSS75.34GB; minimum
available55.04GB, free157.11GB. Swap peaks75MiB,74MiB above original baseline;
no clean-stop sample. Current fan1 798RPM, fan2 1204, fan3 1200. Mutable trace
byte cut and conservative reservation-estimate limitations are recorded.
The deadline remains October19 03:29:18UTC; report/archives target remains
October19 evening Madrid/October20 upload contingency. Evaluation/transfer
costs still need measurement.

## Recovery running — October16,19:45UTC

[Startup verification](hu200-capacity-cygnus-artifacts/recovery-startup-verification.json)
confirms all three actual native commands, process identities and CPU affinities.
Recipe flags and5B resume hashes match the interrupted10B attempts; only
max-seconds/output/telemetry/stop-file values change. Each trainer reserves two
CPUs (0–1,2–3,4–5),38GiB RSS and20GB prospective output. Clean sourcec48e64ce
is immutable while supervisor4325 and observer4367 run. The original clock hash
is unchanged; the effective deadline is the approved fixed+3600s value.

At19:45:25UTC CPU/VRM63/61°C, available115.021GB, family RSS17.623GB while
parents load, NVMe free220.645GB, swap0, no stop requested. Existing webhook-only
task remains enabled/bound to this thread, with the old sent-event ledger reused;
no recurring timer or new task is created. New10B and terminal events use the
same owned root receipt names, now referencing fresh recovery output paths.
No final hands or output archive acceptance yet.

## Approved recovery admission — October16,19:41UTC

The owner explicitly approved the documented common5B rollback and single one-hour
extension.53 checks pass on bb128 under the resource guard. An earlier test
assertion and source-staging setup failures are preserved in
[validation](hu200-capacity-cygnus-artifacts/recovery-validation.json); no scientific
job ran during those corrections. Neither trainer nor exporter code changed.

[Fresh admission](hu200-capacity-cygnus-artifacts/recovery-admission-20261016.json)
revalidates the qualified binary, allK50 tables, all three5B parent hashes,
unchanged5B averages/twins and prior gate receipts. Available132.841GB versus
130.192GB conservative requirement leaves2.649GB margin. Owned94.597GB plus
60GB initial output reservations,20GB growth and50GB floor totals224.597GB
within259GB. NVMe free220.646GB; CPU/VRM55/44°C, swap0 after reboot. The
original1MiB swap baseline is retained. The readmission guard exits0 in46.43s.
[Single fixed extension ledger](hu200-capacity-cygnus-artifacts/training-deadline-amendment-20261016.json).
[Re-quoted launch ETA](hu200-capacity-cygnus-artifacts/eta-at-recovery-launch.json).
All scientific launch/output and observer receipts follow separately.

## Accidental power-off — October16,18:19UTC

The last complete trace sample is18:19:00.665UTC, CPU54°C/VRM67°C, no stop;
the final JSONL row is truncated and preserved. Reboot began about18:21:15UTC.
Both prior scheduler/observer processes are absent. No scientific work restarts.
The guarded post-boot readback passes all11 files /76,175,040,480B, including
both completed10B checkpoints, in168.5s; CPU/VRM55/53°C, RSS33.44MB, no swap.
[Readback](hu200-capacity-cygnus-artifacts/power-off-byte-readback.json),
[guard](hu200-capacity-cygnus-artifacts/power-off-readback-result.json),
[assessment](hu200-capacity-cygnus-artifacts/power-off-assessment.json).

Seed1302's partial temporary checkpoint header reports10,000,000,240 nodes,
but the6.708GB partial file cannot be used for recovery.99m52s elapsed on that
attempt before the last trace. Seeds1421/1422 saved complete10B files that match
their native hashes, but neither completed its every-row audit. No final10B
milestone is claimed. All audited5B recovery/average bytes and restored twins
remain intact; no confirmed corruption among the checked files.

The owner's proposed common5B restart repeats approximately15B physical
main-stage nodes across the three seeds, with the same20B endpoint per seed.
The separate proposal names exact parent/reference hashes, preserves old
attempts in place, budgets their extra30.676GB, admits lower audit concurrency
if needed, and proposes exactly+3600s once to the main deadline. Original
clock remains unchanged; the subsequently approved extension is a separate ledger. Evaluation's24h and every other guard
remain fixed. Completion target is conditional while recovery is pending.

## All three5B saves audited — October16,16:40UTC

All three5B saves pass full byte/SHA/EOF/CRC/order and every-row current/average
audits. Trainer/audit/retirement exit codes are0. All three required descriptive
5B averages remain, totaling16,526,079,059B; their sizes/mtimes still match the
audited receipts. Exact model SHA256s and parent checkpoint hashes are pinned in
[verification and source receipts](hu200-capacity-cygnus-artifacts/milestone-5b-verification.json).
Each10B continuation binds its own audited5B checkpoint SHA. All three runs are
training toward10B, under the clean pinned scheduler and original clock.

| Seed | Completed nodes | Entries | Mean visits/key | Zero visits | Zero average mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026101302 | 5,000,000,521 | 199,420,131 | 4.8513 | 34.614% | 38.061% |
| 2026101421 | 5,000,001,828 | 200,364,896 | 4.8277 | 34.657% | 38.157% |
| 2026101422 | 5,000,001,240 | 197,914,660 | 4.8863 | 34.578% | 38.074% |

Tables remain unfrozen. Over each2B→5B interval, traverser river missing share is
7.517–7.828%, opponent river3.723–3.910%. These are interval support observations,
with no playing-strength conclusion. [Every street/role and exact counters](hu200-capacity-cygnus-artifacts/coverage-5b.csv).

Actual triple overlap lasts3h05m; each audit job takes3h06m–3h30m. Conservative
worker CPU fractions0.877–0.999 and host NVMe busy0.245% indicate the measured
pipeline remains CPU-bound in this cache state. No matched serial ordinary-audit
control or threefold speedup is claimed. During triple overlap CPU/VRM peak
**75/88°C**. There are no CPU>=85°C or VRM>=95°C samples during5B audits.
There are4,275 samples at VRM>=83°C:83°C is the existing admission hold for new
jobs, while running jobs keep the95°C/30s stop. Neither threshold is changed.
Current16:39:48UTC readings are59/68.5°C. Fan1 during5B audits731–1171RPM,
fan2 1080–1889, fan3 1074–1872.

Since scheduler handoff, aggregate RSS peaks56.526GB, available memory stays
>=75.832GB, free NVMe>=219.359GB, and at most seven CPU reservations occur.
Swap reaches1.75MiB, **0.75MiB growth** over the original1MiB baseline, below
the256MiB growth guard. No clean stop or scientific failure is recorded. Nominal
1s resource sampling has a maximum observed2.587s gap including probe overhead;
fixed byte cuts and measurement limitations are in the verification receipt.

Approved per-seed retirement removes32,401,881,132B of unchanged disposable5B
current exports and old2B recoveries after the5B audits. Exact paths/bytes/mtime/
SHA are recorded; the5B recoveries and all5B averages remain. Protected roots,
failures/partials and synced sources are unchanged.

The slowest5B row rate projects a full301M audit to5.294h.
Six remaining full-cap audit projections total31.76worker-hours, ideally
10.59–15.88wall hours at two-to-three workers before admission gaps. Keep the
9–12h native/save allowance and conservative Saturday17 evening–Sunday18 morning
training/audit forecast. Complete report/archives target remains **Monday19
evening Madrid**, **Tuesday20 upload contingency**. Evaluation cost, upload and
final unpacked retention-copy timing remain unmeasured; all original scientific
deadlines and guards stay fixed. [Updated ETA assumptions](hu200-capacity-cygnus-artifacts/eta-after-5b.json).
No final evaluation hands have started.

## All three2B saves audited — October16,12:05UTC

All three2B checkpoints pass independent byte/SHA, complete gzip EOF/CRC,
strict ordering and every-row current-regret/average-total validation. All
trainer/audit/retirement jobs exit0, with no stop. Native pre-audit telemetry's
"unaudited" field stays immutable; separate audit receipts establish verification.
The original clock hash and clean live source94b84b32 match their pins.
Each5B continuation names its own audited2B checkpoint SHA as parent.

| Seed | Completed nodes | Entries | Mean visits/key | Zero visits | Zero average mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026101302 | 2,000,000,044 | 135,824,554 | 2.8920 | 39.066% | 37.167% |
| 2026101421 | 2,000,000,115 | 135,429,210 | 2.9002 | 39.017% | 37.186% |
| 2026101422 | 2,000,000,080 | 135,893,163 | 2.8914 | 39.042% | 37.270% |

All three tables are still growing. Missing-key shares are measured over each
500M→2B save interval, separately by role/street; they are not lifetime shares.
Traverser river missing share is14.411–14.595%, opponent river8.634–8.718%.
[Exact numerators/denominators and all streets](hu200-capacity-cygnus-artifacts/coverage-2b.csv),
[full save/header/audit receipts and resource verification](hu200-capacity-cygnus-artifacts/milestone-2b-verification.json).
These are training support measurements, with no playing-strength inference.

Three actual audits overlap for2h03m, independently measuring0.884–0.999
cores/worker and host NVMe busy0.125% during common intervals. Individual full
jobs take2h03m–2h17m, including validation/export. Their combined sampled window
is09:44:49–12:02:45UTC. Triple-overlap peaks CPU65°C/VRM74°C; since scheduler
handoff, peak aggregate RSS40.846GB, helper RSS141.84MB, minimum available
91.532GB/free260.592GB, swap unchanged1MiB. No clean-stop sample; at most seven
reserved CPUs. Fan1 ranges658–920RPM, fan2 898–1388, fan3 907–1401 during2B
audits. Maximum observed resource sample gap2.587s is disclosed; nominal guard
sampling is1s plus probe overhead. Raw traces remain active under the owned root;
receipts record fixed byte cuts rather than claiming sealed full-trace hashes.
This is an actual triple-overlap measurement, without a matched serial ordinary
audit control or a claimed threefold speedup.

After successor audits, the approved scheduler retires 37,540,138,760B of
unchanged owned disposable2B current/average exports and old500M recoveries.
Every exact path/bytes/mtime/SHA is recorded in the per-seed retirement receipts;
the verified2B recoveries remain, and their size/mtime still match. Required5B/20B
averages will be retained. No other root, failed output or synced source is touched.

The slower measured2B row rate projects a full301M audit to5.089h.
Nine remaining audits, each conservatively projected at full cap, imply
45.80worker-hours, or15.27–22.90wall hours at two-to-three workers before scheduling gaps.
Keep9–12h allowance for remaining native training/saves and the conservative
Saturday17 evening–Sunday18 morning training/audit forecast. Re-quoted complete
report/archives target remains Monday19 evening Madrid, Tuesday20 upload
contingency. Evaluation still needs its cost pilot; upload and final unpacked
retention-copy timing are unmeasured. Original deadlines/budgets stay unchanged.
[Updated arithmetic/assumptions](hu200-capacity-cygnus-artifacts/eta-after-2b.json).

## Earlier preflight and operations record

The owner stopped bb128 for BIOS work and resumed October16 after adding five
fans, modifying the curve and enabling XMP3200. XMP is owner-reported; memtest86+
remains incomplete as accepted. The new2TB disk remains excluded from scientific work; the October16 owner
amendment below adds only a final unpacked retention copy. #292 remains open;
its58,415,280,128-byte root, archives, uploads and shared dependencies remain untouched.
Unrelated preservation files remain untouched. Restart NVMe free263.760GB;
post-audit free259.927GB. [Interrupted-clock receipt](hu200-capacity-cygnus-artifacts/owner-interruption-restart.json).

The no-step500M re-save preserves all69,084,815 keys and row bytes; the only
header difference is config.max_entries. No traversal was repeated after reboot.
[Full identity](hu200-capacity-cygnus-artifacts/memory-row-identity.json).
The resumed export/audit passes on **all69,084,815 rows**, independently checking
checkpoint bytes/SHA256, current regrets and normalized average accumulators.
[Audit and exact hashes](hu200-capacity-cygnus-artifacts/memory-postbios-audit.json).
All old partial exports and the truncated final pre-shutdown resource row remain.

Measured save RSS7,528,763,392 bytes; sampled family peak7,572,365,312 bytes.
After1,384,031,032 bytes of fixed card tables, this is89.6bytes/key; allocating
all100,690,818 currently absent regret slots adds11.66bytes/key, giving about
101.3bytes/key at the observed menu mixture. Admission reserves130bytes/key,
including a five-action fully aged table, index/chunk growth and save workspace.
Three planned tables plus three card-table residents,2GiB helper and6GiB
available-memory floor total130,192,069,748 bytes. Admission must recheck actual
available RAM and the conservative forecast for all three fresh500M prefixes.
No capacity or outcome result is inferred from the zero-step measurement.

The preliminary operation used4,734.229 active seconds of its original5,400:
1,285 before shutdown and3,449.229 after restart. The measured owner-stop interval
of10,429.086seconds is excluded. The72h gates/training/audit clock started October16 at02:29:18UTC;
its fixed deadline is October19 at02:29:18UTC. The unchanged24h evaluation
ceiling follows; archive closeout is separate. [Clock](hu200-capacity-cygnus-artifacts/training-clock.json).
The earlier provisional report plus archive-closeout ETA, updated October16 after the
BIOS interruption, is Tuesday, October20 (Europe/Madrid). Variable upload
throughput may move closeout to Wednesday, October21. This forecast changes
neither scientific runtime ceiling; evaluation cost admission remains pending.

[Resource summary](hu200-capacity-cygnus-artifacts/memory-resource-summary.json):
pre-change CPU/VRM peaks65/60°C, fan1 335–645RPM; resumed export/audit54/58°C,
fan1 479–722RPM. These are different phases, not a controlled cooling comparison.
No recorded resource breach. Twenty-one post-BIOS guard/identity/retention/audit
tests pass, including differential checks against the original every-row auditor.

The revised disk plan reserves259GB including50GB free floor, serial sorted-stream
audits and per-seed retirement of redundant recoveries after an audited successor.
All actual writes require fresh admission. Evaluation uses two seeds concurrently,
then the third, within the declared258.4GB plan. No other PR reclamation is assumed.

At that earlier preflight snapshot, restoration was pending. Connected Drive raw and streamed-reference requests both
reject the3.16GB archive with HTTP413 before invocation (256MiB limit). Read-only
synced-archive/SSH copies timed out; partial bytes remain, and explicit transport
resumption continues with longer connection tolerance. All hashing, extraction,
training and evaluation run on bb128. This is an input-transfer failure, not a
model or training failure. No strength result or restored-input verification is claimed.

Campaign source3f6eb6cb and [resource admission](hu200-capacity-cygnus-artifacts/admission.json)
are pinned before launch. The first three20M saves completed in59.62s wall time,
including writes, with CPU/VRM peaks52/43°C and6.25GB aggregate sampled RSS.
All gate saves receive full serial audits. [All three exact compressed fresh/resume identities](hu200-capacity-cygnus-artifacts/resume-gates.json)
pass. The three fresh500M train/save processes finish in833.77s wall time;
this is a measured prefix cost, not a forecast for frozen large-table throughput.
The301,153,954-entry cap remains conditional on exact twin prefix parity and the
three measured mature-memory forecasts; no dependent2B continuation precedes those checks.

Evaluation helper preparation preserves the existing policy/harness behavior.
Compatible play/reproduction cases reuse registries; independent replay uses
separate policy instances. A global byte-verified barrier requires all six
seed/model groups before any mean. No final evaluations or pilots have run.
The first evaluation fixture checks failed because a K50 table path was omitted
from the test fixture; the actual evaluator correctly rejected it. The corrected
fixture's three checks pass on bb128. Original failure log/guard remain, with
[compact failure receipt](hu200-capacity-cygnus-artifacts/evaluation-tests-initial-failure.json).

The500M stage has826 one-second samples: CPU55°C/VRM64°C maxima,
fan1/2/3 ranges478–735/872–1091/877–1111RPM,22,468,595,712-byte
aggregate RSS peak,109,889,511,424-byte minimum available RAM and unchanged
1MiB swap baseline. [Full compact summary](hu200-capacity-cygnus-artifacts/prefix500m-resource-summary.json).
[Four-check evaluation smoke receipt](hu200-capacity-cygnus-artifacts/evaluation-smoke.json)
includes two toy K50 cache builds and direct/random play/replay/reproduction;
these fixture hands are separate from reserved real-model pilots/finals.

A webhook-only continuation task is bound to this thread for this already
running bounded campaign. It does not schedule future experiments or recurring
timers. The metadata observer runs on CPUs6–7 and expires one hour after the
training deadline; terminal delivery ends it. Endpoint/private token files are
excluded from Git and archives. Initial default-User-Agent requests failed
HTTP403/1010; explicit service identification returnsHTTP202 on both bb128 and
the coordinator. The failed observer log remains; only that metadata observer
was replaced, without signaling scientific work. Verify delivery receipts before
relying on wake-up. Remove the callback task and private files at closeout.
Finder automation also failed cgWindowNotFound while inspecting input downloads;
no UI settings were changed. Public NSFileCoordinator read coordination now
requests full materialization of the existing indexed archive; other inputs are
queued serially with coordinator disk headroom. Restoration remains pending.

[Replacement metadata observer](hu200-capacity-cygnus-artifacts/event-observer-launch-v2.json)
starts06:11UTC; [HTTP202 plus successful app delivery](hu200-capacity-cygnus-artifacts/continuation-delivery.json)
is verified. The first two full archive materializations finish; their old
partial-read clients are stopped after identity checks, their receivers confirmed
closed, and copies resume from retained byte offsets using newly opened fully
local source files. The original conservative2x-size coordinator admission
refused seed1302 before its read began. That refusal remains logged; a coordinated
read without an upload snapshot needs8,921,067,848 bytes including3GB margin,
versus16,324,145,152 bytes free, and is admitted separately. Only read coordination
changes; source archives, model bytes, scientific budgets and guards stay fixed.
Full input archive/model hashing and restoration on bb128 are still pending.

Seed1302 coordinated archive-read readiness was reported October16 at06:15UTC.
Its old owned transport client56817 was stopped after exact command identity
verification; the matching remote receiver was confirmed absent before one
resume from the retained remote offset. This is transport progress only: whole
archive SHA and twin model restoration remain pending on bb128. The synced
source still reports dataless, so the callback alone does not establish durable
local hydration. Sources and partials remain intact.

Seed1422 transport exits0; the owned bb128 archive has exactly3,162,066,289
bytes at06:35UTC ([receipt](hu200-capacity-cygnus-artifacts/transport-complete-2026101422.json)).
Whole-archive/member hashes and restoration remain pending under the serial
controller gate. Seeds1302/1421 continue from retained offsets; no restart or
scientific retry is launched. Two fresh500M prefix audits pass; the third
and normalized twin parity remain pending. The October20/21 closeout ETA holds.

At06:38UTC /08:38Madrid all three fresh500M full audits pass (59–61min
each). The controller runs parallel no-step old-cap normalization before the
remaining exact twin/hash gates and measured mature-memory admission. No2B
continuation or final evaluation hands yet. Active sample: CPU56°C, VRM62°C,
fan1/2/3 738/1107/1109RPM, aggregate workload RSS11.54GB, available120.98GB,
swap unchanged1MiB and272.15GB free NVMe. Archives1302/1421 have reached
1.94/2.07GB of5.92/3.16GB;1422 is complete by size. No campaign failure receipt.
[Status receipt](hu200-capacity-cygnus-artifacts/status-20261016-0638.json).

The October16 owner asks to shorten checkpoint scheduling within eight CPUs.
[Proposed amendment](../hu200-capacity-scheduling-amendment.md) is awaiting the
owner decision required by that request point4: the committed protocol explicitly
forbids overlapping audits/exports. The running schedule has not changed.
[Measured operations evidence](hu200-capacity-cygnus-artifacts/scheduling-observation-20261016.json):
current export consumes0.99946 cores, with host NVMe busy only0.19% over30s;
completed streaming audit RSS peaks below49MB. This supports a concurrency trial,
not a measured threefold speedup. Full sorted-validator utilization is pending.
Qualified source already uses two-worker seat traversal, with serial delta apply
and checkpoint I/O; the earlier six-busy-thread wording overstated continuous
utilization. No live training traversal was available for per-thread measurement.
Owner-reported real-trainer15min maxima69°C VRM/59°C CPU at125W PL1/PL2 are
recorded separately. Conditional revised closeout target October19, upload
contingency October20, requires approval and subsequent throughput admission;
current October20–21 estimate and scientific runtime ceilings remain unchanged.

The owner explicitly approves the scheduling amendment as written on October16.
Protocol execution, save/continuation and disk wording now reflect independent
seed queues and variable utilization of two-worker trainers. One global queue
watchdog replaces the serial controller at a successful audit boundary, keeping
the original clock and receipts. The unchanged audit/export code runs one-core
workers, initially two. A third is admitted only after a completed parallel
batch has row-normalized speedup>=1.4, conservative CPU utilization>=0.8, host
NVMe busy<50%, and the original RAM/disk/output reservations fit. Native output
bytes, all scientific checks and evaluation remain unchanged.
[Validation](hu200-capacity-cygnus-artifacts/scheduling-validation.json):36 distinct
checks pass, including real-process overlap and handoff, global stop propagation,
clock preservation and serial/parallel export byte identity. Test-harness failures
(missing pytest locally, then missing temporary parent/pinned test metadata in
the older isolated checkout) are retained; no scientific attempt was repeated.
The handoff is queued until the current audit exits0; no incomplete audit is
interrupted. October19/20 remains provisional until the first two-audit batch
completes and supplies the requested ETA re-quote.

[Deployment launch](hu200-capacity-cygnus-artifacts/scheduler-launch.json):
clean source94b84b3255f16daa529eae8250ca9a391249b2ac is deployed separately
from the live old source. Detached scheduler PID65291 starts07:21UTC, waiting
for the current normalized prefix audit to finish successfully. Original
controller61705 remains active and supervising that audit until the verified
boundary. The unchanged deadline is1792376957.996109; no third audit is yet
admitted. The first parallel-batch measurement will emit an event into this
same bounded campaign for the requested ETA update. All raw scheduler validation
logs are also retained under the owned bb128 planning directory.

Seed1421 archive transfer is complete; owned bb128 archive size3,162,672,661B
matches the index ([receipt](hu200-capacity-cygnus-artifacts/transport-complete-2026101421.json)).
Both1421/1422 transports are complete by size;1302 progresses to4.337GB/5.921GB.
Whole hashes/member restoration remain pending. Scheduler65291 is healthy and
waiting for the current successful audit boundary; no campaign failure receipt.

All three archive transports are complete by expected byte size at08:19UTC;
[1302 transport receipt](hu200-capacity-cygnus-artifacts/transport-complete-2026101302.json).
Whole-archive/member hashes and guarded restoration remain pending; transport
completion is not model verification.
[Successful handoff](hu200-capacity-cygnus-artifacts/scheduling-handoff.json) occurs
07:52:59UTC after normalized1302 audit exits0, with no scientific child interrupted.
The original clock SHA/deadline are unchanged. Scheduler65291 now owns two
concurrent full audits,1421 onCPU6 and1422 onCPU7. At08:19UTC CPU51°C/VRM60.5°C,
combined audit RSS93.7MB, swap unchanged1MiB, free disk280.36GB;
[live snapshot](hu200-capacity-cygnus-artifacts/parallel-audits-live-20261016.json).
Neither third-worker admission nor the requested measured ETA re-quote occurs
before this first two-audit batch completes.

The first two-audit batch is verified against its completed every-row audit and
identity receipts:69,180,237 and69,055,433 rows pass in4,470.523s wall, versus
4,008.892s for the comparable69,084,815-row serial normalized audit. Row-normalized
throughput is1.7943x; worker CPU utilization0.9002, host NVMe busy0.1585%,
peak CPU/VRM61/61°C, peak helper RSS95,367,168B, minimum available132.353GB
and free disk268.865GB. Initial/extraction/final-validation phases each use
about0.88–0.92 cores per worker; the pipeline is CPU-bound in this measurement.
[Batch](hu200-capacity-cygnus-artifacts/parallel-audit-batch.json),
[independent verification](hu200-capacity-cygnus-artifacts/parallel-batch-verification.json),
[HTTP202 delivery](hu200-capacity-cygnus-artifacts/parallel-audit-batch-delivery.json).
This meets the published thresholds for an audit limit of three; actual writers
still require disjoint CPUs,512MiB family RSS and original disk reservations.
No measured three-worker speedup is claimed. Per-process I/O may include reaped
children, so host disk counters, rather than summed process writes, govern the
disk-bound conclusion. Raw utilization/resources remain in owned guards/scheduler.

[All three exact prefixes](hu200-capacity-cygnus-artifacts/twin-gates.json) match
the indexed twin500M compressed hashes and every row, excluding only header
config.max_entries. [Mature-memory admission](hu200-capacity-cygnus-artifacts/prefix-memory-admission.json)
is100.00–100.03bytes/key versus the unchanged130 budget; cap301,153,954 remains.
Native telemetry retains its original pre-audit "unaudited" label; independent
full audit/identity receipts establish the verified status.

[Restoration](hu200-capacity-cygnus-artifacts/twins-restored.json) passes whole
archive SHA, embedded manifest SHA, selected-member bytes/SHA/CRC and20B
seed/schema/uniform-fallback identity for every twin. All three model hashes
match the owner's pins. [Transport retirement](hu200-capacity-cygnus-artifacts/transport-copy-retirements.json)
removes only12,245,806,798B of unchanged owned input ZIP replicas after all
restorations pass; Drive originals and9,489,299,657B of restored models remain.
Owning PR265/278 are independently confirmed MERGED; PR301 stays OPEN. No other
root or synced original is removed. Retrieval pins remain in drive-inputs.json.

[Live30-second trainer observation](hu200-capacity-cygnus-artifacts/trainer-thread-observation.json)
confirms three native processes on0–1/2–3/4–5, environment thread limits2.
Each has a main thread and two CPU-active traversal workers: worker pairs each
consume about0.85 cores combined, main about0.30, total1.14–1.15 cores/run.
This confirms configured two-worker traversal with variable utilization, not
continuous two-core saturation. No binary or scientific option changes.

[Post-batch ETA arithmetic](hu200-capacity-cygnus-artifacts/eta-after-parallel-batch.json)
conservatively projects the slower normalized audit to5.415hours at301M rows.
Twelve full-cap audits would cost64.98 worker-hours, about21.66–32.49 wall hours
with two-to-three concurrent workers before scheduling gaps. Allow9–12hours
for native training/saves; overlap can reduce the sum, while disk admission
can lower concurrency. Training/audits therefore project to Saturday17 evening
through Sunday18 morning, Madrid. Evaluation/replay/reproduction retains24h
and still requires its cost pilot. Complete report plus archive target is
Monday19 evening, with Tuesday20 upload contingency; upload bandwidth is
unmeasured. Original Monday19 02:29:18UTC training deadline is unchanged.

The real queue integration fixture is explicitly bb128-only: it relies on
qualified physical CPU0–7 and a50GB free-disk floor. Generic CI cannot execute
that hardware fixture on its smaller runner. This changes only test collection,
not live source94b84b32 or its already passed36-check validation.

Owner storage amendment, October16: final uploaded evidence will also be retained
unpacked under `/mnt/data2tb/research/PR-301-hu200-capacity-cygnus/` on bb128,
with the archive member hierarchy and manifests. Read-only inventory confirms
`/mnt/data2tb` is the2TB `/dev/sda1` ext4 mount, owner-writable, with1.968TB
available. No destination files have been created or copied. At closeout,
recheck device/space/PR status, verify member sizes/SHA256 against sealed
manifests, and record the retained copy in the index. This owner-authorized
retention exception changes neither scientific storage nor compute limits;
ETA remains the current October19 evening target/October20 upload contingency,
subject to final copy timing. [Plan](hu200-capacity-cygnus-artifacts/local-retention-plan.json).
