# Frozen HU200 three-seed blend run

Owner-authorized continuation of merged #297 on October 16, 2026. Its disk
admission stop was plumbing: no full blend, cache, pilot or final hand existed.
This campaign follows [the frozen protocol](../hu200-v055-blend-protocol.md)
and preserves [the original report](hu200-v055-blend.md).

**Current status (October 16, 14:53 Madrid): the blend is built and fully audited.**
All156,208,331 rows pass the unchanged independent checkpoint/current/average
audit on the 16 GiB M4. The average inference model is6,282,968,150 bytes;
its raw synthetic checkpoint is6,042,193,732 bytes. Caching is running; fixed
pilots and final evaluation remain pending. This PR stays draft until the
planned research actually runs or a research failure is recorded. Plumbing
interruptions remain pending work, with their evidence preserved.

All construction, export, cache and evaluation work runs on the base Apple M4
with 16 GiB physical memory, under both exclusive research locks and the existing
five-second resource guards. M1 handles Git, PR and archive administration.

The raw `fsum(Node.average_s / W_s)` rule, synthetic identity, original-average
reproduction prerequisites, roots, translation 512/128, final counts and global
replay/reproduction barrier are unchanged. No outcome-dependent tuning, pass/fail
gate, Slumbot, server, release, tag or default change is authorized. The owner
decides merge.

Initial ETA: inventory and admission within 30 minutes if inputs remain local;
missing cloud inputs can add several hours. A full-run ETA is conditional on
actual admission and the cost-only pilots and will be posted before final play.
The frozen 16-hour active campaign envelope includes a two-hour preservation
reserve; an interruption for plumbing does not constitute a research result.

Completed evidence includes all fresh input proofs, actual admission, union
coverage/root diagnostics and full model audit. Pending: caches and cost pilots,
the three fixed direct comparisons and 95% intervals, the seven-opponent panel
for all four models, independent replay/reproduction of every final hand before
readout, final resource/failure summaries, accepted Research-Cloud evidence and
its owner-requested unpacked 2 TB copy. The measured full-run ETA follows pilots.

## Initial inventory and plumbing

[Current inventory](hu200-v055-blend-run-artifacts/initial-inventory.json) confirms
#303 retired the two seed1302/1422 raw checkpoints and all three reproduced
averages: 19,407,460,871 bytes. The protected seed1421 checkpoint, frozen exporter
and pinned tables remain. The source PRs #163, #265, #278 and #297 are currently
merged. Both missing-checkpoint archives remain online-only and their current
independent Drive IDs, names, sizes and parent folders match the index.

Initial M4 free disk was about 83.38 GB. Restoration consumes about 19.41 GB,
leaving roughly 64 GB against the prior 74.474 GB construction requirement.
The owner was alerted to the provisional 10.5 GB shortfall and acknowledged the
plumbing handoff. The later inventory was captured after the first reproduction
had begun writing its output; its lower free-space snapshot includes that partial
output. Actual restored sizes and the fresh union gate will determine admission.
No full blend, cache or poker hand exists. PR remains draft and pending.

The retained seed1421 checkpoint passed its fresh SHA256 check; a fresh frozen
exporter reproduction is running under both M4 locks and the unchanged guards.
Read-only APFS clones isolate its checkpoint/exporter/tables in this campaign's
own ignored root without allocating duplicate payloads.

[Root reservation](hu200-v055-blend-run-artifacts/root-reservation.json) checks
main and every other current open head. Only byte-identical inherited #297
declarations match; no conflicting reservation exists. #297's unplayed roots
remain this continuation's fixed roots.

Administrative failures: the first M1 repository artifact check used system
Python 3.9, which lacks `zip(strict=True)`; Python 3.11 passed before the initial
commit. M4's branch fetch populated `FETCH_HEAD` without a remote tracking ref;
worktree creation was corrected to use that fetched commit. Neither failure
started research. All scientific failures, if any, will be recorded separately.

## First fresh input proof and improved storage

Seed2026101421 freshly reproduces full average SHA256
`c08d86a9615022df7015d34e5dc3de2705bbb8346ed6976bd72284e71d91f444`
from raw checkpoint `c70e8e01b3b27c7fa8b8e0561bdbe5b77b321125de646f04614c82ecacacc676`.
The frozen exporter handles 85,373,916 entries in 713.16 seconds; export plus
hashing is 714.38 seconds. All 169 small-blind root keys exist; mass is exactly
140,199,312,593,278.0, equal to W. Full proof takes 769.18 seconds, 154 clear
samples, 75,415,552-byte sampled family peak, zero swap growth and minimum
80% free memory. [Exact reproduction](hu200-v055-blend-run-artifacts/local-average-reproduced.json),
[root/resource summary](hu200-v055-blend-run-artifacts/local-proof-summary.json).

During that proof M4 free disk rises to about 93.816 GB. After the remaining
16.245 GB of checkpoint/average pairs, the provisional 77.571 GB free projection
would exceed the unchanged prior construction admission by 3.097 GB. This is
provisional: the actual restored sizes and fresh union gate still decide.
Seed1302 archive restoration starts 07:23:30 UTC /09:23:30 Madrid under the
same resource baseline and campaign clock. No full blend, cache or hand exists.

The [new campaign clock](hu200-v055-blend-run-artifacts/campaign-clock.json)
starts this owner-authorized intake at 07:10:28 UTC /09:10:28 Madrid, with the
frozen 16-hour envelope: science ends 23:10:28 Madrid, preservation ends
01:10:28 Madrid October 17. The original #297 clocks/evidence remain unchanged.
No old scientific outcome or failed hand exists to repeat. This campaign's
[failures and pending plumbing](hu200-v055-blend-run-artifacts/failures.json)
remain disclosed; no guard threshold, weight, rule or evaluation count changes.

The [designated evidence folder](https://drive.google.com/drive/folders/1FZiDigTX-_Do4pPihTF4pdZTDaxNRxPK)
is created and independently verified under the Research-Cloud project parent;
both Macs observe the same Drive ID. [Folder receipt](hu200-v055-blend-run-artifacts/drive-folder.json).
Final scientific-output archive acceptance awaits the full run; input preservation is recorded separately below.
At the first archive's observed 4–4.5 MB/s arrival rate, the provisional intake
ETA is 13:00–15:00 Madrid, conditional on network throughput. Final-run ETA still
awaits actual union admission and cost-only pilots.

## Seed1421 raw-checkpoint cloud dependency

The old #278 full cloud object is still only 592,464,795 bytes; its expected
complete ZIP is 20,952,808,971 bytes. The retry object is absent from the current
folder listing. This known upload problem is plumbing, not a model discrepancy.
The M4's retained raw checkpoint already passes its fresh SHA256 and exact
average reproduction above.

M1 archive administration verifies its retained complete local ZIP's whole
SHA256 and embedded manifest, then streams the unchanged checkpoint directly
into a checkpoint-only ZIP without extracting another raw file. All members
and the new whole ZIP pass local readback in 32.50 seconds. All source originals
remain. No blend construction, export, cache or evaluation runs on M1.
[Archive receipt](hu200-v055-blend-run-artifacts/seed1421-checkpoint-archive-receipt.json).
The 4,957,760,250-byte archive is published atomically to this PR's shared
evidence folder. Native uploaded1/uploading0/no-conflicts and independent Drive
ID/name/size/direct-parent metadata now accept it. Remote archive bytes were
not downloaded. [Acceptance](hu200-v055-blend-run-artifacts/seed1421-checkpoint-upload-acceptance.json),
[exact retrieval command](hu200-v055-blend-run-artifacts/seed1421-checkpoint-restoration.json).
Its raw checkpoint remains byte-identical to the M4-verified input. The old
full cloud object remains explicitly unaccepted.

The [complete checkpoint-only archive](https://drive.google.com/file/d/1Kfp_fSXHg7LRTHffUn8Hv_Snb9pzcJFi/view?usp=drivesdk)
fixes this run's required seed1421 raw restoration dependency. It does not
repair the old full cloud object or authorize removal of that retained local
full archive, which contains other evidence. Input-provenance metadata is
added to the new campaign's intake configuration; every raw/average hash,
iteration weight, model rule and evaluation count remains unchanged.

### October 16, 11:07–11:10 Madrid: intake configuration repair

The first full 26.698-GB archive downloaded and passed its whole SHA256.
Restoration then raised `KeyError: manifest_sha256` in the continuation
control configuration before any checkpoint extraction or export. The missing
field is now populated from the existing indexed manifest pin for each pending
source; all required fields are checked before another archive read. The first
repair launch was refused by the original generic STOP sentinel before its
worker started. Both failures and the exact original sentinel are retained;
the baseline and held-state bytes, thresholds and fixed campaign clock remain
unchanged. A distinct guarded intake continuation is running. This is a
plumbing repair with no model mismatch, research failure or hand retry.
[Configuration failure and repair](hu200-v055-blend-run-artifacts/intake-configuration-recovery.json),
[guard-state preservation](hu200-v055-blend-run-artifacts/intake-guard-sentinel-preservation.json).

### October 16, 11:23 Madrid: second exact average and final restore

Seed2026101302 now verifies its full source ZIP, indexed manifest and raw
checkpoint, then exactly reproduces all 85,373,907 original average rows:
`9128f71c602f18a9cb7219b18cd43ff4e3f2b7ed6a30f9babe7ed033840ac7b6`.
The unchanged exporter takes 715.87 seconds; its kernel RSS peak is 4,603,904
bytes. The complete cached-archive continuation takes 739.26 seconds, with
148 clear guard samples and a 58,048,512-byte family RSS peak. The initial
download/configuration stop remains separate in the failure record.

After fresh merged-owner, native upload, independent cloud metadata and open-handle
checks, one normal FileManager request releases the inactive 26.698-GB ZIP
cache. Cloud ID, placeholder size, inode and modification time remain the same;
raw checkpoint and exact average stay protected. The final seed2026101422
archive restoration is now running. Both completed average proofs precede any
full blend or hand.
[Exact reproduction](hu200-v055-blend-run-artifacts/seed1302-input-verified.json),
[resource receipt](hu200-v055-blend-run-artifacts/seed1302-reproduction-resource-receipt.json),
[normal cache-release receipt](hu200-v055-blend-run-artifacts/seed1302-archive-cache-released.json).

### Owner addition: retain unpacked evidence on the 2 TB disk

The owner additionally requests a local unpacked copy of the final uploaded
research evidence. The verified ext4 disk on `bb128` is mounted at
`/mnt/data2tb`; this run will retain its bundle contents under
`/mnt/data2tb/research/PR-306-HU200-v055-blend-run/`, one directory per bundle,
with the same member paths and embedded manifest. Every copied member will
be checked against its recorded size/SHA256, with the resulting receipt and
retrieval commands indexed. The directory does not contain final evidence yet.
This storage addition leaves M4-only construction/evaluation and the frozen
research protocol unchanged. Drive archive/readback/upload acceptance remains
required. [Retention plan and mount observation](hu200-v055-blend-run-artifacts/local-unpacked-mirror-plan.json).

### October 16, 12:59 Madrid: all original averages reproduce

Seed2026101422 verifies its full ZIP, indexed manifest and raw checkpoint,
then exactly reproduces all 85,373,928 original average rows:
`b191c5ed8b943d138e6076fb4a1b0117c98964d92dd2d6253e157f696f691f6f`.
The frozen exporter takes 719.01 seconds, with 4,620,288-byte kernel RSS peak;
full archive restoration/raw verification/reproduction takes 5,613.14 seconds.
All three fresh original-average proofs now pass, before any full blend.

The final inactive 20.953-GB source ZIP cache is released with one normal
request after fresh merged-owner, native/cloud and open-handle checks. Its
same-size placeholder, cloud ID, inode and modification time remain; all
three raw inputs and reproduced averages stay protected. The fresh union
scan and construction-admission calculation are running.
[Third exact proof](hu200-v055-blend-run-artifacts/seed1422-input-verified.json),
[resource receipt](hu200-v055-blend-run-artifacts/seed1422-reproduction-resource-receipt.json),
[normal cache-release receipt](hu200-v055-blend-run-artifacts/seed1422-archive-cache-released.json).

### October 16, 13:05 Madrid: construction admitted and started

The fresh bounded union scan takes 324.31 seconds and again finds
156,208,331 union keys /482,975,338 action slots. Keys present in exactly
one, two and three seeds are 88,880,705 /34,741,832 /32,585,794. The fixed
unplayed 38,103-key compression probe passes the unchanged every-row auditor.
Actual free space is 77,940,490,240 bytes against 74,474,620,158 required:
3,465,870,082 bytes above the complete peak reservation, which already
includes the 16-GiB floor and 2-GiB margin. Construction is admitted and the
full fixed-rule merge starts at 13:05:43 Madrid. No pilot or final hand exists.

The gate reserves 22,654,759,739 audit-workspace bytes, 9,331,111,441 blend-cache
bytes and one seed cache at a time. Compression projections retain the fixed
1.5 multiplier; the small difference from #297's byte total follows the new
run's path/header bytes. Canonical raw/table APFS clones are verified against
their accepted indexed dependencies when sealing, matching the gate's exclusion of separately indexed source inputs; all originals remain.
Union guard: 68 clear readings, 72,941,568-byte family RSS peak.
[Full fresh scan and root diagnostics](hu200-v055-blend-run-artifacts/union-scan.json),
[actual capacity gate](hu200-v055-blend-run-artifacts/construction-capacity-gate.json),
[scan resources](hu200-v055-blend-run-artifacts/union-resource-receipt.json),
[unchanged required sample audit cost](hu200-v055-blend-run-artifacts/union-sample-audit-time.json).

### October 16, 13:28 Madrid: first unpacked HDD bundle verified

The already accepted seed1421 checkpoint-only input bundle is now retained
unpacked at
`bb128:/mnt/data2tb/research/PR-306-HU200-v055-blend-run/hu200-v055-seed-2026101421-checkpoint-20261016/`.
Its source whole ZIP and embedded manifest match the accepted archive pins;
both copied members pass exact size/SHA256 and a separate disk readback.
No ZIP container is present in that final bundle directory. This is storage
administration of an existing input, not blend construction on bb128.
The final blend bundle remains pending. [Unpacked receipt](hu200-v055-blend-run-artifacts/seed1421-unpacked-retention.json);
[direct HDD retrieval](../../RESULTS_INDEX.md#pr306--frozen-hu200-three-seed-blend-run).

### October 16, 13:34 Madrid: complete raw blend checkpoint

The full fixed-rule native merge writes all 156,208,331 rows in
1,732.83 seconds (28 min 52.83 s). The raw synthetic checkpoint is
**6,042,193,732 bytes**, SHA256
`59a6dd353fb4ef06d77a86783c7251521d8c9f8b92a785bb7e11a9ae724d4123`.
The frozen exporter is now writing the current and average inference exports;
full every-row audit follows. The model is not yet claimed audited or ready
for play, and no hand has been generated.
[Complete native merge and pinned output](hu200-v055-blend-run-artifacts/native-merge.json),
[execution timing](hu200-v055-blend-run-artifacts/native-merge-execution.json).

### October 16, 14:03 Madrid: complete inference exports, audit running

The frozen exporter completes in 1,724.08 seconds. The average inference
export is **6,282,968,150 bytes**; the paired current export required by the
existing auditor is 2,960,585,129 bytes. Kernel exporter RSS peak is 4,702,208
bytes. The unchanged independent every-row checkpoint/average/current audit
is now running over the full 156,208,331-key model. No audited-model-ready
claim, pilot or final hand exists yet. Expected audited-model availability
is14:50–15:00 Madrid, conditional on audit success.
[Full export execution](hu200-v055-blend-run-artifacts/full-export-execution.json).

### October 16, 14:53 Madrid: audited blend ready

The full unchanged auditor verifies all **156,208,331 nodes** in2,952.32 s
(49 min12.32 s), including exact accumulator normalization, current regrets,
key uniqueness and complete current/average coverage. It verifies124,272,760
positive-mass rows and31,935,571 zero-mass uniform rows. It does not reconstruct
historical training increments; regrets/visits are deliberately zero in this
synthetic blend, with no training or resume claim.

| File | Bytes | SHA256 |
|---|---:|---|
| average | 6,282,968,150 | `ce3ecc2cb8ebffa1fb3e0a7f11c79d0900b74008b55d85093e1915ff4dde6485` |
| checkpoint | 6,042,193,732 | `59a6dd353fb4ef06d77a86783c7251521d8c9f8b92a785bb7e11a9ae724d4123` |
| current | 2,960,585,129 | `948b7dff67a52a1c2b475cc4da2d7153f00735448d5efc5360611d4a967ed860` |

The complete construction/export/audit controller returns0 after6,420.16 s.
Its1,278 five-second samples have a69,206,016-byte family RSS peak and no
resource stop; all originals are preserved. The inference model and checkpoint
sizes were reported to the owner immediately on readiness. Blend caching is
now running, before unchanged disjoint-root pilots and any final hand.
[Model manifest](hu200-v055-blend-run-artifacts/blend-model-manifest.json),
[every-row audit](hu200-v055-blend-run-artifacts/blend-every-row-audit.json),
[construction resources](hu200-v055-blend-run-artifacts/blend-construction-resource-receipt.json).

Cache retirement additionally checks current GitHub PR status retrieved by M1
after each completed comparison group, and retains the cache until that receipt
is atomically supplied. This is an automatic metadata handoff, requiring no new
owner confirmation. Exact source pins, seven-file inventory, no handles/links
and unchanged size/mtime are checked before retiring only the inactive derived
seed cache. No model, evaluator, count, root, weight or rule changes.
