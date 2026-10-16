# Frozen HU200 three-seed blend run

Owner-authorized continuation of merged #297 on October 16, 2026. Its disk
admission stop was plumbing: no full blend, cache, pilot or final hand existed.
This campaign follows [the frozen protocol](../hu200-v055-blend-protocol.md)
and preserves [the original report](hu200-v055-blend.md).

Status: input restoration in progress; construction awaits disk admission. This PR remains
draft until the planned research actually runs or a research failure is recorded.
Plumbing interruptions remain pending work, with their evidence preserved.

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

Pending evidence: inventory/restoration hashes, current admission, union coverage,
root masses, resource peaks, three direct comparisons and 95% intervals, the
seven-opponent panel for all four models, independent verification of every final
hand, failure history, and accepted Research-Cloud archives/retrieval records.

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
