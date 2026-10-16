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
