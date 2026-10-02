# Owner-authorized turn-check resumption

On 2026-10-02 the owner authorized continuing because the M4 is free again.
The original memory stop, owner-requested cleanup and verified M1 archive remain
retained. This note amends resource admission only; it does not change the
[frozen scientific protocol](hu20-exact-turn-check-protocol.md).

Restore the external pinned AGPL tool and six hash-verified exports outside the
MIT repository. Restore the two atomic results from `main-01`, verify their
request/response/result hashes, and continue in `main-02`. No completed job is
re-solved or selected according to its value. All 48 roots, six exports, sampling
weights, schedule seed, convergence thresholds, hypothesis rules and reporting
estimands remain unchanged. The incomplete report is a retained stop record,
not a completed scientific conclusion.

Fresh machine admission records measured free/reclaimable memory, processes,
pressure and swap. Keep 4 GiB owned RSS, one external solver, two threads and
nice priority 10. The resumed swap baseline is the measured admission level;
growth above that baseline by 1 GiB stops owned work. Other jobs remain untouched.
Restart only the port-6006 TensorBoard needed for the owner-requested monitoring;
the unrelated port-16008 server stays stopped. Use a new `main-02` sidecar name.

The original stop consumed 80.6997572502587 main seconds. Deduct that from the
same 86,400-second total allowance. The owner-requested interruption and
workspace restoration do not consume solver-campaign hours. A restored admission
sets its clock to the current launch time minus those already consumed seconds;
subsequent uninterrupted wall time remains charged as before. Preserve the
old admission and failure hashes in the new admission and resume inventory.
No new 24-hour allowance is added to completed compute. The remaining allowance
is 86,319.30024274974 seconds; reserve bounded worker time before starting jobs.

The Python driver now keeps per-hand fold/projection evidence on disk and only
small counter records in memory, releasing the last verbose response before the
next worker. The stopped-run report also counts every unstarted frozen root.
These reporting/resource fixes do not change solver requests or metric formulas.
The new source inventory distinguishes them from the original admission.
The external solver binary and qualified strategy/BR mappings are unchanged.

The config `configs/diagnostics/hu20-exact-turn-check-resume.json` records the
new baseline, source inventory, this amendment hash and unchanged corpus/policy
hashes. Commit and push admission before remaining main values. Preserve every
old attempt, post remaining PR milestones, finish the report, and leave both
PRs as drafts. No training, promotion, rental or automatic merge.
