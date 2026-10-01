# HU20 500M analysis order and progressive publication

Owner authorized this execution-priority change on October 1, 2026, after
training and verified retrieval completed. It supersedes the original queue's
seed-first execution order. The [scientific protocol](hu20-500m-campaign-protocol.md)
and canonical JSON digest remain unchanged: this is an operational order and
publication plan, not a revision of counts, models, randomness or gates.

## Order

One heavy scientific child runs at a time on M4. Finish the currently running
`broad-2026093001-100000000` child without pausing, restarting or altering it.
Adopt it into the replacement supervisor; replace only the queue owner. Once
that child has closed, validate queue priority before starting another child.

1. **100M broad baseline:** seeds 3001, 3002, 3003.
2. **500M broad endpoint:** seeds 3001, 3002, 3003.
3. **Intermediate broad curves:** 150M, 200M, 300M, 400M, all three seeds at
   each checkpoint in that order.
4. **Fresh final confirmation:** each seed's own 100M and 500M pair, seeds
   3001, 3002, 3003, after all broad exploratory work.

Within every model task the original panel order is unchanged: original-cap2
bounded LBR, native pressure, selective stackoff, pot pressure, passive and four
scripted styles. All 75 light tasks are already complete. Every original task
remains scheduled exactly once: 18 broad plus six held-out model tasks, retaining
the total 907,776 campaign hands. Failed/partial tasks remain explicit and are
never silently repeated. No checkpoint or lineage is dropped because of scores.

The replacement supervisor invokes the **original frozen**
`scripts.evaluate_hu20_500m task` executor at
`17b4c9a08ed0765d0fb8f05240c0409b21e43977`. It runs from a separate `/tmp` copy;
the active clone is not pulled or edited. The live child's identity and original
task document are retained in `endpoint-order-handoff.json`. Adoption does not
recover its OS exit code; the record explicitly distinguishes that unavailable
code from a closed complete result and observed process closure.

The same lock and RSS/swap/disk guards remain in effect. Reporting may acquire
the shared lock between scientific children; it cannot run alongside a heavy
evaluation. Four focused priority tests must pass on M4 before the next task:
complete endpoint groups first, unchanged specifications/counts without replay
of completed IDs, duplicate-task rejection and M4-retrieval enforcement.

## Publication cadence

On each 30-minute check, coalesce newly closed task groups. Publish verified
compact results after each complete three-seed checkpoint group; intermediate
individual results may be labelled incomplete, but never substitute for a
three-seed aggregate. The first publication is the complete broader B100M
profile; the next is B500M plus its paired changes versus own B100M.

Every group includes all declared opponents, absolute returns and paired
changes where the baseline exists, per-seed and position intervals, fallback,
large-call/raise/jam exposure and full-stack tails. Check frozen coordinates,
chip/replay evidence, input hashes and independent block arithmetic before
publication. Retain all negative and inconclusive results and every pending
task/block. Fresh confirmation stays separate and uses its original 97.5%
primary LBR and native-pressure safeguard gates; broad curves remain exploratory.

## Initial ETA, October 1 at approximately 09:15 Madrid

These are estimates, not deadlines. The first LBR panel measured about 1.07
seconds per paired block. Each broader model needs 2,048 LBR blocks plus the
other panels; later policy behavior can change the rate. Publication includes
the next coalesced check and bounded arithmetic/hash verification.

| Available data | Expected Madrid time |
| --- | --- |
| All three broader 100M baseline profiles | October 1, 10:30–11:30 |
| All three broader 500M profiles and paired 100M→500M comparison | October 1, 13:00–14:30 |
| Each subsequent complete intermediate checkpoint | Approximately every two hours after the endpoint group |
| Fresh confirmation and final verified report | October 2, 00:00–05:00 |

The endpoint publication is preliminary, not the fresh final confirmation.
Update forecasts using execution timing, not poker returns. All paid rentals
are terminated; analysis runs on M4. B100M remains unchanged, and no promotion,
merge or follow-on experiment is implied.
