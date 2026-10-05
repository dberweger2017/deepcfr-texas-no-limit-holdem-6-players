# Superseded October 4: M4 revision 3

No pod was ever rented. The $5 budget is unused. This quote is retained as history; do not deploy from it. See [revision 3](hu20-board-pooling-protocol.md).

# RunPod quote revision 2 — rental deferred

The owner approved **up to $5**, then withdrew immediate reservation and asked
for code preparation followed by waiting for an efficient offer. **No pod was
deployed; no paid compute or storage exists for this task.** Do not rent until
an owner resume. The previous one-pod $4 quote and temporary three-pod proposal
are superseded by this single-pod plan.

| Item | Revised candidate / cost |
|---|---|
| CPU | **3-GHz General Purpose, 8 vCPU, 32 GB**, currently **Unavailable** |
| Compute | $0.32/hour, observed in the signed-in console October 3 |
| Container disk | 100 GB, $0.014/hour; ephemeral, no network volume |
| Combined | **$0.334/hour**; plan rate ceiling $0.34/hour |
| Workers | Four independent solver jobs, two Rayon threads each |
| Expected wall | Tentative **3–6 hours**, subject to native Linux pilots |
| Hard rental clock | **12 hours total**, including all setup, failed compute and idle time |
| Retrieval/shutdown reserve | Last **2 hours**; production stops before hour 10 |
| Expected subtotal | $1.002–$2.004 at 3–6 hours |
| Worst clock subtotal | $4.008 at the listed rate; $4.08 at the rate ceiling |
| All-in owner ceiling | **$5 maximum**, no credit purchase or replacement rental |
| Transfer | $0 provider fee; transfer time included in the clock |

[RunPod billing](https://docs.runpod.io/accounts-billing/billing) states no data
transfer fees. Preserve itemized provider rates/times and actual billing, and
stop before any mandatory charges would exceed the $5 ceiling. At the rate
ceiling, $0.92 remains for rounding/tax/fees. Availability is not reserved. The
unavailable card cannot be selected: the saved screenshot proves its listed
CPU price and availability, not a deployable checkout. Disk cost is the same
100-GB itemized quote observed earlier; recheck the full checkout on resume.

This is a resource recommendation, not a measured throughput optimum. General
Purpose offers four GB per vCPU: one 32-GB pod admits four 5-GiB solver workers
more naturally than three separately billed 16-GB pods, uses all eight cores,
and avoids three builds/transfers. Compute-Optimized 8-vCPU/16-GB is too small
for four such workers. No GPU benefits this native CPU solver. Clock labels
alone do not predict solver throughput; the Linux pilots decide admission.

Use `nice`, **4 GiB arena / 5 GiB owned RSS per worker / 21 GiB aggregate**;
aggregate must also be <=80% measured available cgroup/host memory. One worker
per root/lineage job, no shared solver state. If this shape does not admit four
workers, stop rather than quietly reducing threads/workers or changing the
science. Preserve >=20 GiB free disk; stop on >1-GiB swap growth, cgroup events,
nonfinite values, gate failure or the immutable clock.

The prospective revision freezes two 20-board halves; held-out v1/equity-50
loss drives D. Fit policies and codebooks on the opposite half only. In-sample
loss remains secondary. Phase 2 computes BR on fresh locked trees without CFR;
it uses phase 1's hash-linked equilibrium EV. Six frozen replay boards across
three lineages add **18** re-solves (15%), rather than **120** re-solves.

After qualification, forecast remaining main time as
`1.5 * (max_solve_seconds * 138 + max_lock_seconds * 138) / 4`.
This conservatively accounts for 120 primary solves, 18 replay solves, locked
evaluation and replay comparisons; actual support exclusions never justify
selecting a smaller corpus. Admit only if this fits the unreset remaining clock
minus reserve. Preflight includes Linux/macOS fixtures, three real-export
20,000-deal V4 checks, full convergence and fresh/solved-tree BR comparisons.

Retrieve all attempts/resources/configs/pooled policies and verify every hash,
then terminate the owned pod and its ephemeral storage. Keep AGPL code outside
the MIT checkout. No training, promotion, M4 use or merge.

Screenshot: `/Users/dberweger/Local/hu20-board-pooling-20261003/runpod-gp8-quote.jpg`.
The tracked quote records its hash; the private account header is not published.
