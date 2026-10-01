# C/D resource admission and proposed paid budget

**Superseded paid scope:** owner approved [C-only pivot](../dr2x2-c-only-pivot.md) October1. Active rentals are three C jobs only; D deferred. Retained six-job tables/plans below describe the earlier proposal, not current launch permission.

October1,2026. **D prefixes and recovery checks complete. The owner approved the additional
$16 all-in cap ($8–11 expected) on October1. Production controller and Linux
parity are being prepared; no C/D rental has been created at this snapshot.**

## Frozen integration and measurements

Runtime `5f8a3941c0bf009c00695a5a0fbb4ef9a0c0c1e8` combines the exact #143
card descriptor (SHA `190d530ce66d65a031334d95600ce0f005d81324a7353170dcebf1d64a6ccd92`)
with the unchanged compressed-history schema. D preflop keys equal B's full-v2
keys. A/B/C checkpoints and current exports match their retained implementations
byte for byte after16fixed iterations;26focused correctness/isolation/recovery
checks passed. No observation, legal menu, game, seeds or trainer math changed.

Three D workers completed250k/500k/1M/2M from-zero prefixes, serially on M1,
Python3.11.14/engine5db20e3, with no poker-strength outcomes. The supervisor and
all children closed after9.02minutes. All72sealed file hashes were independently
rechecked, along with source/plan, contiguous iterations, completed work, visits,
entries and checkpoint receipts. Each final checkpoint reloaded/exported exact
bytes, and two fresh processes produced the same next work/state/export bytes.

| D seed | Completed nodes | Entries | Step nodes/sec | Peak including export | Final checkpoint save | Export |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3001 | 2,000,127 | 326,747 | 15,098 | 549.47MiB | 3.39s | 2.75s |
| 3002 | 2,000,130 | 324,563 | 15,359 | 553.80MiB | 3.41s | 2.67s |
| 3003 | 2,000,250 | 324,792 | 15,602 | 556.52MiB | 3.35s | 2.63s |

All four prefixes, street histograms and hashes are in the
[resource evidence](dr2x2-history-artifacts/d-resource-20261001/resource-summary.json)
and [CSV](dr2x2-history-artifacts/d-resource-20261001/resources.csv).
D's2M table is only7.1–7.7% smaller than #143's retained351,592-entry seed1 B
prefix. That cross-run comparison is descriptive, with different seeds/visited
paths; it does not identify a causal mechanism or a mature reduction.
The approved C10M density result remains passed, its original2M failure retained.

## RAM choice, separately measured for each cell

C's three10M tables contain276–278kentries and peak460–469MiB. Scaling the
largest complete-prefix peak tenfold and adding50% headroom gives about6.86GiB,
so16GB is a reasonable conservative starting shape.

D's last1M→2M increments project15.34–15.45M entries at100M if absolute growth
continues; local power fits give11.90–12.09M. Neither is a mature-table guarantee.
Scaling D's measured complete-prefix peak by the linear entry projection and
adding50% gives38.06–38.49GiB. This includes corpus/classification/export
measurement overhead and is deliberately conservative. It exceeds32GB;
**choose64GB for D from this measurement**, rather than inheriting B's shape
without checking. Actual B100M peaks of about11.1GiB show the projection may
overestimate mature usage; D's different paths still need operational guards.

Use one worker per pod. The8vCPU D shape buys RAM; no eight-core speedup is
assumed. Record actual CPU model/topology, affinity, cgroup CPU/RAM, host and
throughput. vCPUs are not independent physical-core counts.

## Live quote and requested incremental ceiling

Authenticated read-only provider quotes around18:41–18:45UTC report CPU5 memory
at$0.065/vCPU/hour and HIGH availability in EU-RO-1/EUR-IS-1. This is a live flavor
availability observation, not a capacity reservation. Requote/admit actual shape
and total price at creation. The [retained provider response](dr2x2-history-artifacts/d-resource-20261001/live-cpu-availability.json)
and [D-shape request](dr2x2-history-artifacts/d-resource-20261001/live-d-shape-availability.json)
contain no credentials/account balances. The [provider catalog API](https://www.runpod.io/blog/runpods-rest-api-v2-is-here-one-api-for-your-entire-gpu-stack)
is used for live resource admission.

| Cell | Jobs | CPU5 memory shape | Compute/hour each | Admitted compute+disk ceiling/hour |
| --- | ---: | --- | ---: | ---: |
| C | 3 | 2vCPU /16GB | $0.13 | $0.18 |
| D | 3 | 8vCPU /64GB | $0.52 | $0.57 |

30GB ephemeral disk per pod; no persistent/network volume proposed. The$0.05/h
storage component is a conservative allowance, not a separately verified posted
disk price. Reject a pod whose actual total rate exceeds its declared ceiling.

Prefix throughput projects1.8–1.9pure training hours/job. Planning at10k
nodes/sec instead gives2.78h, plus45minutes for setup, parity, recovery,
serialization, hashes and retrieval: about3.5h. Four-hour allocations are
cost scenarios, not scientific wall deadlines. Growing tables/host variability
may extend elapsed time; the cost/resource watchdog, not an arbitrary node
reduction, controls safety.

- Expected six-job training/recovery cost: **$6–8**, including disk allowance.
- Six full4hour allocations at admitted rates: `3 ×4 ×($0.18+$0.57) = $9.00`.
- Training/recovery subcap: **$12**, leaving$3 above that conservative allocation
  for failures, replacement, slow transfers and safe retrieval.
- Reserve **$4 for common four-cell analysis** if paid RAM is needed. Six hours
  on one D-sized host at$0.57/h gives$3.42. This is a planning reserve, not a
  frozen evaluation count or a claim of measured D evaluation timing. Use M1
  where resource admission permits; M4 remains exclusively #136.
- **Owner-approved additional all-in hard cap: $16. Expected total: $8–11.** Start
  safe checkpoint/retrieval/shutdown at a conservative$14upper estimate,
  retaining$2reserve. Include every failed/retired attempt and storage/analysis
  charge; never reset the ledger. Owner approval is recorded; source/controller and Linux parity admission still apply.

Already completed B/#143's$3.203879upper estimate is separate historical spend,
not charged again against this additional cap. Billing estimates are not settled
invoices. Account sufficiency is checked privately before any creation.

## Frozen work and launch admission

The [budget proposal](../../configs/blueprint/dr2x2-cd-budget-proposal.json) fixes
three C and three D seeds2026093001/02/03 at100M TOTAL completed nodes each,
from zero. Prefix checkpoints are measurement artifacts, not main-training
parents. First wave C1/C2/D1/D2, then C3/D3 after verified retrieval/termination;
at most four owned pods, one worker each, all names/volumes/artifact roots
`dr2x2-`. Record exact ownership before creation/after teardown; New Guy's pods
and ledger are untouched.

Before main work each cell must pass Linux direct/resume parity with hash-before-
load, exact meaningful state/RNG/work and deterministic checkpoint/export checks.
Distinguish #129's gzip OS byte from any payload/state divergence. Freeze the
production controller and engineering entry admission before launch; the4M
prefix guard is not carried into100M. Do not evict keys or reduce work to fit
RAM. Any infrastructure/engineering-limit revision is explicit, retains all
incidents, preserves scientific settings and requires recovery parity after a
material environment change.

Atomic recovery every10M with off-pod destination hashes before rotation;
retain permanent25M/50M/100M checkpoints and exports. M1 is the C/D off-pod
host, with81.7GiB free at admission and30GiB retained-artifact allowance.
Preserve source copies until destination hashes pass; terminate promptly after
closed logs/final retrieval. Do not delete unrelated research or unverified Drive
originals. No automatic model promotion, merge or300M extension.

## Common analysis admission remains prospective

Commit one fresh common A/B/C/D schedule after outcome-free timing and before
common strength outcomes: exact counts/deal roots, five simultaneous primary
contrasts, seed/position/block averaging, controls/tails/fallback, common river
range law/#142 roots and feasible #141 current/average readouts. Previous #143
and #136 outcomes are not pooled as factorial confirmation. All negative and
inconclusive results remain. The$4analysis reserve does not authorize silently
reducing a frozen count, changing opponents or omitting a cell; if safe resource/
cost admission cannot fit it, report that before paid analysis and obtain a
revised concrete budget. Training settings do not depend on observed strength.

Raw D measurements stay at
`/Users/dberweger/Local/dr2x2-d-resource-20261001/results/dr2x2-d-resource-m1-20261001`.
Consult the [72-file final inventory](dr2x2-history-artifacts/d-resource-20261001/final-manifest.json)
before copying/hashing exact checkpoints. Compact evidence is separately
transport-hash-verified and committed. #136 continues its90-minute M4 schedule.

## Approved launch forecast

Allow1–2hours to freeze and verify the controller and Linux setup. After launch,
expect5–8hours for two waves and verified retrieval, with4–6hours of pure
training across the waves at the measured/projection rates. This is an elapsed
time forecast, not a deadline. Common four-cell evaluation and sealing follow
training and have a separate runtime admission. No extra parallel worker is
implied by D’s8vCPU shape.
