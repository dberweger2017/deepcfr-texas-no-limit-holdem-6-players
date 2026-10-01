# HU20 three-lineage 100M→500M campaign

Status: **preliminary light checkpoint curves published; broader tests and fresh
final confirmation pending**. The October 1, 06:43 UTC snapshot contains 74
closed light tasks / 265,216 hands. All three lineages have light results through
480M; two have individual 500M results. The missing third 500M task stays
pending, so this snapshot has no three-lineage 500M aggregate.

The owner authorized three retained B100M continuations on October 1, 2026,
with a $10 target and a firm $15 all-in ceiling. #132/#133 are merged. The
[frozen protocol](../hu20-500m-campaign-protocol.md) and JSON plan retain the
original seeds, iteration/RNG lineages, regrets/averages/visits, K1, abstraction,
uncapped menu and current extraction. B100M is unchanged; no promotion.

## Analysis order and next publications

The owner authorized an [endpoint-first analysis/publication plan](../hu20-500m-analysis-publication-plan.md)
on October 1. Finish the current child, then complete all three broad100M
baselines, all three broad500M endpoints, the150/200/300/400M groups, and fresh
confirmation last. The scientific executor/plan stay frozen; only queue priority
changes. All retained archives are hash-verified on M4 and all rentals have
terminated. All75 light tasks are complete; the snapshot below deliberately
retains its original74-task capture.

Initial Madrid-time ETA: full broader baseline **10:30–11:30 October1**;
broader500M versus own100M **13:00–14:30 October1**; fresh confirmation/final
verification **00:00–05:00 October2**. These are timing forecasts, not deadlines.
Publish each newly completed group with all opponents, seed/position intervals,
paired changes and tails; pending groups cannot become smaller aggregates.

## Preliminary playing profiles

The [complete checkpoint tables](hu20-500m-campaign-artifacts/preliminary-20261001/curves.md)
retain every available checkpoint against random, passive, four scripted styles
and selective-stackoff-v1. The [per-seed/position CSV](hu20-500m-campaign-artifacts/preliminary-20261001/per-seed-role.csv)
and [machine-readable summary](hu20-500m-campaign-artifacts/preliminary-20261001/summary.json)
retain individual results, paired changes, tails, fallback by street and
whole-hand return partitions. No available light opponent/checkpoint is omitted.

Here is the latest complete three-lineage endpoint, compared with each lineage's
own B100M on the same light schedule. All units are **BB/100**, and intervals
are exploratory, unadjusted 95% Student-t intervals over 256 paired deal blocks.
The three seed returns/contrasts are averaged within each block, together with
both positions; seeds and positions do not multiply the independent sample count.

| Fixed light opponent | B100M | B480M | Paired change [95% interval] |
| --- | ---: | ---: | --- |
| Random | +108.69 | +126.37 | +17.68 [−1.47, +36.83] |
| Selective stackoff | +48.73 | +62.30 | +13.57 [+2.28, +24.87] |
| Passive | +144.08 | +166.86 | +22.79 [−15.47, +61.04] |
| Loose passive | +10.03 | +11.78 | +1.76 [−18.97, +22.48] |
| Loose aggressive | −8.63 | +16.76 | +25.39 [−3.74, +54.52] |
| Tight passive | +46.19 | +48.63 | +2.44 [−4.92, +9.80] |
| Tight aggressive | +46.09 | +47.43 | +1.33 [−10.34, +13.01] |

These profiles are mixed and fluctuate across checkpoints. Selective-stackoff's
480M paired interval excludes zero on this exploratory panel; the other six
endpoint intervals include zero. With many checkpoints/opponents and no
multiple-comparison correction here, this is preliminary evidence, not final
confirmation or proof of a monotonic learning curve/plateau. The selective
opponent was designed after Luna and is a regression/stress opponent, not an
independent Luna replication.

The selective-stackoff seed point estimates at 100M→480M are
59.18→62.70, 43.85→53.13 and 43.16→71.09 BB/100. This does not establish
improvement separately for every seed or position. Full 20BB losses total
**3/1,536 hands at both endpoints**: by seed, 0/1/2 becomes 0/2/1. Thus the
aggregate profit increase has not demonstrated a reduction in full-stack loss
frequency. The small panel recorded no large/all-in calls at either endpoint;
large target raises were 12→19 and jams 9→13. Rare-tail counts cannot establish
that catastrophic errors have disappeared. Fallback was 3 target decisions at
100M and zero at 480M against this opponent; full denominators are in the tables.

### Preliminary validation and scope

The reporting pass ran on M4 under the shared one-worker lock. Three focused
statistical tests passed: seeds/positions do not multiply block count, a missing
lineage cannot enter an aggregate, and mismatched block sets fail. All closed
raw records were checked for exact frozen coordinates, model identity,
native-replay evidence, zero-sum chip accounting, recomputed tails, complete
paired blocks and independent return sums. Raw/result hashes and model hashes
are retained in the summary; the [publication manifest](hu20-500m-campaign-artifacts/preliminary-20261001/manifest.json)
was verified after compact transfer. This pass did not rerun gameplay or alter
the frozen executable, seeds, checkpoint cadence, opponents, schedules or gates.

Broader bounded-LBR/native-pressure tests and disjoint fresh final confirmation
remain pending. No inference about their results follows from these light
profiles. B100M remains the unchanged preview model.

## Validated source and startup evidence

Executable source: `17b4c9a08ed0765d0fb8f05240c0409b21e43977`.
Canonical plan SHA-256:
`74f0c18024a4a409d223276d4781c2464b27fc8d5d32a51ccb22a224332f2219`.
Both full GitHub CI shards, aggregate test gate and GitGuardian passed before
rental. **24 focused M4 tests passed**, including actual driver recovery/work
chain equivalence, partial-state preservation, failed-export acknowledgement
rejection, backup rotation and native replay of the evaluator's hand path.

All three same-source M4 mature validation prefixes and fresh-process resumes
matched checkpoint/export bytes, complete work and next RNG streams. The
[compact preflight evidence](hu20-500m-campaign-artifacts/preflight.json) records
input/final hashes and every phase. All three Linux workers passed this
reference and their fresh-process resume checks before main training. The initial observed continuation rate is
17.0–18.7k completed nodes/sec per pod, with about 1.9 GB sampled aggregate
startup RSS and zero swap growth. These are early resource observations, not
a forecast that accounts for later table growth. The only allowed cross-platform
export difference is the known gzip OS byte; normalized compressed bytes and complete policy payload
must both agree. A meaningful divergence blocks that lineage.

The earlier two outcome-free validation attempts are retained on M4. They
validate implementation and reporting additions; their repeated prefixes are
neither main training work nor independent diversity. No confirmation playing
outcomes were inspected to select source, settings or counts.

## Schedule and spending

Initial rentals are three CPU5 memory 2-vCPU/16-GB pods, one per lineage, with
live compute quote $0.13/hour each. Owned pod IDs are `mnze3gf23y1ded`,
`qvvpd424l32rjo` and `1di6o94izsf3gj`, for seeds 3001/3002/3003 respectively.
Actual topology/cgroup limits are retained in each worker record;
vCPU count is not a claim about independent physical cores. Mature pilot
throughput suggests ~6.25 hours of pure continuation; growing tables and
save/export/verification overhead can extend this. There is no arbitrary
wall-time cutoff. Actual throughput and costs will replace that estimate.

The independent watchdog conservatively includes a $0.05/hour/pod storage
allowance and all retired attempts. It requests safe stop at $13 upper estimate,
reserving $2 and at most ten minutes for checkpoint retrieval/shutdown. Account
sufficiency was privately verified; credentials and balances are not published.
Final cost will distinguish posted ledger charges from conservative estimates.

At the 06:44 UTC operational read, seeds 3002/3003 had reached
500,000,321 / 500,000,576 completed lifetime nodes, passed final reload and
verified off-pod archive retrieval, and their rentals were terminated. Seed
3001 was still running with a hash-verified 490,000,257-node recovery. The
watchdog's conservative all-in upper estimate was **$3.65**, not a settled
invoice. This preliminary publication does not wait for or select on subsequent
playing outcomes; the original controller continues the fixed campaign.

Each 10M recovery is destination-hash-verified before rotation; each 20M has a
current export and queued light evaluation; each 50M is a permanent full save.
The fixed 907,776 evaluation hands include random/styles/passive/selective
stackoff, broader 150/200/300/400/500M panels including unchanged bounded LBR,
and separate fresh 500M-versus-own100M confirmation. All curves before final
confirmation are exploratory. Report full-stack tails, large calls/raises,
jam/response opportunities, fallback, roles and individual seeds, retaining
absolute losses and every failed/unattempted block.

## Operations and retrieval

M4 source: `/Users/dberweger/Local/hu20-500m-campaign-20261001`.
Main artifacts: `results/hu20-three-lineage-500m-20261001` below that source.
External controller log:
`/Users/dberweger/Local/hu20-500m-controller-20261001.log`.

The durable controller/provider watchdog handle backups and guard resources;
30-minute agent checks coalesce milestones/incidents and remain quiet while
healthy. M4 performs evaluations serially; training and backups take priority.
The M1 emergency route was authenticated and available capacity verified. Its
root is `/Users/dberweger/Local/hu20-500m-emergency-backups-20261001`; it performs
transfers/transport hashing only. Earlier research and unverified Drive copies
remain retained.

Retrieve a particular artifact only after consulting its immutable receipt:

```sh
scp -o HostName=100.122.216.94 -o BatchMode=yes \
  m4:/Users/dberweger/Local/hu20-500m-campaign-20261001/results/hu20-three-lineage-500m-20261001/artifacts/SEED/attempt-1/FILE \
  ./FILE
shasum -a 256 ./FILE
```

Replace `SEED`/`FILE` using the receipt's exact name/hash. Emergency M1 receipts
record the alternate absolute destination. Final sealed manifests, learning
curves for the broader/confirmation schedules, resource measurements, billing
and incident table remain pending until
execution and independent verification finish. This PR stays draft.

The preliminary raw hands remain on M4 under
`results/hu20-three-lineage-500m-20261001/evaluation/light-SEED-NODES/hands.jsonl.gz`.
Use an exact task/model entry and `hands_sha256` from the published summary:

```sh
scp -o HostName=100.122.216.94 -o BatchMode=yes \
  m4:/Users/dberweger/Local/hu20-500m-campaign-20261001/results/hu20-three-lineage-500m-20261001/evaluation/light-SEED-NODES/hands.jsonl.gz \
  ./light-SEED-NODES-hands.jsonl.gz
shasum -a 256 ./light-SEED-NODES-hands.jsonl.gz
```

Reproduce the reporting pass from this PR's reporting source on M4 (use a fresh
output directory; the original attempt remains retained):

```sh
python -m scripts.report_hu20_500m_preliminary \
  --plan configs/blueprint/hu20-500m-campaign.json \
  --root results/hu20-three-lineage-500m-20261001 \
  --out results/hu20-three-lineage-500m-20261001/preliminary-NEW
```

The reporting script was run from `/tmp/dr-research-pr136-preliminary.py`,
importing the unchanged frozen clone through `PYTHONPATH`, rather than updating
the running campaign source. A later snapshot may include tasks that closed
after this publication's fixed input list.
