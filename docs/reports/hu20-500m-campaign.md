# HU20 three-lineage 100M→500M campaign

Status: **launch in progress; no new playing-strength result yet**.

The owner authorized three retained B100M continuations on October 1, 2026,
with a $10 target and a firm $15 all-in ceiling. #132/#133 are merged. The
[frozen protocol](../hu20-500m-campaign-protocol.md) and JSON plan retain the
original seeds, iteration/RNG lineages, regrets/averages/visits, K1, abstraction,
uncapped menu and current extraction. B100M is unchanged; no promotion.

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
input/final hashes and every phase. Each Linux worker must match this reference
before main training. The only allowed cross-platform export difference is the
known gzip OS byte; normalized compressed bytes and complete policy payload
must both agree. A meaningful divergence blocks that lineage.

The earlier two outcome-free validation attempts are retained on M4. They
validate implementation and reporting additions; their repeated prefixes are
neither main training work nor independent diversity. No confirmation playing
outcomes were inspected to select source, settings or counts.

## Schedule and spending

Initial rentals are three CPU5 memory 2-vCPU/16-GB pods, one per lineage, with
live compute quote $0.13/hour each. Capture actual model/topology/cgroup limits;
vCPU count is not a claim about independent physical cores. Mature pilot
throughput suggests ~6.25 hours of pure continuation; growing tables and
save/export/verification overhead can extend this. There is no arbitrary
wall-time cutoff. Actual throughput and costs will replace that estimate.

The independent watchdog conservatively includes a $0.05/hour/pod storage
allowance and all retired attempts. It requests safe stop at $13 upper estimate,
reserving $2 and at most ten minutes for checkpoint retrieval/shutdown. Account
sufficiency was privately verified; credentials and balances are not published.
Final cost will distinguish posted ledger charges from conservative estimates.

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
curves, resource measurements, billing and incident table remain pending until
execution and independent verification finish. This PR stays draft.
