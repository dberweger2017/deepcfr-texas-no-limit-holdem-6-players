# CPU pilot: parallel experiments and strategy fitting

## Decision and budget

The owner authorized **up to $10 total on Runpod on September 15, 2026**, including
storage. The initial choice is one 5 GHz compute-optimized CPU pod with 8 vCPUs,
16 GB RAM, and 10 GB container disk. The displayed compute rate is $0.28/hour;
verify the final deployment quote. Allow at most four hours for the experiment
runner and stop earlier when its fixed work is complete. Setup and retrieval
also consume rental time; stop the rental within five hours of provisioning.
At the quoted rate, five hours of compute is $1.40, plus disk charges. Do not
spend the remaining budget automatically or enable automatic account top-ups.

The runner starts and stops worker processes. **It does not provision a pod or
stop Runpod billing.** The operator must copy results back, verify their hashes,
and terminate the pod after completion, failure, or the rental deadline. Temporary
container disk is erased on termination; the original replay exports stay local.
If a network volume is added later, its retained storage needs a separate cleanup
decision. No account credentials belong in the repository or run bundles.

## Questions this pilot answers

1. How long does the same small Leduc training workload take locally and remotely?
2. Do four independent processes improve throughput while preserving each job's
   exact results on the same machine and runtime?
3. Which fitting changes deserve a new convergence campaign?

This is preparation for milestone 3. It does not complete milestone 5, qualify
a Hold'em model, or change the failed Leduc result from PR #48.

## Fixed protocol

[The versioned plan](../configs/solver/cpu-pilot-v1.json) is recorded before remote
execution. Four seeds (101, 103, 107, 109) each run eight Leduc iterations with
256 traversals per player, 1,000 advantage fitting steps per update, and 6,000
final strategy fitting steps. Each process owns its random streams and uses one
Torch/BLAS thread. This workload exercises collection, fitting, exact evaluation,
and checkpoint/export I/O together; it is not a component-level profiler.

Run the same four jobs serially, then with four workers. Require identical
non-timing training reports, including policy hashes, before running the fitting
sweep. Use four fitting workers only if measured throughput improves by at least
20%; otherwise use one. Record wall time including startup, CPU time, peak RSS
per worker, CPU description, runtime versions, and source/configuration hashes.
Per-worker peak RSS is not simultaneous aggregate RAM use. This single ordered
comparison is a preliminary sizing check; cache warm-up and shared-host variation
can affect it. Repeat/order-balance before making a general performance claim.

### Frozen replay experiments

Use all three original Leduc seed-11/29/47 strategy reservoirs at iteration 120.
Export them on the original machine through the strict checkpoint loader. The
portable format contains only replay arrays and provenance, with no executable
pickle, optimizer, RNG recovery, or model weights. Its loader checks the data
hash, counts, types, legal probability targets, training source and public
information catalog. The runtime may differ; the original checkpoint loader's
cross-environment restriction remains intact.

| Variant | Hidden width | Steps | Batch | Learning rate |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 64 | 6,000 | 256 | 0.001 |
| Longer fit | 64 | 24,000 | 256 | 0.001 |
| Wider network | 128 | 24,000 | 256 | 0.001 |
| Larger batch | 64 | 6,000 | 1,024 | 0.001 |
| Lower rate | 64 | 24,000 | 256 | 0.0003 |

Compare longer fit and larger batch to baseline; compare wider network and lower
rate to longer fit. The larger-batch arm processes as many sampled rows as the
longer-fit arm, but uses fewer optimizer updates. These are diagnostic contrasts,
not equal-cost algorithm rankings. Initialization/minibatch seeds follow the
original strategy-fit rule for each source seed. Different batch sizes or network
shapes do not imply identical gradients or initial parameters.

For every variant and source seed, retain exact exploitability, value, fitting
error, sample noise, policy hash, replay hash, and per-information-set counts,
weighted targets, predictions, and squared errors. Unobserved targets are null,
not fabricated zero-probability strategies. The baseline is refitted on the
remote runtime; compare other variants to that baseline, not to a claimed
bitwise reproduction of the Mac's model.

**Selection rule:** shortlist a variant only if its worst-seed exploitability
improves and none of the three source seeds regresses against the remote baseline.
Use fitting error and information-set coverage to explain results, not to replace
playing-strength measurements. These already-observed replay seeds are exploratory
data. Even if all refits fall below 0.15, milestone 3 remains open until a newly
declared end-to-end campaign passes the original exploitability/value limits with
fresh training seeds. No policy is promoted by this pilot.

## Commands

Use Python 3.11. A minimal environment needs only the pinned pilot dependencies:

```bash
python3.11 -m venv .venv-pilot
.venv-pilot/bin/python -m pip install -r requirements-pilot.txt
```

On the original training machine, export each completed bundle. For example:

```bash
.venv/bin/python -m scripts.cpu_pilot export \
  --training results/neural-convergence-leduc-11/training \
  --out results/cpu-pilot-replays/11
```

Repeat for seeds 29 and 47. Keep the three numbered folders under one directory.
Copy that directory to the rental alongside a checkout of the reviewed pilot
commit. No original training snapshots need to leave the local machine.

Local timing baseline (serial jobs only):

```bash
.venv/bin/python -m scripts.cpu_pilot run \
  --plan configs/solver/cpu-pilot-v1.json \
  --out results/cpu-pilot-local-v1
```

Remote full pilot:

```bash
.venv-pilot/bin/python -m scripts.cpu_pilot run \
  --plan configs/solver/cpu-pilot-v1.json \
  --replays results/cpu-pilot-replays \
  --out results/cpu-pilot-remote-v1
```

The parent imposes a four-hour wall limit and an 840-second limit per subprocess,
including startup and evaluation. A worker failure or timeout stops the pilot
and reaps active process groups; partial logs and completed outputs remain. The
runner does not silently retry or extend limits. SIGINT also triggers cleanup.
Host loss or SIGKILL cannot guarantee parent cleanup; the operator must enforce
the rental deadline independently. A new attempt uses a fresh output directory
and retains the previous failure. Existing outputs cannot be overwritten.

## Results to retain

Copy the full result directory back before terminating compute. Preserve
`manifest.json`, `report.json`, per-phase `progress.json`, job specifications/logs,
training reports, and fitting `information_sets.json` files. Large checkpoints
remain outside Git. Commit a compact report with:

- Actual CPU, memory allocation, image, runtime, and source revision.
- Local/remote serial timing, remote parallel speedup and exact-result check.
- All 15 fitting outcomes, including failures and timeouts.
- A shortlist under the declared rule, or an explicit inconclusive result.
- Provision/termination times, observed charges, and storage left behind.

Provider references: [Runpod pricing](https://docs.runpod.io/pods/pricing) and
[CPU pod storage support](https://www.runpod.io/blog/enhanced-cpu-pods-docker-network).
