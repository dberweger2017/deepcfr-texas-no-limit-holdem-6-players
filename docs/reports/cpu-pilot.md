# CPU pilot: results

**Four independent workers delivered 3.63× the throughput of serial execution,
with identical training results on the remote machine. A wider strategy network
is the strongest candidate from the frozen-replay sweep. Milestone 3 remains open.**

Date: September 15, 2026. [PR #49](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/49)
contains the runner, portable replay format, tests, and this report.
See the [declared protocol](../cpu-pilot.md),
[configuration](../../configs/solver/cpu-pilot-v1.json), and
[machine-readable results](cpu-pilot.json).

## Hardware and execution

| Measurement | Result |
| --- | --- |
| Local CPU | Apple M1; serial experiment jobs only |
| Rental | Runpod CPU pod, EUR-IS-1, AMD EPYC 4564P host |
| Allocation | Eight logical CPUs in the container affinity mask; 16,000,000,000-byte RAM limit |
| Disk | 10 GB temporary container disk; no persistent volume |
| Runtime | Python 3.11.15, Torch 2.5.1+cpu, NumPy 1.26.4, SciPy 1.17.1 |
| Remote source revision | `ff86350161827042d93b1977eaa3380c768e050d` |
| Local serial benchmark | 59.98 seconds |
| Remote serial benchmark | 58.14 seconds |
| Remote four-worker benchmark | 16.02 seconds |
| Parallel throughput gain on rental | 3.63× |
| Full 15-job fitting sweep | 64.74 seconds |
| Whole remote pilot, excluding provisioning | 139.30 seconds |
| Largest observed fitting-worker peak RSS | 342.7 MB |

The remote serial result is close to the laptop result. The useful improvement
is parallel throughput and freeing the laptop for development. The protocol uses
one Torch/BLAS thread per process; it does not distribute one training run across
cores. RAM is ample for these small-game experiments. This does not establish
memory requirements for full Hold'em training.

All four serial/parallel pairs matched their non-timing training reports,
including exported policy hashes. Every phase completed without a worker failure,
timeout, replay mutation, or dropped seed. Cross-platform bitwise equivalence was
not required or claimed. This is one preliminary serial-then-parallel measurement,
not a statistically controlled comparison of CPU vendors.

The original checkpoint loader verified all three exports on the Mac. On Linux,
the portable loader accepted the same solver-source fingerprint and public
catalog while recording the different runtime. The remote checkout's dirty flag
is retained: its only untracked file was the uploaded `cpu-pilot-replays.tar.gz`.
Tracked source files were unchanged. The full dependency listing and cgroup
allocation are included in the JSON report.

## All fitting outcomes

Exact exploitability in Leduc, in ante units per hand; lower is better. Each cell
uses the final fit specified before execution. No intermediate checkpoint was
selected after looking at results.

| Variant | Seed 11 | Seed 29 | Seed 47 | Worst seed |
| --- | ---: | ---: | ---: | ---: |
| Baseline: width 64, 6k steps | 0.168006 | 0.344337 | 0.212707 | 0.344337 |
| Longer fit: width 64, 24k steps | 0.167523 | 0.132373 | 0.161669 | 0.167523 |
| Wider network: width 128, 24k steps | **0.147863** | **0.118400** | **0.149616** | **0.149616** |
| Larger batch: 1,024 rows, 6k steps | 0.174197 | 0.127069 | 0.156285 | 0.174197 |
| Lower rate: 0.0003, 24k steps | 0.163393 | 0.124007 | 0.149738 | 0.163393 |

The replay-weighted excess mean squared error measures how well the network
matches the conditional mean targets in the retained replay:

| Variant | Seed 11 | Seed 29 | Seed 47 |
| --- | ---: | ---: | ---: |
| Baseline | 0.003766 | 0.006472 | 0.002931 |
| Longer fit | 0.001486 | 0.002229 | 0.001772 |
| Wider network | **0.000793** | **0.001161** | **0.001139** |
| Larger batch | 0.001856 | 0.002060 | 0.001197 |
| Lower rate | 0.001987 | 0.001985 | 0.001934 |

The larger batch reduces seed 11's fitting error but worsens its exploitability.
This independently repeats the warning from the previous diagnosis: lower loss
alone is not a sufficient selection rule. Under the declared no-seed-regression
rule, longer fitting, the wider network, and the lower learning rate qualify for
the shortlist; larger batches do not. The wider network has the best worst-seed
exploitability and is the preferred next candidate.

Coverage stays at 287/288, 287/288, and 285/288 information sets for seeds 11, 29,
and 47. Refitting cannot add missing observations. Per-information-set counts,
weighted targets, predictions, and errors are retained for every job; absent
targets remain explicitly unknown. Their presence makes a later coverage
investigation possible, but this pilot does not establish that missing coverage
is the remaining cause of error.

## What this establishes, and the next change

The frozen replay can support a neural strategy below 0.15 exploitability on all
three existing seeds. The weakest result, 0.149616, has only a 0.000384 margin.
These are already-observed exploratory seeds, and the sweep did not collect new
self-play data. It therefore does **not** pass the end-to-end convergence gate or
establish professional Hold'em strength.

The next PR should make strategy-network capacity independently configurable
from advantage-network capacity, preserve the existing advantage training, and
test the width-128/24k strategy fit in a newly declared multi-seed campaign. Keep
the original exploitability and value-error limits, retain failures, and require
fresh seeds before closing milestone 3. Separating the two capacities matters:
changing the current shared width would alter advantage learning as well as the
fitting change measured here.

## Cost and cleanup

The owner authorized $10 for CPU investigation, separate from a future GPU budget.
The quoted combined rate was $0.281/hour. The pod was provisioned around 14:30 UTC,
stopped after results were retrieved, and termination was verified before 14:42 UTC.
The console showed $0.00/hour after stopping and no retained billable storage.

The displayed account balance moved from $10.87 to $10.84. Those balances are
rounded to cents, so the $0.03 displayed change is not an exact itemized invoice;
the rental cost only a few cents. The remaining CPU budget is available for
subsequent declared experiments. The four-hour pilot limit was a ceiling, not a
requirement to leave an idle machine running.

The full remote archive was copied locally and its SHA-256 matched on both ends
before stopping the pod:

```text
9ef9a46e3828b8d14bab4220459e04f460c07b683ea9ab65077c59087e965892
```

Raw artifacts remain under `results/cpu-pilot-local-v1`,
`results/cpu-pilot-remote-v1`, `results/cpu-pilot-replays`, and
`results/cpu-pilot-remote-v1.tar.gz`. They are excluded from Git. The compact JSON
contains the run manifests, input/report hashes, allocation details, all fitting
metrics, and hashes for the detailed information-set reports. No pod credentials
or private keys are included.

## Validation

- 266 tests passed locally, including fresh-process serial/parallel equality,
  corrupt and illegal replay rejection, deterministic refits, and timeout cleanup.
- GitHub's Linux test job passed, including the existing neural reproduction and
  fresh-process checkpoint recovery checks.
- The declared local benchmark completed, all original replay exports loaded,
  and the full remote pilot completed with exact serial/parallel agreement.
- Source diff and lint checks passed. No solver, game-rule, or policy-input code
  changed, and no trained model was promoted.
