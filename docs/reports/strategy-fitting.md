# Strategy fitting: results

**All 144 fits completed. Decaying the learning rate passes the absolute limits
in all 36 fits, but no recipe qualifies under the complete predeclared screen.**
The decayed minibatch recipe exceeds the paired-regression limit on one fit.
The exact-gradient recipes are diagnostic controls and cannot be promoted.
Milestone 3 remains open; confirmation seeds remain unused.

[Protocol](../strategy-fitting-study.md) · [Frozen plan](../../configs/solver/strategy-fitting-v1.json) · [Summary and final fits](strategy-fitting.json) · [All 432 checkpoint records](strategy-fitting-checkpoints.jsonl)

## What we tested

The same twelve final Leduc replays from the capacity study, each with 200,000
samples collected through iteration 480, were fitted with four recipes. Each
recipe used three paired fitting seeds, a width-128 two-hidden-layer network,
48,000 Adam updates, the original iteration-weighted objective, and float32 CPU
execution. The sampled recipes used identical batches of 256 rows within each
pair. All four recipes shared their initial weights within each replicate.

The changes were gradient calculation (sampled minibatches versus the full replay
objective) and learning rate (constant 0.001 versus cosine decay to 0.00001).
There was no new traversal collection, change of network capacity, target
reweighting, game-rule change, or fresh confirmation. Only the final checkpoint
was screened; earlier 12k and 24k results are preserved but cannot rescue a fit.

## Final comparison

Exploitability is measured by exact best responses in Leduc, in ante units per
hand; lower is better. Passing the absolute limits requires exploitability at
most 0.15 and value error at most 0.10. The sampled candidate also cannot worsen
any paired result by more than 0.005 against the fixed-rate control.

| Recipe | Absolute limits passed | Mean exploitability | Worst exploitability | Largest paired worsening | Eligible? |
| --- | ---: | ---: | ---: | ---: | --- |
| Minibatch, fixed rate | 31/36 | 0.126274 | 0.174001 | 0 | No: absolute failures |
| Minibatch, decayed rate | 36/36 | 0.115920 | 0.147944 | 0.015791 | No: paired regression |
| Exact gradient, fixed rate | 36/36 | 0.112816 | 0.142632 | 0.008155 | Diagnostic only |
| Exact gradient, decayed rate | 36/36 | 0.112618 | 0.142218 | 0.008498 | Diagnostic only |

Every value-error check passed, including the intermediate evaluations; the
largest error across all 432 checkpoints was 0.024051. The exact-gradient
recipes would also exceed the paired-regression limit if they were eligible.

The decayed minibatch recipe improves 31/36 paired fits, with mean change
−0.010354. Exact fixed-rate fitting improves 35/36 pairs and exact decayed-rate
fitting improves 34/36. These are paired observations on reused data, not 36
independent collection runs. We do not attach confidence intervals treating them
as independent training trajectories.

### The failed guardrail is a different case from the old failure

Seed 211 was the previous study's remaining failure. All three fixed-rate fitting
replicates fail there (0.151373, 0.174001, 0.156712). All three decayed-rate fits
pass (0.147708, 0.141046, 0.147944), as do all exact-gradient fits.

The regression failure is **seed 239, fitting replicate 0**:

| Recipe | Exploitability | Change against fixed minibatches |
| --- | ---: | ---: |
| Minibatch, fixed rate | 0.087157 | — |
| Minibatch, decayed rate | 0.102948 | +0.015791 |
| Exact gradient, fixed rate | 0.095312 | +0.008155 |
| Exact gradient, decayed rate | 0.095655 | +0.008498 |

All four remain below 0.15 on this case. Fitting the replay more faithfully does
not guarantee improvement over every approximate policy: approximation can also
happen to yield a less exploitable policy. This explains why low fitting loss is
not a sufficient promotion criterion; it does not excuse the failed guardrail.
The committed outcome remains `no_candidate`, with no retrospective relaxation.

## What the fitting diagnostics tell us

| Recipe | Mean final replay-weighted excess MSE |
| --- | ---: |
| Minibatch, fixed rate | 0.000682702 |
| Minibatch, decayed rate | 0.000070492 |
| Exact gradient, fixed rate | 0.000009929 |
| Exact gradient, decayed rate | 0.000001141 |

Decay reduces mean fitting error by roughly 90% relative to fixed minibatches.
Removing minibatch noise reduces it further. These results demonstrate that this
width-128 model can fit the observed replay much more closely with the same
48,000-update budget. Another width sweep is not the immediate question.

At each checkpoint, we separately replaced the predictions in one observed
sample-count band with that band's replay means and recomputed a best response.
For the fixed minibatch recipe, the average final exploitability changes were:

| Replaced band | Mean change | Range across 36 fits |
| --- | ---: | --- |
| 1–9 samples | −0.000008 | −0.000274 to +0.000092 |
| 10–99 samples | −0.000601 | −0.005653 to +0.003338 |
| 100+ samples | −0.013311 | −0.038918 to +0.004828 |

The largest average effect comes from the frequently represented situations.
This weighs against rare observed situations being the main explanation for
this particular fitting gap. It does not prove rare or missing situations are
unimportant. Missing targets were left unknown and never replaced. Band
interventions interact; their changes must not be added as causal contributions.

With decayed minibatches, replacing the 100+ band changes exploitability by
−0.003446 on average; with exact decayed gradients the average is −0.000382.
The full per-situation evidence retains both players, all 288 information sets,
and every scheduled checkpoint, including improvements and regressions.

## Every collection seed and fitting replicate

The table gives the range over all three final fitting replicates. The linked
JSON records each individual fit, and the JSONL records every checkpoint.

| Seed | Minibatch fixed | Minibatch decay | Exact fixed | Exact decay |
| --- | --- | --- | --- | --- |
| 211 | 0.151373–0.174001 | 0.141046–0.147944 | 0.141128–0.142632 | 0.141652–0.142218 |
| 223 | 0.118899–0.122663 | 0.101318–0.108801 | 0.097531–0.103002 | 0.098173–0.098357 |
| 227 | 0.113222–0.119411 | 0.113763–0.115903 | 0.108073–0.110592 | 0.109805–0.110868 |
| 229 | 0.116812–0.145916 | 0.105660–0.108146 | 0.104756–0.107815 | 0.105742–0.106081 |
| 233 | 0.111416–0.118729 | 0.111059–0.115284 | 0.110060–0.110593 | 0.110530–0.110644 |
| 239 | 0.087157–0.106133 | 0.095653–0.102948 | 0.095312–0.095995 | 0.095494–0.095655 |
| 241 | 0.109554–0.123427 | 0.107721–0.108490 | 0.104766–0.105147 | 0.104249–0.104500 |
| 251 | 0.107758–0.125888 | 0.109732–0.111510 | 0.107247–0.109288 | 0.108446–0.108673 |
| 257 | 0.120107–0.125589 | 0.114951–0.116013 | 0.111739–0.112218 | 0.111608–0.111713 |
| 263 | 0.111139–0.119275 | 0.106673–0.111839 | 0.107885–0.109161 | 0.107972–0.108203 |
| 269 | 0.132713–0.141317 | 0.121017–0.121857 | 0.117994–0.120198 | 0.119145–0.119308 |
| 271 | 0.144030–0.167861 | 0.135385–0.146577 | 0.136745–0.137126 | 0.136529–0.136673 |

## Execution, provenance, and cost

- Run revision: `11f7e71d9288612d4f1b6dda1abd24d5905cc702`.
- Frozen plan SHA256: `43ff1c80ebcf8d9b99dca4c1f57072ef700a9148512bdec95574cf137e6f0899`.
- Runtime: Linux, Python 3.11.16, Torch 2.5.1+cpu, NumPy 1.26.4, SciPy 1.17.1.
- Hardware: eight allocated vCPUs on AMD EPYC 4564P, 16 GB RAM, 20 GB temporary
  disk, no GPU or persistent volume. Each worker used one Torch/BLAS thread.
- Full execution: **1,325.73 seconds (22.10 minutes)**. The initial reproduction
  control took 43.69 seconds; the first eight concurrent fits took 73.60 seconds;
  the remaining 135 fits took 1,207.12 seconds. Worker fitting times sum to
  9,895.05 seconds; peak recorded RSS was about 345 MiB per worker.
- All twelve original replicate-0 controls reproduced their pinned policy hashes.
  Paired initialization and sampled minibatch hashes matched. All 144 model
  files and 432 information-set files passed their hashes on the host and again
  after local retrieval. Recomputed local screening matches the remote screen.
- Source and input manifests, replay hashes, final weights, optimizer diagnostics,
  full information-set reports, and all checkpoints are retained in the archive.

The fresh quote was $0.28/hour for compute plus $0.003/hour for storage, with no
transfer fees under the provider's [billing documentation](https://docs.runpod.io/accounts-billing/billing).
The pod was requested around 16:41 UTC on 15 September 2026; termination was
verified by 17:23:31 UTC, within the one-hour ceiling. The run finished around
17:05:58 UTC. The rental interval includes provisioning, the delay until the next
status check, retrieval, and shutdown; runtime alone is not the billed duration.

The account display moved from $9.68 to $9.50. The rounded $0.18 difference is not
an itemized invoice. The quote multiplied by the full observed rental interval
is approximately **$0.198**; retain **$0.21** as the conservative allowance for
this rental. Combined with the previous $1.22 allowance, conservative total CPU
usage is **$1.43**, leaving **$8.57** of the original $10 authorization. No top-up
was made. This is separate from the account balance and any future GPU budget.

After artifact verification, the console showed stopped compute and storage at
$0.00/hour. The pod was then terminated; no billable storage was retained.

### Artifact retrieval

Local archive: `results/strategy-fitting-remote.tar.gz`, **41,128,008 bytes**.

SHA256:

```text
8f749a96feb9bb1870a726a98130d13fc0cd17b7378631ba467db3eefc5e837c
```

It contains `strategy-fitting-remote/` and the exact exported
`strategy-fitting-inputs/`. Verify the archive hash before extracting. Model and
full information-set artifacts remain outside Git; their hashes and all scalar
checkpoint results are in the committed reports. The remote copy was disposable
only after the local archive had matched its independently read remote hash.

## Decision

Preserve the failed selection screen and end this campaign. **No additional size
sweep, no confirmation run, and no model promotion follow automatically.**

The evidence supports better optimization as a useful change: decay fixes the
absolute-limit failures across the declared replicates, and exact gradients show
that the current representation can closely fit the replay. The remaining
selection issue is a paired strength regression, not an inability to reach the
absolute limit or an obvious need for more width.

The next task is a bounded decision record for production strategy fitting and
its confirmation criteria. Assess the seed-239 regression using this completed
report, compare the engineering cost of decayed fitting with an alternative such
as Single Deep CFR, and specify what evidence would justify a fresh end-to-end
confirmation. Any proposed change to the regression criterion must be explicit
and prospective; it cannot turn this failed screen into a pass. Keep the original
Kuhn/Leduc absolute limits and the unused confirmation seeds. Do not spend on
another sweep to resolve a protocol decision.

This result does not establish six-player convergence or professional Hold'em
strength. It narrows the next implementation decision before scaling training.

## Validation

- 279 local tests passed before rental; GitHub's Linux checks passed at the run
  revision, including headless rules, arena reproduction, and training recovery.
- The four-recipe local smoke completed in 13.19 seconds within its 120-second
  ceiling. It was unscored and did not change the frozen recipes.
- The complete run passed all worker watchdogs, provenance, immutable replay,
  numerical, oracle, paired-stream, control-reproduction, and artifact checks.
- All declared seeds, recipes, fitting replicates, and scheduled diagnostics are
  retained. No timeout, failed job, or incomplete fit was excluded.
