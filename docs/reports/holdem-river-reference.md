# Six-player river learning diagnostic

**The existing network learns these reliable targets. An exact own-information
baseline substantially reduces sampling noise. Generalization remains weak enough
to make costly card-dependent mistakes on the held-out board.**

The complete local run finished in **118.72 seconds** (1m59s), below its
15-minute limit, without failures, retries or rental spending. No production
optimizer, encoder, sampler or model was changed. The new check/fold policy is an
additional arena control, not a replacement agent.

## What was tested

The [protocol](../holdem-river-reference.md) and
[plan](../../configs/holdem/river-reference.json) were committed at `7bef598` before
measurements. The executed implementation is `1a6b214e5937d982106471c90692d3de2e980308`.
The [complete compact results](holdem-river-reference.json) retain every context,
hidden assignment, reference target, sampling cell, seed, prediction and artifact
hash.

- 24 six-player river contexts: three boards, four hero holdings per board, and
  unopened/facing-all-in situations. All six players are live at every root.
- Two equally weighted, explicit joint hidden-card assignments per context;
  reference action values integrate both worlds and subsequent opponent actions.
- Two frozen public policies, producing 48 exact reference cells and 69,792
  enumerated nodes. These are finite diagnostic policies, not trained opponents.
- Three sampling seeds, 32 replicates per cell and four sampler/baseline arms:
  **18,432 traversal estimates**, 576 sampling cells, 210,148 sampled nodes.
- Six fresh network fits, with exact versus sampled targets and matched
  initialization for each seed. Sixteen contexts train; all eight contexts on
  board three are held out, including their hidden worlds.
- A separate 768-hand arena accounting control at six-player 100 BB.

The river reference starts from legal 2 BB hands, with calls/checks leading to
1 BB remaining. It is deliberately much smaller than full 100 BB poker. Exact
regret tables and neural fits use the same fixed policy schedule with weights
1 and 2. This is a matched target-recovery experiment, **not online tabular versus
neural CFR convergence**, not an exact multiplayer solver, and not a poker-strength
comparison.

## Can the neural learner recover the signal?

Relative error below is regret RMSE divided by RMS exact regret, expressed as a
percentage. It is **not percentage improvement in replay loss**. The predeclared
training threshold was 10%; the separate held-out threshold was 50%.

| Seed | Fit targets | Training relative error | Held-out relative error | Held-out policy TV | Held-out decision cost (BB) |
| --- | --- | ---: | ---: | ---: | ---: |
| 421 | exact | 0.36% | 46.15% | 0.260 | 0.217 |
| 421 | sampled | 4.86% | 48.45% | 0.317 | 0.403 |
| 431 | exact | 1.10% | 47.00% | 0.250 | 0.182 |
| 431 | sampled | 7.57% | 47.03% | 0.250 | 0.183 |
| 433 | exact | 0.72% | 47.95% | 0.250 | 0.183 |
| 433 | sampled | 3.56% | 48.14% | 0.250 | 0.183 |

**All three exact fits and all three sampled fits pass both declared thresholds.**
The exact fits recover the training mapping with 0.36–1.10% relative error.
Sampled fits have 3.56–7.57% error. The direct sampled tables have 5.34–8.94%
training error against the exact table; their held-out lookup errors are
3.11–4.89%. Those held-out table lookups use samples collected at those contexts;
they do not demonstrate table generalization.

For training contexts, the clean models distinguish strong and weak holdings with
the public board, action prefix and opponent assignments fixed. A neural model
can recover this finite mapping using the current encoder; a broad claim that
neural regret fitting cannot work is not supported.

### Passing the coarse held-out screen does not mean good generalization

Exact fits have 46.15–47.95% held-out error and mean policy TV 0.250–0.260.
All three clean models call with **Ac Kd on Qs Js 8d 5c 2h** when facing the all-in,
although the reference says folding is better by **1 BB** under the declared
continuations and ranges. Their calling probability is 1. The sampled fits make
the same call. The full per-decision predictions retain this failure.

That is a concrete transfer error, not a professional-level nuance. The 50%
screen was deliberately permissive; it remains passed rather than being changed
after seeing results. One held-out board and two joint deals cannot select a new
encoder, establish broad generalization or identify the full-game failure.

Decision cost compares a policy's expected payoff with the best immediate action
under the fixed reference continuations. Those continuations are held fixed even
after a different hero action; this is not a best response over the whole game or
exploitability. Even the exact regret-matching reference has mean training cost
0.00256 BB because the weighted regret policy is not defined by greedily maximizing
the weighted Q values.

## Does a better baseline have room to help?

Each row averages per-context/profile/seed statistics. Variance is the trace of
the centered root-regret sample covariance, including hidden-world and continuation
randomness. The work proxy is each cell's variance multiplied by its mean node
count, then averaged. Raw samples and all individual cells remain available.

| River situation | Sampler/baseline | Mean variance (BB²) | Mean nodes | Mean variance × nodes |
| --- | --- | ---: | ---: | ---: |
| open | single-zero | 114.192 | 7.64 | 872.056 |
| open | single-oracle | 4.812 | 7.64 | 36.660 |
| open | first-zero | 11.389 | 20.96 | 238.870 |
| open | first-oracle | 1.807 | 20.96 | 37.816 |
| facing | single-zero | 17.146 | 6.00 | 102.874 |
| facing | single-oracle | 0.606 | 6.00 | 3.637 |
| facing | first-zero | 0.327 | 11.00 | 3.601 |
| facing | first-oracle | 0.327 | 11.00 | 3.601 |

For unopened rivers under the production first-decision sampler, the exact
own-information baseline reduces mean variance × nodes by **84.17%**, clearing
the predeclared 25% diagnostic threshold. Per-seed reductions are 83.50%, 84.51%
and 84.56%. This is descriptive finite-sample screening, not a confidence bound
for a full-game effect.

There are **two worse cells out of 72 unopened first-decision comparisons**:
board-0/hand-2 (Qh Jh), seed 431, under both frozen profiles. Uniform cost proxy
rises from 113.921 to 123.576; increasing-policy cost rises from 78.617 to 79.496.
They are retained. The aggregate is not a claim of universal improvement.

Facing an all-in, first-decision expansion already evaluates both hero choices
and hero has no later decision. Zero and oracle baselines produce **identical
estimates on every paired draw**. This negative control passes. Every baseline
pair also has identical executed-path hashes and node counts; changing a baseline
has not changed visitation or street coverage.

The single-action oracle and first-decision oracle have similar aggregate work
proxies here. That does not select a replacement sampler: the exact table is not
an available learned baseline, and these are shallow river trees.

### Costs and limits of the oracle

Exact enumeration took 11.79 seconds; sampling took about 45.20 seconds across all
arms; fitting took about 53.95 seconds. The full run also includes control play,
serialization and artifact publication. Oracle construction is charged separately
and its tables are queried in memory. No persistent critic was trained, and its
training/inference overhead has not been measured.

This oracle conditions on hero's own information and averages over compatible
worlds; it is not a history-aware critic that sees opponents' dealt cards. Its
result shows available benefit in this family of baselines, not a universal
minimum variance. It does not remove every source of continuation noise or create
missing postflop decisions.

## Check/fold accounting control

The new `check_fold` policy checks whenever possible and otherwise folds, making
no voluntary payments. Against the five existing styles, 64 fresh blocks with
six rotations per arm give:

| Policy | Hands | BB/100 | Worst balanced block |
| --- | ---: | ---: | ---: |
| Check/fold | 384 | -17.708 | -25.000 |
| Existing fold | 384 | -19.922 | -25.000 |

Every hand settles to zero sum; every balanced block obeys the -25 BB/100 posted-
blind bound. This is an accounting control, not evidence of skill or a new paired
comparison against the previously evaluated models. The old model point estimates
were much worse on their own schedules, motivating the control without treating
those different schedules as a paired experiment.

## Decision

**Retain neural CFR as a candidate and proceed to a bounded persistent-critic
prototype.** The accurate-baseline screen passes, and clean-target fitting does
not fail. Both clean and noisy training fits pass, so this diagnostic has not
reproduced the full-game learning failure or proved its cause.

The prototype should have an explicit current-policy value objective, separate
recent replay, immutable collection-time baselines and complete recovery. Compare
it with zero and the existing historical value head on frozen six-player profiles,
including the retained river references and full-game postflop probes. Include a
cheap known-payoff baseline so a learned critic must justify its added complexity. Charge
critic training and inference in the measured cost. Record collector versus arena
street visitation separately; a fixed-profile baseline cannot fix coverage.

Keep the held-out Ac Kd mistakes as regression probes. They justify measuring
card-sensitive transfer, but not choosing a transformer or larger network on this
single board. A new encoder remains a controlled alternative if reliable-target
transfer continues to fail. Only after a learned baseline earns a cost-aware
comparison should one short, predeclared online multi-seed test follow. No larger
campaign, paid compute or model promotion is authorized by this report.

## Verification and artifacts

Twelve focused tests cover compatible hidden worlds, own-information baseline
queries, exact sample expectations across both hidden deals, hero payoff
perspective, weighted target construction, baseline path identity, final-decision
invariance, deterministic fitting, complete artifacts and failure recording.
The broader sampler/arena selection passed 48 tests before execution.

Post-run verification re-read all 18,432 estimates, checked the 576 cell counts,
recomputed means/centered variances/reference MSE, verified all paired path hashes,
checked all 768 arena outcome digests, zero voluntary payments and balanced blind
bounds, and reloaded all six models to reproduce every final prediction and
metric exactly. All recorded artifact hashes match. All 597 tests and the end-to-end reproduction checks pass in CI on the executed
implementation. Final-head CI is required again before merge.

Local raw artifacts are in `results/river-reference/`: `report.json`,
`samples.jsonl`, the control schedule/outcomes, six model state dictionaries and
`verification.txt`. The compact JSON contains their hashes and the exact executed
revision. Model and raw-hand files are intentionally not committed. The regression
weights are diagnostic models, not resumable full-game training checkpoints.
