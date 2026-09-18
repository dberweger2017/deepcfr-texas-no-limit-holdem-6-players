# Multi-street representation campaign results

Completed September 18, 2026. **Neither candidate qualified.** The learned
card branch improved validation decisions on two training seeds but regressed
on the third. Explicit visible-card features regressed on all three. Production
models and training defaults remain unchanged.

This was supervised learning on restricted six-player Hold'em reference
problems, with 2 BB initial stacks and fixed continuation policies. It was
not full-game self-play, a win-rate measurement, or evidence of professional
playing strength. The separate M4 full-game experiment belongs to PR #86.

## What ran

The [frozen protocol](../holdem-multistreet-campaign.md) used 48 fresh board
families, split 24/8/8/8 into training, tuning, validation and sealed test.
Four holdings, three streets and open/facing situations produced 1,152 contexts:
576 training and 192 in each other split. Uniform and increasing continuation
profiles retained their 1:2 target weighting.

All 1,152 references, nine independent fits and 27 saved fit checkpoints
completed. Seeds were 941, 947 and 953. Checkpoints at 1,024, 2,048 and 4,096
optimizer steps were scored on tuning boards; the lowest mean tuning decision
cost selected one duration per architecture before validation. Checkpoints from
one fit are correlated observations, not additional training seeds.

Decision cost is the difference between the highest reference action value
and the value of the model's mixed action under the same fixed continuations.
Lower is better. It is measured in BB per benchmark decision, not BB/100 hands
or full-game exploitability.

## Duration selection

| Representation | 1,024 tuning cost | 2,048 | 4,096 | Selected steps |
| --- | ---: | ---: | ---: | ---: |
| Scaled baseline | 0.20537 | 0.34646 | 0.46717 | 1,024 |
| Explicit visible features | 0.17272 | 0.22801 | 0.22663 | 1,024 |
| Learned card branch | 0.28121 | 0.26579 | 0.26246 | 4,096 |

The earliest checkpoint was best for two architectures. Longer fitting would
have worsened their tuning decisions. Explicit features won the tuning
comparison but did not preserve that advantage on the untouched validation
families. This illustrates why tuning and validation were kept separate.

## Selected validation results

| Representation | Seed 941 cost | Seed 947 cost | Seed 953 cost | Mean cost | Qualifying seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scaled baseline | 0.07903 | 0.08360 | 0.14996 | 0.10420 | Control |
| Explicit visible features | 0.13164 | 0.09012 | 0.32546 | 0.18240 | 0/3 |
| Learned card branch | 0.18187 | 0.05795 | 0.05406 | 0.09796 | 2/3 |

Positive paired gain below means lower cost than the same-seed baseline.
The SE measures paired Monte Carlo reference uncertainty conditional on these
contexts and fitted policies. It does not include board-population uncertainty
or variation across training seeds.

| Candidate | Seed | Paired gain BB | Reference SE BB | Gain minus 2 SE | Passed |
| --- | ---: | ---: | ---: | ---: | --- |
| Explicit features | 941 | -0.05261 | 0.00271 | -0.05803 | No |
| Explicit features | 947 | -0.00651 | 0.00236 | -0.01124 | No |
| Explicit features | 953 | -0.17550 | 0.00287 | -0.18124 | No |
| Card branch | 941 | -0.10284 | 0.00354 | -0.10993 | No |
| Card branch | 947 | +0.02566 | 0.00185 | +0.02197 | Yes |
| Card branch | 953 | +0.09590 | 0.00300 | +0.08990 | Yes |

Every candidate passed the relative regret-error non-regression screen. That
was insufficient: the frozen rule also required practical decision improvement
of at least 0.02 BB and 10%, and a positive reference-noise lower bound on
**every** training seed. Relative RMSE could increase by at most 0.02. The card
branch's slightly better mean cannot override its large seed-941 regression.

At its selected duration, the card branch fits training targets closely:
relative regret RMSE is 2.2–3.5%, versus 22.1–24.1% for the baseline. Its
validation error remains 62.4–64.7%, versus 65.4–68.9% for the baseline.
This is evidence of a large transfer gap in this benchmark; it does not isolate
capacity, coverage or reference noise as the sole cause. Lower aggregate regret
error also does not guarantee better regret-matched decisions.

### Where the validation mistakes occur

This descriptive breakdown averages the selected models across seeds and the
32 validation contexts in each stratum. It did not influence selection.

| Stratum | Baseline cost BB | Card branch | Explicit features |
| --- | ---: | ---: | ---: |
| Flop/open | 0.05114 | 0.07189 | 0.05526 |
| Flop/facing | 0.11646 | 0.08286 | 0.15698 |
| Turn/open | 0.02430 | 0.02377 | 0.04698 |
| Turn/facing | 0.09715 | 0.09339 | 0.18615 |
| River/open | 0.11210 | 0.09080 | 0.24570 |
| River/facing | 0.22402 | 0.22503 | 0.40335 |

River mistakes remain substantial even though the river calibration strata
met the precision criterion. The model result therefore cannot be dismissed
solely because flop and turn references missed their precision target. Equally,
calibration on training contexts does not certify every validation river label.

## Sealed test

No candidate qualified, so only the baseline was evaluated on sealed test.
The saved baseline decision costs are 0.06711, 0.13383 and 0.08179 BB for seeds
941, 947 and 953. Relative regret RMSE is 51.1%, 51.5% and 56.4%.
There is no candidate test comparison and no post-hoc exception to the gate.

## Reference precision

The [calibration report](holdem-multistreet-calibration.md) remains part of the
result. Its frozen budgets and p90 worst-action-pair standard errors were:

| Street/situation | Worlds | Calibration SE BB | Status |
| --- | ---: | ---: | --- |
| Flop/open | 128 | 0.30742 | Unresolved at cap |
| Flop/facing | 128 | 0.35244 | Unresolved at cap |
| Turn/open | 128 | 0.27551 | Unresolved at cap |
| Turn/facing | 128 | 0.33331 | Unresolved at cap |
| River/open | 32 | 0.07031 | Resolved |
| River/facing | 16 | 0.07762 | Resolved |

The four unresolved strata were an anticipated protocol outcome, retained
without extending world budgets after seeing model results. Calibration and
production used independent streams; alternative actions within a context
shared worlds. Per-context worst-pair errors can be much larger than the SE of
an average paired model difference. These measure different quantities.

## Independent-reference stability audit

A supplementary audit, proposed before campaign completion, compares the saved
calibration and production streams for their 72 overlapping training contexts.
It uses the optimized calibration cache (identical numerical results to the
original) and the frozen production world budgets. River calibration is truncated
to its first 32 or 16 worlds, so both batches use the same N. No new worlds,
model fits, validation choices or sealed comparisons were added.

The table reports the combined 1:2 profile targets used for learning. A means
calibration and B means production. RMSE pools action entries within each
stratum; policy and cost summaries weight its 12 contexts equally.

| Stratum | Q RMSE BB | Regret RMSE BB | Mean policy TV | Top-action agreement | Cost of A policy on B, BB | Cost of B policy on A, BB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flop/open | 0.12873 | 0.08107 | 0.01058 | 12/12 | 0.00000 | 0.00341 |
| Flop/facing | 0.32686 | 0.23761 | 0.00000 | 12/12 | 0.00000 | 0.00000 |
| Turn/open | 0.12078 | 0.07597 | 0.00126 | 12/12 | 0.00137 | 0.00149 |
| Turn/facing | 0.13167 | 0.09547 | 0.00000 | 12/12 | 0.00000 | 0.00000 |
| River/open | 0.24032 | 0.15033 | 0.00044 | 12/12 | 0.00127 | 0.00142 |
| River/facing | 0.17334 | 0.12612 | 0.00000 | 12/12 | 0.00000 | 0.00000 |

Best-action sets agree in **72/72 contexts**. Mean policy TV is 0.00205;
the largest individual TV is 0.12691. Cross-batch costs average 0.00044 BB
(A on B) and 0.00105 BB (B on A); the largest individual cross-batch cost is
0.04096 BB. Within-batch costs average 0.00041 and 0.00069 BB respectively.
Thus differing numerical estimates usually preserve the strategically relevant
action ordering in this overlap. Flop/open pairwise sign agreement is 97.22%;
the other five strata have 100% agreement.

The JSON retains each individual profile, pooled Q/regret correlations,
tie-aware rankings, within-batch costs and signed policy comparisons evaluated
on the **same** Q batch. Ties use absolute tolerance 1e-12. Regret matching
need not be greedy, so even identical reference policies can incur nonzero
cost relative to the best single action. Maximizing noisy Q estimates also
can inflate decision cost. Neither quantity is exploitability.

The individual profiles are not identical to their combined target. Uniform
turn/open has best-action agreement in 11/12 contexts and mean TV 0.06193.
Uniform river/open has cross-batch costs around 0.095–0.096 BB, but its
within-batch costs are also around 0.094–0.097 BB and mean TV is only 0.00020.
That is a concrete example of non-greedy regret-matched policy cost, rather
than evidence that the two reference batches imply very different decisions.

These are descriptive results on twelve training families shared across
streets, not 72 independent families. Calibration helped select world budgets;
this is not an independent post-selection precision guarantee. The audit does
not certify held-out contexts, and it examines the controlled reference builder,
not the production self-play sampler's noisy replay targets.

**Interpretation:** strategically unstable labels are a weak explanation for
the model failures on this audited training overlap. Generalization and data
coverage deserve priority in interpreting the benchmark; we have not established
which is the main full-game bottleneck. The earlier river-only feature advantage
did not reproduce consistently in this larger multi-street comparison.

Reproduce locally from the retained cache with:

```bash
python -m scripts.audit_multistreet_reference_stability
```

The output is `results/multistreet-reference-stability.json`. It reads saved
training references and preserves the original campaign's source fingerprints.
The output SHA-256 is
`4d6e8442324f7ff3d9062a9114dbdcf327de6b11b631b79ed733dfb7fd24f8fc`.
Eleven focused audit tests cover signed policy costs, frozen-prefix truncation,
pooled error aggregation, cache tampering, source/stream mismatches and a missing
campaign context. The audit plus full-game experiment/reproduction subset passed
all 20 tests after the audit source was frozen.

## Decision and next task

Retain all three architectures and the frozen protocol as controls. Do not
promote the two successful card-branch seeds while ignoring the third, increase
fitting duration across the board, or spend another rental merely to repeat this
comparison. The current narrow benchmark does not justify abandoning neural CFR.

The next bounded task is to locate the expensive generalization errors using
saved training, tuning and validation records: action margins, visible hand/board
structure and coverage of those structures in training. Keep the sealed candidate
models closed. Use that analysis to select one targeted data or representation
change and predeclare a fresh confirmation comparison. Full-game replay noise
and postflop coverage remain separate open problems; this audit does not clear
them. Another paid campaign needs its own measured cost and decision rule.

## Operations and artifact recovery

The rental used 32 vCPU and 64 GB RAM in Runpod's 3 GHz pool (reported AMD EPYC
9654). The 5 GHz pool was unavailable. Compute was $0.96/hour, with approximately
$0.003/hour storage overhead and an unchanged $10 campaign ceiling.

Original calibration took 2,321.30 seconds and reached 36.53 GB cgroup memory.
Root-only accumulation removed unused later-decision statistics; twelve saved
prefix comparisons matched exactly, and the complete optimized calibration
dictionary equalled the original. Optimized calibration took 1,774.26 seconds
with 7.46 GB sampled peak memory. The failed original admission and both
calibrations remain retained. A training-only fitting timing probe supported
the revised resource admission documented before production launch.

Production collection plus fitting took 20,308.28 seconds (5 h 38 min).
The fitting runner itself took 168.02 seconds with parallel jobs. Reference
construction dominated runtime. These timings exclude earlier setup and both
calibrations. The driver completed at 19:30:43 UTC / 21:30 Madrid.

Fresh campaign verification and strict provenance reconstruction both passed:
144 calibration and 2,304 production context/profile cache entries. The archive
was downloaded and all 2,709 inventoried files were independently hash-checked
locally before storage deletion. Compute terminated after retrieval at about
19:31 UTC; the disposable network volume was subsequently deleted and the
provider storage page showed no remaining volumes.

The displayed remaining balance was **$18.07**, down $7.31 from the owner-funded
$25.38. This is an observed account-balance difference, not an itemized invoice.
No new rental or training campaign was launched.

### Retained evidence

These raw artifacts are in the local ignored `results/` directory; committing
this report does **not** upload them to GitHub or provide an off-machine backup.

- Archive: `results/multistreet-campaign.tar.gz` (20,793,652 bytes).
- SHA-256: `ce48e72f3ba634e6a5a7de93a9dc9079a09b482abce3355ae69741650ab2e94f`.
- Extraction: `results/multistreet-retrieved/`, including original calibration,
  operational logs, exact source archives, references, checkpoints and reports.
- Main report: `poker/results/multistreet-campaign/fit/report.json` within extraction.
- Manifest: `poker/results/multistreet-campaign/campaign-manifest.json`.
- Completion and inventory: `multistreet-operations/`.
- Scientific revision: `c27342736dce8525d0e151fb4bc14507019cef16`.
- Scientific source fingerprint: `7567c413f2920aa78d09b8ec926c1d8368b1f1a603e5f07c62f568a33d19eaaf`.
- Production plan fingerprint: `a4bc0326e86476acf912c9033df572dd53f8e2400f9bb221c184f3cb5d2dd1b9`.

The timestamped [first](holdem-multistreet-preliminary.md) and
[second](holdem-multistreet-preliminary-2.md) preliminary reports remain historical
snapshots. Their pending-work statements are superseded by this report.
