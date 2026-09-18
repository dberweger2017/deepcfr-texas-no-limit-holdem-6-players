# Multi-street calibration and host admission

## Original collector

The first calibration completed all 72 training-only contexts on September 18,
2026, at revision `dfc94714861d277fe7dc52af7135ea1a7b8aa037`. It used the
unchanged 48-family campaign plan and nested 8/16/32/64/128-world streams.
No production references or model fits were started by this invocation.

| Stratum | Frozen worlds | p90 worst-pair SE (BB) | Precision status |
| --- | ---: | ---: | --- |
| Flop, open | 128 | 0.307424 | Unresolved at cap |
| Flop, facing | 128 | 0.352443 | Unresolved at cap |
| Turn, open | 128 | 0.275513 | Unresolved at cap |
| Turn, facing | 128 | 0.333313 | Unresolved at cap |
| River, open | 32 | 0.070311 | Resolved |
| River, facing | 16 | 0.077625 | Resolved |

The target was one paired standard error <= 0.10 BB under the declared p90
summary, not a confidence bound on every context. The flop and turn strata
remain unresolved, as the protocol explicitly permits. Their reference noise
must stay visible in any representation comparison; this result does not
justify describing all targets as precise.

Calibration took 2,321.30 seconds (38.69 minutes), with 32,227.95 summed
reference-worker seconds. The production projection was 495,112.18 worker
seconds: 4.30 hours at an ideal 32-way concurrency, before overhead or fitting.
The cgroup memory high-water mark was 36.53 GB on the 64 GB rental. Only twelve
unopened-flop contexts were available in calibration; using this measurement
to assume that 32 simultaneous heavy contexts fit would be unsafe.

Inspection found that this root-decision benchmark accumulated candidate and
regret-value statistics for every later hero information set, although its
output only uses the initial decision. Full production admission is held while
we verify a collector optimization that retains only the required root
statistics. This is an implementation change to the restricted reference
benchmark, not to the full-game Deep CFR sampling algorithm.

The original calibration, caches and source archive are retained. The revised
collector must reproduce the original world values, root targets and node
counts before a fresh calibration with the same plan and streams. A new source
fingerprint gets a separate cache; the original results are not relabelled.

## Review disposition for this rental

The independent review found no hidden-world input leakage or split
contamination. It identified three operational hardening issues:

- The reference timeout restarts with each invocation. The rental driver will
  explicitly subtract completed calibration work from the reference allowance;
  the independent provider cutoff continues to cap total paid time.
- An explicit zero CLI timeout falls through to the default. The CLI now rejects zero, negative and non-finite allowances before
  creating outputs; this rental never used zero.
- Standalone verification checks cache hashes but does not independently
  reconstruct every provenance field. A separate fresh-process check will
  rebuild each declared context and use the strict cache loader to validate
  streams, worlds and action ordering before final artifact acceptance.

No model has been fitted or promoted. Final resource admission and results
will be recorded after the optimization checks.

## Collector equivalence and resource check

Commit `6fcfb3957dbbd4f7cd987fc5bdcac99216edfb15` retains only the root
statistics needed by this benchmark. Local tests compare the complete returned
reference, including node counts, with the original enumerator on open and
facing flop/turn contexts. All eight optimization/timeout checks pass; the
preceding reference/invariance checks passed 25 tests.

A fresh process on the rental then recomputed the first eight worlds of one
original calibration context in each of the six strata, under both profiles.
All twelve comparisons exactly matched saved per-world values and the derived
root value/regret targets. The process completed in 110.19 seconds and peaked
at 201.00 MiB RSS. Its unopened-flop profiles each traversed 264,248 nodes in
40.50 and 41.06 seconds. The original full-stream timings for those profiles
were 8.48 and 8.07 seconds per world, versus 5.06 and 5.13 in this small check;
the different concurrency means this is an admission estimate, not a controlled
speedup claim.

Using these timings gives a preliminary 325,665.62 worker-second production
projection, or 4.24 hours at 32 workers with a 1.5 allowance. Full admission
still requires the repeated 72-context calibration to reproduce the original
precision results exactly and its measured memory/runtime to fit the remaining
rental allowance. The scientific source for that repetition is `c273427`.
The external driver reserves 90 minutes for fitting and 45 minutes before the
provider cutoff for completion, retrieval and shutdown.

## Completed optimized calibration and revised time allocation

The optimized collector completed all 72 contexts in 1,774.26 seconds
(29.57 minutes). Its complete calibration decision/trace dictionary exactly
matches the original, including all four unresolved strata. Sampled cgroup
usage peaked at 7.46 GB; conservatively scaling that whole measurement from
12 to 32 heavy contexts gives 19.89 GB, below the 48 GB admission limit.
Production projects to 420,019.28 worker-seconds, or 19,688.40 elapsed seconds
at 32 workers with the 1.5 allowance (5.47 hours).

The initial automatic runtime gate failed. Its six-hour cumulative reference
allowance left 17,504.44 seconds after both calibrations, less than the padded
projection. The original 90-minute fitting reserve also exceeded the remaining
combined work allowance. That failed admission record is retained.

Before any production reference collection or model selection, a timing-only
probe ran all nine 128-step fitting jobs concurrently. It used only the 72
training-split calibration contexts, repeated to match the actual 576/192
training/metric row counts; no tuning, validation or test context was opened.
Weights and quality metrics from this probe are not campaign candidates.
The slowest fit took 23.36 seconds. Scaling to 4,096 steps with a 1.5 allowance
and another 300 seconds for overhead projects to 1,421.18 seconds (23.69
minutes). This projection supports a 30-minute fitting reserve, subject to the
unchanged absolute shutdown deadline.

The revised operations allocation permits seven hours of cumulative reference
work, including both calibrations, leaving 21,104.44 seconds for production.
That is still below the configured six-hour limit for a single reference
invocation. Production's 19,688.40-second padded estimate fits this allowance;
production plus the separate 1,800-second fitting reserve must also fit the
remaining time before 20:10 UTC. The driver checks both conditions immediately
before launch.

The $10 all-in campaign ceiling, provider cutoff at 20:55 UTC, and 45-minute
completion/retrieval reserve are unchanged. The original failed gate is not
relabelled as passed. Models, seeds, contexts, reference world counts,
checkpoint selection and sealed-test rules are unchanged. This amendment
reallocates time within the already authorized rental; it does not add a new
experiment or increase spending authority.
