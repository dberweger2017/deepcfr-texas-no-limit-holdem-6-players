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
