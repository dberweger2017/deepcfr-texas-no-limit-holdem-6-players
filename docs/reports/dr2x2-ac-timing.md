# A/C outcome-blind evaluation timing

All six frozen current-policy inputs passed hash/schema checks and completed the
[prospective timing plan](../../configs/diagnostics/dr2x2-ac-evaluation-timing.json)
at evaluation source `a21125af0239b70615624611ce11b54058dfe291` on M1,
Python3.11.14 with NumPy1.26.4 and the pinned engine. No payoff output was
recorded or inspected, no paid job created, no M4 allocation.

Four paired validation blocks per model/panel, both positions:480 timing hands.
All completed. The entire pilot took **58.91seconds**, peak **2.394GiB**, zero
swap growth. Each model loaded sequentially, freed before the next. A loads
11.31–11.65seconds; C loads4.41–4.58seconds. Original training artifacts stayed
unchanged. Supervisor completed exit0 and released the M1 heavy lock.

## Projection, not a deadline or scientific result

The frozen projection sums each model's maximum measured hand time × both
positions × its panel's declared blocks, then multiplies by2. Load overhead
also receives the2× margin. The proposed104,448current-policy hands project to
**7,313.57seconds / 2.03hours on this M1**, including96seconds load allowance.
This estimate does not include full raw snapshot/native replay/tail serialization
cost (timing suppresses outcomes and skips those exports), provider setup,
Linux speed differences, or the separately pending mechanism diagnostics.
Those costs must enter the final runtime/quote; this is not a promised completion
time or permission to reduce counts if execution takes longer.

| Panel | Projected seconds including2× margin |
| --- | ---: |
| lbr | 6632.96 |
| loose_aggressive | 30.28 |
| loose_passive | 29.52 |
| native-pressure | 222.55 |
| passive | 33.52 |
| pot_pressure | 28.83 |
| selective-stackoff | 147.44 |
| tight_aggressive | 29.08 |
| tight_passive | 23.77 |
| uniform | 39.85 |

Timing roots use validation streams and are disjoint from the proposed test
roots. LBR keeps originalcap2/K4/five-soft-seconds and cached probability queries;
limited batches/overruns remain reportable. No strength, coverage-improvement,
convergence or factorial-interaction conclusion follows from these timings.

## Checks and remaining admission

Three pinned focused checks passed: compressed river visit telemetry uses its
own schema/key; both arms/rotations pair identical deals while validation/test
deals differ; the timing result contains no payoffs. An initial test collection
lacked NumPy in the pinned parity environment, before any candidate was loaded.
The launcher adds the existing NumPy1.26.4 dependency directory with
`site.addsitedir`, preserving the pinned Python/engine and standard-library
precedence. The timing pilot itself succeeded on its first attempt.

[Summary](dr2x2-history-artifacts/ac-timing-20261002/summary.json),
[prospective plan](dr2x2-history-artifacts/ac-timing-20261002/plan.json),
[transport/publication hashes](dr2x2-history-artifacts/ac-timing-20261002/publication-manifest.json).
Large inputs remain at their exact read-only M1 paths. Original A inputs are
#116's B100M v1 states, not #143's v2 cell despite their B filename prefix.

Next admit the #142 common uniform-compatible restricted river-root BR
measurement and #141 stored-current/average diagnostic for both schemas.
Freeze the final strength executor/schedules before payoffs; then publish any
separate paid quote with Linux/setup/retrieval/storage/failure margin inside
remaining approved$16. No rental or strength launch by this report. D remains
deferred; no automatic extension/promotion/merge and no #136 restart.
