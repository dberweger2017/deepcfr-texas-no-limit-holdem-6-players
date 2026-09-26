# River CFR development sequence in draft PR #108

The first [M4 preflight](reports/river-quality-m4.md) established one
full-range resource point. This frozen development run asks how the
own-reach-weighted average strategy's restricted-game exploitability changes
with more work and across varied river roots. It is exploratory development,
not the fresh paired confirmation or a full-game strength evaluation.

The [plan](../configs/blueprint/river-development-m4.json) fixes the saved
12M checkpoint hash, all cases and budgets before the run. The 12 existing
two-player reference fixtures run sequentially without loading the checkpoint
through 8,192, 16,384 and 32,768 full sweeps. Every snapshot records actual
completed sweeps, the stop reason, average and final-played exploitability,
and agreement with the independently enumerated best response. A missed
milestone is recorded as incomplete and stops the run.

The checkpoint then loads once. Five declared full-range six-seat river
roots vary board, button, prior betting, root pot (2, 4, 8, 12 or 32 BB),
remaining stack, and range shape. The default `blueprint` shape uses the
adapter's action-likelihood marginals. `uniform` replaces those marginals
with equal weights; `squared` squares and renormalizes them. Every shape
uses the same card-collision rule. These are explicitly stipulated games;
the shape variants do not assert a true opponent posterior.

For each root, measure cold range and tree/payoff construction, then complete
full-range CFR sweeps up to elapsed decision times of 5, 15, 30 and 60
seconds, including setup. Record actual sweeps, segment wall time per sweep,
average and final-played exploitability, range effective support, process RSS,
system swap and memory pressure. Save the final complete profiles. The
runner keeps every row, failure, input and engine hash, and artifact checksum.

The run is sequential on the M4 with a 10.5-GiB process RSS ceiling,
30-GiB minimum free disk and a 15-minute overall wall limit. It uses no paid
host and does not train or modify a blueprint. If that cap prevents a
milestone, retain the partial result; do not silently extend work.

After inspecting these development results, select **average extraction**
and one practical decision work/time limit. Freeze that choice, independent
river confirmation roots, rollout settings (both ordinary and calibrated
compute-matched), evaluation deals, uncertainty analysis and the campaign
wall budget in a separate protocol before running any paired comparison.
No playing-strength or model-promotion claim follows from this development
run alone.

## Range-shape amendment after the first run

The completed first run showed that four of its five full-range roots had
uniform effective marginals despite their labeled `blueprint`, `uniform` or
`squared` input shapes. Squaring a uniform prior leaves it uniform. To test
the requested range variation, the separately frozen
[amendment](../configs/blueprint/river-range-amendment-m4.json) adds three
full-range cases only. It keeps the solver, action menu, checkpoint and
5/15/30/60-second measurements unchanged. The stipulated `suited-bias`,
`pair-bias` and `high-card-bias` laws multiply the blueprint marginal by a
fixed factor of four for suited hands, four for pocket pairs, or three for
hands containing J/Q/K/A respectively, then renormalize and apply card
compatibility. They deliberately create nonuniform development distributions;
they are not claims about real opponent beliefs. A deep mixed board replaces
the board-playing-straight texture as a nontrivial deep-stack stress case.
The amendment is exploratory and must not be treated as a fresh confirmation
after inspecting the first development run. It has a separate seven-minute
M4 wall cap and retains its own artifact hashes and failures.
