# Separate trainer performance hypothesis

This is a handoff for a separate engineering task, not a trainer change or
training authorization in PR #126. Claude v7 reports an early M1 profile at
`a2053bb`, resolved here to `a2053bbeea8a5e5170a1d83bf5d440684f82283d`.
His `/tmp/profile_training.py`, `/tmp/profile_engine.py` and `/tmp/engine_raw.py`
were absent at the checked paths on both Macs; an executable/hash-sealed
archive was not recovered. The supplied measurements remain **unreplicated**.

His 1M-node warmed table had 116,186 entries; another profiled 1M nodes showed
overlapping cumulative shares: `Hand.apply` 47.8%, `Hand.observe` 46.5%,
`information_key` 22.4%, `_history` 8.6%, `_postflop` 7.5%, `choices` 6.6%,
`json.dumps` 3.0%, and `hand_value` 1.7%. These shares cannot be added.
Reported replay time was 26.3 of 49.8 instrumented seconds for 300k nodes.
The simplified native-only no-raise transition microbenchmark does not
establish an achievable 22× trainer gain. His M1 results cannot be used as an
M4 or mature-table throughput forecast.

Actual source inspection supports a specific duplicate-work hypothesis:
`src/blueprint/solver.py` `visit` builds the acting observation before key/menu
work; `src/game/hand.py` `Hand.apply` calls `self.observe(self.actor)` again
before legal-action validation. `Hand.observe` reconstructs from public events
and that seat's private cards using `observation.replay`. Measure the precise
paths, including branches and leaf observations, before estimating savings.

The preferred first candidate reuses an observation only for the **same
immutable Hand instance and same acting seat** at the traversal/apply
boundary, while keeping full legal-action validation. Bind reuse to the
concrete state/event sequence, not an abstract information key. Reject reuse
after a transition, across seats, across hand instances or altered histories.
Do not begin with a broad incremental replay/observation rewrite.

Freeze native and candidate fixed-work streams on the actual uncapped HU20
recipe with K1, roots/seat, seeds and current extraction unchanged. Require
identical observations, concrete keys, ordered menus, regrets, averages,
visits, completed/discarded nodes, iteration weighting, traversal and RNG
states. Check save/export/reload and interruption/resume equivalence in fresh
processes. Require byte-identical deterministic artifacts; list timestamp,
timing and other genuinely nonsemantic metadata explicitly before comparison.
Never normalize away unexplained state differences.

Benchmark on M4 only, in separate fresh processes, using both a small table
and a representative retained mature checkpoint. Include load/save/export
overhead, throughput, peak aggregate RSS, swap/disk and recovery. Use the
10.5-GiB/0.5-GiB/8-GiB guards and coordinate the exclusive M4 phase through
`/tmp/DR_RESEARCH_M4_COORDINATION.txt`. There is no automatic 3× adoption
threshold. Neither this handoff nor Claude's advice authorizes a 100M control
arm, new recipe, paid host or concurrent heavy research job.
