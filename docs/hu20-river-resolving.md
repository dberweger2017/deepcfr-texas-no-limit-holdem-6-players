# Opt-in HU20 river re-solving

This diagnostic player uses B500M current before the river and the average
strategy of a full-range two-player river CFR solve afterward. It is separate
from the six-seat #108 adapter and is not a new default web player. See the
[frozen protocol](hu20-river-resolving-protocol.md) and
[report](reports/hu20-river-resolving.md) for measured cost and limits.

## What the solve means

Each seat has every one of the 1,081 unordered holdings compatible with the
river board. For holding `h`, its factor is the product of its own observed
pre-river blueprint action likelihoods. The shared law is:

`Q(h0,h1) ∝ factor0(h0) × factor1(h1) × 1[disjoint cards]`.

There are no folded third-player cards to approximate. This is exact card
compatibility **under the declared likelihood model**, not a true opponent
posterior. Missing blueprint keys use the existing uniform policy. On-menu
zero likelihoods stay zero. Earlier off-menu raises retain the existing
corrected likelihood kernel and are counted; no wager is translated. Wholly
zero/incompatible ranges fail instead of silently inventing a new prior.

The betting tree uses exact native settlement and #108's restricted
min/pot/conditional-jam menu with two ordinary river raises and no free fold.
All observed river raises are inserted exactly, including raises beyond that
ordinary cap. Full-range solving does not mean every possible future wager.
Information sets distinguish exact holdings at each public node. Solve work
is a fixed count of complete simultaneous Linear CFR sweeps; play uses the
linearly own-reach-weighted **average**, including declared uniform behavior
at zero-average-denominator rows.

## Later off-tree actions

An on-tree response reuses the profile. An off-tree opponent action triggers
a new solve from the river-round root with observed sizes inserted. Every
prior hero decision fixes its entire old probability matrix for all holdings
at that public node, even at zero own reach. Thus the already-used action
likelihoods cannot change during re-solving. Opponent policies elsewhere can
change: this is consistent nested re-solving, **not safe-solving guarantees**.
No failed/unfinished river solve quietly delegates back to the blueprint.

## Cache and boundaries

One immutable cache belongs to one blueprint source. Profiles are keyed by
public root, model/range identity, sweep/tree configuration, inserted public
raises and prior hero matrices. Actual holdings, seeds and opaque hand labels
are absent from the key. A full profile serves every hypothetical holding at
that root; the live player retains its own per-hand state and random stream.
The LRU bounds both range and profile stores. `profile_identity` in diagnostic
records identifies the cache inputs; it is not a hash of serialized output
strategy bytes. Raw action records retain the returned distributions.

The watchdog is an emergency abort, not the scientific solve setting. The
player rejects partial fixed-sweep results, records a failure and raises it.
Per-root and whole-experiment deadlines and process-peak RSS limits apply.
Recorded lookup exposure describes the assumed model, not posterior accuracy.
Range counters attached to repeated requests can reference the same cached
construction; they are not counts of distinct independent observations.

## Reproduce

Use the source-install environment and pinned native engine in the
[README](../readme.md#quick-start). Retrieve the research B500M current exports
listed in [the timing plan](../configs/diagnostics/hu20-river-timing.json) and
[comparison plan](../configs/diagnostics/hu20-river-comparison.json); verify
their pinned compressed byte lengths/hashes before parsing. The runner does
this again. These research exports are not claimed to be release downloads.
No retained training checkpoint or M4 computation is needed.

```sh
python -m pytest -q tests/test_hu20_river.py tests/test_hu20_river_evaluation.py tests/test_river_cfr_reference.py tests/test_river_conditional.py tests/test_river_reference.py
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.evaluate_hu20_river --plan configs/diagnostics/hu20-river-timing.json --inputs B500M_EXPORT_DIRECTORY --out results/river-timing-new
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.evaluate_hu20_river --plan configs/diagnostics/hu20-river-comparison.json --inputs B500M_EXPORT_DIRECTORY --out results/river-comparison-new
python -m scripts.report_hu20_river --directory results/river-comparison-new --out results/river-report-new
```

Use new output directories; existing evidence is never overwritten. Only
closed successful records produce final paired tables. Failed runs retain
their partial gzip/JSONL and failure status. Generated fixture checks do not
claim real-model playing strength. Timing and paired comparison run on M1;
other platforms and live UI integration are untested by this experiment.

For a headless observation-only caller, opt in explicitly:

```python
from src.blueprint.hu20_river import HU20RiverConfig, HU20RiverPlayer
player = HU20RiverPlayer(verified_blueprint, seed=42,
                        config=HU20RiverConfig(sweeps=250))
action = player.choose_action(acting_player_observation)
```

Do not pass engine state or hidden opponent cards. Share immutable full-profile
caches only between players using the same immutable source; never share live
player state/randomness across sessions. No LBR, training, new abstraction,
model promotion or automatic merge is included.
