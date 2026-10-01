# Model-free HU20 postflop bucket study

I inspect the current `_postflop` descriptor unchanged on M1. No saved models,
trainer state, M4 computation, new poker evaluation hands or policy intervention.
This is a uniform-card diagnostic, not a distribution of states reached by B100M.

Freeze root `202610030301`, 64 independently sampled boards for each street and
16 distinct uniform compatible own holdings per board: 1,024 rows per street,
3,072 total. Flop/turn/river are sampled separately, not reused prefixes.
Use `stream_seed(root,'test','deal',street,board_index)` for each board+holding
sample and `stream_seed(root,'test','opponent',street,board_index,holding_index)`
for equity simulation. Preserve exact boards, own cards and seeds in the output.

For flop/turn equity sample 512 independent compatible worlds per holding:
uniform opponent pair and uniform remaining board without replacement within
one world, ties worth half. River equity enumerates all 990 compatible pairs
exactly. This is HU showdown equity against a **uniform compatible range**,
without betting/folds/rake. Use the previously verified exact ranker, cross-check
small generated cases against the independent native-reference evaluator in tests.

Group by street and the actual descriptor `(category, top_band, flush_draw,
straight_draw, board_paired)`. Publish every observed bucket's count/share,
equity p10/median/p90, p90−p10 and min/max. Quantiles use NumPy's linear method.
Report concrete value/kicker collisions on identical boards within one bucket,
with both exact holdings, best-hand tuples and equity. Rank common buckets with
n≥20 by `share*(p90-p10)`; also retain small buckets with their denominators.
Use the largest same-board spread to choose one collision per bucket, with
stable card/index tie-breaking. This selection is explicitly descriptive.

Flop/turn Monte Carlo SE is at most 0.5/sqrt(512)≈0.0221 per holding; extreme
or selected pair differences have additional sampling/selection noise. Boards
are clusters (16 holdings each); do not treat rows as independent confidence
samples. Frequency is an empirical uniform-card share, not a trained-policy
occupancy estimate. Unobserved buckets are unmeasured, not absent in the game.

One sequential bounded worker. Fixed work has roughly 2.1 million worlds/pairs,
no outcome-based expansion. Report source, kernel/schema identity, runtime,
process RSS, full rows and checksums. Explain future distinctions a v2 study
would need, without designing, changing or training v2 in this PR.
