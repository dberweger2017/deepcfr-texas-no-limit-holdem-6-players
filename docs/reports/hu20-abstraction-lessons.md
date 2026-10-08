# HU20 card abstraction: what earlier attempts taught us

Read this before changing the HU20 information key. The repo has tried to change or measure the v1 card abstraction five times; two real changes made play worse, for reasons that are now avoidable. This page records what happened, why, and the rules that follow. October 7, 2026.

## The record

| Work | What was tried | Result | Why |
|---|---|---|---|
| [Bucket coarseness](hu20-bucket-coarseness.md) | Model-free equity spread inside v1's card buckets | Common buckets mix very different hands; for example, river two pair on paired boards ranges from 15% to 82% equity | v1's descriptor `(category, top band, flush draw, straight draw, board paired)` drops kickers and secondary ranks |
| [#143 card v2](hu20-card-v2.md) | A finer descriptor (contribution, kicker, draw detail), 100M nodes, three lineages, current policy | **Worse on all 13 panels:** min-raise −151.30, native pressure −109.15, LBR −69.53 BB/100 | **About 7× the keys (1.50M → 10.5M), so visits per key fell from ~13.5 to ~1.95 at the same node budget.** The A/B measured sparsity, not the representation |
| [#144 DR2x2 C](dr2x2-ac-comparison.md) ([phase 0](hu20-history-factorial-phase0.md), [10M](hu20-history-10m-followup.md)) | Compress the betting history to make keys denser | **Worse:** bounded LBR C−A −31.60 [−52.42, −10.78] BB/100 | Density improved, but the lost history information cost more |
| [#145 exact turn](hu20-exact-turn-check.md) / [flop](hu20-exact-flop-check.md) | Exact solves with an equity-200 witness | Equity-200 retains about 37% of full-v1 loss on turn roots (headroom ratio 0.368); flop solving was blocked by memory | A constructed upper bound, not a trained result |
| [#149 board pooling](hu20-board-pooling.md) | Held-out witness built from exact per-root solves, pooled by 50 equity buckets | **0.3874 BB** held-out loss, vs 0.6567 for the v1 witness and 1.4194 for the blueprint | Also constructed from exact solves, not from CFR training |
| [#163 equity buckets](../hu20-equity-buckets.md) | Full-deck equity-histogram tables, K = 50 and 200 per street, k-means on earth mover's distance | Tables built and checked exactly | Not yet validated on #149's roots, and never trained |

## Since then

- **Training is cheap.** The native trainer ([#164](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/164)) is about 60–120× faster than Python, with exact parity: 1B nodes in about 10 min, 10B in about 1 h on one M4 core, at about 1.5 GB ([learning curve](hu20-learning-curve.md), [O@10B](hu20-o-10b.md)).
- **The average policy, with opponent-sampled averaging, is the production recipe** (v0.4.1). The current policy oscillates and gets worse with training ([trainer bench](hu20-trainer-bench.md)).
- **v1 now limits training.** Under v1, O plateaus by about 2B nodes: 2B, 5B and 10B are all about +4 BB/100 over 1B. Its turn/river loss stays near Q 0.5 (E ≈ 1.04 BB at 10B), far above the per-root solution (0.47 BB).
- **Direct matches decide; probes can mislead.** CFR+ ("0.4.0-shield") had the best LBR and bench numbers but lost directly to v0.4.0 ([#173](hu20-cfr-plus.md), [#175](hu20-floor-control.md)).
- **Exported averages load compactly** (about 50 bytes per key, [#186](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/186)), so larger key sets no longer block evaluation or the web table.

## Rules for the next abstraction change

1. **Compare at matched visits per key, or train each abstraction to its plateau.** Never compare a finer abstraction with v1 at the same node budget. Report the key count and the visits-per-key distribution for every run, and give each abstraction its own learning curve.
2. **Use the production recipe:** linear CFR, opponent-sampled average and average export. No current-policy comparisons and no regret floor.
3. **Change only the card part of the key.** Keep the ordered betting history, menus and action abstraction identical, so the comparison isolates card information.
4. **A witness isn't a trained result.** #149's 0.39 BB is an upper bound built from exact solves. Before any full-game run, train bucket keys on the turn/river bench and score them exactly on held-out boards. Only a trained improvement counts.
5. **Gate on direct play.** A full-game bucket model must beat the incumbent in a fresh, predeclared direct match, plus the existing arena release rule.
6. **Version the key schema.** Give it a new `abstraction` name, with exact Rust/Python key parity tests against `src/blueprint/equity_buckets.py`. Check the exports, loaders, web table, turn search (which takes ranges from blueprint keys) and LBR before claiming compatibility.
7. **Choose K on evidence.** K = 200 has more headroom (#145) but more keys. Pick it on the bench, at matched visits per key, not up front.

## Planned sequence

1. **Validate #163's full-deck tables on #149's 40 limped turn roots:** reproduce the held-out equity witness at K = 50 and 200.
2. **Add bucket keys to the native trainer** as a labeled alternative schema (card part only), with parity tests.
3. **Bench:** train v1 and bucket keys on #162's fixed turn roots, each to its plateau with visits per key reported, and score them exactly on held-out boards. Gate: bucket E clearly below v1's trained E at plateau.
4. **Full game, only if the bench passes:** three bucket lineages, each with its own learning curve, opponent-sampled average, then a direct match against the current release plus the arena rule.
