# HU20 equity-bucket abstraction tables

#149's strongest side result was that a 50-bucket equity abstraction fitted on *other* boards lost 0.39 BB on the held-out limped turn roots. That beat even a v1 strategy fitted to the same board (0.47 BB). Equity buckets keep information that v1's categorical card descriptors drop. This builds Pluribus-style buckets for every postflop situation, as lookup tables the trainer and the evaluators can share. Nothing here trains a policy.

## Method

`native/hu20-buckets` (Rust, MIT) builds the tables; `src/blueprint/equity_buckets.py` reads them.

- **Situations and symmetry:** a situation is a two-card holding plus a flop, turn or river board. Suit-isomorphic situations form one class, keyed by the sorted per-suit (holding ranks, board ranks) signatures. The class weight is its orbit size: 24 suit permutations divided by the stabilizer. On every street the weights sum exactly to the raw situation count, and the tests check this.
- **River:** each class's feature is equity against a uniform random opponent holding, with exact card removal and ties counted one half. It's computed once per canonical board for all its holdings, using sorted ranks and per-card counts.
- **Turn and flop:** each class's feature is the histogram of its river equities over every runout (46 river cards on the turn; 1,081 turn-and-river pairs on the flop), in 50 equal-width bins. This follows #149's turn feature at finer resolution. It is distribution-aware, not Pluribus's potential-aware next-street clustering.
- **Clustering:** weighted k-means with K = 50 and 200 per street. River clusters scalar equity; turn and flop use L1 between cumulative histograms (earth mover's distance) with mean centroid updates. Seeding is k-means++ on a weighted sample, then up to 100 Lloyd iterations; the seed is fixed.
- **Tables:** `<street>-k<K>.bin` holds the `HU20BKT1` header, the street code and K, then sorted 64-bit class hashes and `u16` buckets. A hash collision aborts the build. `summary.json` records class counts, weights, the objective per iteration, and each bucket's mass and mean equity.

## Checks before the full build

- **Evaluator:** all 133,784,560 seven-card hands give exactly the known category counts (0.85 s on four M1 threads). Across 200,000 random pairs of hands, its ordering never disagrees with the engine's `src.game.showdown.hand_value`.
- **River equity:** matches brute-force enumeration.
- **Small decks:** class weights sum to the exact raw situation counts on every street. 24-card and 32-card decks build end to end on the M1 in 1.2 s and 15 s. The Python reader finds every randomly drawn situation on every street of the 32-card build.
- **Python keys:** match the Rust builder bit for bit on golden cases.

## Full build

`native/hu20-buckets/pod_build.sh COMMIT` runs on a disposable Linux pod. It sizes threads from the cgroup quota, runs the tests including the full seven-card gate, builds 52-card tables at K = 50 and 200, and writes `SHA256SUMS`. Expect about 120M river, 13M turn and 1.3M flop classes, 12–15 GB peak memory, and well under an hour on about 24 CPUs. The tables total about 2.7 GB.

Retrieve `/workspace/hu20-buckets-out` to the M4, verify `SHA256SUMS`, then terminate the pod. Keep the tables out of Git: index them in RESULTS_INDEX and archive them to Drive.

## Next: validation before any training

Score the global (not corpus-fitted) tables on #149's 40 limped turn roots, held out, with the existing exact lock-only evaluator, against v1 (P = 0.66 BB) and #149's corpus-fitted equity witness (0.39 BB). If global buckets also beat v1 there, the abstraction change is worth a training run, using whatever trainer fix #162 indicates.
