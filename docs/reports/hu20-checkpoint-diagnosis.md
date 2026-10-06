# HU20 checkpoint diagnosis: averaging rule and regret floor

#175 found two separate effects:
- the CFR+ regret floor causes 0.4.0-shield's direct-match loss (shield vs T −14.26 BB/100);
- traverser-reach averaging causes the native-pressure drop (T ≈ shield ≈ 104 vs O ≈ R ≈ 128 BB/100).

This diagnosis compares the three seed-2026100601 1B-node checkpoints key by key, as their exports would play:
- **T:** linear CFR, traverser-reach average (`~/Local/hu20-native-average-20261005/training/T-2026100601-1000000000.json.gz`).
- **O:** its lockstep opponent-sampled partner.
- **F:** shield, regret floor 0 (`~/Local/hu20-cfr-plus-pr171-20261006/training/F-2026100601-1000000000.json.gz`).

T and O have bit-identical regrets and visits at every key T stores; the script checks this. So any O−T difference is the averaging rule alone. F ran 2,940,243 iterations against T's 2,126,271 for the same 1B nodes.

**Situations** come from the stored menu: facing a jam is exactly `fold, call`, facing a bet or raise has `fold` and more, and unopened starts with `check`.

**Bands** are T's traverser visits:
- `T-missing` holds keys that O stores only from opponent-sampled visits. T has no entry there, so its export plays uniformly.
- `F-only` holds keys only the floor run reached.

**Policies** follow the export rule: a zero-mass average plays uniformly.

Full per-key and visit-weighted tables: [summary.md](hu20-checkpoint-diagnosis-artifacts/summary.md), [summary.json](hu20-checkpoint-diagnosis-artifacts/summary.json). Script: `scripts/diagnose_hu20_checkpoints.py`.

## 1. Traverser-reach averaging leaves much of the table with no average

Share of keys whose exported average is uniform for lack of mass, visit-weighted:

| Situation, T visits | T zero-mass | O zero-mass |
|---|---:|---:|
| facing jam, 2–9 | 0.898 | 0.464 |
| facing jam, 10–99 | 0.520 | 0.154 |
| facing jam, 100–999 | 0.146 | 0.039 |
| facing bet, 10–99 | 0.430 | 0.089 |
| facing bet, 100–999 | 0.075 | 0.013 |
| unopened, 10–99 | 0.331 | 0.038 |

On top of these, O holds **407,793 facing-jam keys and 433,399 facing-bet keys that T lacks entirely** (`T-missing`). T plays all of them uniformly.

The traverser-reach average adds `t × own reach × policy`. External sampling explores every traverser action, including those its current policy gives probability exactly 0, so whole subtrees reached through such actions add nothing. Their keys have trained regrets but an empty average. The export then plays them uniformly, which facing a jam means folding 50%. The opponent-sampled average adds `t × policy` at every sampled opponent visit, so it has mass wherever the policy was actually played.

The effect on play is more folding:

| Situation, T visits | T fold | O fold |
|---|---:|---:|
| facing bet, 2–9 | 0.243 | 0.171 |
| facing bet, 10–99 | 0.160 | 0.104 |
| facing jam, 2–9 | 0.482 | 0.414 |
| facing jam, 10–99 | 0.394 | 0.363 |

That fits T winning about 24 BB/100 less against relentless minimum raises. Well-visited keys barely differ: the O−T distance is 0.04–0.05 at 1000+ visits.

## 2. The floor shifts common spots; it doesn't flatten sparse ones

The expected sparse-key mechanism is **not** what the data show:
- **Almost no all-zero regrets:** F has essentially no keys whose regrets are all zero (≤ 0.4% in every facing band).
- **F has fewer zero-mass keys than T, not more:** at 10–99 facing-jam visits, 0.5% against T's 52%.
- **F does not fold more than T at the same keys.** Facing jams it folds less at every band; facing bets it folds a little more only at well-visited keys:

| Situation, T visits | T fold | F fold | F−T distance |
|---|---:|---:|---:|
| facing jam, 100–999 | 0.422 | 0.388 | 0.180 |
| facing bet, 100–999 | 0.165 | 0.203 | 0.369 |
| facing bet, 1000+ | 0.303 | 0.339 | 0.254 |
| unopened, 100–999 | – | – | 0.376 |
| unopened, 1000+ | – | – | 0.229 |

- **The floor rewrites the strategy in common spots.** F−T distances of 0.23–0.42 at keys with hundreds or thousands of visits are several times the O−T distances there (0.05–0.19).

Shield's direct-match and pressure losses therefore come from a different strategy in frequently reached situations. The arena showed it re-raising more and getting into more raise wars. It is not caused by undertrained keys. The key-level view can't say which reached situations cost the chips. Replaying the arena hands to attach street, raise depth and payoff is batch 2.

## Implications

1. **Export rule.** The uniform fallback for zero-mass average keys discards trained regrets. Falling back to the key's current (regret-matched) policy instead would remove most of T's gap without retraining. It would also apply to any traverser-reach checkpoint. It is cheap to test: re-export T and play it directly against T, O and shipped R1.
2. **Averaging.** Opponent-sampled averaging, as O uses, already avoids the empty-average problem. It remains the preferred rule.
3. **Floor.** A zero floor changes the common-spot strategy for the worse in direct play. A Pluribus-style deep negative floor, which rarely binds, should behave like linear CFR. Its value is storage bounds and pruning, so it should be judged by direct play against O, not by bench Q or LBR.

Limits: one seed. Key-level means weight keys by training visits, not by how often play reaches them. Fold rates mix very different situations within a band.
