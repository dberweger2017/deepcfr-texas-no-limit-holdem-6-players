# Frozen HU200 20B seed model card

These are v0.5.5's three original ingredients, with no seed selection or new
training. Each is a frozen opponent-sampled average at the predeclared 20B-node
endpoint, paired with the exact raw training checkpoint needed for sequence-form
blending. Small node-count overshoots complete whole iterations. They cover
heads-up equal 200 BB stacks, no rake or ante, with schema
`hu200-native-reopening-ordered-history-equity-k50-v1` and PR163 K50 cards.

| Seed | Average bytes | Average SHA256 |
|---|---:|---|
| 2026101302 | 3,164,563,975 | `9128f71c602f18a9cb7219b18cd43ff4e3f2b7ed6a30f9babe7ed033840ac7b6` |
| 2026101421 | 3,162,671,080 | `c08d86a9615022df7015d34e5dc3de2705bbb8346ed6976bd72284e71d91f444` |
| 2026101422 | 3,162,064,602 | `b191c5ed8b943d138e6076fb4a1b0117c98964d92dd2d6253e157f696f691f6f` |

| Seed | Raw checkpoint bytes | Raw checkpoint SHA256 |
|---|---:|---|
| 2026101302 | 4,960,844,586 | `c5929bda17457911347e47cd77a83dd8b34618a5542731bed96ece1fb4288b6f` |
| 2026101421 | 4,957,752,665 | `c70e8e01b3b27c7fa8b8e0561bdbe5b77b321125de646f04614c82ecacacc676` |
| 2026101422 | 4,957,316,628 | `e265c384686f003521e49158c5162e57175d3b53e405140cb494d30a8ec9660c` |

The seed iterations are 16,670,844 /16,745,107 /16,827,728; entries are
85,373,907 /85,373,916 /85,373,928. Per-seat scheduled weights `T*(T+1)/2` are
138,958,528,171,590 /140,199,312,593,278 /141,586,223,234,856.
These pins come from #306's input proofs. K50 table bytes and hashes, translation
512/128 and the publication binding are fixed in the verifier and manifest.
Raw checkpoints are supplied for exact reconstruction; this package makes no
new training or resume guarantee and the table app does not load the seed collection.

## Evidence and failed qualification

Seed 2026101302 was the original v0.5.5 candidate. #304's Slumbot raw results
were −23.08 [−56.53, +10.38], −10.66 [−40.53, +19.21] and −7.87 [−39.39, +23.65]
BB/100 over 10,000 hands each. Protocol 1 failed on 1302 by **6.5 BB/100**;
the recipe **did not qualify under its declared rule**. #316's AIVAT analysis
attributes that mostly to card luck, without changing the verdict.

The later blend qualified under prospective protocol v2, not protocol 1:
−19.67 [−38.57, −0.76] BB/100 over 30,000 fresh Slumbot hands. It loses by a
small detectable margin. #318 identified flop-fold losses of −12.9 BB/100 and
big-blind results of −32.8 ±26.7. No claim to beat Slumbot is made.

#306's blend-minus-seed comparisons were +3.61 [−1.15, +8.37], +0.37 [−4.30,
+5.05] and +1.85 [−2.83, +6.52] BB/100, all within noise. Its scripted 1.5× pot
pressure result was −29.39 [−55.24, −3.53]. Fixed-policy intervals do not measure
training-seed population uncertainty or establish general strength.

## Exact reconstruction

The blend uses `fsum(raw Node.average_s / (T_s*(T_s+1)/2))`, absent keys zero,
followed by the frozen exporter. Average exports alone lose information and
cannot reproduce the raw blend exactly. The raw checkpoints are therefore part
of this public collection. The original configuration retains provenance paths;
`--input-dir` changes file lookup while preserving the output header byte for byte.

The target average is 6,282,968,150 bytes, SHA256
`ce3ecc2cb8ebffa1fb3e0a7f11c79d0900b74008b55d85093e1915ff4dde6485`.
[INSTALL.md](INSTALL.md) runs the repository's native blend tool, including fresh
seed-average reproduction and the final whole-model check, from these parts.
The blend was **built and tested on a base 16 GB MacBook Pro M4**; #306 measured
12,748,324,864 bytes peak simultaneous family RSS for its research pipeline.
No HU100, multiway, search, exploitability or professional-strength claim follows.
