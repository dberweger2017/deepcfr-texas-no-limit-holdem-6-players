# v0.5.5 model card

## Intended use and identity

A local, unraked heads-up no-limit Hold'em opponent at equal **200 BB** stacks,
50/100 chip blinds and 0.01 table units per chip. Human restricted/free sizing
and self-play spectator use the same fixed inference identity. Other stack
depths, multiple opponents, rake, antes and search are outside this release.

#306's raw sequence-form blend combines three frozen 20B-node seeds:
2026101302, 2026101421 and 2026101422. For each key/action it sums
`fsum(Node.average_s / (T_s*(T_s+1)/2))`; absent keys contribute zero, then the
existing exporter normalizes each row. It does not average already normalized
policies. Regrets/visits are zero; synthetic iteration 1 and seed 2026101529701
identify an inference blend, not a trained seed or a resumable checkpoint.

- Model: `HU200-v055-three-seed-blend.average.gz`, 6,282,968,150 bytes,
  SHA256 `ce3ecc2cb8ebffa1fb3e0a7f11c79d0900b74008b55d85093e1915ff4dde6485`.
- 156,208,331 entries; schema `hu200-native-reopening-ordered-history-equity-k50-v1`.
- Source blend checkpoint SHA256 `59a6dd353fb4ef06d77a86783c7251521d8c9f8b92a785bb7e11a9ae724d4123`.
- Translation `hu200-public-menu-translation-equity-k50-v1`, 512 states /128 events.
  It searches public-history witnesses for off-menu wagers without changing real
  amounts or the random sampling stream. Unresolved missing keys and zero-mass
  rows play uniformly in the legal menu. There is no silent check/call override.
- Cards: PR163 equity-histogram K50 tables, pinned below.

| Table | Bytes | SHA256 |
|---|---:|---|
| flop-k50.bin | 12,867,944 | `084e243ded9fe93ad03cd59601f22f16c3210592f51e1e3646a60983b572c111` |
| turn-k50.bin | 139,600,524 | `c0bc6a8aa0535553118109d18a32d3b4dc6880e937c263cdc87472b1ae9f168f` |
| river-k50.bin | 1,231,562,564 | `70c1cb5ecf4cd292e4c89dc2200f7c8381100be3b0288ceeb0a12a3e83977629` |

## Evidence

Protocol v2 qualified this exact model against Slumbot at **−19.67 [−38.57,
−0.76] BB/100 over 30,000 fresh hands** (#318). The lower bound exceeds −50;
the upper bound below zero means it **loses to Slumbot by a small, detectable
margin**. The declared description is not an embarrassing opponent, not a
claim to beat Slumbot. All journals verify; no abandoned/rejected hands counted.

The first protocol failed on seed 2026101302 (#304): −23.08 [−56.53, +10.38],
while seeds 1421 and 1422 gave −10.66 [−40.53, +19.21] and −7.87 [−39.39, +23.65].
The recipe did not qualify under that declared rule. [#316's later AIVAT
analysis](https://github.com/dberweger2017/CFR-NLHE-6max/blob/v0.5.5/docs/reports/slumbot-aivat-real-runs.md) attributes that failure mostly
to card luck; it does not revise the raw-statistic verdict. Protocol v2 was
predeclared for the blend with fresh hands and a pooled decision statistic.

[#306's internal comparisons](https://github.com/dberweger2017/CFR-NLHE-6max/blob/v0.5.5/docs/reports/hu200-v055-blend-run.md) are
+3.61 [−1.15, +8.37] /+0.37 [−4.30, +5.05] /+1.85 [−2.83, +6.52] BB/100 over
the three seeds, all within noise. The [release notes](RELEASE_NOTES.md) give
the full scripted panel, including 1.5× pot pressure **−29.39 [−55.24, −3.53]**.
All 21 paired scripted contrasts include zero. Every one of 2,162,688 stored
primary hand records independently replays and reproduces before readout;
every model row audits. Fixed-model intervals do not establish a population gain.

Weaknesses found in #318 include flop folds (−12.9 BB/100) and big-blind play
(−32.8 ±26.7 BB/100). Neither these results nor the scripted wins establish
HU100, multiway, search, exploitability or professional-strength performance.

## Resources and reproducibility

The blend was **built and tested on a base 16 GB MacBook Pro M4**, with #306's
measured peak simultaneous family RSS **12,748,324,864 bytes** (12.75 GB /11.87 GiB).
The runtime builds a binary cache by streaming the verified model on first
start, then verifies and maps that cache read-only on every start. A 16 GB
machine is recommended. On Linux with one CPU core and local SSD storage,
first-start verification, reassembly and cache building took **20 min 15 s**
at **0.52 GB peak process RSS**; a fresh later start took **81 s** at **0.59 GB**.
The peaks include play and exclude operating-system file caching. Load time
varies by CPU and storage. Allow 25 GB free disk including the 9.33 GB derived
cache (9,331,096,671 bytes with its manifest). Cache corruption or model mismatch
stops startup rather than falling back to unverified data.
No model is trained, re-exported or selected for packaging. Six ordered <=1 GiB
parts reassemble to the pinned whole hash. The standalone verifier pins the
model, tables, translation and publication source/approval. The three public
documents ship byte for byte from the tagged source.

[INSTALL.md](INSTALL.md) describes download, reassembly, memory, disk and local
play. The separate non-Latest `hu200-frozen-20b-seeds` research pre-release
retains the three ingredients and their failed original qualification history.
Code is MIT; the upstream engine licensing question is recorded in the
[licence audit](https://github.com/dberweger2017/CFR-NLHE-6max/blob/v0.5.5/docs/releases/v0.4.0/LICENSE_AUDIT.md).
