# v0.5.5 — HU200 three-seed blend vs Slumbot

v0.5.5 qualified under [protocol version 2](https://github.com/dberweger2017/CFR-NLHE-6max/blob/v0.5.5/docs/hu200-v055-acceptance-protocol-v2.md):
**−19.67 [−38.57, −0.76] BB/100 against Slumbot over 30,000 fresh hands**, 10,000
on each of three fresh accounts (#318). Its lower 95% bound clears the declared
−50 BB/100 threshold. In the protocol's wording it is not an embarrassing opponent,
but it **loses to Slumbot by a small, detectable margin**. It does not beat Slumbot.

The original protocol required every one of three training seeds to clear the
threshold. It failed on seed 2026101302 (#304), whose lower bound was −56.53.
[#316's AIVAT analysis](https://github.com/dberweger2017/CFR-NLHE-6max/blob/v0.5.5/docs/reports/slumbot-aivat-real-runs.md) attributes that
failure mostly to card luck: about −9 BB/100 for that seed. This later analysis
does not change the original failed verdict. Protocol v2 was declared before
any qualifying blend hand; exploratory hands were excluded.

The model is #306's one audited sequence-form blend of the frozen 20B seeds
2026101302, 2026101421 and 2026101422. Its 156,208,331 entries, pinned PR163 K50
card tables and public-history translation (512 states /128 events) are unchanged.
There was no retraining, re-export or model selection for this release.

## Internal results and limitations

[#306](https://github.com/dberweger2017/CFR-NLHE-6max/blob/v0.5.5/docs/reports/hu200-v055-blend-run.md) found blend-minus-seed results of
+3.61 [−1.15, +8.37], +0.37 [−4.30, +5.05] and +1.85 [−2.83, +6.52] BB/100,
respectively, over 131,072 duplicate blocks per seed. All are within noise;
no direct improvement is established. Its scripted panel used 16,384 paired
blocks per opponent, translation on throughout:

| Opponent | Blend BB/100 [nominal 95% interval] |
|---|---:|
| Check/call | +160.50 [148.05, 172.96] |
| Half-pot pressure | +62.89 [47.24, 78.53] |
| Loose aggressive | +120.78 [97.44, 144.12] |
| Matched-pot pressure | +64.95 [49.60, 80.30] |
| 1.5× pot pressure | **−29.39 [−55.24, −3.53]** |
| Random | +186.17 [111.34, 261.01] |
| Tight aggressive | +68.10 [56.60, 79.60] |

The adverse 1.5× pot result remains part of the evidence. None of the 21 paired
scripted blend-minus-seed contrasts establishes a difference. These intervals
condition on fixed policies and deals; they do not measure training-seed population uncertainty.

#318 identifies flop folds (−12.9 BB/100) and big-blind play (−32.8 ±26.7 BB/100)
as weaknesses. This release covers heads-up, equal 200 BB stacks only: no HU100,
multiway, rake, search or general professional-strength claim.

## Local play and assets

Human play supports restricted sizing and any legal free sizing; the spectator
shows self-play decisions. The runtime fixes 200 BB stacks and translation 512/128.
The HU20 default remains v0.4.2; existing HU100 releases remain available.

The blend was **built and tested on a base 16 GB MacBook Pro M4**, with #306's
measured simultaneous process-family peak RSS of **12,748,324,864 bytes**
(12.75 GB /11.87 GiB). That is the research pipeline peak, not a runtime-load measurement.
First start builds a low-memory binary play cache from the verified model.
Every start verifies its model binding and file hashes and maps it read-only;
later starts avoid parsing the 156 million JSON rows. Corrupt or mismatched
caches stop startup and are preserved for diagnosis. Allow 25 GB free disk,
including the 9.33 GB derived cache, and a 16 GB machine.
Two fresh CLI starts on Linux with one CPU core and local SSD storage measured
**20 min 15 s / 0.52 GB peak RSS** for first-start verification, reassembly and
cache building, then **81 s / 0.59 GB peak RSS** with the existing cache. These
process peaks include play; operating-system file caching is additional.
The measured cache occupies **9,331,096,671 bytes** including its manifest.
Load time varies by CPU and storage.
See [INSTALL.md](INSTALL.md) for disk/RAM requirements and the verifier commands.

The release contains six ordered model parts of at most 1 GiB, three K50 tables,
these three public documents, `release-manifest.json`, `SHA256SUMS` and the
stdlib-only `verify_hu200_bundle.py`. The whole model is **6,282,968,150 bytes**,
SHA256 `ce3ecc2cb8ebffa1fb3e0a7f11c79d0900b74008b55d85093e1915ff4dde6485`.
The separate `hu200-frozen-20b-seeds` research pre-release contains the ingredients;
it is not v0.5.5 and is not Latest.
