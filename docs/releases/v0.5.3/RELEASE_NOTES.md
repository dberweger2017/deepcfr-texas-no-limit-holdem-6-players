# v0.5.3: frozen heads-up 100 BB K50 menu v2

Unpublished owner-review bundle. The owner decides merge and separately
permits publication. No release or tag is created by this preparation.

The candidate is PR272’s frozen 5B opponent-sampled average, original seed
2026100601, declared without seed selection. It uses K50 postflop cards,
97,290,578 frozen entries and min/half-pot/pot/1.5× pot/2× pot/conditional
all-in sizing, with duplicate targets removed.

Merged PR296 completed the three-seed evaluation with v2 translation off:
direct gains over translated v0.5.2 were +47.04 [40.13,53.94],
+40.95 [34.02,47.87] and +45.42 [38.49,52.36] BB/100. No scripted matchup
had a negative estimated absolute return. Margins against half-pot and
matched-pot pressure were smaller; the original matched-pot contrast versus
the release was −26.57 [−40.79,−12.34].

This package adds `hu100-public-menu-translation-equity-k50-menu-v2`, with
512 public states /128 history events and the policy’s own pinned tables.
The declared safeguard passes: seven 16,384-block paired on-minus-off
contrasts are exactly 0.00 [0.00,0.00] BB/100. The fresh 131,072-block direct
match, translation on for both models, is +37.10 [30.16,44.04] BB/100.
Every one of 720,896 original final hands independently replays and exactly
reproduces before readout; independent statistics agree exactly.

No v2 witness was selected in these matches. Random alone bet off-menu:
8,231 all-ins per arm outside the conditional menu, with 8,222 unresolved
lookup attempts. Other styles were on-menu. This establishes no measured
translation gain against human wagers; unresolved histories remain uniform.

The standalone verifier checks three ordered model parts (each at most 1 GiB),
the unchanged whole average and separate flop/turn/river K50 assets. Runtime
requires the explicit translation identity and rejects changed inputs.
The browser’s restricted human menu follows v2; free sizing accepts legal
integer-chip wagers. Old release manifests and v1 trace hashes stay identical.

Internal model matches and scripted opponents only; no external HU100
benchmark. Intervals condition on fixed seeds and policies. There is no
human, multiplayer or exploitability claim. HU20’s default remains v0.4.2.
[Science report](https://github.com/dberweger2017/CFR-NLHE-6max/blob/PACKAGE_SOURCE_COMMIT/docs/reports/hu100-v053-translation.md),
[measured package readiness and asset inventory](https://github.com/dberweger2017/CFR-NLHE-6max/pull/298), [install](INSTALL.md).
