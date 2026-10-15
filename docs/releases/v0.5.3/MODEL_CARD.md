# v0.5.3 candidate model card: heads-up 100 BB

**Unpublished owner-review bundle. Translation passes the declared safeguard.**
The owner declared PR272's original seed as the candidate, without seed selection.
No further training or export is part of this task.

Linear CFR with opponent-sampled averaging; seed 2026100601; 5,000,006,023
cumulative nodes, iteration 1,794,658; frozen 97,290,578-entry table.
Schema `hu100-native-reopening-ordered-history-equity-k50-menu-v2`.
K50 postflop bucket tables are the unchanged SHA-pinned PR163 assets.
The betting menu includes min, half-pot, pot, 1.5× pot, 2× pot and conditional
all-in. The stored normalized average is played; zero-mass and unresolved
missing keys use uniform menu probabilities. Search is not included.

Average: **3,196,291,551 bytes**, SHA256
`c3a8edc4cc78098f07ac3e4ecbbb3dacae298a266b9fcd5fc788d0e2b6fcfc8e`.
Checkpoint SHA256
`401fe31263c763c5b805b0cff2f6af332d2dc7a3e9b7d8b99f1120b0216f58d6`.
The task's initial byte count belonged to the earlier parent; archive verification
matched this average's exact whole hash. Downloads split the unchanged gzip
bytes into ordered parts; the standalone verifier checks every part and the whole.

Table: two seats, reset 10,000 chips each hand, blinds 50/100, chip unit 0.01,
no ante or rake. The policy sees only its seat's legal observation, public history
and private hole cards. Translation changes lookup history labels only; real
wagers stay untouched. The package’s explicit adapter
`hu100-public-menu-translation-equity-k50-menu-v2` uses at most 512 public states
and 128 history events, without hidden cards, deck access or extra RNG draws.

PR296's direct comparisons against v0.5.2 with translation on used 131,072
duplicate blocks per seed, with v2 translation off:

| Training seed | Frozen v2 minus v0.5.2 BB/100 [95%] |
|---|---:|
| 2026100601 (declared candidate) | +47.04 [40.13,53.94] |
| 2026101501 | +40.95 [34.02,47.87] |
| 2026101502 | +45.42 [38.49,52.36] |

The candidate's absolute scripted returns in the original panel were random
+106.91 [71.01,142.81], check_call +268.70 [252.02,285.37], tight_aggressive
+37.29 [27.87,46.71], loose_aggressive +34.87 [15.09,54.65], pot_pressure
+27.24 [15.27,39.21], half_pot_pressure +16.18 [2.60,29.75] and
matched_pot_pressure +20.46 [7.61,33.30] BB/100. Half-pot and matched-pot margins
are smaller. Relative to the release, the original matched-pot contrast is
−26.57 [−40.79,−12.34]; passing the historical upper-bound rule does not establish
20-BB/100 non-inferiority. All PR296 final hands independently replay/reproduce.

The new fixed-count experiment passes the declared safeguard: all seven paired
on-minus-off contrasts are **0.00 [0.00, 0.00] BB/100**, at 16,384 duplicate
blocks each. The fresh descriptive direct match, both models with translation
on, is **+37.10 [30.16, 44.04] BB/100** over 131,072 blocks. All 720,896 original
hands independently replay and exactly reproduce before readout; a separate
statistics recount agrees exactly. Pilots are unpooled.

The v2 adapter selected no witness in these matches. Random made 8,231 off-menu
raises per arm, all all-ins outside v2's conditional all-in menu; 8,222 attempted
lookups found no witness and kept uniform fallback. The other six opponents
used on-menu sizes. No translation strength gain is measured here. The
v0.5.2 baseline selected 81,783 translations in direct play. See the
[fixed protocol](https://github.com/dberweger2017/CFR-NLHE-6max/blob/PACKAGE_SOURCE_COMMIT/docs/releases/v0.5.3/TRANSLATION_PROTOCOL.md) and
[full report](https://github.com/dberweger2017/CFR-NLHE-6max/blob/PACKAGE_SOURCE_COMMIT/docs/reports/hu100-v053-translation.md).

Adoption uses exactly the declared rule: no opponent contrast’s nominal 95%
upper bound below −20 BB/100. This does not establish non-inferiority against
arbitrary opponents or benefit against human off-menu wagers.

Limits: scripted opponents only; no external HU100 benchmark; per-fixed-seed
intervals cover deal/action variation, not a training-seed population. Sparse
late-street training support remains despite denser retained keys; this is CFR
on a restricted frozen game. No general human, multiplayer or exploitability
claim. Measured M1 load time, peak RSS and human/spectator/restart journal evidence
are maintained in [PR298’s package readiness record](https://github.com/dberweger2017/CFR-NLHE-6max/pull/298). The HU20 default stays v0.4.2.
