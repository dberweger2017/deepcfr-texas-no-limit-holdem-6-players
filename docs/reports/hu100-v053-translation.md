# Frozen HU100 menu-v2 action translation for v0.5.3

The declared seven-opponent safeguard passes: no paired contrast's nominal
Student-t 95% upper bound is below −20 BB/100. This adopts the explicit bounded
adapter for this package; gains and the direct match remain descriptive.
The owner decides merge and separate publication.

## Frozen design and verification

Candidate: PR272's original seed 2026100601, fixed without seed selection.
The average is 3,196,291,551 bytes, SHA256
`c3a8edc4cc78098f07ac3e4ecbbb3dacae298a266b9fcd5fc788d0e2b6fcfc8e`.
The task's initial 3,488,922,486-byte count belonged to the earlier parent.
Checkpoint `401fe31263c763c5b805b0cff2f6af332d2dc7a3e9b7d8b99f1120b0216f58d6`,
iteration 1,794,658 /97,290,578 entries. All model/archive/manifest and K50 table
pins were verified on the M1; the comparator is the verified released v0.5.2.

Scientific source `3bfabb9d1d46f95544826f602686292acc2b538a`; canonical plan SHA256
`a2eec9a4ca9499f63df906ef61c8bcb4b205b410e2d65d99e9f994ea253cc5b4`. The full plan was posted and exactly read back before
final hands. [Protocol](../releases/v0.5.3/TRANSLATION_PROTOCOL.md),
[canonical plan](../releases/v0.5.3/verification/canonical-plan.json),
[global barrier](../releases/v0.5.3/verification/all-final-verified.json).
Every one of the 720,896 original final hands independently replays and exactly reproduces
before readout; raw hashes, source, roots, model identities and complete counts
are rechecked across all eight comparisons. Elapsed telemetry alone is omitted
from equality. Pilots of 1,024 direct blocks and 64 per opponent are cost-only and unpooled.
No count change, extension, failed-hand rerun or outcome-dependent selection.

Seven 16,384-block paired panels use the same average on minus off. Direct uses
131,072 duplicate blocks, v0.5.3 translation on against v0.5.2 translation on.
Root seeds are `{'direct': 20261015530103, 'panel': 20261015530104, 'pilot_direct': 20261015530101, 'pilot_panel': 20261015530102}`. Hands reset to 100 BB, blinds 50/100, no rake.
The adapter is `hu100-public-menu-translation-equity-k50-menu-v2`, 512 states /128 events. It changes public history lookup labels only and uses the policy's
own pinned K50 tables, legal witness menus and one unchanged sampling draw.
Exact/zero-mass behavior and unresolved uniform fallback remain. Search is absent.

## Final paired results

| Opponent | Translation on BB/100 [95%] | Off | Paired on minus off | Flag |
|---|---:|---:|---:|---|
| check_call | +250.41 [+233.97, +266.84] | +250.41 [+233.97, +266.84] | +0.00 [+0.00, +0.00] | False |
| half_pot_pressure | +17.48 [+5.05, +29.90] | +17.48 [+5.05, +29.90] | +0.00 [+0.00, +0.00] | False |
| loose_aggressive | +40.84 [+21.62, +60.06] | +40.84 [+21.62, +60.06] | +0.00 [+0.00, +0.00] | False |
| matched_pot_pressure | +26.39 [+14.18, +38.61] | +26.39 [+14.18, +38.61] | +0.00 [+0.00, +0.00] | False |
| pot_pressure | +40.02 [+27.41, +52.63] | +40.02 [+27.41, +52.63] | +0.00 [+0.00, +0.00] | False |
| random | +136.17 [+99.70, +172.65] | +136.17 [+99.70, +172.65] | +0.00 [+0.00, +0.00] | False |
| tight_aggressive | +29.52 [+19.99, +39.04] | +29.52 [+19.99, +39.04] | +0.00 [+0.00, +0.00] | False |

Direct descriptive v0.5.3 on minus translated v0.5.2:
**+37.10 [+30.16, +44.04] BB/100**.
Intervals condition on each fixed policy pair; this is no non-inferiority
certificate or uncertainty estimate over a training-seed population.

## Observed sizes and lookup work

| Opponent | On-arm raises /off menu | Off-arm raises /off menu |
|---|---:|---:|
| check_call | 0 /0 | 0 /0 |
| half_pot_pressure | 8,163 /0 | 8,163 /0 |
| loose_aggressive | 12,847 /0 | 12,847 /0 |
| matched_pot_pressure | 6,960 /0 | 6,960 /0 |
| pot_pressure | 6,623 /0 | 6,623 /0 |
| random | 16,550 /8,231 | 16,550 /8,231 |
| tight_aggressive | 3,252 /0 | 3,252 /0 |

Full per-arm/street counts and off-menu examples remain in the replay receipts.
Random alone bet off-menu: **8,231 raises per arm**, all all-ins outside the
conditional all-in menu. A separate fresh engine recount classifies every one.
The v2 adapter attempted 8,222 lookups against random and selected no witness;
unresolved histories kept their uniform fallback. The other six opponents
placed no off-menu raises. Every paired block return difference is zero. Thus this
safeguard result measures no v2 translation strength gain.

| Comparison | Policy | Decisions | Search attempts | Selected translations |
|---|---|---:|---:|---:|
| direct | v052-on | 588,717 | 95,393 | 81,783 |
| direct | v053-on | 574,886 | 0 | 0 |
| panel-check_call | v053-on | 117,126 | 0 | 0 |
| panel-half_pot_pressure | v053-on | 34,523 | 0 | 0 |
| panel-loose_aggressive | v053-on | 52,256 | 0 | 0 |
| panel-matched_pot_pressure | v053-on | 32,430 | 0 | 0 |
| panel-pot_pressure | v053-on | 31,873 | 0 | 0 |
| panel-random | v053-on | 48,933 | 8,222 | 0 |
| panel-tight_aggressive | v053-on | 27,725 | 0 | 0 |

The 81,783 selected translations in direct belong to the v0.5.2 baseline;
the v2 candidate selected none in this entire experiment.
[Per-street counts, modes and all-decision lookup timing](../releases/v0.5.3/verification/lookup-telemetry.csv),
[timing conditional on a selected translation](../releases/v0.5.3/verification/translated-only-lookup-seconds.json).
The v2 random search attempts took mean **0.383 ms**, median **0.254 ms**, p99 **1.419 ms**, max **3.125 ms** on the M1.
[Attempt-conditional per-street latency](../releases/v0.5.3/verification/attempted-lookup-seconds.json)
also records v0.5.2’s direct search cost. Other timing fields report seconds,
with mean/median/tails and observed maxima. These are observed working-laptop
latencies, without a portability or worst-case guarantee.
The [independent statistics recount](../releases/v0.5.3/verification/independent-statistics-recount.json)
reproduces every estimate, interval and safeguard with maximum absolute difference 0.
The [off-menu size recount](../releases/v0.5.3/verification/off-menu-size-classification.json)
retains exhaustive wager amount counts and all-in classification.
The complete telemetry retains state counts, bounds, distances and all-in suffix
changes. Zero selected translations means no selected-lookup latency sample.

## Context and limits

Merged PR296 evaluated v2 with translation off against the release: direct gains
were +47.04 [40.13,53.94], +40.95 [34.02,47.87], +45.42 [38.49,52.36] BB/100
across three fixed training seeds. No scripted matchup had a negative estimated
absolute return; half-pot and matched-pot margins were smaller. The original
matched-pot contrast versus the release was −26.57 [−40.79,−12.34]. Passing an
upper-bound safeguard does not establish 20 BB/100 non-inferiority.

Evidence covers internal model matches and the scripted panel only; no external
HU100 benchmark. On-menu controls cannot establish gains against arbitrary
human wagers. Real off-menu human functionality and all-in/restart/spectator
journal verification belong to the separate M1 package readiness record.
No external, human, multiplayer or exploitability performance is established.

The measured finite science allowance was 6,099.59 s:
fixed 255.20 s + scaled block work
3,184.47 s, multiplied by 1.25, plus 1,800 s
scientific closeout. Actual resource/runtime/archive receipts are recorded in
[readiness](../releases/v0.5.3/READINESS.md). All failures and cleanup/restoration
receipts remain indexed. No release, tag, publication, HU20 default change or
owner merge is performed by this work.

Final guarded play/replay/reproduction took 2,951.81 s; official readout 30.40 s,
independent statistics recount 15.25 s, size classification 10.18 s and
attempt-latency extraction 20.30 s. All fit the unchanged 6,099.59 s allowance.
