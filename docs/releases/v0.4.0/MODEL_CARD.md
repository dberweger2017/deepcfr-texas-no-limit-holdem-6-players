# B100M HU20 fixed-first-seed inference export

**Release:** v0.4.0 research preview. **Artifact:** `B100M-HU20-current-seed-2026093001.json.gz`, 40,144,034 unchanged compressed bytes, SHA-256 `4534e7db2f69bedd54098b7eaa3c9bd82450838405ae162270a3b7684db9bedf`.

This is the **current-policy inference export** from training lineage B-2026093001 at 100,000,029 completed traversal nodes. It is tabular external-sampling CFR, not a neural checkpoint. It cannot resume training. The original retained path is `training/B-2026093001/current-100000000.json.gz` in the [#116 sealed recovery](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/blob/v0.4.0/docs/reports/hu20-scaling-m4-recovery.md); the [manifest](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/blob/v0.4.0/docs/reports/hu20-scaling-recovery-artifacts/final-manifest.json) records its size and hash. Full resumable checkpoints, raw hands and audits remain separately on M4 at the path in that report; this small release bundle does not contain them all.

| Identity | Value |
| --- | --- |
| Format | `holdem-hu20-native-reopening-blueprint-v1` |
| Game | `hu20-native-reopening-20bb-52card-no-ante-rake-v1` |
| Abstraction | `hu20-native-reopening-ordered-history-card-v1` |
| Seats | Two |
| Stacks | 2,000 chips = 20 BB each; reset each hand |
| Extraction | `current` |
| Restricted menu | Native reopening, no raise-count cap; selected abstract sizes, not arbitrary wagers |

The web service verifies bytes and identity before accepting play. Its **restricted** mode reproduces the trained menu. Its **free-sizing** mode applies arbitrary native-legal human raises exactly, but the bot still uses the original menu and missing-key fallback. The export is not a six-player, 100 BB or tournament model. It does not establish human strength, exact exploitability or the v0.5/v1.0 release criteria.

The [#116 report](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/blob/v0.4.0/docs/reports/hu20-scaling-m4-recovery.md) measures an aggregate over three saved lineages, not this first seed alone. Aggregate B100M-minus-own-B20M improvement was +25.94 BB/100 [97.5% +7.43, +44.45] against a bounded LBR, but B100M still earned −73.14 BB/100 in absolute terms against it. The [completed diagnostics](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/blob/v0.4.0/docs/reports/hu20-scaling-diagnostics.md) record mixed small secondary panels, frequent fallback against pot pressure and sparse late-street coverage. Do not apply aggregate estimates to this seed as a standalone strength estimate.

**Redistribution:** the repository's own code is MIT. The pinned `pokers` fork has no published upstream license grant. I license my changes in that fork under MIT, while the original authors' terms remain unverified pending direct confirmation; the upstream-derived engine is not represented as MIT. No engine binary or paper is bundled with this model. I approved publication of this fixed export with the canonical v0.4.0 release, subject to the reviewed/merged source and final checks.
