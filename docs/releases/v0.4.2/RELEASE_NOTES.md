# v0.4.2 candidate — 10B-node opponent-sampled average

Prepared for owner review; **no tag or release has been published**. The fixed first seed 2026100601 uses the same heads-up 20-BB linear-CFR opponent-sampled-average recipe as v0.4.1, trained to 10B nodes instead of 1B. Its inference export bytes are unchanged from #185. Compact storage changes memory representation only.

The matched-seed three-lineage direct comparison is **+3.50 [+1.63, +5.37] BB/100**. Native pressure and severe-regression checks pass from [#185](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/185). Fresh bounded LBR in [#188](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/188) is **−1.092 [−4.965, +2.782] BB/100** target-profit difference, clearing the unchanged −5 safeguard narrowly; half-width **3.874** meets ≤5. This establishes the declared safeguard, not an LBR improvement. All pilot/final actions and settlements independently audit. Historical and pilot hands are excluded from the final estimate.

The inference package has six files: fixed 10B average export, `MODEL_CARD.md`, these notes, `release-manifest.json`, `SHA256SUMS` and `verify_v042_bundle.py`. After retrieving it, use Python 3.11:

```sh
python verify_v042_bundle.py . --expect-source PREPARATION_COMMIT
```

The expected source is recorded in the preparation manifest and the archived verification receipt. The verifier pins the exact first-seed model hash, lineage, game, extraction, all assets and explicit publication hold. This preparation does not update runtime defaults or spectator release choices. Publication and any subsequent runtime integration require the owner's separate explicit go and an approved source binding. v0.4.1 remains Latest; v0.4.0 remains available.

LBR is bounded, intervals are conditional on saved lineages, and the game remains two players at 20 BB with no rake. No full-game exploitability, human-strength, six-player or 100-BB claim follows. See the model card for all evidence and the inherited engine license disclosure.
