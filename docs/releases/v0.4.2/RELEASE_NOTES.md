# v0.4.2 — 10B-node opponent-sampled average

Owner-authorized stable Latest release; v0.4.0 and v0.4.1 remain available. The fixed first seed 2026100601 uses the same heads-up 20-BB linear-CFR opponent-sampled-average recipe as v0.4.1, trained to 10B nodes instead of 1B. Its inference export bytes are unchanged from #185 and retrieved from #188's canonical verified `research/package/` archive directory. Compact storage changes memory representation only.

The matched-seed three-lineage direct comparison is **+3.50 [+1.63, +5.37] BB/100**. Native pressure and severe-regression checks pass from [#185](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/185). Fresh bounded LBR in [#188](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/pull/188) is **−1.092 [−4.965, +2.782] BB/100** target-profit difference, clearing the unchanged −5 safeguard narrowly; half-width **3.874** meets ≤5. This establishes the declared safeguard, not an LBR improvement. All pilot/final actions and settlements independently audit. Historical and pilot hands are excluded from the final estimate.

The inference package has seven files: fixed 10B average export, `MODEL_CARD.md`, these notes, `release-manifest.json`, `catalog-manifest.json`, `SHA256SUMS` and `verify_v042_bundle.py`. After retrieving it, use Python 3.11:

```sh
python verify_v042_bundle.py . --expect-source TAGGED_COMMIT --require-publication
```

The expected source is the exact commit referenced by tag `v0.4.2` and `approved_release_source_commit` in the publication manifest. The verifier pins the first-seed model hash, lineage, game, extraction, original #188 preparation provenance, every asset and consistent explicit publication approval/source/tag binding. The fixed catalog identity manifest is separately byte-pinned by the runtime and recorded in the publication manifest, avoiding a source-commit self-reference. The released local table defaults to v0.4.2; spectator pairings and human sessions retain all three versions. Missing/corrupt pinned models fail rather than silently substitute an older release.

LBR is bounded, intervals are conditional on saved lineages, and the game remains two players at 20 BB with no rake. No full-game exploitability, human-strength, six-player or 100-BB claim follows. See the model card for all evidence and the inherited engine license disclosure.
