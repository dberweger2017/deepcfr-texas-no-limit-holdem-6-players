# v0.5.0 candidate model card

Unpublished candidate for usable heads-up 100 BB cash play, internal and scripted
evaluation only. No established external benchmark, professional-strength claim,
or publication approval. The release remains v0.4.2 until an owner-approved
publication workflow changes it.

## Exact identity

Fixed #207 opponent-sampled linear-CFR average, original training seed
**2026100601**, **1,000,002,065 actual nodes**, iteration **885307**,
**41,010,014 entries**. This is the existing export, not a re-extraction or the
best-scoring #215 seed. File `O1B-HU100-opponent-sampled-average-seed-2026100601.jsonl.gz`:
**1,173,264,021 bytes**, SHA256
`47d493c2ca0a750ffec8ba5490bd8fdec0a582e0cf2fe3e4309868f6ae620fa9`.
Checkpoint SHA256
`cca0b54a609f47c60b29e9fe5a920475a91ec641df2ff0e3ea48fdac615147ec`.
Scientific source `bd0e7a417064f736091dc2b667954b50becb4b69`;
[model index](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/blob/PACKAGE_SOURCE_COMMIT/docs/reports/native-hu100-growth-1b-artifacts/model-index.json).

Candidate identity **v0.5.0-candidate-pr207-translation-v1** includes
**hu100-public-menu-translation-v1, max_states=512, max_events=128, enabled**,
exactly #215's evaluated settings. Translation uses bounded public history
witnesses without hidden cards or policy randomness; real legal wagers remain
unchanged. Missing/zero-mass unsupported histories retain uniform fallback.
Average extraction is lifetime iteration opponent-sampled normalization;
card/history schema is `hu100-native-reopening-ordered-history-card-v1` and
format remains `holdem-hu100-stored-cfr-average-research-v1`. No search,
opponent adaptation or arbitrary stack scaling.

## Table and information

Two seats, 10,000 chips each, blinds 50/100, chip unit 0.01, no ante/rake,
uncapped no-limit betting, fixed reset each hand, alternating button. Human
restricted/free and self-play spectator modes are supported. Each bot sees only
its legal seat observation. Private journals retain deal/sampling state for
independent replay; human public responses exclude this state and hidden bot
cards. Spectator inspection is privileged to the viewer, not an agent input.
HU20/HU200 or mixed-depth combinations are rejected.

## Evidence and limits

[#207](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/blob/PACKAGE_SOURCE_COMMIT/docs/reports/native-hu100-growth-1b.md) supplies exact audited retained
bytes, recovery, evaluation and replay provenance. [#212](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/blob/PACKAGE_SOURCE_COMMIT/docs/hu100-local-play.md)
verified 80 retained-model runtime hands and durable recovery. Neither selects a
new policy or establishes general strength.

[#215](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/blob/PACKAGE_SOURCE_COMMIT/docs/reports/hu100-independent-stages.md) completed three fixed lineages
on fresh 4,096 paired blocks/opponent, with no sample extension. Loose growth
and pot translation pass practical support across all three seeds. The original
**overall recipe qualification failed**: seed 2026100902 tight growth
+32.54 [-5.11, 70.20] BB/100 at its predeclared 99.1667% adjusted interval is
inconclusive. A positive descriptive 95% interval cannot replace that decision.
The selected seed's translation gain is +90.15 [53.53,126.77] at the adjusted
interval; absolute translated pot result is **-9.33 [-39.10,20.44] BB/100**
(descriptive 95%). Other seeds' translated pot estimates are also negative with
intervals crossing zero. Profitability against pot pressure is unproven;
seed 2026100901 loose absolute profitability is also inconclusive. Reported
intervals concern these fixed policies/deals, not training-population uncertainty.

The revised v0.5.0 milestone requires usable HU100 and internal/scripted evidence,
not all-opponent profitability, a successful all-nine-contrast growth recipe,
or an external benchmark. Those failures remain explicit limitations, not
waived pass results. v0.8's every-seed opponent-profitability requirement and
later milestones remain unchanged. [Readiness](https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/blob/PACKAGE_SOURCE_COMMIT/docs/releases/v0.5.0/READINESS.md) maps each requirement
and records integration gates and owner decisions. #218's HU200 experiment is
independent, not a dependency. No new strength experiment is part of preparation.
